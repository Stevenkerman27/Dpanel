"""Verify bank orientation against the captured CAD mesh and the old exporter."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np

from panel.half_surface_mesh import build_half_surface
from panel.half_native import prepare_half_topology

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case", type=Path, required=True, help="te_bank_audit output directory")
parser.add_argument("--corrected", type=Path, help="Optional successful cad_half output directory for exact CLI comparison")
parser.add_argument("--base", default="b12a3ff")
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--logs", type=Path, help="Optional local regression/solve evidence directory")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
old_source = subprocess.run(["git", "show", args.base+":panel/half_surface_mesh.py"],
                            cwd=root, check=True, text=True, capture_output=True).stdout
old = types.ModuleType("old_half_surface_mesh")
exec(compile(old_source, "baseline_half_surface_mesh", "exec"), old.__dict__)
with np.load(args.case/"half_mesh_input.npz") as data:
    options = {key: data[key] for key in data.files}
try:
    old.build_half_surface(**options)
except ValueError as error:
    old_error = str(error)
    assert old_error == "TE cut must have distinct upper and lower banks"
else:
    raise AssertionError("Captured input no longer reproduces the original error")
result = build_half_surface(**options)
compared = ("coordinates", "faces", "wake_faces", "trailing_edges", "upper_te_faces", "lower_te_faces",
            "cut_edges", "cut_kind", "gamma_stations", "gamma_station_dof")
if args.corrected is not None:
    cli = json.loads((args.corrected/"surface.json").read_text(encoding="utf-8"))
    for key in compared:
        np.testing.assert_array_equal(result[key], cli[key])
topology = prepare_half_topology(
    result["coordinates"], result["faces"], result["wake_faces"], result["trailing_edges"],
    result["upper_te_faces"], result["lower_te_faces"], result["gamma_stations"],
    result["gamma_station_dof"], result["component_ids"], cut_edges=result["cut_edges"])
fixture = json.loads((root/"cases/panel_half/thick_wing_two_strip.json").read_text(encoding="utf-8"))
control = dict(points=fixture["coordinates"], wall_faces=fixture["faces"],
               wall_tags=fixture["component_ids"], wake_faces=fixture["wake_faces"],
               te_polyline=[[1., 0., 0.], [1., .5, 0.], [1., 1., 0.]],
               stations=fixture["gamma_stations"], contact_intervals=[],
               reference_area=fixture["reference_area"], reference_chord=fixture["reference_chord"],
               reference_point=fixture["moment_reference"], freestream=fixture["freestream"],
               density=fixture["density"], wake_end_x=5.)
assert old.build_half_surface(**control) == build_half_surface(**control)
receipt = dict(
    command=[sys.executable, *sys.argv], baseline=args.base, original_exception=old_error,
    raw_input_sha256=hashlib.sha256((args.case/"half_mesh_input.npz").read_bytes()).hexdigest(),
    old_source_sha256=hashlib.sha256(old_source.encode()).hexdigest(),
    new_source_sha256=hashlib.sha256((root/"panel/half_surface_mesh.py").read_bytes()).hexdigest(),
    corrected_CLI_fields_identical_to_frozen_mesh_replay=list(compared) if args.corrected is not None else None,
    original_control_dictionary_exactly_unchanged=True,
    exported_wall_faces=len(topology.faces), exported_te_edges=len(result["trailing_edges"]),
    physical_accuracy_accepted=False)
if args.logs is not None:
    current_log = (args.logs/"full_pytest.log").read_text(encoding="utf-8")
    previous_log = (root/"runs/verification/native_blas/fixed_pytest.log").read_text(encoding="utf-8")
    current_failed = {line.split(" - ")[0] for line in current_log.splitlines() if line.startswith("FAILED ")}
    previous_failed = {line.split(" - ")[0] for line in previous_log.splitlines() if line.startswith("FAILED ")}
    receipt["regression"] = dict(
        command=[sys.executable, "-m", "pytest", "-q", "test", "-p", "no:cacheprovider"],
        summary=current_log.splitlines()[-1], failed_ids=sorted(current_failed),
        new_failed_ids=sorted(current_failed-previous_failed),
        reference_log="runs/verification/native_blas/fixed_pytest.log",
        full_log_sha256=hashlib.sha256((args.logs/"full_pytest.log").read_bytes()).hexdigest())
    solve = json.loads((args.logs/"fixed_solve/result.json").read_text(encoding="utf-8"))
    receipt["fixed_solve"] = {key: solve[key] for key in ("CL", "CD", "CM", "panels", "unknowns", "residual")}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print("original error reproduced; frozen-mesh replay and control unchanged; topology valid; CLI compared=%s"
      % (args.corrected is not None))
