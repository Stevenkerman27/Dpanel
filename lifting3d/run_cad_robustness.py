"""Run explicit CAD geometry acceptance cases, preserving every attempt."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--size", type=float, default=.3)
parser.add_argument("--mach", type=float, default=0.)
parser.add_argument("--input-root", type=Path, default=root/"results"/"cad_inputs_v1")
parser.add_argument("--cases", nargs="+", default=[
    "naca0012_wing", "naca4412_wing", "naca0012_wing_body", "naca4412_wing_body",
    "naca4412_scaled_translated", "naca4412_taper060_sweep15"])
args = parser.parse_args()
assert not args.out.exists(), "Preserve previous experiments"
args.out.mkdir(parents=True)
(args.out/"run_cad_robustness.py.snapshot").write_bytes(Path(__file__).read_bytes())
records = []
for name in args.cases:
    inputs = (args.input_root/name).resolve()
    target = args.out/name
    geometry = {"wing-cad": str(inputs/"wing.step"),
                "trailing-edge": str(inputs/"trailing_edge.json"),
                "size": args.size, "wake-strips": 8, "radius": 5., "far-size": 1.5}
    if (inputs/"body.step").exists():
        geometry["body-cad"] = str(inputs/"body.step")
    case = dict(method="lifting-cad-p2", geometry=geometry, flow=dict(alpha=5., mach=args.mach),
                numerics=dict(ncpu=8, kutta="weak", quadrature=3))
    case_path = args.out/(name+".json")
    case_path.write_text(json.dumps(case, indent=2), encoding="utf-8")
    started = time.perf_counter()
    command = [sys.executable, str(root.parent/"aero.py"), "--case", str(case_path.resolve()),
               "--out", str(target.resolve()), "--export-surface"]
    run = subprocess.run(command, capture_output=True, text=True)
    (args.out/(name+".log")).write_text(run.stdout+run.stderr, encoding="utf-8")
    manifest = json.loads((target/"manifest.json").read_text(encoding="utf-8"))
    row = dict(case=name, returncode=run.returncode, status=manifest["status"],
               geometry_accepted=manifest.get("geometry_accepted", False),
               converged=manifest.get("converged", False), elapsed_seconds=time.perf_counter()-started)
    mesh_path = target/"mesh"/"mesh.json"
    if mesh_path.exists():
        mesh = json.loads(mesh_path.read_text(encoding="utf-8"))
        row.update(points=mesh["points"], tets=mesh["tets"],
                   contacts=mesh["wall_contact_intervals"],
                   min_dense_jacobian=mesh["min_dense_jacobian"],
                   mesh_sha256=mesh["mesh_sha256"])
    result_path = target/"solution"/"result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        row.update({key: result.get(key) for key in [
            "cl_pressure", "cl_jump", "weak_mass_residual_norm", "kutta_max",
            "potential_jump_max_error", "cp_min", "cp_max", "wall_normal_rms"]})
    records.append(row)
    (args.out/"results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(row), flush=True)
summary = dict(total=len(records), accepted=sum(row["returncode"] == 0 for row in records),
               criterion="CAD topology, finite subsonic field, pressure loads and raw surface export; no accuracy threshold",
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(args.out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
assert summary["accepted"] == summary["total"], "Inspect preserved failures"
