"""Fixed-topology CAD coordinate invariance, including body contact units."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--mesh", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
assert not args.out.exists(), "Preserve previous checks"
args.out.mkdir(parents=True)
original = dict(np.load(args.mesh))
metadata = json.loads(args.mesh.with_name("mesh.json").read_text(encoding="utf-8"))
assert metadata["geometry_family"] == "imported_cad_planar_te"
records = []
for name, scale, shift in [("base", 1., [0., 0., 0.]),
                           ("small_shifted", .001, [100., -200., 50.]),
                           ("large_shifted", 1000., [100., -200., 50.])]:
    target = args.out/name
    target.mkdir()
    mesh = dict(original)
    for key in ["points", "trailing_edge_points", "cad_original_coordinates"]:
        if key in original:
            mesh[key] = original[key]*scale+shift
    mesh["gamma_stations"] = original["gamma_stations"]*scale+shift[1]
    np.savez_compressed(target/"mesh.npz", **mesh)
    transformed = dict(metadata)
    transformed.update(
        chord=metadata["chord"]*scale, span=metadata["span"]*scale,
        reference_area=metadata["reference_area"]*scale**2,
        normalization_origin=(np.asarray(metadata["normalization_origin"])*scale+shift).tolist(),
        wall_contact_intervals=(np.asarray(metadata["wall_contact_intervals"]).reshape(-1, 2)*scale+shift[1]).tolist(),
        min_dense_jacobian=metadata["min_dense_jacobian"]*scale**3,
        mesh_sha256=hashlib.sha256((target/"mesh.npz").read_bytes()).hexdigest(),
        transform_source_mesh_sha256=hashlib.sha256(args.mesh.read_bytes()).hexdigest())
    (target/"mesh.json").write_text(json.dumps(transformed, indent=2), encoding="utf-8")
    audit_command = [sys.executable, str(root/"test_mesh.py"), "--mesh", str(target/"mesh.npz")]
    audit = subprocess.run(audit_command, capture_output=True, text=True)
    (target/"audit.log").write_text(audit.stdout+audit.stderr, encoding="utf-8")
    command = [sys.executable, str(root/"solve.py"), "--mesh", str(target/"mesh.npz"),
               "--alpha", "5", "--kutta", "weak", "--ncpu", "8", "--out", str(target/"solution")]
    run = subprocess.run(command, capture_output=True, text=True)
    (target/"solve.log").write_text(run.stdout+run.stderr, encoding="utf-8")
    result = json.loads((target/"solution"/"result.json").read_text(encoding="utf-8"))
    accepted = audit.returncode == run.returncode == 0 and result["converged"]
    row = dict(test=name, audit_command=audit_command, solve_command=command,
               geometry_accepted=audit.returncode == 0, converged=result["converged"], passed=accepted)
    if accepted:
        field = np.load(target/"solution"/"field.npz")
        if name == "base":
            base_cp, base_velocity = field["cp"].copy(), field["surface_velocity"].copy()
            base_gamma = np.asarray(result["potential_jump"])
        row.update(cp_max_difference=float(np.max(abs(field["cp"]-base_cp))),
                   velocity_max_difference=float(np.max(abs(field["surface_velocity"]-base_velocity))),
                   normalized_jump_max_difference=float(np.max(abs(np.asarray(result["potential_jump"])/scale-base_gamma))))
        row["passed"] = (row["cp_max_difference"] < 1e-6 and row["velocity_max_difference"] < 1e-6
                         and row["normalized_jump_max_difference"] < 1e-6)
    records.append(row)
    (args.out/"results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(row), flush=True)
(args.out/"driver.py.snapshot").write_bytes(Path(__file__).read_bytes())
assert all(row["passed"] for row in records)
