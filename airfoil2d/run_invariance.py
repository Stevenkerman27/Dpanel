"""Coordinate and input-path robustness checks; use a fixed mesh to isolate the PDE transformation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

root = Path(__file__).resolve().parent
out = root/"results"/"invariance_v1"
out.mkdir(parents=True, exist_ok=True)
assert not (out/"summary.json").exists(), "Preserve the previous study"
source = root/"meshes"/"j02_mirror"/"mesh.npz"
source_metadata = json.loads(source.with_name("mesh.json").read_text())
source_mesh = dict(np.load(source))
records = []
results = []
commands = []
for label, scale, shift in [
    ("base", 1., [0., 0.]), ("small_translated", 1e-3, [100., -200.]),
    ("large_translated", 1e3, [100., -200.]),
]:
    target = out/label
    target.mkdir(parents=True, exist_ok=True)
    mesh = dict(source_mesh)
    mesh["points"] = source_mesh["points"]*scale+shift
    mesh["contour"] = source_mesh["contour"]*scale+shift
    np.savez_compressed(target/"mesh.npz", **mesh)
    metadata = dict(source_metadata)
    metadata.update(chord=source_metadata["chord"]*scale,
                    farfield_center=(np.array(source_metadata["farfield_center"])*scale+shift).tolist(),
                    normalization_origin=(np.array(source_metadata["farfield_center"])*scale+shift).tolist(),
                    coordinate_scale=scale, coordinate_translation=shift,
                    far_radius=source_metadata["far_radius"]*scale,
                    transform_source_mesh_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (target/"mesh.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    command = [sys.executable, str(root/"solve.py"), "--mesh", str(target/"mesh.npz"),
               "--boundary", "exact", "--out", str(target/"solve")]
    run = subprocess.run(command, capture_output=True, text=True)
    (target/"run.log").write_text(run.stdout+run.stderr, encoding="utf-8")
    commands.append(dict(command=command, returncode=run.returncode))
    result = json.loads((target/"solve"/"result.json").read_text())
    field = np.load(target/"solve"/"field.npz")
    results.append(result)
    if label == "base":
        base = result
        base_cp = field["cp"].copy()
    cp_error = float(np.max(abs(field["cp"]-base_cp)))
    lift_error = abs(result["cl_pressure"]-base["cl_pressure"])
    normal_error = abs(result["wall_normal_rms"]-base["wall_normal_rms"])
    circulation_error = abs(result["circulation_ccw"]/scale-base["circulation_ccw"])
    records.append(dict(test=label+"_fixed_mesh", cp_error=cp_error, lift_error=lift_error,
                        normal_rms_error=normal_error, circulation_error=circulation_error,
                        passed=bool(run.returncode == 0 and cp_error < 1e-7 and lift_error < 1e-8
                                    and normal_error < 1e-7 and circulation_error < 1e-7)))

contour = source_mesh["contour"]*1e-3+[100., -200.]
np.savetxt(out/"transformed_contour.dat", contour)
command = [sys.executable, str(root/"build_mesh.py"), "--shape", "file", "--airfoil",
           str(out/"transformed_contour.dat"), "--size", ".02", "--mirror", "--out", str(out/"file_mesh")]
run = subprocess.run(command, capture_output=True, text=True)
(out/"file_mesh.log").write_text(run.stdout+run.stderr, encoding="utf-8")
commands.append(dict(command=command, returncode=run.returncode))
records.append(dict(test="translated_small_external_file_mesh", passed=run.returncode == 0))
if run.returncode == 0:
    command = [sys.executable, str(root/"test_p2.py"), "--mesh", str(out/"file_mesh"/"mesh.npz")]
    run = subprocess.run(command, capture_output=True, text=True)
    (out/"file_topology.log").write_text(run.stdout+run.stderr, encoding="utf-8")
    commands.append(dict(command=command, returncode=run.returncode))
    records.append(dict(test="external_file_mesh_topology", passed=run.returncode == 0))
    for label, mesh_path in [("original_uniform", source), ("file_uniform", out/"file_mesh"/"mesh.npz")]:
        command = [sys.executable, str(root/"solve.py"), "--mesh", str(mesh_path), "--out", str(out/label)]
        run = subprocess.run(command, capture_output=True, text=True)
        (out/(label+".log")).write_text(run.stdout+run.stderr, encoding="utf-8")
        commands.append(dict(command=command, returncode=run.returncode))
        records.append(dict(test=label+"_status", passed=run.returncode == 0))
    original = json.loads((out/"original_uniform"/"result.json").read_text())
    external = json.loads((out/"file_uniform"/"result.json").read_text())
    error = abs(original["cl_pressure"]-external["cl_pressure"])
    records.append(dict(test="external_remesh_lift", absolute_error=error, tolerance=.002, passed=error < .002))

np.savetxt(out/"crossed.dat", [[1., 0.], [.8, .1], [.6, -.1], [.3, -.08],
                              [0., 0.], [.3, .08], [.6, .1], [.8, -.1]])
for label, command, expected in [
    ("self_intersection", [sys.executable, str(root/"build_mesh.py"), "--shape", "file",
                          "--airfoil", str(out/"crossed.dat"), "--out", str(out/"crossed")],
     "Self-intersecting contour"),
    ("outside_center", [sys.executable, str(root/"solve.py"), "--mesh", str(source),
                        "--center", "10", "10", "--out", str(out/"outside")],
     "Circulation center must lie inside the body"),
]:
    run = subprocess.run(command, capture_output=True, text=True)
    (out/(label+".log")).write_text(run.stdout+run.stderr, encoding="utf-8")
    commands.append(dict(command=command, returncode=run.returncode))
    path = out/("crossed" if label == "self_intersection" else "outside")/"failure.json"
    failure = json.loads(path.read_text()) if path.exists() else {}
    records.append(dict(test=label+"_explicit_rejection", passed=bool(run.returncode != 0
                        and expected in failure.get("message", "")), failure=failure))
summary = dict(records=records, commands=commands, results=results,
               passed=sum(row["passed"] for row in records), total=len(records))
(out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(out/"driver.py.snapshot").write_bytes(Path(__file__).read_bytes())
print(json.dumps(records, indent=2))
assert summary["passed"] == summary["total"]
