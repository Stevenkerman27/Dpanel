"""Fixed-mesh coordinate, symmetry, thread-count and rejection checks for the lifting prototype."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

root = Path(__file__).resolve().parent
out = root/"results"/"invariance_v1"
out.mkdir(parents=True, exist_ok=True)
assert not (out/"summary.json").exists()
source = root/"meshes"/"wing03"/"mesh.npz"
source_mesh = dict(np.load(source))
source_metadata = json.loads(source.with_name("mesh.json").read_text())
records, results, commands = [], {}, []
for name, scale, shift, alpha, ncpu in [
    ("base", 1., [0., 0., 0.], 5., 8),
    ("small_shifted", .001, [100., -200., 50.], 5., 8),
    ("large_shifted", 1000., [100., -200., 50.], 5., 8),
    ("zero", 1., [0., 0., 0.], 0., 8),
    ("negative", 1., [0., 0., 0.], -5., 8),
    ("one_cpu", 1., [0., 0., 0.], 5., 1),
]:
    target = out/name
    target.mkdir(parents=True, exist_ok=True)
    mesh = dict(source_mesh)
    mesh["points"] = source_mesh["points"]*scale+shift
    mesh["gamma_stations"] = source_mesh["gamma_stations"]*scale+shift[1]
    np.savez_compressed(target/"mesh.npz", **mesh)
    metadata = dict(source_metadata)
    metadata.update(chord=scale, span=source_metadata["span"]*scale,
                    reference_area=source_metadata["reference_area"]*scale**2,
                    normalization_origin=shift, transform_source_mesh_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (target/"mesh.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    command = [sys.executable, str(root/"solve.py"), "--mesh", str(target/"mesh.npz"), "--alpha", str(alpha),
               "--ncpu", str(ncpu), "--out", str(target/"solve")]
    run = subprocess.run(command, capture_output=True, text=True)
    (target/"run.log").write_text(run.stdout+run.stderr, encoding="utf-8")
    commands.append(dict(command=command, returncode=run.returncode))
    result = json.loads((target/"solve"/"result.json").read_text())
    results[name] = result
    records.append(dict(test=name+"_status", passed=bool(run.returncode == 0 and result["converged"])))
    if not result["converged"]:
        continue
    field = np.load(target/"solve"/"field.npz")
    if name == "base":
        baseline = result
        base_cp = field["cp"].copy()
    if alpha == 5.:
        cp_error = float(np.max(abs(field["cp"]-base_cp)))
        cl_error = abs(result["cl_pressure"]-baseline["cl_pressure"])
        gamma_error = float(np.max(abs(np.array(result["potential_jump"])/scale-baseline["potential_jump"])))
        mass_error = max(abs(result["raw_mass_by_tag"][tag]/scale**2-baseline["raw_mass_by_tag"][tag])
                         for tag in baseline["raw_mass_by_tag"])
        records.append(dict(test=name+"_coordinate_and_units", cp_error=cp_error, cl_error=cl_error,
                            gamma_error=gamma_error, raw_mass_error=mass_error,
                            passed=bool(cp_error < 1e-6 and cl_error < 1e-7 and gamma_error < 1e-7 and mass_error < 1e-6)))
    if alpha == 0.:
        records.append(dict(test="zero_lift", error=abs(result["cl_pressure"]), passed=abs(result["cl_pressure"]) < 1e-9))
    if alpha == -5.:
        error = abs(result["cl_pressure"]+baseline["cl_pressure"])
        records.append(dict(test="positive_negative_symmetry", error=error, passed=error < 1e-9))

target = out/"invalid_nan"
target.mkdir(parents=True, exist_ok=True)
mesh = dict(source_mesh)
mesh["points"] = source_mesh["points"].copy()
mesh["points"][0, 0] = np.nan
np.savez_compressed(target/"mesh.npz", **mesh)
(target/"mesh.json").write_text(json.dumps(source_metadata), encoding="utf-8")
command = [sys.executable, str(root/"solve.py"), "--mesh", str(target/"mesh.npz"), "--out", str(target/"solve")]
run = subprocess.run(command, capture_output=True, text=True)
(target/"run.log").write_text(run.stdout+run.stderr, encoding="utf-8")
failure = json.loads((target/"solve"/"result.json").read_text())
commands.append(dict(command=command, returncode=run.returncode))
records.append(dict(test="nonfinite_mesh_explicit_rejection",
                    passed=bool(run.returncode == 2 and "finite coordinates" in failure.get("message", "")),
                    failure_message=failure.get("message")))
summary = dict(records=records, commands=commands, results=results,
               passed=sum(row["passed"] for row in records), total=len(records),
               scope="Algebra, coordinate units, symmetry, rejection; no absolute 3D aerodynamic accuracy certification")
(out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(out/"driver.py.snapshot").write_bytes(Path(__file__).read_bytes())
print(json.dumps(records, indent=2))
assert summary["passed"] == summary["total"]
