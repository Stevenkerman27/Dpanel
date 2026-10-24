"""Run prescribed tapered/swept planforms and symmetry checks, preserving all failed stages."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true", help="Reuse saved converged solutions after a reporting failure")
args = parser.parse_args()
out = root/"results"/"planforms_v1"
assert not out.exists() or args.resume, "Preserve previous planform evidence"
out.mkdir(parents=True, exist_ok=args.resume)
started = time.perf_counter()
records, rows, commands = [], [], []
for name, taper, sweep in [("taper", .5, 0.), ("sweep", 1., 20.), ("combined", .5, 20.)]:
    target = out/name
    target.mkdir(exist_ok=args.resume)
    mesh = target/"mesh"/"mesh.npz"
    stages = [
        ("mesh", [sys.executable, str(root/"build_mesh.py"), "--size", ".2", "--curvature", "32",
                  "--wake-strips", "24", "--taper", str(taper), "--sweep", str(sweep), "--out", str(mesh.parent)]),
        ("mesh_test", [sys.executable, str(root/"test_mesh.py"), "--mesh", str(mesh)]),
    ]
    for stage, command in stages:
        if stage == "mesh" and args.resume and mesh.exists():
            commands.append(dict(command=command, returncode=0, reused_existing_mesh=True))
            records.append(dict(test=name+"_"+stage, passed=True))
            continue
        run = subprocess.run(command, capture_output=True, text=True)
        (target/(stage+".log")).write_text(run.stdout+run.stderr, encoding="utf-8")
        commands.append(dict(command=command, returncode=run.returncode))
        records.append(dict(test=name+"_"+stage, passed=run.returncode == 0))
        if run.returncode:
            break
    if run.returncode:
        continue
    metadata = json.loads(mesh.with_name("mesh.json").read_text())
    records.append(dict(test=name+"_reference_area", passed=abs(metadata["reference_area"]-4.*(1.+taper)/2.) < 1e-12))
    for alpha in ([5., 0., -5.] if name == "combined" else [5.]):
        solve = target/("a"+str(alpha).replace("-", "m").replace(".", "p"))
        command = [sys.executable, str(root/"solve.py"), "--mesh", str(mesh), "--alpha", str(alpha),
                   "--kutta", "weak", "--ncpu", "8", "--out", str(solve)]
        reused = args.resume and (solve/"result.json").exists()
        if reused:
            assert json.loads((solve/"result.json").read_text())["converged"], "Do not reuse a failed solve"
            returncode = 0
        else:
            run = subprocess.run(command, capture_output=True, text=True)
            (target/(solve.name+".log")).write_text(run.stdout+run.stderr, encoding="utf-8")
            returncode = run.returncode
        commands.append(dict(command=command, returncode=returncode, reused_existing_solution=reused))
        result = json.loads((solve/"result.json").read_text())
        passed = bool(returncode == 0 and result["converged"])
        records.append(dict(test=name+"_"+solve.name+"_solve", passed=passed))
        if not passed:
            continue
        field = np.load(solve/"field.npz")
        records.append(dict(test=name+"_"+solve.name+"_residual_and_finite_cp",
                            passed=bool(result["history"][-1]["residual_norm"] < 1e-9
                                        and result["kutta_max"] < 1e-9 and np.isfinite(field["cp"]).all())))
        rows.append(dict(name=name, alpha=alpha, taper=taper, sweep=sweep,
                         nodes=result["nodes"], tets=result["elements"],
                         cl=result["cl_pressure"], cl_jump=result["cl_jump"], cd=result["cd_pressure"],
                         wall_rms=result["wall_normal_rms"], wall_max=result["wall_normal_max"],
                         cp_min=result["cp_min"], cp_max=result["cp_max"], wake=result["wake_diagnostics"],
                         mesh_sha256=result["mesh_sha256"], solver_sha256=result["solver_sha256"],
                         seconds=result["elapsed_seconds"]))
        if name == "combined" and alpha == 5.:
            baseline = result
        if alpha == 0.:
            error = float(max(abs(result["cl_pressure"]), abs(result["cl_jump"]), max(abs(np.array(result["potential_jump"])))))
            records.append(dict(test="combined_zero_lift", error=error, passed=error < 1e-9))
        if alpha == -5.:
            error = max(abs(result["cl_pressure"]+baseline["cl_pressure"]),
                        abs(result["cl_jump"]+baseline["cl_jump"]),
                        float(np.max(abs(np.array(result["potential_jump"])+baseline["potential_jump"]))))
            records.append(dict(test="combined_positive_negative_symmetry", error=error, passed=error < 1e-9))
        (out/"progress.json").write_text(json.dumps(dict(rows=rows, records=records, commands=commands), indent=2), encoding="utf-8")
summary = dict(rows=rows, records=records, commands=commands,
               passed=sum(row["passed"] for row in records), total=len(records),
               seconds=time.perf_counter()-started,
               scope="Planform geometry, area, algebra and symmetry; no absolute aerodynamic accuracy acceptance")
(out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(out/"driver.py.snapshot").write_bytes(Path(__file__).read_bytes())
print(json.dumps(summary, indent=2))
assert summary["passed"] == summary["total"]
