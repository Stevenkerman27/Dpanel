"""Sequential, reproducible general-airfoil validation; preserve every subprocess outcome."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

root = Path(__file__).resolve().parent
out = root/"results"/"general_v1"
meshes = root/"meshes"/"general_v1"
out.mkdir(parents=True, exist_ok=True)
meshes.mkdir(parents=True, exist_ok=True)
assert not (out/"summary.json").exists(), "Choose a new study directory to preserve prior evidence"
records = []
commands = []
started = time.perf_counter()
for name, options in [
    ("mapping", [str(root/"test_mapping.py")]),
    ("p2_patch", [str(root/"test_p2.py")]),
]:
    command = [sys.executable]+options
    run = subprocess.run(command, capture_output=True, text=True)
    (out/(name+".log")).write_text(run.stdout+run.stderr, encoding="utf-8")
    commands.append(dict(name=name, command=command, returncode=run.returncode))
    records.append(dict(test=name, passed=run.returncode == 0))

for shape, size, name in [
    ("joukowski", .08, "j08"), ("joukowski", .04, "j04"), ("joukowski", .02, "j02"),
    ("trefftz", .04, "t04"), ("trefftz", .02, "t02"),
    ("naca", .04, "n12_04"), ("naca", .02, "n12_02"),
    ("naca", .04, "n24_04"), ("naca", .02, "n24_02"),
]:
    command = [sys.executable, str(root/"build_mesh.py"), "--shape", shape, "--size", str(size),
               "--mirror", "--thickness", ".24" if "n24" in name else ".12", "--out", str(meshes/name)]
    run = subprocess.run(command, capture_output=True, text=True)
    (out/(name+"_mesh.log")).write_text(run.stdout+run.stderr, encoding="utf-8")
    commands.append(dict(name=name+"_mesh", command=command, returncode=run.returncode))
    records.append(dict(test=name+"_mesh", passed=run.returncode == 0))
    if run.returncode:
        continue
    command = [sys.executable, str(root/"test_p2.py"), "--mesh", str(meshes/name/"mesh.npz")]
    run = subprocess.run(command, capture_output=True, text=True)
    (out/(name+"_topology.log")).write_text(run.stdout+run.stderr, encoding="utf-8")
    commands.append(dict(name=name+"_topology", command=command, returncode=run.returncode))
    records.append(dict(test=name+"_topology", passed=run.returncode == 0))

cases = []
for name in ["j08", "j04", "j02", "t04", "t02"]:
    for alpha in [-5., 0., 5.]:
        cases.append((name, alpha, 0., "exact", []))
for name in ["n12_04", "n12_02", "n24_04", "n24_02"]:
    for alpha, mach in [(0., 0.), (5., 0.), (5., .1), (5., .2)]:
        cases.append((name, alpha, mach, "uniform", []))
cases += [("j02", 5., 0., "exact", ["--quadrature", "6"]),
          ("n12_02", 5., .8, "uniform", []),
          ("j02", 5., 0., "exact", ["--center", ".2", "0"])]
results = {}
for index, (name, alpha, mach, boundary, options) in enumerate(cases):
    case_name = f"{index:02d}_{name}_a{alpha:g}_m{mach:g}"
    target = out/case_name
    command = [sys.executable, str(root/"solve.py"), "--mesh", str(meshes/name/"mesh.npz"),
               "--alpha", str(alpha), "--mach", str(mach), "--boundary", boundary,
               "--ncpu", "8", "--out", str(target)]+options
    run = subprocess.run(command, capture_output=True, text=True)
    target.mkdir(parents=True, exist_ok=True)
    (target/"run.log").write_text(run.stdout+run.stderr, encoding="utf-8")
    commands.append(dict(name=case_name, command=command, returncode=run.returncode))
    result_path = target/"result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    results[case_name] = result
    expected_sonic_rejection = mach == .8
    passed = (run.returncode == 2 and "sonic" in result.get("status", "")) if expected_sonic_rejection else (
        run.returncode == 0 and result.get("converged", False))
    records.append(dict(test=case_name+"_status", passed=bool(passed),
                        status=result.get("status"), expected_sonic_rejection=expected_sonic_rejection))
    if result.get("converged") and name in ["j02", "t02"] and mach == 0. and not options:
        lift_error = abs(result["cl_pressure"]-result["cl_exact"])
        circulation_error = abs(result["cl_circulation"]-result["cl_exact"])
        tolerance = .01*abs(result["cl_exact"]) if abs(alpha) > 0. else 1e-7
        records.append(dict(test=case_name+"_physical_gates",
                            passed=bool(lift_error < tolerance and circulation_error < tolerance
                                        and result["cp_rmse_exact"] < .03 and abs(result["cd_pressure"]) < .005),
                            lift_absolute_error=lift_error, circulation_absolute_error=circulation_error,
                            lift_tolerance=tolerance, cp_rmse=result["cp_rmse_exact"]))
    print(case_name, result.get("status", "missing_result"), flush=True)

for name in ["j08", "j04", "j02", "t04", "t02", "n12_04", "n12_02", "n24_04", "n24_02"]:
    subset = [value for value in results.values() if value.get("converged") and value["mach"] == 0.
              and not value["quadrature_order"] == 6
              and value.get("mesh_sha256") == hashlib.sha256((meshes/name/"mesh.npz").read_bytes()).hexdigest()]
    zero = [value for value in subset if value["alpha_deg"] == 0.]
    if zero:
        records.append(dict(test=name+"_zero_lift", error=abs(zero[0]["cl_pressure"]),
                            passed=abs(zero[0]["cl_pressure"]) < 1e-7))
    negative = [value for value in subset if value["alpha_deg"] == -5.]
    positive = [value for value in subset if value["alpha_deg"] == 5.]
    if negative and positive:
        error = abs(negative[0]["cl_pressure"]+positive[0]["cl_pressure"])
        records.append(dict(test=name+"_angle_symmetry", error=error, passed=error < 1e-7))

summary = dict(records=records, commands=commands, results=results,
               passed=sum(row["passed"] for row in records), total=len(records),
               elapsed_seconds=time.perf_counter()-started,
               driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(out/"run_study.py.snapshot").write_bytes(Path(__file__).read_bytes())
print(f"{summary['passed']}/{summary['total']} checks passed", flush=True)
assert summary["passed"] == summary["total"], "Retain failures; diagnose without changing old gates"
