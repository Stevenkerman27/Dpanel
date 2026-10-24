"""One preregistered quadrature comparison on the saved thick wingbody mesh."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

root = Path(__file__).resolve().parent
baseline = root/"results"/"wingbody_matrix_v1"/"t024_r035_a090"
out = root/"results"/"wingbody_quadrature_v2"
assert not out.exists(), "Preserve earlier quadrature runs"
out.mkdir(parents=True)
protocol = (root/"QUADRATURE_PROTOCOL.md").read_bytes()
(out/"protocol.md").write_bytes(protocol)
(out/"driver.py.snapshot").write_bytes(Path(__file__).read_bytes())
q4_directory = root/"results"/"wingbody_quadrature_v1"/"solution"
q4_result = json.loads((q4_directory/"result.json").read_text())
assert q4_result["solver_sha256"] == hashlib.sha256((root/"solve.py").read_bytes()).hexdigest()
assert q4_result["p2_sha256"] == hashlib.sha256((root.parent/"volume"/"p2.py").read_bytes()).hexdigest()
stages = []
command = [sys.executable, str(root/"solve.py"), "--mesh", str(baseline/"mesh"/"mesh.npz"),
           "--alpha", "5", "--mach", "0", "--kutta", "weak", "--quad-order", "3",
           "--ncpu", "8", "--out", str(out/"q3")]
run = subprocess.run(command, capture_output=True, text=True)
(out/"q3_solve.log").write_text(run.stdout+run.stderr, encoding="utf-8")
stages.append(dict(command=command, returncode=run.returncode))
if run.returncode == 0:
    command = [sys.executable, str(root/"analyze.py"), "--mesh", str(baseline/"mesh"/"mesh.npz"),
               "--run", str(out/"q3")]
    run = subprocess.run(command, capture_output=True, text=True)
    (out/"analysis.log").write_text(run.stdout+run.stderr, encoding="utf-8")
    stages.append(dict(command=command, returncode=run.returncode))
records, rows = [], []
for name, directory in [("q3", out/"q3"), ("q4", q4_directory)]:
    result = json.loads((directory/"result.json").read_text())
    records.append(dict(test=name+"_converged", passed=bool(result["converged"])))
    if not result["converged"] or not (directory/"analysis.json").exists():
        continue
    analysis = json.loads((directory/"analysis.json").read_text())
    with np.load(directory/"field.npz") as field:
        finite = bool(np.isfinite(field["cp"]).all())
    records += [
        dict(test=name+"_finite_cp", passed=finite),
        dict(test=name+"_component_closure", passed=analysis["component_sum_error"] < 1e-10),
        dict(test=name+"_final_residual", passed=result["history"][-1]["residual_norm"] < 1e-9
                                                   and result["kutta_max"] < 1e-9)]
    rows.append(dict(name=name, cl=result["cl_pressure"], cl_jump=result["cl_jump"],
                     wall_rms=result["wall_normal_rms"], wall_max=result["wall_normal_max"],
                     junction_rms=analysis["regions"]["junction"]["normal_rms"],
                     wake=result["wake_diagnostics"], loads=result["component_loads"],
                     seconds=result["elapsed_seconds"], solver_sha256=result["solver_sha256"],
                     mesh_sha256=result["mesh_sha256"]))
if len(rows) == 2:
    assert rows[0]["solver_sha256"] == rows[1]["solver_sha256"], "Controlled comparison requires identical solver source"
    assert rows[0]["mesh_sha256"] == rows[1]["mesh_sha256"]
    difference = abs(rows[1]["cl"]-rows[0]["cl"])
    records.append(dict(test="pressure_cl_quadrature_difference", error=difference,
                        limit=.002, passed=difference < .002))
summary = dict(rows=rows, records=records, stages=stages,
               historical_q3_directory=str(baseline/"m0_a5"),
               reused_q4_directory=str(q4_directory),
               correction="v1 mixed historical and current solver sources; v2 recomputes q3 with the q4 source",
               protocol_sha256=hashlib.sha256(protocol).hexdigest(),
               passed=sum(row["passed"] for row in records), total=len(records))
(out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
assert len(rows) == 2 and all(row["passed"] for row in records)
