"""Controlled strip-resolution sequence; preserve each mesh, command and raw result."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--strips", type=int, nargs="+", default=[12, 24, 48])
parser.add_argument("--size", type=float, default=.15)
args = parser.parse_args()
assert not args.out.exists(), "Use a new output directory"
args.out.mkdir(parents=True)
started = time.perf_counter()
commands, rows, records = [], [], []
for strips in args.strips:
    target = args.out/("s"+str(strips))
    mesh = target/"mesh"/"mesh.npz"
    stages = [
        ("mesh", [sys.executable, str(root/"build_mesh.py"), "--size", str(args.size),
                  "--curvature", "32", "--wake-strips", str(strips), "--out", str(mesh.parent)]),
        ("mesh_test", [sys.executable, str(root/"test_mesh.py"), "--mesh", str(mesh)]),
        ("solve", [sys.executable, str(root/"solve.py"), "--mesh", str(mesh),
                   "--alpha", "5", "--ncpu", "8", "--out", str(target/"solve")]),
    ]
    target.mkdir(parents=True)
    for name, command in stages:
        tick = time.perf_counter()
        run = subprocess.run(command, capture_output=True, text=True)
        (target/(name+".log")).write_text(run.stdout+run.stderr, encoding="utf-8")
        commands.append(dict(stage=name, strips=strips, command=command, returncode=run.returncode,
                             seconds=time.perf_counter()-tick))
        if run.returncode:
            break
    result_path = target/"solve"/"result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    records.append(dict(test=f"s{strips}_mesh_and_solver_status",
                        passed=bool(run.returncode == 0 and result.get("converged"))))
    if result.get("converged"):
        rows.append(dict(strips=strips, nodes=result["nodes"], tets=result["elements"],
                         cl=result["cl_pressure"], cl_jump=result["cl_jump"], cd=result["cd_pressure"],
                         wall_rms=result["wall_normal_rms"], wall_max=result["wall_normal_max"],
                         cp_min=result["cp_min"], cp_max=result["cp_max"],
                         wake=result["wake_diagnostics"], seconds=result["elapsed_seconds"],
                         mesh_sha256=result["mesh_sha256"]))
    (args.out/"progress.json").write_text(json.dumps(dict(commands=commands, rows=rows), indent=2), encoding="utf-8")
if len(rows) == 3 and args.strips == [12, 24, 48]:
    first_change = abs(rows[1]["cl"]-rows[0]["cl"])
    last_change = abs(rows[2]["cl"]-rows[1]["cl"])
    records.append(dict(test="CL_strip_refinement_last_change_below_point005",
                        error=last_change, passed=last_change < .005))
    records.append(dict(test="CL_strip_refinement_changes_decrease",
                        first_change=first_change, last_change=last_change, passed=last_change < first_change))
summary = dict(rows=rows, records=records, commands=commands,
               passed=sum(row["passed"] for row in records), total=len(records),
               elapsed_seconds=time.perf_counter()-started,
               driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               scope="Strip discretization and meshing consistency only; no absolute aerodynamic accuracy")
(args.out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(args.out/"driver.py.snapshot").write_bytes(Path(__file__).read_bytes())
print(json.dumps(summary, indent=2))
if summary["passed"] != summary["total"]:
    raise SystemExit(2)
