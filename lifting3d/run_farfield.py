"""Farfield sequence with a fixed outer element size; remeshing remains an explicit uncertainty."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--radius", type=float, nargs="+", default=[8., 12.])
args = parser.parse_args()
assert not args.out.exists(), "Preserve previous outputs"
args.out.mkdir(parents=True)
commands, records, rows = [], [], []
started = time.perf_counter()
for radius in args.radius:
    target = args.out/("r"+str(radius).replace(".", "p"))
    target.mkdir()
    mesh = target/"mesh"/"mesh.npz"
    stages = [
        ("mesh", [sys.executable, str(root/"build_mesh.py"), "--size", ".15", "--curvature", "32",
                  "--radius", str(radius), "--far-size", str(5./3.), "--wake-strips", "48",
                  "--out", str(mesh.parent)]),
        ("mesh_test", [sys.executable, str(root/"test_mesh.py"), "--mesh", str(mesh)]),
    ]
    for mode in ["node", "weak"]:
        stages.append((mode, [sys.executable, str(root/"solve.py"), "--mesh", str(mesh),
                              "--kutta", mode, "--alpha", "5", "--ncpu", "8",
                              "--out", str(target/mode)]))
    for name, command in stages:
        tick = time.perf_counter()
        run = subprocess.run(command, capture_output=True, text=True)
        (target/(name+".log")).write_text(run.stdout+run.stderr, encoding="utf-8")
        commands.append(dict(radius=radius, stage=name, command=command,
                             returncode=run.returncode, seconds=time.perf_counter()-tick))
        records.append(dict(test=f"R{radius}_{name}_status", passed=run.returncode == 0))
        if run.returncode:
            break
        if name in ["node", "weak"]:
            result = json.loads((target/name/"result.json").read_text())
            rows.append(dict(radius=radius, kutta=name, nodes=result["nodes"], tets=result["elements"],
                             cl=result["cl_pressure"], cl_jump=result["cl_jump"], cd=result["cd_pressure"],
                             wall_rms=result["wall_normal_rms"], wall_max=result["wall_normal_max"],
                             cp_min=result["cp_min"], cp_max=result["cp_max"], wake=result["wake_diagnostics"],
                             seconds=result["elapsed_seconds"], mesh_sha256=result["mesh_sha256"]))
        (args.out/"progress.json").write_text(json.dumps(dict(commands=commands, rows=rows), indent=2), encoding="utf-8")
for mode in ["node", "weak"]:
    selected = [row for row in rows if row["kutta"] == mode]
    if len(selected) >= 2:
        change = abs(selected[-1]["cl"]-selected[-2]["cl"])
        records.append(dict(test=mode+"_last_CL_change_below_point002", change=change, passed=change < .002))
summary = dict(rows=rows, records=records, commands=commands,
               passed=sum(row["passed"] for row in records), total=len(records),
               seconds=time.perf_counter()-started,
               scope="Fixed farfield size 5/3 chords, h=.15, 48 strips, curvature32. Unstructured remeshing changes the nearfield too.")
(args.out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(args.out/"driver.py.snapshot").write_bytes(Path(__file__).read_bytes())
print(json.dumps(summary, indent=2))
if summary["passed"] != summary["total"]:
    raise SystemExit(2)
