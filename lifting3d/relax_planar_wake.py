"""Determine one planar-wake angle from mean flow tangency, without changing the solid or fitting loads."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--mesh", type=Path, required=True)
parser.add_argument("--alpha", type=float, default=5.)
parser.add_argument("--mach", type=float, default=0.)
parser.add_argument("--ncpu", type=int, default=8)
parser.add_argument("--maxiter", type=int, default=8)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
assert not args.out.exists() or not any(args.out.iterdir()), "Preserve previous experiments"
args.out.mkdir(parents=True, exist_ok=True)
started = time.perf_counter()
history = []
status = "iteration_limit"
commands = []
angle = 0.
for iteration in range(args.maxiter):
    target = args.out/f"{iteration:02d}"
    command = [sys.executable, str(root/"tilt_mesh.py"), "--mesh", str(args.mesh.resolve()),
               "--angle", str(angle), "--out", str(target/"mesh")]
    run = subprocess.run(command, capture_output=True, text=True)
    target.mkdir(parents=True, exist_ok=True)
    (target/"mesh.log").write_text(run.stdout+run.stderr, encoding="utf-8")
    commands.append(dict(command=command, returncode=run.returncode))
    if run.returncode:
        status = "mesh_rejected"
        break
    command = [sys.executable, str(root/"solve.py"), "--mesh", str(target/"mesh"/"mesh.npz"),
               "--alpha", str(args.alpha), "--mach", str(args.mach), "--ncpu", str(args.ncpu),
               "--out", str(target/"solve")]
    run = subprocess.run(command, capture_output=True, text=True)
    (target/"solve.log").write_text(run.stdout+run.stderr, encoding="utf-8")
    commands.append(dict(command=command, returncode=run.returncode))
    if run.returncode:
        status = "flow_rejected"
        break
    field = np.load(target/"solve"/"field.npz")
    result = json.loads((target/"solve"/"result.json").read_text())
    up, low = field["tags"] == 201, field["tags"] == 202
    distance = np.linalg.norm(field["surface_points"][up, :, None]-field["surface_points"][low, None, :], axis=3)
    paired = distance.argmin(axis=2)
    assert distance.min(axis=2).max() < 1e-10
    low_velocity = np.take_along_axis(field["surface_velocity"][low], paired[:, :, None], axis=1)
    average_velocity = .5*(field["surface_velocity"][up]+low_velocity)
    normal_speed = np.sum(average_velocity*field["surface_normal"][up], axis=2)
    weights = field["surface_weight"][up]
    mean_normal_speed = float(np.average(normal_speed, weights=weights))
    rms_normal_speed = float(np.sqrt(np.average(normal_speed**2, weights=weights)))
    history.append(dict(iteration=iteration, angle_deg=angle, mean_normal_speed=mean_normal_speed,
                        rms_normal_speed=rms_normal_speed, cl_pressure=result["cl_pressure"],
                        cd_pressure=result["cd_pressure"], flow_result=str(target/"solve"/"result.json"),
                        solver_sha256=result["solver_sha256"], p2_sha256=result["p2_sha256"]))
    print(json.dumps(history[-1]), flush=True)
    if abs(mean_normal_speed) < 1e-5:
        status = "mean_planar_tangency_converged"
        break
    if iteration == 0:
        angle = args.alpha
        continue
    brackets = [(left, right) for left in history for right in history
                if left["angle_deg"] < right["angle_deg"]
                and left["mean_normal_speed"]*right["mean_normal_speed"] < 0.]
    if not brackets:
        status = "mean_tangency_not_bracketed_by_zero_and_freestream_angles"
        break
    left, right = min(brackets, key=lambda pair: pair[1]["angle_deg"]-pair[0]["angle_deg"])
    angle = ((left["angle_deg"]*right["mean_normal_speed"]-right["angle_deg"]*left["mean_normal_speed"])
             /(right["mean_normal_speed"]-left["mean_normal_speed"]))
    if iteration % 3 == 0:
        angle = .5*(left["angle_deg"]+right["angle_deg"])
summary = dict(status=status, history=history, commands=commands,
               elapsed_seconds=time.perf_counter()-started,
               model="One planar orientation degree of freedom; mean velocity tangency only, no free-wake rollup",
               acceptance="Mean |q.n|<1e-5; RMS tangency, pressure jump, wall error and load accuracy remain separate",
               driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               source_mesh_sha256=hashlib.sha256(args.mesh.read_bytes()).hexdigest())
(args.out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(args.out/"driver.py.snapshot").write_bytes(Path(__file__).read_bytes())
print(json.dumps(summary, indent=2))
if status != "mean_planar_tangency_converged":
    raise SystemExit(2)
