"""Freeze and reproduce the B3 steady-wake baseline for the position study.

python -m tools.wake_sensitivity_baseline --case swept30 --out runs/verification/issue102_wake_sensitivity
"""
import argparse
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy
import torch
from threadpoolctl import threadpool_info

from panel.p1_lifting_g2 import combine_wings
from panel.p1_steady_free_wake import prepare_wake_material, solve_steady_free_wake
from tools.issue92.b3_dpanel import _wing


root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case", choices=("rect", "swept30", "elliptic"), required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--iterations", type=int, default=60)
args = parser.parse_args()
out = args.out / args.case
out.mkdir(parents=True, exist_ok=True)
if (out / "initial_state.npz").exists():
    raise FileExistsError("Preserve the existing frozen run: " + str(out))
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
spec = json.loads((root / "docs/issue92/b3_cases.json").read_text(encoding="utf-8"))["cases"][args.case]
wing = _wing(spec)
mesh = combine_wings([wing], reference_chord=spec["cref"],
                     reference_area=spec["sref"], reference_point=wing.reference_point)
reference = json.loads((root / "docs/issue92/b3_free_wake.json").read_text(encoding="utf-8"))["cases"][args.case]["dpanel"]
arrays = {f.name: getattr(wing, f.name) for f in fields(wing)
          if isinstance(getattr(wing, f.name), np.ndarray)}
materials = {g.name: prepare_wake_material(g) for g in mesh.groups}
sources = ("tools/wake_sensitivity_baseline.py", "tools/issue92/b3_dpanel.py",
           "panel/p1_lifting_g1.py", "panel/p1_lifting_g2.py", "panel/p1_solve_torch.py",
           "panel/p1_steady_free_wake.py", "panel/p1_pressure_kutta_torch_driver.py",
           "panel/p1_pressure_kutta_torch.py")
manifest = dict(
    schema="dpanel.wake_sensitivity.baseline.v1", case=args.case, spec=spec,
    head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                        capture_output=True, text=True).stdout.strip(),
    source_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources},
    command=[sys.executable, *sys.argv], pid=os.getpid(), started=time.time(),
    environment=dict(os=platform.platform(), cpu=platform.processor(),
                     python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                     torch=torch.__version__, blas=threadpool_info(),
                     threads={k: os.environ.get(k) for k in
                              ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}),
    settings=dict(alpha_deg=4., kutta="pressure", relaxation=.3, iterations=args.iterations,
                  probe_fraction=.055, step_fraction=.009, wake_velocity_block=24,
                  max_nfev=80, geometry_tol=.002, slip_tol=.01, min_axial=.08,
                  frozen_guard=True, max_step_chord=0., slip_gate=0.,
                  probe_floor_fraction=0., relax_min_abs_y=0.),
    historical_initial_CL=reference["history"][0]["CL"], historical_final_CL=reference["CL"],
    physical_accuracy_accepted=False, rows=[], status="running", exit_code=None)
if "surface" in spec:
    manifest["input_sha256"] = hashlib.sha256((root / spec["surface"]).read_bytes()).hexdigest()
    if manifest["input_sha256"] != spec["surface_sha256"]:
        raise AssertionError("CAD input differs from the frozen B3 fixture")
(out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
initial_tri = mesh.groups[0].wake_xyz[mesh.groups[0].wake_faces]
initial_cross = np.cross(initial_tri[:, 1] - initial_tri[:, 0],
                         initial_tri[:, 2] - initial_tri[:, 0])
initial_area2 = np.linalg.norm(initial_cross, axis=1)


def save(iteration, current, state, record):
    snapshot = dict(mu=np.asarray(state["mu"]), gamma=np.asarray(state["gamma"]),
                    cp=np.asarray(state["cp"]), wake_xyz=current.groups[0].wake_xyz)
    path = out / ("iteration_%03d.npz" % iteration)
    np.savez_compressed(path, **snapshot)
    if iteration == 0:
        np.savez_compressed(out / "initial_state.npz", **arrays, **{
            "state_" + k: v for k, v in snapshot.items()})
        manifest["initial_state_sha256"] = hashlib.sha256((out / "initial_state.npz").read_bytes()).hexdigest()
        delta = abs(float(state["CL"]) - manifest["historical_initial_CL"])
        manifest["initial_CL_difference"] = delta
        if delta > 1e-8:
            raise AssertionError("Pressure-Kutta iteration-0 baseline mismatch: " + str(delta))
    tri = current.groups[0].wake_xyz[current.groups[0].wake_faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    row = {k: state[k] for k in ("CL", "CL_gamma", "CM_y", "CD_pressure", "wall_rms",
                                  "jump_rms", "pressure_kutta_rms", "nonlinear")}
    row.update(iteration=iteration, wakes=record["wakes"],
               snapshot=path.name, snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               min_area2=float(np.linalg.norm(cross, axis=1).min()),
               min_reference_signed_jacobian=float(np.min(
                   np.einsum("fi,fi->f", cross, initial_cross) / initial_area2**2)),
               finite_gamma=bool(np.isfinite(snapshot["gamma"]).all()),
               collision_audit="pending", elapsed=time.time() - manifest["started"])
    manifest["rows"].append(row)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("%s iter=%d CL=%.9f wall=%.3e slip=%.3e elapsed=%.1f"
          % (args.case, iteration, state["CL"], state["wall_rms"],
             max(v["normal_slip_rms"] for v in record["wakes"].values()), row["elapsed"]),
          flush=True)


try:
    result = solve_steady_free_wake(mesh, 4., kutta="pressure", iterations=args.iterations,
                                    relaxation=.3, save_callback=save)
    manifest.update(status="completed", exit_code=0, seconds=result["seconds"],
                    converged=result["converged"], geometry_stable=result["geometry_stable"],
                    final_CL_difference=float(result["state"]["CL"] - reference["CL"]),
                    history=result["history"])
except Exception as error:
    manifest.update(status="failed", exit_code=1, error=repr(error))
    raise
finally:
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
