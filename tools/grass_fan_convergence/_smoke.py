"""Smoke test: GRASS-plane CST wing -> DPanel P1 lifting solve."""
from __future__ import annotations

import json
import time

import numpy as np

from tools.grass_fan_convergence.wing_builder import build_wing, naca_code

t0 = time.perf_counter()
code2412 = naca_code("2412")
print("cst code", np.asarray(code2412).shape, "%.1fs" % (time.perf_counter() - t0), flush=True)

sections = [
    dict(y=0.0, chord=1.0, code=code2412),
    dict(y=1.0, chord=0.85, le_x=0.12, z=0.03, code=code2412),
    dict(y=2.0, chord=0.70, le_x=0.24, z=0.06, twist_deg=-1.0, code=code2412),
    dict(y=3.0, chord=0.55, le_x=0.36, z=0.09, twist_deg=-2.0, code=code2412),
]
wing = build_wing(sections, span_panels=3, chord_panels=8, wake_panels=3,
                  wake_length=5.0, name="smoke")
print("nodes", len(wing.xyz), "faces", len(wing.faces),
      "dof", wing.ndof_mu, "gamma", wing.gamma_hat.shape[1],
      "wake", len(wing.wake_xyz), len(wing.wake_faces), flush=True)
print("bank counts", np.bincount(wing.face_bank + 1), flush=True)

tri = wing.xyz[wing.faces]
n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
up = wing.face_bank == 1
print("upper mean nz>0", float(np.mean(n[up, 2] > 0)),
      "lower mean nz<0", float(np.mean(n[~up & (wing.face_bank == -1), 2] < 0)),
      "caps", int(np.sum(wing.face_bank == 0)), flush=True)

from panel.p1_lifting_g2 import combine_wings, solve_multi
mesh = combine_wings([wing], names=["grass_wing"], reference_chord=wing.chord,
                     reference_point=np.zeros(3))
print("mesh faces", len(mesh.faces), "groups", [g.name for g in mesh.groups], flush=True)
t1 = time.perf_counter()
state = solve_multi(mesh, 5.0, kutta="pressure")
print("%.1fs CL=%+.6f CLg=%+.6f CD_p=%+.6f CMy=%+.6f" % (
    time.perf_counter() - t1, state["CL"], state["CL_gamma"],
    state["CD_pressure"], state["CM_y"]), flush=True)
print("kutta_residual", float(state.get("pressure_kutta_rms", float("nan"))),
      "wall_rms", float(state.get("wall_rms", float("nan"))), flush=True)
print(json.dumps({k: float(v) for k, v in state.items()
                  if isinstance(v, (int, float, np.floating))}, indent=2), flush=True)
