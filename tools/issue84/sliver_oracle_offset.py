"""Why do the fan and a uniform Gauss rule disagree on the sliver point?

On the sheet the area kernel is ~1/rho^2, so the integral is only logarithmically
regularised by the core: total ~ ln(R/core). A uniform rule smears the core scale to
the element size h, so its value drifts like 1/ln(R/h) instead of converging. Lifting
the target off the sheet removes the log and both families should agree.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from panel.p1_lifting_g2 import assemble_multi, solve_multi
from panel.p1_coupled_material_wake import TotalVelocity
from tools.issue84.wake_node_rejection import build_case
from tools.issue84.sliver_subdivision import (reference_integral, area_velocity_split,
                                              point_triangle_distance)

SURFACE = "D:/3D/Projects/panel/runs/half_cad_wingbody_v1/surface.json"
BASE = np.array([22.333, 0.238, 0.0])
CORE = 0.01

mesh, alpha_deg = build_case(SURFACE, case="g2", alpha_deg=5.0)
ops = assemble_multi(mesh)
state = solve_multi(mesh, alpha_deg, kutta="pressure", operators=ops)
total = TotalVelocity(mesh, state, method="rm_regularized_p1", core_radius=CORE)
field = total._regularized
tri = np.asarray(field.triangles, float)
gam = np.asarray(field.gamma, float)

rows = []
for offset in (0.0, 0.01, 0.05, 0.2):
    point = BASE + np.array([0.0, 0.0, offset])
    dist = np.empty(len(tri)); scale = np.empty(len(tri))
    for i, t in enumerate(tri):
        dist[i], scale[i] = point_triangle_distance(t, point)
    near = (dist <= 2.0 * scale) | (dist <= CORE)
    ntri, ngam = tri[near], gam[near]
    record = dict(offset=float(offset), near=int(near.sum()), oracle={}, fan={})
    for depth in (5, 6, 7):
        begin = time.perf_counter()
        value, nodes = reference_integral(ntri, ngam, point, CORE, depth=depth)
        record["oracle"][str(depth)] = dict(norm=float(np.linalg.norm(value)),
                                           nodes=int(nodes),
                                           seconds=time.perf_counter() - begin)
    for order in (16, 48, 96):
        begin = time.perf_counter()
        value, pieces = area_velocity_split(ntri, ngam, point, CORE, depth=2,
                                            order=order, strips=8)
        record["fan"][str(order)] = dict(norm=float(np.linalg.norm(value)),
                                        pieces=int(pieces),
                                        seconds=time.perf_counter() - begin)
    rows.append(record)
    fine = record["oracle"]["7"]["norm"]
    print("offset %.3f near %3d | oracle d5/6/7 %.8e %.8e %.8e | fan 16/48/96 %.8e %.8e %.8e"
          % (offset, near.sum(),
             record["oracle"]["5"]["norm"], record["oracle"]["6"]["norm"],
             record["oracle"]["7"]["norm"],
             record["fan"]["16"]["norm"], record["fan"]["48"]["norm"],
             record["fan"]["96"]["norm"]), flush=True)
    print("    oracle d6->7 %.2e  fan 48 vs oracle7 %.2e  fan 48 vs 96 %.2e"
          % (abs(record["oracle"]["6"]["norm"] - fine) / fine,
             abs(record["fan"]["48"]["norm"] - fine) / fine,
             abs(record["fan"]["48"]["norm"] - record["fan"]["96"]["norm"])
             / record["fan"]["96"]["norm"]), flush=True)

Path("runs/issue84/oracle_offset.json").write_text(json.dumps(dict(rows=rows), indent=2),
                                                   encoding="utf-8")
print("wrote runs/issue84/oracle_offset.json")
