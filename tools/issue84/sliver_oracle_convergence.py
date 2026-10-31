"""Oracle convergence check for the sliver point: uniform subdivision + degree-5 rule."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import panel.p1_projection_duffy as duffy
from panel.p1_lifting_g2 import assemble_multi, solve_multi
from panel.p1_coupled_material_wake import TotalVelocity
from tools.issue84.fan_quadrature_gate import _wake_geometry
from tools.issue84.wake_node_rejection import build_case
from tools.issue84.sliver_subdivision import (reference_integral, split_faces,
                                              area_velocity_split, aspect,
                                              point_triangle_distance)

SURFACE = "D:/3D/Projects/panel/runs/half_cad_wingbody_v1/surface.json"
POINT = np.array([22.333, 0.238, 0.0])
CORE = 0.01

t0 = time.perf_counter()
mesh, alpha_deg = build_case(SURFACE, case="g2", alpha_deg=5.0)
ops = assemble_multi(mesh)
state = solve_multi(mesh, alpha_deg, kutta="pressure", operators=ops)
groups = [g for g in mesh.groups if len(g.wake_faces)]
_ = _wake_geometry(groups)
total = TotalVelocity(mesh, state, method="rm_regularized_p1", core_radius=CORE)
field = total._regularized
tri = np.asarray(field.triangles, float)
gam = np.asarray(field.gamma, float)
dist = np.empty(len(tri)); scale = np.empty(len(tri))
for i, t in enumerate(tri):
    dist[i], scale[i] = point_triangle_distance(t, POINT)
near = (dist <= 2.0 * scale) | (dist <= CORE)
ntri, ngam = tri[near], gam[near]
print("wall %d wake %d near %d  aspect max %.3f" % (len(mesh.faces), len(tri), near.sum(), aspect(ntri).max()), flush=True)
print("setup %.1f s" % (time.perf_counter() - t0), flush=True)

rows = []
for depth in (4, 5, 6, 7):
    begin = time.perf_counter()
    value, nodes = reference_integral(ntri, ngam, POINT, CORE, depth=depth)
    rows.append(dict(depth=depth, value=[float(v) for v in value],
                     norm=float(np.linalg.norm(value)), nodes=int(nodes),
                     seconds=time.perf_counter() - begin))
    print("oracle depth %d nodes %9d |v| %.8e  %.2f s" % (depth, nodes, rows[-1]["norm"], rows[-1]["seconds"]), flush=True)

for depth, order, strips in ((4, 24, 8), (4, 32, 8), (5, 24, 8), (5, 16, 8)):
    begin = time.perf_counter()
    value, pieces = area_velocity_split(ntri, ngam, POINT, CORE, depth=depth, order=order, strips=strips)
    rows.append(dict(depth=depth, order=order, strips=strips, kind="fan",
                     value=[float(v) for v in value], norm=float(np.linalg.norm(value)),
                     pieces=int(pieces), seconds=time.perf_counter() - begin))
    print("fan    depth %d %d:%d |v| %.8e  %.2f s" % (depth, order, strips, rows[-1]["norm"], rows[-1]["seconds"]), flush=True)

finest = np.asarray(rows[-1]["value"], float)
Path("runs/issue84/oracle_convergence.json").write_text(
    json.dumps(dict(rows=rows, wall=float(time.perf_counter())), indent=2), encoding="utf-8")
for row in rows:
    err = float(np.linalg.norm(np.asarray(row["value"], float) - np.asarray(rows[3]["value"], float))
                / np.linalg.norm(np.asarray(rows[3]["value"], float)))
    print("%-22s rel vs oracle depth 7 %.3e" %
          ((("oracle d%d" % row["depth"]) if "order" not in row else ("fan d%d %d:%d" % (row["depth"], row["order"], row["strips"]))), err))
