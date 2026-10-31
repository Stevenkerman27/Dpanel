"""Diagnose the T3 true_max failure at b2_free_node xyz=(1.1413,-0.2,0)."""
from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np

from tools.issue84.adaptive_area_acceptance import adaptive_field
from tools.issue84.fan_quadrature_gate import _field
from tools.issue84.wake_node_rejection import build_case
from panel.p1_coupled_material_wake import limits, triangle_data
from panel.p1_lifting_g2 import assemble_multi, solve_multi

CORE = 0.01
TARGET_XYZ = np.array([1.1413036132966763, -0.2, 0.0])
SURFACE = ["D:/3D/Projects/panel/runs/half_cad_wingbody_v1/surface.json"]

mesh, alpha = build_case(SURFACE, case="g2", alpha_deg=5.0)
ops = assemble_multi(mesh)
state = solve_multi(mesh, alpha, kutta="pressure", operators=ops)
groups = [g for g in mesh.groups if len(g.wake_faces)]

# Rebuild the free-node normal/scale exactly as b2_points does.
best = None
for group in groups:
    tri, normal, area, _shape, edges = triangle_data(group.wake_xyz, group.wake_faces)
    node_normals = np.zeros_like(group.wake_xyz)
    node_scale = np.full(len(node_normals), np.inf)
    for k in range(3):
        np.add.at(node_normals, group.wake_faces[:, k], normal * area[:, None])
        np.minimum.at(node_scale, group.wake_faces[:, k], edges.min(axis=1))
    distance = np.linalg.norm(group.wake_xyz - TARGET_XYZ, axis=1)
    node = int(np.argmin(distance))
    if best is None or distance[node] < best[0]:
        best = (float(distance[node]), group, node, node_normals, node_scale)
distance, group, node, node_normals, node_scale = best
point = group.wake_xyz[node][None, :]
vector = node_normals[node]
normal = (vector / np.linalg.norm(vector))[None, :]
scale = np.array([float(node_scale[node])])
print("node %d group %s xyz %s (distance %.2e) normal %s scale %.6f" % (
    node, group.name, point[0].tolist(), distance, normal[0].tolist(), scale[0]), flush=True)

ladder = np.array([1.0, .5, .25, .125])
kwargs = dict(eps_fraction=.05, ladder=ladder, order=48)

def evaluate(label, field):
    t0 = time.perf_counter()
    sample = limits(field, point, normal, scale, tag="diag", **kwargs)
    dt = time.perf_counter() - t0
    v = np.asarray(sample["mean"][0], float)
    print("%-26s v=[% .10f % .10f % .10f] q=%.3e eps=%.3e %5.1fs" % (
        label, v[0], v[1], v[2], float(sample["quadrature_error"][0]),
        float(sample["epsilon_error"][0]), dt), flush=True)
    return v

results = {}
results["adaptive 28:8 far8"] = evaluate('adaptive 28:8 far8', adaptive_field(
    mesh, state, CORE, fan_order=28, fan_strips=8, refinement=48, refinement_strips=8,
    near_ratio=2.0, far_order=8, far_levels=1, budget=2e-4, escalations=2))
for far_order in (16, 24):
    results["adaptive 28:8 far%d" % far_order] = evaluate(
        "adaptive 28:8 far%d" % far_order, adaptive_field(
            mesh, state, CORE, fan_order=28, fan_strips=8, refinement=48,
            refinement_strips=8, near_ratio=2.0, far_order=far_order, far_levels=1,
            budget=2e-4, escalations=2))
for label, order, strips, ref_order, ref_strips in (
        ("fan 28:8", 28, 8, 48, 8),
        ("fan 48:8", 48, 8, 96, 8),
        ("fan 48:64", 48, 64, 48, 8),
        ("fan 96:64", 96, 64, 48, 8)):
    results[label] = evaluate(label, _field(mesh, state, CORE, order, strips,
                                            refinement=ref_order,
                                            refinement_strips=ref_strips))

ref = results["fan 48:64"]
print()
for key, value in results.items():
    print("%-24s rel_vs_48:64 %.3e" % (
        key, float(np.linalg.norm(value - ref) / max(np.linalg.norm(ref), 1e-30))))
out = Path("docs/issue84/truth_point_1162.json")
out.write_text(json.dumps(dict(
    point=point[0].tolist(), normal=normal[0].tolist(), scale=float(scale[0]),
    node=int(node), group=group.name, velocities={k: [float(x) for x in v]
                                                  for k, v in results.items()}),
    indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("wrote", out, flush=True)
