"""Static vs swept-hull false positive for the G2 wall gate on wake triangle 293."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import panel.p1_coupled_material_wake as m
from panel.p1_steady_free_wake import prepare_wake_material
from tools.issue84.wake_node_rejection import build_case

mesh, alpha = build_case(["D:/3D/Projects/panel/runs/cases/wingbody/half_cad_wingbody_v1/surface.json"],
                         case="g2", alpha_deg=5.0)
gates = m.GeometryGates()
materials = {g.name: prepare_wake_material(g) for g in mesh.groups}

baseline = {g.name: g.wake_xyz.copy() for g in mesh.groups}
print("unchanged reference geometry ->", m.geometry_failure(mesh, baseline, materials, gates))

# Move only the nodes of wake triangle 293 (main wing) by a controlled amount.
group = mesh.groups[0]
tri = group.wake_faces[293]
centroid = group.wake_xyz[tri].mean(axis=0)
print("triangle 293 nodes", tri.tolist(), "centroid", centroid.tolist())
rows = []
for scale in (0.0, 1e-12, 1e-9, 1e-7, 1e-5, 1e-3):
    candidates = {g.name: g.wake_xyz.copy() for g in mesh.groups}
    candidates[group.name][tri] += np.array([0.0, scale, 0.0])
    rows.append(dict(shift=scale, reason=m.geometry_failure(mesh, candidates, materials, gates)))
print(json.dumps(rows, indent=1))

# Also test the pure old geometry SAT against wall triangle 13 for triangle 293.
old = m.triangle_data(group.wake_xyz, group.wake_faces)[0][293]
wall = mesh.xyz[mesh.faces][13]
print("old tri 293", np.round(old, 6).tolist())
print("wall tri 13", np.round(wall, 6).tolist())
print("old-vs-wall overlap", bool(m.convex_overlap(old, wall, gates.tolerance_fraction * mesh.reference_chord)))
