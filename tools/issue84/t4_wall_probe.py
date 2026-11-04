"""Diagnose why the G2 audit-mode geometry trial never satisfies the step/wall gates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import panel.p1_coupled_material_wake as m
from tools.issue84.wake_node_rejection import build_case

OUT = Path("docs/issue84/t4_wall_probe.json")
original = m.geometry_failure
traces = []


def _wall_hits(mesh, candidates, gates):
    tol = gates.tolerance_fraction * mesh.reference_chord
    shrink = 1e-8
    hits = []
    wall = mesh.xyz[mesh.faces]
    for group in mesh.groups:
        old, new = group.wake_xyz, candidates[group.name]
        tri0, _n0, _a0, _g0, _e0 = m.triangle_data(old, group.wake_faces)
        tri1, _n1, _a1, _g1, _e1 = m.triangle_data(new, group.wake_faces)
        t0 = (1 - shrink) * tri0 + shrink * tri0.mean(axis=1)[:, None]
        t1 = (1 - shrink) * tri1 + shrink * tri1.mean(axis=1)[:, None]
        hulls = np.concatenate((t0, t1), axis=1)
        for i, hull in enumerate(hulls):
            for j, tri in enumerate(wall):
                if not m._bounds_overlap(hull, tri, tol):
                    continue
                if m.convex_overlap(hull, tri, tol):
                    hits.append(dict(group=group.name, wake_triangle=int(i),
                                     wake_nodes=[int(n) for n in group.wake_faces[i]],
                                     wall_triangle=int(j),
                                     wake_centroid=[float(v) for v in tri0[i].mean(axis=0)],
                                     displacement=[float(v) for v in
                                                  (new[group.wake_faces[i]] - old[group.wake_faces[i]]).mean(axis=0)]))
                    break
            if len(hits) >= 12:
                return hits
    return hits


def patched(mesh, candidates, materials, gates):
    trial = len(traces)
    trace = dict(trial=trial, reason=None, wall_hits=None, step_bad=None)
    if trial >= 4:  # step gate passes only after several halvings; inspect the wall stage
        trace["wall_hits"] = _wall_hits(mesh, candidates, gates)
    reason = original(mesh, candidates, materials, gates)
    trace["reason"] = reason
    traces.append(trace)
    return reason


m.geometry_failure = patched
mesh, alpha = build_case(["D:/3D/Projects/panel/runs/cases/wingbody/half_cad_wingbody_v1/surface.json"],
                         case="g2", alpha_deg=5.0)
result = m.solve_coupled_material_wake(mesh, alpha, kutta="signed_speed", iterations=1,
                                       max_backtracks=12, gate_mode="audit")
m.geometry_failure = original
payload = dict(schema="dpanel.issue84.t4_wall_probe.v1", status=result["status"],
               trials=traces, partial=False)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
for trace in traces:
    hits = trace["wall_hits"]
    print(trace["trial"], trace["reason"],
          "hits", None if hits is None else len(hits),
          "" if not hits else json.dumps(hits[0]))
print("wrote", OUT)
