"""Diagnose why the G2 audit-mode geometry trial never satisfies the step gate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import panel.p1_coupled_material_wake as m
from tools.issue84.wake_node_rejection import build_case

OUT = Path("docs/issue84/t4_step_probe.json")
original = m.geometry_failure
traces = []


def patched(mesh, candidates, materials, gates):
    reason = original(mesh, candidates, materials, gates)
    trace = dict(reason=reason, groups={})
    for group in mesh.groups:
        mat = materials[group.name]
        old, new = group.wake_xyz, candidates[group.name]
        tri0, n0, a0, _g0, e0 = m.triangle_data(old, group.wake_faces)
        scale = np.full(len(old), np.inf)
        for k in range(3):
            np.minimum.at(scale, group.wake_faces[:, k], e0.min(axis=1))
        disp = np.linalg.norm(new[mat.used] - old[mat.used], axis=1)
        limit = gates.max_step_fraction * scale[mat.used]
        ratio = disp / np.maximum(limit, 1e-300)
        bad = np.flatnonzero(ratio > 1.0)
        trace["groups"][group.name] = dict(
            used=int(len(mat.used)), bad=int(len(bad)),
            max_step_fraction=float(gates.max_step_fraction),
            scale_min=float(np.min(scale[mat.used])),
            scale_median=float(np.median(scale[mat.used])),
            bad_nodes=[int(mat.used[i]) for i in bad[:12]],
            bad_scale=[float(scale[mat.used[i]]) for i in bad[:12]],
            bad_disp=[float(disp[i]) for i in bad[:12]],
            bad_ratio_max=float(ratio.max()),
            zero_scale_nodes=int(np.count_nonzero(scale[mat.used] <= 1e-12)))
    traces.append(trace)
    return reason


m.geometry_failure = patched
mesh, alpha = build_case(["D:/3D/Projects/panel/runs/half_cad_wingbody_v1/surface.json"],
                         case="g2", alpha_deg=5.0)
result = m.solve_coupled_material_wake(mesh, alpha, kutta="signed_speed", iterations=1,
                                       max_backtracks=4, gate_mode="audit")
m.geometry_failure = original
payload = dict(schema="dpanel.issue84.t4_step_probe.v1", status=result["status"],
               trials=traces, partial=False)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
for i, trace in enumerate(traces):
    for name, info in trace["groups"].items():
        print(i, trace["reason"], name, "bad", info["bad"], "ratio", round(info["bad_ratio_max"], 3),
              "scale_med", round(info["scale_median"], 6), "zero_scale", info["zero_scale_nodes"])
print("wrote", OUT)
