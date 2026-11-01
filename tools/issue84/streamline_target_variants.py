"""Issue #84: which part of the 23-chord streamline request is mesh, which is map?"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import panel.p1_coupled_material_wake as m
from tools.issue84.wake_node_rejection import build_case

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/issue84/streamline_target_variants.json")
captured = {}
original = m.streamline_target


def spy(xyz, material, velocity, eligible):
    target = original(xyz, material, velocity, eligible)
    if "xyz" not in captured:
        captured.update(xyz=np.asarray(xyz, float), velocity=np.asarray(velocity, float),
                        eligible=np.asarray(eligible, bool), target=np.asarray(target, float),
                        reference=np.asarray(material.reference_xyz, float),
                        predecessor=np.asarray(material.predecessor, int),
                        order=np.asarray(material.order, int), roots=np.asarray(material.roots, int))
    return target


m.streamline_target = spy
mesh, alpha = build_case(["D:/3D/Projects/panel/runs/half_cad_wingbody_v1/surface.json"],
                         case="g2", alpha_deg=5.0)
result = m.solve_coupled_material_wake(mesh, alpha, kutta='signed_speed', iterations=1,
                                       max_backtracks=12, gate_mode='audit')
group = mesh.groups[0]
chord = float(mesh.reference_chord)
xyz, velocity, eligible = captured["xyz"], captured["velocity"], captured["eligible"]
reference, pred, order, roots = (captured["reference"], captured["predecessor"],
                                 captured["order"], captured["roots"])
faces = np.asarray(group.wake_faces, int)
node_scale = m.node_step_scale(group.wake_xyz, group.wake_faces)
incident = np.zeros(len(xyz), int)
for k in range(3):
    np.add.at(incident, faces[:, k], 1)
link = np.zeros(len(xyz))
for n in order:
    p = int(pred[n])
    if p >= 0:
        link[n] = np.linalg.norm(reference[n]-reference[p])
long_link = link > 4.0*node_scale
boundary = incident < 6
print("long-link nodes", int(long_link.sum()), "boundary nodes", int(boundary.sum()),
      "free+long", int((long_link & eligible).sum()))


def run(*, step_mode="reference", reanchor=False, freeze=(), resolved=0.):
    target = np.asarray(xyz, float).copy()
    target[roots] = reference[roots]
    fixed = np.zeros(len(xyz), bool)
    for flag in freeze:
        fixed |= flag
    fixed[roots] = True
    for node in order:
        p = int(pred[node])
        if p < 0 or not eligible[node] or fixed[node]:
            if fixed[node] and p >= 0:
                target[node] = xyz[node]          # frozen truncation/far row
            continue
        v = velocity[node].copy()
        if pred[p] >= 0 and eligible[p]:
            v = (v+velocity[p])/2
        speed = np.linalg.norm(v)
        ds = link[node]
        if step_mode == "local":
            ds = min(ds, 4.0*node_scale[node])
        if resolved > 0.:
            ds = min(ds, resolved*node_scale[node])
        if (reanchor and long_link[node]) or (resolved > 0. and long_link[node]):
            base = xyz[p]
        else:
            base = target[p]
        target[node] = base+ds*v/speed
    d = np.linalg.norm(target-xyz, axis=1)/chord
    moved = eligible & ~fixed
    cap = m.GeometryGates().max_step_fraction*node_scale
    return dict(max=float(d.max()), p99=float(np.percentile(d[moved], 99)),
                p90=float(np.percentile(d[moved], 90)),
                median=float(np.median(d[moved])),
                above_cap=int(np.count_nonzero(d > cap)),
                moved=int(moved.sum()),
                ratio=int(np.count_nonzero(d > cap)),
                max_over_cap=float(np.max(np.divide(d[moved], cap[moved]))))


variants = dict(
    V0_production=dict(),
    V1_no_long_links_in_moving_set=dict(freeze=(long_link,)),
    V2_local_step_cap=dict(step_mode="local"),
    V3_reanchor_long_links=dict(reanchor=True),
    V4_local_step_and_freeze=dict(step_mode="local", freeze=(long_link,)),
    V5_resolved_reanchor_4=dict(resolved=4.),
    V6_resolved_reanchor_2=dict(resolved=2.),
    V7_resolved_reanchor_1=dict(resolved=1.),
)
out = dict(schema="dpanel.issue84.streamline_target_variants.v1", case="g2", alpha_deg=5.0,
           status=result["status"], reference_chord=chord,
           geometry=dict(wake_faces=int(len(faces)), nodes=int(len(xyz)),
                         roots=int(len(roots)), free=int(eligible.sum()),
                         truncation_nodes=int(long_link.sum()),
                         boundary_nodes=int(boundary.sum()),
                         link_chord=dict(median=float(np.median(link[link > 0])),
                                         max=float(link.max())),
                         node_scale_chord=dict(median=float(np.median(node_scale)),
                                               min=float(node_scale.min()))),
           variants={k: run(**v) for k, v in variants.items()})
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, indent=2, sort_keys=True)+chr(10), encoding="utf-8")
print(json.dumps(out["geometry"], indent=1))
for k, v in out["variants"].items():
    print(k, json.dumps(v))
print("wrote", OUT)
