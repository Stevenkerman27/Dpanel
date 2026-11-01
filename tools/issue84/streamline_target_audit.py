"""Issue #84: why does the rooted fixed-arclength streamline map ask for 23 chords?

Records the exact production inputs of streamline_target() on the first B2
iteration of the G2 CAD case, then decomposes the requested displacement into

  * the local tangent error (angle between the sampled velocity and the
    reference arclength direction),
  * the off-sheet component of that velocity (a streamline map needs a
    tangential direction; a normal component is spurious),
  * the accumulated chain drift (the map marches from the TE root, so one bad
    tangent is carried downstream by every successor).

Also evaluates the same map with three alternative tangents to separate "the
map is wrong" from "the sampled tangent is wrong".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import panel.p1_coupled_material_wake as m
from tools.issue84.wake_node_rejection import build_case

SURFACE = ["D:/3D/Projects/panel/runs/half_cad_wingbody_v1/surface.json"]
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/issue84/streamline_target_audit.json")

captured = {}
original = m.streamline_target


def spy(xyz, material, velocity, eligible):
    target = original(xyz, material, velocity, eligible)
    if "xyz" not in captured:
        captured.update(xyz=np.asarray(xyz, float), velocity=np.asarray(velocity, float),
                        eligible=np.asarray(eligible, bool), target=np.asarray(target, float),
                        reference=np.asarray(material.reference_xyz, float),
                        predecessor=np.asarray(material.predecessor, int),
                        order=np.asarray(material.order, int),
                        roots=np.asarray(material.roots, int),
                        used=np.asarray(material.used, int),
                        faces=None)
    return target


m.streamline_target = spy
mesh, alpha = build_case(SURFACE, case="g2", alpha_deg=5.0)
result = m.solve_coupled_material_wake(mesh, alpha, kutta='signed_speed', iterations=1,
                                       max_backtracks=12, gate_mode='audit')
chord = float(mesh.reference_chord)
group = mesh.groups[0]
cap = m.node_step_scale(group.wake_xyz, group.wake_faces)
gates = m.GeometryGates()
cap = gates.max_step_fraction*cap

xyz, velocity, eligible = captured["xyz"], captured["velocity"], captured["eligible"]
target, reference = captured["target"], captured["reference"]
pred, order, roots = captured["predecessor"], captured["order"], captured["roots"]

delta = target-xyz
disp = np.linalg.norm(delta, axis=1)/chord
guess = np.zeros_like(xyz)
guess[roots] = reference[roots]
for node in order:
    p = int(pred[node])
    if p < 0 or not eligible[node]:
        continue
    v = velocity[node].copy()
    if pred[p] >= 0 and eligible[p]:
        v = (v+velocity[p])/2
    ds = np.linalg.norm(reference[node]-reference[p])
    guess[node] = guess[p]+ds*v/np.linalg.norm(v)
map_disp = np.linalg.norm(guess-xyz, axis=1)/chord
print("reproduced map:", float(np.max(np.abs(map_disp-disp))))
# --- reference geometry: local arclength step and its direction
t_ref = np.zeros_like(xyz)
ds_ref = np.zeros(len(xyz))
depth = np.zeros(len(xyz), int)
for node in order:
    p = int(pred[node])
    if p < 0:
        continue
    ds_ref[node] = np.linalg.norm(reference[node]-reference[p])
    t_ref[node] = (reference[node]-reference[p])/max(ds_ref[node], 1e-30)
    depth[node] = depth[p]+1
# reference sheet normals (for the off-sheet component)
tri, normal, area, _g, _e = m.triangle_data(reference, group.wake_faces)
node_normal = np.zeros_like(xyz)
count = np.zeros(len(xyz))
for k in range(3):
    np.add.at(node_normal, group.wake_faces[:, k], normal)
    np.add.at(count, group.wake_faces[:, k], 1.)
node_normal /= np.maximum(count, 1)[:, None]
speed = np.linalg.norm(velocity, axis=1)
tangential = np.einsum('ij,ij->i', velocity, t_ref)
off_sheet = np.einsum('ij,ij->i', velocity, node_normal)
cos_tangent = np.divide(tangential, speed, out=np.zeros_like(speed), where=speed > 1e-30)
free = np.array([n for n in order if int(pred[n]) >= 0], int)
# --- alternative tangents: reference direction, and RM-regularized sampling
alt_reference = xyz.copy()
alt_reference[roots] = reference[roots]
for node in order:
    p = int(pred[node])
    if p < 0 or not eligible[node]:
        continue
    d = reference[node]-reference[p]
    alt_reference[node] = alt_reference[p]+ds_ref[node]*d/max(np.linalg.norm(d), 1e-30)
record = dict(schema="dpanel.issue84.streamline_target_audit.v1", case="g2", alpha_deg=5.0,
              kutta="signed_speed", gate_mode="audit", velocity_method="analytic_edges",
              reference_chord=chord, status=result["status"],
              nodes=dict(total=int(len(xyz)), free=int(len(free))),
              requested_chord=dict(max=float(disp.max()), p99=float(np.percentile(disp, 99)),
                                   p90=float(np.percentile(disp, 90)),
                                   median=float(np.median(disp[free])),
                                   rms=float(np.sqrt(np.mean(disp[free]**2)))),
              cap_chord=dict(min=float(cap.min()), median=float(np.median(cap))),
              map_residual_chord=float(np.max(disp)),
              tangent=dict(speed_median=float(np.median(speed[free])),
                           speed_p01=float(np.percentile(speed[free], 1)),
                           cos_tangent_median=float(np.median(cos_tangent[free])),
                           cos_tangent_below_0=int(np.count_nonzero(cos_tangent[free] < 0)),
                           cos_tangent_below_0p5=int(np.count_nonzero(cos_tangent[free] < 0.5)),
                           cos_tangent_below_0p9=int(np.count_nonzero(cos_tangent[free] < 0.9)),
                           normal_fraction_median=float(np.median(
                               np.abs(off_sheet[free])/np.maximum(speed[free], 1e-30))),
                           normal_fraction_max=float(np.max(
                               np.abs(off_sheet[free])/np.maximum(speed[free], 1e-30)))),
              reference_direction_map_chord=dict(
                  max=float((np.linalg.norm(alt_reference-xyz, axis=1)/chord).max()),
                  rms=float(np.sqrt(np.mean((np.linalg.norm(alt_reference-xyz, axis=1)[free]/chord)**2)))),
              worst=[])
for idx in np.argsort(-disp)[:15]:
    record["worst"].append(dict(node=int(idx), depth=int(depth[idx]), accepted=bool(eligible[idx]),
                                requested_chord=float(disp[idx]), cap_chord=float(cap[idx]),
                                ds_ref=float(ds_ref[idx]), speed=float(speed[idx]),
                                cos_tangent=float(cos_tangent[idx]),
                                normal_fraction=float(abs(off_sheet[idx])/max(speed[idx], 1e-30))))
# chain drift: correlate requested displacement with chain depth
if len(free):
    record["chain"] = dict(depth_max=int(depth.max()),
                           corr_depth_disp=float(np.corrcoef(depth[free], disp[free])[0, 1]),
                           disp_by_depth_quartile=[float(np.median(disp[free[(depth[free] > a)
                                                                             & (depth[free] <= b)]]))
                                                   if np.any((depth[free] > a) & (depth[free] <= b))
                                                   else None
                                                   for a, b in ((0, 5), (5, 15), (15, 30), (30, 10**9))])
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(record, indent=2, sort_keys=True)+chr(10), encoding="utf-8")
print(json.dumps({k: v for k, v in record.items() if k != "worst"}, indent=2)[:2500])
print("worst:", json.dumps(record["worst"][:5], indent=1))
print("wrote", OUT)
