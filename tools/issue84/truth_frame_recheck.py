"""Recompute the T3 true errors with the sampling frame each point was recorded with.

T3's truth stage replaced every recorded point frame by the nearest wake
triangle's normal and edge length, so value and reference were sampled at
different sites. This tool rebuilds the recorded frame from the geometry and
repeats the 48:64 reference comparison.
"""
from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np

from tools.issue84.adaptive_area_acceptance import _field, _wake_geometry, load_points
from tools.issue84.wake_node_rejection import build_case
from panel.p1_coupled_material_wake import limits, triangle_data
from panel.p1_lifting_g2 import assemble_multi, solve_multi

CORE = 0.01
SURFACE = ["D:/3D/Projects/panel/runs/half_cad_wingbody_v1/surface.json"]
T3 = Path("docs/issue84/adaptive_area_acceptance_t3.json")
OUT = Path("docs/issue84/truth_frame_recheck.json")

data = json.loads(T3.read_text(encoding="utf-8"))
records = {int(p["index"]): p for p in data["points"]}
truth = data["truth"]

mesh, alpha = build_case(SURFACE, case="g2", alpha_deg=5.0)
ops = assemble_multi(mesh)
state = solve_multi(mesh, alpha, kutta="pressure", operators=ops)
groups = [g for g in mesh.groups if len(g.wake_faces)]

frames = {}
for group in groups:
    tri, normal, area, _shape, edges = triangle_data(group.wake_xyz, group.wake_faces)
    node_normals = np.zeros_like(group.wake_xyz)
    node_scale = np.full(len(group.wake_xyz), np.inf)
    for k in range(3):
        np.add.at(node_normals, group.wake_faces[:, k], normal * area[:, None])
        np.minimum.at(node_scale, group.wake_faces[:, k], edges.min(axis=1))
    centroid = tri.mean(axis=1)
    frames[group.name] = dict(xyz=group.wake_xyz, tri=tri, centroid=centroid,
                              normal=normal, edges=edges, node_normals=node_normals,
                              node_scale=node_scale)

csv_rows = []
for path in (Path("runs/issue84/projection_gate_quad.csv"),
             Path("runs/issue84/projection_gate_eps.csv")):
    for row in load_points(path):
        csv_rows.append(row)

def frame_of(record):
    xyz = np.asarray(record["xyz"], float)
    name = record.get("group")
    if name is None:
        name = min(frames, key=lambda key: float(np.min(np.linalg.norm(
            frames[key]["centroid"] - xyz, axis=1))))
    frame = frames[name]
    if record["set"] == "b2_free_node":
        dist = np.linalg.norm(frame["xyz"] - xyz, axis=1)
        node = int(np.argmin(dist))
        vector = frame["node_normals"][node]
        scale = float(frame["node_scale"][node])
        normal = vector / np.linalg.norm(vector)
        return normal, scale, "free_node %.1e" % dist[node]
    if record["set"] == "b2_wake_face":
        dist = np.linalg.norm(frame["centroid"] - xyz, axis=1)
        tri = int(np.argmin(dist))
        return frame["normal"][tri], float(frame["edges"][tri].min()), "wake_face %.1e" % dist[tri]
    best = None
    for row in csv_rows:
        d = float(np.linalg.norm(np.asarray(row["xyz"], float) - xyz))
        if best is None or d < best[0]:
            best = (d, row)
    d, row = best
    normal = np.asarray(row["normal"], float)
    return normal / np.linalg.norm(normal), float(row["scale"]), "csv %.1e" % d

ladder = np.array([1.0, .5, .25, .125])
kwargs = dict(eps_fraction=.05, ladder=ladder, order=48)
reference = _field(mesh, state, CORE, 48, 64, refinement=48, refinement_strips=8)

rows = []
for entry in truth:
    record = records[int(entry["index"])]
    normal, scale, provenance = frame_of(record)
    point = np.asarray(record["xyz"], float)[None, :]
    t0 = time.perf_counter()
    ref = limits(reference, point, normal[None, :], np.array([scale]),
                 tag="recheck", **kwargs)
    value = np.asarray(record["velocity"], float)
    denominator = max(float(np.linalg.norm(ref["mean"][0])), 1e-30)
    fresh = float(np.linalg.norm(value - ref["mean"][0]) / denominator)
    rows.append(dict(index=int(record["index"]), set=record["set"],
                     provenance=provenance, true_error_original=entry["true_error"],
                     true_error_recorded_frame=fresh,
                     adaptive_quadrature_error=entry["adaptive_quadrature_error"],
                     seconds=time.perf_counter() - t0))
    print("%-18s idx %4d t3 %8.3e recheck %8.3e gate %8.3e  %s" % (
        record["set"], record["index"], entry["true_error"], fresh,
        entry["adaptive_quadrature_error"], provenance), flush=True)

OUT.write_text(json.dumps(dict(rows=rows), indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("max original %.3e  max recheck %.3e" % (
    max(r["true_error_original"] for r in rows),
    max(r["true_error_recorded_frame"] for r in rows)), flush=True)
print("wrote", OUT, flush=True)
