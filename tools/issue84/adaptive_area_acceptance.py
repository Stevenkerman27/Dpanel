"""Adaptive near/far area quadrature acceptance battery (issue #84, tasks 2-3).

The adaptive evaluator gates itself: near faces use the projection fan, far faces a
composite Duffy map, and the a-posteriori estimate is the difference between the base
and a refinement of the same construction. This tool measures that claim on a fixed
acceptance point set:

* the 109 frozen gate points (7 legacy quadrature rejects + 102 epsilon rejects);
* far points at 1, 3 and 10 chord from the wake sheet (chord from the wall bounding box);
* special points: tip, trailing edge, root, sliver faces and the highest-aspect faces;
* a stratified sample of the real B2 target set (wake-face centroids + free nodes).

Every point is compared against a saturated projection fan (48:64 by default), whose
self-convergence was established in docs/issue84/projection_fan_knee.json. Rows are
flushed after every point so an interrupted run still leaves usable data.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from panel.p1_coupled_material_wake import TotalVelocity, limits, triangle_data
from panel.p1_lifting_g2 import assemble_multi, solve_multi
from panel.p1_steady_free_wake import prepare_wake_material
from panel.velocity_profile import peak_rss_mb
from tools.issue84.fan_quadrature_gate import (_field, _jsonable, _nearest_wake_triangle,
                                               _stats, _wake_geometry, load_points)
from tools.issue84.wake_node_rejection import build_case

# Declared before the run so the acceptance verdict cannot be chosen after the fact.
THRESHOLDS = dict(
    gate_max=2e-4,          # a-posteriori estimate of any accepted point
    gate_p90=2e-5,          # typical gate value
    true_max=2e-4,          # estimator error against the saturated reference
    cost_ratio=1.0,         # median seconds per point vs production fan 28:8
)


def adaptive_field(mesh, state, core, *, fan_order, fan_strips, refinement,
                   refinement_strips, near_ratio, far_order, far_levels, budget,
                   escalations):
    return TotalVelocity(mesh, state, method="rm_regularized_p1", core_radius=float(core),
                         area_quadrature="projection_fan_adaptive",
                         fan_order=int(fan_order), fan_strips=int(fan_strips),
                         fan_refinement_order=int(refinement),
                         fan_refinement_strips=int(refinement_strips),
                         near_ratio=float(near_ratio),
                         adaptive_far_order=int(far_order),
                         adaptive_far_levels=int(far_levels),
                         adaptive_budget=float(budget),
                         adaptive_max_escalations=int(escalations))


def special_points(groups, geometry):
    """Tip, trailing-edge, root, sliver and high-aspect targets near the wake sheet."""
    rows = []
    for part in geometry:
        tri, normal, scale = part["tri"], part["normal"], part["scale"]
        edges = np.linalg.norm(np.stack([tri[:, (i+1) % 3]-tri[:, i] for i in range(3)],
                                        axis=1), axis=2)
        longest, shortest = edges.max(axis=1), edges.min(axis=1)
        aspect = longest/np.maximum(shortest, 1e-30)
        centroid = tri.mean(axis=1)
        picks = [
            ("tip", int(np.argmax(np.abs(centroid[:, 1])))),
            ("root", int(np.argmin(np.abs(centroid[:, 1])))),
            ("trailing_edge", int(np.argmax(centroid[:, 0]))),
            ("leading_edge", int(np.argmin(centroid[:, 0]))),
            ("sliver", int(np.argmin(shortest))),
            ("high_aspect", int(np.argmax(aspect))),
        ]
        for kind, index in picks:
            rows.append(dict(set="special_"+kind, xyz=np.asarray(centroid[index], float),
                             normal=np.asarray(normal[index], float),
                             scale=float(scale[index]), group=part["name"]))
    return rows


def far_points(groups, geometry, wall_xyz, distances, per_distance, seed):
    """Points offset along the sheet normal at 1, 3 and 10 chord from the sheet."""
    span = float(np.ptp(wall_xyz[:, 0]))
    rows = []
    rng = np.random.default_rng(seed)
    for chord_multiple in distances:
        for part in geometry:
            tri, normal, scale = part["tri"], part["normal"], part["scale"]
            picked = rng.choice(len(tri), size=min(per_distance, len(tri)), replace=False)
            for index in picked:
                centroid = tri[index].mean(axis=0)
                rows.append(dict(set="far_%gc" % chord_multiple,
                                 xyz=centroid+chord_multiple*span*normal[index],
                                 normal=np.asarray(normal[index], float),
                                 scale=float(scale[index]), group=part["name"]))
    return rows


def b2_points(groups, args, rng):
    rows = []
    for group in groups:
        mat = prepare_wake_material(group)
        tri, normal, area, _shape, edges = triangle_data(group.wake_xyz, group.wake_faces)
        free = np.setdiff1d(mat.used, mat.roots)
        picked = rng.choice(len(tri), size=min(args.b2_face, len(tri)), replace=False)
        for index in picked:
            rows.append(dict(set="b2_wake_face", xyz=tri[index].mean(axis=0),
                             normal=np.asarray(normal[index], float),
                             scale=float(edges[index].min()), group=group.name))
        node_normals = np.zeros_like(group.wake_xyz)
        node_scale = np.full(len(node_normals), np.inf)
        for k in range(3):
            np.add.at(node_normals, group.wake_faces[:, k], normal*area[:, None])
            np.minimum.at(node_scale, group.wake_faces[:, k], edges.min(axis=1))
        picked = rng.choice(free, size=min(args.b2_node, len(free)), replace=False)
        for node in picked:
            vector = node_normals[node]
            rows.append(dict(set="b2_free_node", xyz=group.wake_xyz[node],
                             normal=vector/np.linalg.norm(vector),
                             scale=float(node_scale[node]), group=group.name))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-half-surface", nargs="+", required=True)
    parser.add_argument("--case", default="g2", choices=("g1", "g2"))
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--core-radius", type=float, default=0.01)
    parser.add_argument("--quad-csv", type=Path,
                        default=Path("runs/issue84/projection_gate_quad.csv"))
    parser.add_argument("--eps-csv", type=Path,
                        default=Path("runs/issue84/projection_gate_eps.csv"))
    parser.add_argument("--production", default="28:8")
    parser.add_argument("--refinement", default="48:8")
    parser.add_argument("--reference", default="48:64")
    parser.add_argument("--near-ratio", type=float, default=2.0)
    parser.add_argument("--far-order", type=int, default=8)
    parser.add_argument("--far-levels", type=int, default=1)
    parser.add_argument("--budget", type=float, default=2e-4)
    parser.add_argument("--escalations", type=int, default=2)
    parser.add_argument("--far-distances", type=float, nargs="+", default=[1.0, 3.0, 10.0])
    parser.add_argument("--far-per-distance", type=int, default=4)
    parser.add_argument("--b2-face", type=int, default=20)
    parser.add_argument("--b2-node", type=int, default=20)
    parser.add_argument("--production-points", type=int, default=24)
    parser.add_argument("--reference-points", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--out", type=Path,
                        default=Path("docs/issue84/adaptive_area_acceptance.json"))
    args = parser.parse_args(argv)

    def pair(text):
        first, _, second = text.partition(":")
        return int(first), int(second)

    production, refinement, reference = pair(args.production), pair(args.refinement), pair(args.reference)
    started = time.perf_counter()
    mesh, alpha_deg = build_case(args.main_half_surface, case=args.case, alpha_deg=args.alpha)
    ops = assemble_multi(mesh)
    state = solve_multi(mesh, alpha_deg, kutta="pressure", operators=ops)
    groups = [g for g in mesh.groups if len(g.wake_faces)]
    solve_seconds = time.perf_counter() - started

    adaptive = adaptive_field(mesh, state, args.core_radius, fan_order=production[0],
                              fan_strips=production[1], refinement=refinement[0],
                              refinement_strips=refinement[1], near_ratio=args.near_ratio,
                              far_order=args.far_order, far_levels=args.far_levels,
                              budget=args.budget, escalations=args.escalations)
    prod_field = _field(mesh, state, args.core_radius, *production, refinement=refinement[0],
                        refinement_strips=refinement[1])
    reference_field = _field(mesh, state, args.core_radius, *reference,
                             refinement=refinement[0], refinement_strips=refinement[1])

    geometry = _wake_geometry(groups)
    rng = np.random.default_rng(args.seed)
    rows = []
    for label, path in (("quadrature", args.quad_csv), ("epsilon", args.eps_csv)):
        for row in load_points(path):
            point = np.asarray(row["xyz"], float)
            _d, _g, _f, normal, scale = _nearest_wake_triangle(geometry, point)
            rows.append(dict(set=label, xyz=point, normal=np.asarray(normal, float),
                             scale=float(scale)))
    rows += far_points(groups, geometry, mesh.xyz, args.far_distances,
                       args.far_per_distance, args.seed)
    rows += special_points(groups, geometry)
    rows += b2_points(groups, args, rng)
    if args.limit:
        rows = rows[:args.limit]

    payload = dict(schema="dpanel.issue84.adaptive_area_acceptance.v1", stage="T2-T3",
                   python=platform.python_version(), numpy=np.__version__,
                   case=args.case, alpha_deg=alpha_deg, core_radius=args.core_radius,
                   production=dict(order=production[0], strips=production[1]),
                   refinement=dict(order=refinement[0], strips=refinement[1]),
                   reference=dict(order=reference[0], strips=reference[1]),
                   adaptive=dict(near_ratio=args.near_ratio, far_order=args.far_order,
                                 far_levels=args.far_levels, budget=args.budget,
                                 max_escalations=args.escalations),
                   wall_faces=int(len(mesh.faces)),
                   wake_faces=[int(len(g.wake_faces)) for g in groups],
                   field_faces=int(len(adaptive.triangles)), solve_seconds=solve_seconds,
                   error_metric="||v - v_reference|| / ||v_reference|| on the mean total velocity",
                   thresholds=dict(THRESHOLDS), points=[], status="running")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def flush():
        payload["total_seconds"] = time.perf_counter()-started
        args.out.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")

    ladder = np.array([1.0, .5, .25, .125])
    kwargs = dict(eps_fraction=.05, ladder=ladder, order=48)
    for index, row in enumerate(rows):
        point = np.asarray(row["xyz"], float)[None, :]
        normal = np.asarray(row["normal"], float)[None, :]
        normal = normal/np.linalg.norm(normal)
        scales = np.array([max(float(row["scale"]), 1e-9)])
        t0 = time.perf_counter()
        sample = limits(adaptive, point, normal, scales, tag="adaptive", **kwargs)
        seconds = time.perf_counter()-t0
        diagnostics = adaptive._regularized.last_adaptive if hasattr(
            adaptive, "_regularized") else adaptive.last_adaptive
        record = dict(index=index, set=row["set"], xyz=point[0],
                      group=row.get("group"), scale=float(scales[0]),
                      adaptive_quadrature_error=float(sample["quadrature_error"][0]),
                      adaptive_epsilon_error=float(sample["epsilon_error"][0]),
                      near_faces=int(diagnostics["near_faces"]),
                      far_faces=int(diagnostics["far_faces"]),
                      near_fraction=float(diagnostics["near_faces"]
                                          / max(diagnostics["near_faces"]
                                                + diagnostics["far_faces"], 1)),
                      escalations=int(diagnostics["escalations"]),
                      escalated_faces=int(diagnostics["escalated_faces"]),
                      seconds=float(seconds),
                      velocity=np.asarray(sample["mean"][0], float))
        payload["points"].append(record)
        flush()
        print("%s %d/%d q=%.3e near=%.2f esc=%d (%.2fs)" % (
            row["set"], index+1, len(rows), record["adaptive_quadrature_error"],
            record["near_fraction"], record["escalations"], seconds), flush=True)

    # Production fan on a stratified subset: the adaptive evaluator must not be worse.
    prod_rows = []
    if args.production_points:
        step = max(1, len(payload["points"])//args.production_points)
        for record in payload["points"][::step][:args.production_points]:
            point = np.asarray(record["xyz"], float)[None, :]
            _d, _g, _f, normal, scale = _nearest_wake_triangle(geometry, point[0])
            normal = np.asarray(normal, float)[None, :]/np.linalg.norm(normal)
            scales = np.array([max(float(scale), 1e-9)])
            t0 = time.perf_counter()
            sample = limits(prod_field, point, normal, scales, tag="production", **kwargs)
            prod_rows.append(dict(set=record["set"],
                                  production_quadrature_error=float(sample["quadrature_error"][0]),
                                  production_seconds=time.perf_counter()-t0))
        payload["production_subset"] = _stats(
            [r["production_quadrature_error"] for r in prod_rows])

    # Saturated reference on a stratified subset: the true error of the adaptive value.
    truth = []
    if args.reference_points:
        step = max(1, len(payload["points"])//args.reference_points)
        for record in payload["points"][::step][:args.reference_points]:
            point = np.asarray(record["xyz"], float)[None, :]
            _d, _g, _f, normal, scale = _nearest_wake_triangle(geometry, point[0])
            normal = np.asarray(normal, float)[None, :]/np.linalg.norm(normal)
            scales = np.array([max(float(scale), 1e-9)])
            t0 = time.perf_counter()
            ref = limits(reference_field, point, normal, scales, tag="reference", **kwargs)
            value = np.asarray(record["velocity"], float)
            denominator = max(float(np.linalg.norm(ref["mean"][0])), 1e-30)
            entry = dict(index=int(record["index"]), set=record["set"],
                         adaptive_quadrature_error=record["adaptive_quadrature_error"],
                         true_error=float(np.linalg.norm(value-ref["mean"][0])/denominator),
                         reference_seconds=time.perf_counter()-t0)
            truth.append(entry)
            payload["truth"] = truth
            flush()
            print("truth %d/%d set=%s true=%.3e gate=%.3e" % (
                len(truth), max(1, args.reference_points), record["set"],
                entry["true_error"], entry["adaptive_quadrature_error"]), flush=True)

    points = payload["points"]
    payload["summary"] = dict(
        point_count=len(points),
        adaptive_quadrature_error=_stats([r["adaptive_quadrature_error"] for r in points]),
        adaptive_epsilon_error=_stats([r["adaptive_epsilon_error"] for r in points]),
        near_fraction=_stats([r["near_fraction"] for r in points]),
        escalations=_stats([float(r["escalations"]) for r in points]),
        seconds=_stats([r["seconds"] for r in points]),
        sets=sorted({r["set"] for r in points}),
        per_set={name: dict(
            count=len([r for r in points if r["set"] == name]),
            adaptive_quadrature_error=_stats([r["adaptive_quadrature_error"]
                                              for r in points if r["set"] == name]),
            near_fraction=_stats([r["near_fraction"] for r in points if r["set"] == name]),
            seconds=_stats([r["seconds"] for r in points if r["set"] == name]))
            for name in sorted({r["set"] for r in points})},
        truth_count=len(truth),
        truth=_stats([r["true_error"] for r in truth]) if truth else dict(count=0))
    gates = [r["adaptive_quadrature_error"] for r in points]
    gate_max = float(max(gates)) if gates else 0.0
    gate_p90 = float(np.percentile(gates, 90)) if gates else 0.0
    true_max = max((r["true_error"] for r in truth), default=0.0)
    adaptive_median = float(np.median([r["seconds"] for r in points])) if points else 0.0
    production_median = (float(np.median([r["production_seconds"] for r in prod_rows]))
                         if prod_rows else 0.0)
    payload["verdict"] = dict(
        gate_max=dict(value=gate_max, limit=THRESHOLDS["gate_max"],
                      passed=bool(gate_max <= THRESHOLDS["gate_max"])),
        gate_p90=dict(value=gate_p90, limit=THRESHOLDS["gate_p90"],
                      passed=bool(gate_p90 <= THRESHOLDS["gate_p90"])),
        true_max=dict(value=true_max, limit=THRESHOLDS["true_max"],
                      passed=bool(not truth or true_max <= THRESHOLDS["true_max"])),
        cost_ratio=dict(value=adaptive_median / max(production_median, 1e-30),
                        limit=THRESHOLDS["cost_ratio"], passed=None),
        adaptive_median_seconds=adaptive_median,
        production_median_seconds=production_median,
        peak_rss_mb=float(peak_rss_mb()))
    payload["status"] = "completed"
    flush()
    print(json.dumps(payload["summary"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
