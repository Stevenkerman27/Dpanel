"""Per-face versus aggregate quadrature gate on the frozen 109-point set (issue #84).

The production gate reports estimate.sum() / |sum(base)|: a sum of per-face
norms.  The functional it is meant to protect is the velocity at one point, which
is a vector sum over faces and therefore allows per-face errors to cancel.  This
tool re-runs the same production limits() call on the 109 frozen G2 points and
records both statistics side by side, so the cost of changing the reported number
can be read off directly.

Read-only: the gate itself still reports the per-face value.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from panel.p1_coupled_material_wake import limits
from panel.p1_lifting_g2 import assemble_multi, solve_multi
from panel.p1_coupled_material_wake import TotalVelocity
from tools.issue84.adaptive_area_acceptance import adaptive_field
from tools.issue84.fan_quadrature_gate import _jsonable, _nearest_wake_triangle, _stats, \
    _wake_geometry
from tools.issue84.projection_gate_set import load_points
from tools.issue84.wake_node_rejection import build_case

TOLERANCE = 2e-4


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
    parser.add_argument("--near-ratio", type=float, default=2.0)
    parser.add_argument("--budget", type=float, default=2e-4)
    parser.add_argument("--escalations", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path("docs/issue84/aggregate_gate_109.json"))
    args = parser.parse_args(argv)

    def pair(text):
        first, _, second = text.partition(":")
        return int(first), int(second)

    production, refinement = pair(args.production), pair(args.refinement)
    started = time.perf_counter()
    mesh, alpha_deg = build_case(args.main_half_surface, case=args.case, alpha_deg=args.alpha)
    ops = assemble_multi(mesh)
    state = solve_multi(mesh, alpha_deg, kutta="pressure", operators=ops)
    groups = [g for g in mesh.groups if len(g.wake_faces)]
    solve_seconds = time.perf_counter() - started

    field = adaptive_field(mesh, state, args.core_radius, fan_order=production[0],
                           fan_strips=production[1], refinement=refinement[0],
                           refinement_strips=refinement[1], near_ratio=args.near_ratio,
                           far_order=8, far_levels=1, budget=args.budget,
                           escalations=args.escalations)
    rows = []
    for label, path in (("quadrature", args.quad_csv), ("epsilon", args.eps_csv)):
        for row in load_points(path):
            rows.append(dict(set=label, xyz=np.asarray(row["xyz"], float),
                             reasons=row.get("reasons")))
    if args.limit:
        rows = rows[:args.limit]

    payload = dict(
        schema="dpanel.issue84.aggregate_gate.v1", stage="aggregate-vs-per-face",
        python=platform.python_version(), numpy=np.__version__,
        case=args.case, alpha_deg=alpha_deg, core_radius=args.core_radius,
        production=dict(order=production[0], strips=production[1]),
        refinement=dict(order=refinement[0], strips=refinement[1]),
        tolerance=TOLERANCE, wall_faces=int(len(mesh.faces)),
        field_faces=int(len(field.triangles)), solve_seconds=solve_seconds,
        per_face_metric="sum_f ||diff_f|| / |sum_f base_f|",
        aggregate_metric="||sum_f gamma_f x diff_f|| / |sum_f base_f|",
        points=[], status="running")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def flush():
        payload["total_seconds"] = time.perf_counter() - started
        args.out.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")

    geometry = _wake_geometry(groups)
    kwargs = dict(eps_fraction=.05, ladder=np.array([1.0, .5, .25, .125]), order=48)

    for index, row in enumerate(rows):
        point = row["xyz"]
        t0 = time.perf_counter()
        _distance, _group, _face, normal, scale = _nearest_wake_triangle(geometry, point)
        sample = limits(field, point[None, :], normal[None, :]/np.linalg.norm(normal),
                        np.array([max(scale, 1e-9)]), tag="aggregate", **kwargs)
        diagnostics = field._regularized.last_adaptive
        payload["points"].append(dict(
            index=index, set=row["set"], reasons=row.get("reasons"), xyz=point.tolist(),
            scale=float(scale),
            per_face_relative=float(np.asarray(sample["quadrature_error"]).reshape(-1)[0]),
            aggregate_relative=float(diagnostics["aggregate_relative_estimate"]),
            per_face_estimate=float(diagnostics["relative_estimate"]),
            aggregate_vector=np.asarray(diagnostics["aggregate_vector"]).tolist(),
            mean=np.asarray(sample["mean"]).reshape(-1).tolist(),
            epsilon_error=float(np.asarray(sample["epsilon_error"]).reshape(-1)[0]),
            seconds=time.perf_counter() - t0))
        flush()
        entry = payload["points"][-1]
        print("%s %d/%d perf=%.3e agg=%.3e (%.2fs)" % (
            row["set"], index + 1, len(rows), entry["per_face_relative"],
            entry["aggregate_relative"], entry["seconds"]), flush=True)

    per_face = [p["per_face_relative"] for p in payload["points"]]
    aggregate = [p["aggregate_relative"] for p in payload["points"]]
    payload["summary"] = dict(
        per_face=_stats(per_face), aggregate=_stats(aggregate),
        per_face_over_tol=int(sum(v > TOLERANCE for v in per_face)),
        aggregate_over_tol=int(sum(v > TOLERANCE for v in aggregate)))
    ratio = [float(p["per_face_relative"]/max(p["aggregate_relative"], 1e-300))
             for p in payload["points"]]
    payload["summary"]["ratio"] = _stats(ratio)
    payload["status"] = "completed"
    flush()
    print(json.dumps(_jsonable(payload["summary"]), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
