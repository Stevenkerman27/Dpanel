"""End-to-end check of the subdivision rung through the production limits() gate.

The kernel-level experiment (docs/ISSUE84_SLIVER_SUBDIVISION.md) split the sliver faces
by hand. This tool asks the same question through the production path: build the G2
coupled case, construct TotalVelocity with projection_fan_adaptive, and call
panel.p1_coupled_material_wake.limits() on the special targets with the rung off
(levels 0) and on (levels 2), recording the gate value, the diagnostics and the cost.

    python -m tools.issue84.subdivision_rung_check \
        --main-half-surface D:/3D/Projects/panel/runs/half_cad_wingbody_v1/surface.json \
        --sets trailing_edge high_aspect sliver tip root \
        --out docs/issue84/subdivision_rung_gate.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from panel.p1_coupled_material_wake import TotalVelocity, limits
from panel.p1_lifting_g2 import assemble_multi, solve_multi
from tools.issue84.adaptive_area_acceptance import special_points
from tools.issue84.fan_quadrature_gate import _wake_geometry
from tools.issue84.wake_node_rejection import build_case


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, bool) or value is None:
        return value
    return str(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-half-surface", nargs="+", required=True)
    parser.add_argument("--case", default="g2", choices=("g1", "g2"))
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--core-radius", type=float, default=0.01)
    parser.add_argument("--fan", default="28:8")
    parser.add_argument("--refinement", default="48:8")
    parser.add_argument("--near-ratio", type=float, default=2.0)
    parser.add_argument("--far-order", type=int, default=8)
    parser.add_argument("--far-levels", type=int, default=1)
    parser.add_argument("--budget", type=float, default=2e-4)
    parser.add_argument("--escalations", type=int, default=2)
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 2])
    parser.add_argument("--subdivision-fan", default="48:8")
    parser.add_argument("--sets", nargs="+",
                        default=["trailing_edge", "high_aspect", "sliver", "tip", "root"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=Path("docs/issue84/subdivision_rung_gate.json"))
    args = parser.parse_args(argv)

    def pair(text):
        first, _, second = text.partition(":")
        return int(first), int(second)

    fan, refinement = pair(args.fan), pair(args.refinement)
    subdivision = pair(args.subdivision_fan)
    started = time.perf_counter()
    mesh, alpha_deg = build_case(args.main_half_surface, case=args.case,
                                 alpha_deg=args.alpha)
    operators = assemble_multi(mesh)
    state = solve_multi(mesh, alpha_deg, kutta="pressure", operators=operators)
    groups = [g for g in mesh.groups if len(g.wake_faces)]
    solve_seconds = time.perf_counter() - started
    geometry = _wake_geometry(groups)

    rows = [row for row in special_points(groups, geometry)
            if row["set"].replace("special_", "") in set(args.sets)]
    if args.limit:
        rows = rows[:args.limit]
    print("targets: %d, solve %.1f s" % (len(rows), solve_seconds), flush=True)

    fields = {}
    for levels in args.levels:
        fields[levels] = TotalVelocity(
            mesh, state, method="rm_regularized_p1", core_radius=float(args.core_radius),
            area_quadrature="projection_fan_adaptive",
            fan_order=fan[0], fan_strips=fan[1],
            fan_refinement_order=refinement[0],
            fan_refinement_strips=refinement[1],
            near_ratio=args.near_ratio, adaptive_far_order=args.far_order,
            adaptive_far_levels=args.far_levels, adaptive_budget=args.budget,
            adaptive_max_escalations=args.escalations,
            adaptive_subdivision_levels=int(levels),
            adaptive_subdivision_fan=subdivision)

    ladder = np.array([1.0, .5, .25, .125])
    kwargs = dict(eps_fraction=.05, ladder=ladder, order=48)
    payload = dict(case=args.case, alpha_deg=float(alpha_deg),
                   core_radius=float(args.core_radius), fan=args.fan,
                   refinement=args.refinement, budget=float(args.budget),
                   escalations=int(args.escalations),
                   subdivision_fan=args.subdivision_fan,
                   solve_seconds=solve_seconds, points=[])

    def flush():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))

    for index, row in enumerate(rows):
        point = np.asarray(row["xyz"], float)[None, :]
        normal = np.asarray(row["normal"], float)[None, :]
        normal = normal/np.linalg.norm(normal)
        scales = np.array([max(float(row["scale"]), 1e-9)])
        record = dict(index=index, set=row["set"], xyz=point[0],
                      scale=float(scales[0]), runs={})
        for levels in args.levels:
            field = fields[levels]
            begin = time.perf_counter()
            sample = limits(field, point, normal, scales, tag="rung_%d" % levels, **kwargs)
            seconds = time.perf_counter() - begin
            diagnostics = field._regularized.last_adaptive
            record["runs"][str(levels)] = dict(
                quadrature_error=float(sample["quadrature_error"][0]),
                epsilon_error=float(sample["epsilon_error"][0]),
                mean=np.asarray(sample["mean"][0], float),
                seconds=float(seconds),
                near_faces=int(diagnostics["near_faces"]),
                far_faces=int(diagnostics["far_faces"]),
                escalations=int(diagnostics["escalations"]),
                escalated_faces=int(diagnostics["escalated_faces"]),
                subdivision_faces=int(diagnostics.get("subdivision_faces", 0)),
                subdivision_pieces=int(diagnostics.get("subdivision_pieces", 0)),
                subdivision_seconds=float(diagnostics.get("subdivision_seconds", 0.0)),
                faces_over_budget=int(diagnostics["faces_over_budget"]))
        payload["points"].append(record)
        flush()
        summary = " ".join("L%s q=%.3e esc=%d sub=%d (%.1fs)" % (
            key, run["quadrature_error"], run["escalations"], run["subdivision_faces"],
            run["seconds"]) for key, run in record["runs"].items())
        print("%s %d/%d %s" % (record["set"], index+1, len(rows), summary), flush=True)

    payload["seconds"] = time.perf_counter() - started
    flush()
    print("wrote %s in %.1f s" % (args.out, payload["seconds"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
