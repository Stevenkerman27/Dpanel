"""Per-configuration kernel profile on the frozen 109-point G2 set (issue #84, P0).

Measures what a *single* total-velocity evaluation costs and decomposes it: how many
triangles are visited, how many fan sub-triangles are generated per target, which
projection region dominates, and how the wall clock splits between the fan partition,
the Duffy node generation, the kernel contraction and the aggregation.

This is a measurement tool only: it calls the production evaluator and never changes
it. profiling counters are inert unless this tool enables them.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from panel import velocity_profile
from panel.p1_coupled_material_wake import TotalVelocity
from panel.p1_lifting_g2 import assemble_multi, solve_multi
from tools.issue84.projection_gate_set import load_points
from tools.issue84.wake_node_rejection import build_case


def _split(counts, targets, faces, seconds):
    """Per-target and per-face normalisations of one profiling block."""
    row = dict(counters=counts, seconds={k: round(v, 6) for k, v in seconds.items()})
    if targets:
        row["calls_per_target"] = round(counts.get("area_terms.calls", 0)/targets, 6)
        row["triangles_per_target"] = round(counts.get("area_terms.triangles", 0)/targets, 3)
        row["subtriangles_per_target"] = round(
            counts.get("fan.subtriangles", counts.get("area_terms.subtriangles", 0))/targets, 6)
        row["strips_per_target"] = round(counts.get("fan.strips", 0)/targets, 6)
        row["fan_calls_per_target"] = round(counts.get("fan.calls", 0)/targets, 3)
        row["node_sets_per_target"] = round(counts.get("area_terms.node_sets", 0)/targets, 3)
        row["milliseconds_per_target"] = round(
            1000.0*sum(seconds.values())/targets, 3)
    row["triangles"] = faces
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-half-surface", nargs="+", required=True)
    parser.add_argument("--csv", action="append", required=True, help="label:path")
    parser.add_argument("--case", default="g2", choices=("g1", "g2"))
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--core-radius", type=float, default=0.01)
    parser.add_argument("--configs", nargs="+", default=["legacy_duffy", "16:8", "24:8", "28:8", "48:8"],
                        help="legacy_duffy, or fan order:strips")
    parser.add_argument("--quadrature-order", type=int, default=48,
                        help="legacy Duffy area quadrature order (the A5 baseline column)")
    parser.add_argument("--limit", type=int, default=0, help="profile only the first N points")
    parser.add_argument("--out", type=Path, default=Path("docs/issue84/velocity_profile_109.json"))
    args = parser.parse_args(argv)

    started = time.perf_counter()
    mesh, alpha_deg = build_case(args.main_half_surface, case=args.case, alpha_deg=args.alpha)
    ops = assemble_multi(mesh)
    state = solve_multi(mesh, alpha_deg, kutta="pressure", operators=ops)
    solve_seconds = time.perf_counter() - started
    sets = {}
    for spec in args.csv:
        label, _, path = spec.partition(":")
        if not path:
            raise SystemExit("--csv wants label:path")
        rows = load_points(path)
        sets[label] = rows[:args.limit] if args.limit else rows

    payload = dict(schema="dpanel.issue84.velocity_profile.v1", stage="P0",
                   python=platform.python_version(), numpy=np.__version__,
                   case=args.case, alpha_deg=alpha_deg, core_radius=args.core_radius,
                   quadrature_order=args.quadrature_order,
                   wall_faces=int(len(mesh.faces)),
                   wake_faces=[int(len(g.wake_faces)) for g in mesh.groups],
                   solve_seconds=solve_seconds, sets={}, configs={})
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def flush():
        payload["total_seconds"] = time.perf_counter() - started
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for label, rows in sets.items():
        payload["sets"][label] = np.array([row["xyz"] for row in rows], float).tolist()

    for spec in args.configs:
        if spec == "legacy_duffy":
            field = TotalVelocity(mesh, state, method="rm_regularized_p1",
                                  core_radius=args.core_radius,
                                  area_quadrature="legacy_duffy")
            name = "legacy_duffy/%d" % args.quadrature_order
        else:
            order, strips = (int(part) for part in spec.split(":"))
            field = TotalVelocity(mesh, state, method="rm_regularized_p1",
                                  core_radius=args.core_radius,
                                  area_quadrature="projection_fan",
                                  fan_order=order, fan_strips=strips)
            name = "%d/%d" % (order, strips)
        block = dict(config=name)
        for label, rows in sets.items():
            points = np.array([row["xyz"] for row in rows], float)
            velocity_profile.reset()
            velocity_profile.enable(True)
            t0 = time.perf_counter()
            values = np.stack([np.asarray(field(point), float) for point in points])
            elapsed = time.perf_counter() - t0
            velocity_profile.enable(False)
            snap = velocity_profile.snapshot()
            row = _split(snap["counters"], len(points), int(len(field.triangles)),
                         snap["seconds"])
            row.update(points=int(len(points)), wall_seconds=round(elapsed, 3),
                       milliseconds_per_point=round(1000.0*elapsed/len(points), 3),
                       finite=bool(np.isfinite(values).all()))
            block[label] = row
            print("%s %s %d pts %.3f s (%.3f ms/pt)" % (name, label, len(points), elapsed,
                                                        1000.0*elapsed/len(points)),
                  flush=True)
            payload["configs"][name] = block
            flush()
        payload["configs"][name] = block
        flush()
    payload["status"] = "completed"
    flush()
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
