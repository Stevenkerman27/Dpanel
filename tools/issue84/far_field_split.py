"""P3 prototype: split the area kernel into a near fan and a cheap far quadrature.

The P0 profile showed that the fan spends 0.62 s per target at 28/8 because every one of
the 1666 triangles is integrated with 8 sub-triangles x order**2 nodes, even when the
target is many circumradii away and the integrand is smooth there. This tool measures the
alternative: keep the fan for the near set and use the plain (non-projection) Duffy map at
a low order for the far set, then report the error against the full fan and the speedup.

Nothing here is wired into production; it is the measurement that decides whether a
near/far split is worth building.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.issue84.projection_gate_set import load_points  # noqa: E402
from tools.issue84.wake_node_rejection import build_case  # noqa: E402

from panel.p1_coupled_material_wake import TotalVelocity  # noqa: E402
from panel.p1_lifting_g2 import assemble_multi, solve_multi  # noqa: E402
from panel.p1_projection_duffy import area_terms  # noqa: E402


def terms(triangles, point, core, *, fan_order, strips, far_order, ratio):
    """Near set with the projection fan, far set with the plain Duffy map."""
    tri = np.asarray(triangles, float)
    p = np.asarray(point, float)
    centroid = tri.mean(axis=1)
    edges = np.linalg.norm(np.stack((tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 1],
                                     tri[:, 0]-tri[:, 2]), axis=1), axis=2)
    longest = edges.max(axis=1)
    distance = np.linalg.norm(p[None, :]-centroid, axis=1)
    near = distance <= ratio*longest
    out = np.zeros((len(tri), 3))
    if np.any(near):
        out[near] = area_terms(tri[near], p, core, order=fan_order, strips=strips)
    if np.any(~near):
        out[~near] = area_terms(tri[~near], p, core, order=far_order, strips=1)
    return out, int(near.sum())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-half-surface", type=Path, required=True)
    parser.add_argument("--csv", action="append", required=True)
    parser.add_argument("--case", default="g2", choices=("g1", "g2"))
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--core-radius", type=float, default=0.01)
    parser.add_argument("--fan-order", type=int, default=28)
    parser.add_argument("--fan-strips", type=int, default=8)
    parser.add_argument("--reference-order", type=int, default=48)
    parser.add_argument("--reference-strips", type=int, default=32)
    parser.add_argument("--far-order", type=int, default=6)
    parser.add_argument("--ratio", type=float, default=4.0)
    parser.add_argument("--ratios", type=float, nargs="+", default=None,
                        help="sweep near/far split ratios (overrides --ratio)")
    parser.add_argument("--far-orders", type=int, nargs="+", default=None,
                        help="sweep far quadrature orders (overrides --far-order)")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    mesh, alpha_deg = build_case(args.main_half_surface, case=args.case, alpha_deg=args.alpha)
    state = solve_multi(mesh, alpha_deg, kutta="pressure", operators=assemble_multi(mesh))
    field = TotalVelocity(mesh, state, method="rm_regularized_p1",
                          core_radius=args.core_radius, area_quadrature="projection_fan",
                          fan_order=args.fan_order, fan_strips=args.fan_strips)
    tri = np.asarray(field._regularized.triangles, float)
    gamma = np.asarray(field._regularized.gamma, float)
    points = []
    for spec in args.csv:
        _label, _, path = spec.partition(":")
        points.extend(np.asarray(row["xyz"], float) for row in load_points(Path(path)))
    if args.limit:
        points = points[: args.limit]
    ratios = args.ratios if args.ratios is not None else [args.ratio]
    far_orders = args.far_orders if args.far_orders is not None else [args.far_order]
    payload = {"schema": "dpanel.issue84.far_field_split.v1", "cad": str(args.main_half_surface),
               "case": args.case, "alpha_deg": alpha_deg, "core_radius": args.core_radius,
               "triangles": len(tri), "points": len(points), "ratios": ratios,
               "far_orders": far_orders, "rows": [], "cells": {}}
    for point in points:
        started = time.perf_counter()
        reference, _ = terms(tri, point, args.core_radius, fan_order=args.reference_order,
                             strips=args.reference_strips, far_order=args.reference_order,
                             ratio=float("inf"))
        reference_seconds = time.perf_counter()-started
        started = time.perf_counter()
        production, _ = terms(tri, point, args.core_radius, fan_order=args.fan_order,
                              strips=args.fan_strips, far_order=args.reference_order,
                              ratio=float("inf"))
        production_seconds = time.perf_counter()-started
        velocity_reference = np.sum(np.cross(gamma, reference), axis=0)
        velocity_production = np.sum(np.cross(gamma, production), axis=0)
        scale = max(float(np.linalg.norm(velocity_reference)), 1e-300)
        reference_norm = scale
        rows = {"xyz": [float(x) for x in point],
                "reference_seconds": reference_seconds,
                "production_seconds": production_seconds,
                "production_relative_error": float(np.linalg.norm(
                    velocity_production-velocity_reference))/reference_norm,
                "cells": {}}
        for ratio in ratios:
            for far_order in far_orders:
                started = time.perf_counter()
                candidate, near_count = terms(tri, point, args.core_radius,
                                              fan_order=args.fan_order,
                                              strips=args.fan_strips,
                                              far_order=far_order, ratio=ratio)
                candidate_seconds = time.perf_counter()-started
                velocity_candidate = np.sum(np.cross(gamma, candidate), axis=0)
                rows["cells"]["%.3f/%d" % (ratio, far_order)] = {
                    "near_faces": near_count,
                    "near_fraction": near_count/len(tri),
                    "candidate_seconds": candidate_seconds,
                    "relative_velocity_error": float(np.linalg.norm(
                        velocity_candidate-velocity_reference))/reference_norm}
        payload["rows"].append(rows)
        sys.stdout.flush()
    for ratio in ratios:
        for far_order in far_orders:
            key = "%.3f/%d" % (ratio, far_order)
            cell = {"ratio": ratio, "far_order": far_order}
            for field, pick in (("relative_velocity_error", "rel_error"),
                                ("candidate_seconds", "seconds"),
                                ("near_fraction", "near_fraction")):
                values = np.array([row["cells"][key][field] for row in payload["rows"]], float)
                cell[pick+"_median"] = float(np.median(values))
                cell[pick+"_max"] = float(values.max())
            cell["speedup_vs_production_median"] = float(np.median(
                [row["production_seconds"]/row["cells"][key]["candidate_seconds"]
                 for row in payload["rows"]]))
            payload["cells"][key] = cell
            print("%s near %.3f relerr med %.2e max %.2e  %.3f s  %.2fx vs production" % (
                key, cell["near_fraction_median"], cell["rel_error_median"],
                cell["rel_error_max"], cell["seconds_median"],
                cell["speedup_vs_production_median"]))
            sys.stdout.flush()
    if payload["rows"]:
        payload["production_relative_error_max"] = float(max(
            row["production_relative_error"] for row in payload["rows"]))
        payload["reference_seconds_median"] = float(np.median(
            [row["reference_seconds"] for row in payload["rows"]]))
        print("production fan %.0f/%d relerr max %.2e vs reference %.1f s/point" % (
            args.fan_order, args.fan_strips, payload["production_relative_error_max"],
            payload["reference_seconds_median"]))
    payload["status"] = "ok"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
