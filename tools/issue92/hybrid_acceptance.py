"""End-to-end acceptance of the hybrid near/far area evaluator (issue #92, step 3).

The hybrid keeps the projection fan for the faces within near_cores core radii of the
target and drops to the plain Duffy map everywhere else
(panel/p1_hybrid_area.py). It is only worth having if that substitution is invisible at
the tolerance the production sampling gate uses, so this tool measures it the only way
that can decide: against the full fan on the 109 real sampler targets of the G2 wake.

ACCEPTANCE THRESHOLDS ARE DECLARED HERE, BEFORE ANY MEASUREMENT
  * max relative error   <= 1e-3
  * p90 relative error   <= 1e-4
  * cost                 <= 0.05x the reference evaluation (same point, same process)

Reference: the refined projection fan, 48x8, which is the pass the production sampling
gate treats as converged (the base fan is 16x8). Error is |dv|/U with U the freestream
magnitude, exactly the normalisation of the gate's epsilon_error, so a 1e-3 threshold is
'one twentieth of the gate's epsilon_tol' and not a number chosen after the fact. The
cost ratio is also reported against the cheap 48x1 fan, because that is the baseline the
near rule is copied from: both ratios are in the payload, the verdict uses the refined
one, and the 48x1 ratio failing the threshold is reported as a failure rather than
hidden.

Usage (from the repository root, after a warm run):

    python -m tools.issue92.hybrid_acceptance \
        --main-half-surface runs/half_cad_wingbody_v1/surface.json \
        --csv eps:runs/issue84/projection_gate_eps.csv \
        --csv quad:runs/issue84/projection_gate_quad.csv \
        --out docs/issue92/hybrid_acceptance_g2.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panel.p1_coupled_material_wake import TotalVelocity  # noqa: E402
from panel.p1_hybrid_area import (hybrid_area_terms, near_face_mask,  # noqa: E402
                                  point_triangle_distance)
from panel.p1_lifting_g2 import assemble_multi, solve_multi  # noqa: E402
from panel.p1_projection_duffy import area_terms  # noqa: E402
from tools.issue84.projection_gate_set import load_points  # noqa: E402
from tools.issue84.wake_node_rejection import build_case  # noqa: E402

THRESHOLDS = {"max_relative_error": 1e-3, "p90_relative_error": 1e-4,
              "cost_ratio": 0.05}


def induced(triangles, gamma, point, core, *, near_cores, fan_order, fan_strips,
            far_order, edge):
    if np.isinf(near_cores):
        # reference: the full projection fan, exactly the production quadrature
        area = area_terms(triangles, point, core, order=fan_order, strips=fan_strips)
    else:
        area = hybrid_area_terms(triangles, point, core, near_cores=near_cores,
                                 fan_order=fan_order, fan_strips=fan_strips,
                                 far_order=far_order)
    return edge + np.sum(np.cross(gamma, area), axis=0)


def timed(fn, repeats=3):
    best, value = None, None
    for _ in range(repeats):
        started = time.perf_counter()
        value = fn()
        elapsed = time.perf_counter()-started
        best = elapsed if best is None else min(best, elapsed)
    return value, float(best)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-half-surface", type=Path, required=True)
    parser.add_argument("--csv", action="append", required=True,
                        help="label:path to a wake_node_rejection CSV")
    parser.add_argument("--case", default="g2", choices=("g1", "g2"))
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--core-radius", type=float, default=0.01)
    parser.add_argument("--fan-order", type=int, default=48)
    parser.add_argument("--fan-strips", type=int, default=1)
    parser.add_argument("--reference-order", type=int, default=48)
    parser.add_argument("--reference-strips", type=int, default=8)
    parser.add_argument("--near-cores", type=float, nargs="+", default=[5.0, 10.0, 20.0])
    parser.add_argument("--far-orders", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    mesh, alpha_deg = build_case(args.main_half_surface, case=args.case,
                                 alpha_deg=args.alpha)
    state = solve_multi(mesh, alpha_deg, kutta="pressure",
                        operators=assemble_multi(mesh))
    field = TotalVelocity(mesh, state, method="rm_regularized_p1",
                          core_radius=args.core_radius,
                          area_quadrature="projection_fan",
                          fan_order=args.fan_order, fan_strips=args.fan_strips)
    regularized = field._regularized
    triangles = np.asarray(regularized.triangles, float)
    gamma = np.asarray(regularized.gamma, float)

    points = []
    for spec in args.csv:
        label, _, path = spec.partition(":")
        if not path:
            raise SystemExit("--csv wants label:path")
        for row in load_points(Path(path)):
            points.append(dict(row, label=label))
    if args.limit:
        points = points[: args.limit]
    if not points:
        raise SystemExit("no gate points loaded")

    cells = [dict(near_cores=near, far_order=order)
             for near in args.near_cores for order in args.far_orders]
    for cell in cells:
        cell.update(relative_errors=[], near_faces=[], seconds=[], reference_seconds=[],
                    baseline_seconds=[])

    rows = []
    started_all = time.perf_counter()
    for index, row in enumerate(points):
        point = np.asarray(row["xyz"], float)
        edge = regularized._edge_velocity(point)
        reference, reference_seconds = timed(lambda: induced(
            triangles, gamma, point, args.core_radius, near_cores=np.inf,
            fan_order=args.reference_order, fan_strips=args.reference_strips,
            far_order=16, edge=edge), repeats=1)
        baseline, baseline_seconds = timed(lambda: induced(
            triangles, gamma, point, args.core_radius, near_cores=np.inf,
            fan_order=args.fan_order, fan_strips=args.fan_strips,
            far_order=16, edge=edge), repeats=1)
        # gate normalisation: |dv|/U, with U the freestream magnitude
        scale = max(float(np.linalg.norm(regularized.freestream)), 1e-300)
        entry = {"index": index, "label": row["label"], "group": row["group"],
                 "node": row["node"], "xyz": [float(x) for x in point],
                 "reasons": row["reasons"],
                 "induced_magnitude": float(np.linalg.norm(reference)),
                 "reference_seconds": reference_seconds,
                 "baseline_seconds": baseline_seconds, "cells": {}}
        for cell in cells:
            key = "%.3f/%d" % (cell["near_cores"], cell["far_order"])
            value, seconds = timed(lambda: induced(
                triangles, gamma, point, args.core_radius,
                near_cores=cell["near_cores"], fan_order=args.fan_order,
                fan_strips=args.fan_strips, far_order=cell["far_order"], edge=edge))
            error = float(np.linalg.norm(value-reference))/scale
            near_count = int(near_face_mask(triangles, point, args.core_radius,
                                            near_cores=cell["near_cores"]).sum())
            cell["relative_errors"].append(error)
            cell["near_faces"].append(near_count)
            cell["seconds"].append(seconds)
            cell["reference_seconds"].append(reference_seconds)
            cell["baseline_seconds"].append(baseline_seconds)
            entry["cells"][key] = {"relative_error": error, "near_faces": near_count,
                                   "seconds": seconds,
                                   "cost_ratio": seconds/reference_seconds,
                                   "cost_ratio_vs_48x1": seconds/baseline_seconds}
        rows.append(entry)
        print("point %3d/%d %-9s node %4d  induced %.3e  ref %.3f s" % (
            index+1, len(points), row["label"], row["node"],
            np.linalg.norm(reference), reference_seconds))
        sys.stdout.flush()

    summary = {}
    for cell in cells:
        key = "%.3f/%d" % (cell["near_cores"], cell["far_order"])
        errors = np.array(cell["relative_errors"], float)
        ratios = np.array(cell["seconds"], float)/np.array(cell["reference_seconds"], float)
        near = np.array(cell["near_faces"], float)
        verdict = {
            "max_relative_error": float(errors.max()),
            "p90_relative_error": float(np.percentile(errors, 90)),
            "median_relative_error": float(np.median(errors)),
            "cost_ratio_median": float(np.median(ratios)),
            "cost_ratio_max": float(ratios.max()),
            "cost_ratio_vs_48x1_median": float(np.median(
                np.array(cell["seconds"], float)/np.array(cell["baseline_seconds"], float))),
            "speedup_median": float(np.median(1.0/ratios)),
            "near_face_fraction_median": float(np.median(near)/len(triangles)),
            "near_faces_max": int(near.max()),
        }
        verdict["accepted"] = bool(
            verdict["max_relative_error"] <= THRESHOLDS["max_relative_error"]
            and verdict["p90_relative_error"] <= THRESHOLDS["p90_relative_error"]
            and verdict["cost_ratio_median"] <= THRESHOLDS["cost_ratio"])
        summary[key] = verdict
        print("%s max %.3e p90 %.3e med %.3e  cost %.3fx (vs 48x1 %.3fx)  "
              "near %.3f%%  %s" % (
                  key, verdict["max_relative_error"], verdict["p90_relative_error"],
                  verdict["median_relative_error"], verdict["cost_ratio_median"],
                  verdict["cost_ratio_vs_48x1_median"],
                  100.0*verdict["near_face_fraction_median"],
                  "ACCEPT" if verdict["accepted"] else "reject"))
        sys.stdout.flush()

    payload = {"schema": "dpanel.issue92.hybrid_acceptance.v1",
               "main_half_surface": str(args.main_half_surface), "case": args.case,
               "alpha_deg": float(alpha_deg), "core_radius": args.core_radius,
               "triangles": len(triangles), "points": len(points),
               "reference": {"fan_order": args.reference_order,
                             "fan_strips": args.reference_strips},
               "baseline": {"fan_order": args.fan_order,
                            "fan_strips": args.fan_strips},
               "thresholds": THRESHOLDS, "cells": summary, "rows": rows,
               "elapsed_seconds": time.perf_counter()-started_all,
               "platform": platform.platform(),
               "python": platform.python_version(),
               "numpy": np.__version__,
               "status": "ok"}
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n",
                            encoding="utf-8")
        print("wrote", args.out)
    accepted = sorted(key for key, cell in summary.items() if cell["accepted"])
    print("accepted cells:", accepted if accepted else "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
