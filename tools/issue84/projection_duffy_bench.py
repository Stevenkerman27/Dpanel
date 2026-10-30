"""Composite-Duffy oracle and strategy comparison for the projection-aware quadrature.

Reference scheme (independent of the projection fan): split the parent triangle into
n*n equal sub-triangles in barycentric space and integrate each one with its own Duffy
map at the given order. The reference is accepted only when doubling n and raising the
order changes the integral by less than --ref-tol.

Error metric: the induced vector cross(gamma, area term) with gamma taken from the shipped
evaluator, so legacy and fan rows are directly comparable and the reported numbers are
the area part of the velocity, not a scalar surrogate.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from panel.p1_projection_duffy import area_terms
from panel.p1_regularized_sheet_velocity import RegularizedP1SheetField


def uniform_subdivide(tri, n):
    """n*n equal sub-triangles of a triangle (barycentric grid, orientation preserved)."""
    a, b, c = np.asarray(tri, float)
    t = np.arange(n + 1) / n
    grid = (a[None, None, :] + t[None, :, None] * (b - a)[None, None, :]
            + t[:, None, None] * (c - a)[None, None, :])
    out = []
    for i in range(n):
        for j in range(n - i):
            out.append([grid[i, j], grid[i + 1, j], grid[i, j + 1]])
            if i + j + 1 < n:
                out.append([grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]])
    return np.asarray(out)


def composite_reference(tri, point, core, *, n=32, order=24):
    subs = uniform_subdivide(tri, n)
    # the integral over the parent is the sum over its sub-triangles
    return area_terms(subs, point, core, order=order, strips=1).sum(axis=0), len(subs)


def _field(tri, core):
    return RegularizedP1SheetField(
        [(np.asarray(tri, float), np.array([[0, 1, 2]]), np.array([[0.0, 1.0, 0.37]]))],
        core_radius=core,
    )


def legacy_induced(field, point, order):
    """Area-only induced vector from the shipped evaluator (single Duffy)."""
    return field._area_velocity(point, order)


def aspect_triangle(aspect, length=1.0):
    return np.array([[0.0, 0.0, 0.0], [length, 0.0, 0.0], [0.0, length / aspect, 0.0]])


def _rel(got, ref):
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/issue84/projection_duffy_bench.json")
    ap.add_argument("--aspects", default="1,10,50,260,520,1405.7")
    ap.add_argument("--core-ratios", default="0.1,0.3,1,3")
    ap.add_argument("--ref-n", type=int, default=32)
    ap.add_argument("--ref-order", type=int, default=24)
    ap.add_argument("--ref-tol", type=float, default=1e-6)
    args = ap.parse_args()
    aspects = [float(x) for x in args.aspects.split(",")]
    ratios = [float(x) for x in args.core_ratios.split(",")]
    report = {"reference": {"n": args.ref_n, "order": args.ref_order, "tol": args.ref_tol}, "cases": []}
    flagged = 0
    for aspect in aspects:
        tri = aspect_triangle(aspect)
        point = np.array([0.5, 0.5 / aspect, 0.01])
        dist = float(point[2])
        for ratio in ratios:
            core = ratio * dist
            field = _field(tri, core)
            gamma = field.gamma[0]
            ref_area, nsub = composite_reference(tri, point, core, n=args.ref_n, order=args.ref_order)
            ref2_area, _ = composite_reference(tri, point, core, n=2 * args.ref_n, order=args.ref_order + 8)
            ref_vec = np.cross(gamma, ref2_area)
            ref_rel = float(np.linalg.norm(np.cross(gamma, ref2_area - ref_area)) / np.linalg.norm(ref_vec))
            entry = {"aspect": aspect, "core_over_d": ratio, "core": core,
                     "n_sub_reference": nsub, "reference_self_change": ref_rel, "strategies": []}
            if ref_rel > args.ref_tol:
                entry["reference_flagged"] = True
                flagged += 1
            for order, strips in ((16, 1), (16, 8), (16, 64), (32, 8), (32, 64), (48, 64)):
                t0 = time.perf_counter()
                area = area_terms(tri[None], point, core, order=order, strips=strips)[0]
                dt = time.perf_counter() - t0
                entry["strategies"].append({"solve": "fan", "order": order, "strips": strips,
                                             "rel_error": _rel(np.cross(gamma, area), ref_vec),
                                             "seconds": dt})
            for order in (16, 32, 48, 96):
                t0 = time.perf_counter()
                legacy = legacy_induced(field, point, order)
                dt = time.perf_counter() - t0
                entry["strategies"].append({"solve": "legacy", "order": order, "strips": 1,
                                             "rel_error": _rel(legacy, ref_vec),
                                             "seconds": dt})
            report["cases"].append(entry)
            cells = []
            for s in entry["strategies"]:
                cells.append("%s%d/%d:%.2e" % (s["solve"][:3], s["order"], s["strips"], s["rel_error"]))
            print("aspect %8.1f core/d %4.1f ref_change %.2e  %s" % (aspect, ratio, ref_rel, " ".join(cells)))
    report["n_reference_flagged"] = flagged
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("wrote", args.out, "flagged", flagged)


if __name__ == "__main__":
    main()