"""P2 measurement: does the projection-fan kernel lose time to temporaries or loop structure?

This tool never touches the production path. It re-implements the kernel section of
panel/p1_projection_duffy.area_terms in two alternative shapes and times them against the
production function on the same targets, same geometry, same order and strips:

* v1 production: quad limited by an inner chunk loop, np.einsum("q,sqi->si", w, r/den)
* v2 inv-den:    same chunk loop, np.einsum("sq,sqi->si", w/den, r) - one fewer (s,q,3)
                 temporary per chunk
* v3 single:     no inner chunk loop at all (chunk = 1 sub-triangle pass)

Every variant must return bitwise-comparable terms; the tool reports max abs difference
against v1 so a speedup that changes the answer cannot be reported as one.
"""
from __future__ import annotations

import argparse
import json
import os
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
from panel.p1_projection_duffy import _duffy_nodes, _sub_area2, projection_fan  # noqa: E402


def kernel_terms(triangles, point, core, *, order, strips, swap, chunk, face_chunk, variant):
    tri = np.asarray(triangles, float)
    p = np.asarray(point, float)
    u, v, w = _duffy_nodes(order)
    uu = u[None, :, None]
    vv = v[None, :, None]
    out = np.zeros((len(tri), 3))
    step = max(1, int(chunk))
    fstep = max(1, int(face_chunk))
    for f0 in range(0, len(tri), fstep):
        f1 = min(len(tri), f0 + fstep)
        subs, owner = [], []
        for f in range(f0, f1):
            piece, _meta = projection_fan(tri[f], p, strips=strips, swap=swap)
            subs.append(piece)
            owner.append(np.full(len(piece), f, dtype=int))
        sub = np.concatenate(subs)
        own = np.concatenate(owner)
        a2 = _sub_area2(sub)
        q = sub[:, 0][:, None, :]
        a = (sub[:, 1] - sub[:, 0])[:, None, :]
        ba = (sub[:, 2] - sub[:, 1])[:, None, :]
        source = q + uu * a + (uu * vv) * ba
        r = p[None, None, :] - source
        den = (np.einsum("sqi,sqi->sq", r, r) + core * core) ** 1.5
        quad = np.empty((len(sub), 3))
        if variant == "v1":
            for first in range(0, len(sub), step):
                sl = slice(first, first + step)
                quad[sl] = np.einsum("q,sqi->si", w, r[sl] / den[sl, :, None])
        elif variant == "v2":
            for first in range(0, len(sub), step):
                sl = slice(first, first + step)
                quad[sl] = np.einsum("sq,sqi->si", w[None, :] / den[sl], r[sl])
        elif variant == "v3":
            quad[:] = np.einsum("sq,sqi->si", w[None, :] / den, r)
        else:
            raise ValueError("unknown variant: " + str(variant))
        np.add.at(out, own, quad * a2[:, None])
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-half-surface", type=Path, required=True)
    parser.add_argument("--csv", action="append", required=True,
                        help="name:path.csv; the points are read with load_points")
    parser.add_argument("--case", default="g2", choices=("g1", "g2"))
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--core-radius", type=float, default=0.01)
    parser.add_argument("--config", action="append", default=None,
                        help="order:strips, repeatable; default 28:8 and 48:8")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--face-chunk", type=int, default=64)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.chunk != 1024:
        raise SystemExit("keep --chunk at the production default 1024")
    configs = [tuple(int(x) for x in item.split(":")) for item in (args.config or ["28:8", "48:8"])]
    surface = args.main_half_surface
    mesh, alpha_deg = build_case(surface, case=args.case, alpha_deg=args.alpha)
    state = solve_multi(mesh, alpha_deg, kutta="pressure", operators=assemble_multi(mesh))
    field = TotalVelocity(mesh, state, method="rm_regularized_p1",
                          core_radius=args.core_radius,
                          area_quadrature="projection_fan",
                          fan_order=configs[0][0], fan_strips=configs[0][1])
    tri = np.asarray(field._regularized.triangles, float)
    points = []
    for spec in args.csv:
        _name, path = spec.split(":", 1)
        for row in load_points(Path(path)):
            points.append(np.asarray(row["xyz"], float))
    if args.limit:
        points = points[: args.limit]
    payload = {"schema": "dpanel.issue84.fan_kernel_bench.v1", "cad": str(surface),
               "case": args.case, "alpha_deg": args.alpha, "core_radius": args.core_radius,
               "triangles": len(tri), "points": len(points), "chunk": args.chunk,
               "face_chunk": args.face_chunk, "configs": {}}
    for order, strips in configs:
        key = "%d/%d" % (order, strips)
        entry = {"order": order, "strips": strips, "variants": {}}
        seconds = {"v1": 0.0, "v2": 0.0, "v3": 0.0}
        differences = {"v2": 0.0, "v3": 0.0}
        for point in points:
            reference = None
            for variant in ("v1", "v2", "v3"):
                started = time.perf_counter()
                terms = kernel_terms(tri, point, args.core_radius, order=order, strips=strips,
                                     swap=False, chunk=args.chunk, face_chunk=args.face_chunk,
                                     variant=variant)
                seconds[variant] += time.perf_counter() - started
                if variant == "v1":
                    reference = terms
                else:
                    scale = max(float(np.abs(reference).max()), 1e-300)
                    differences[variant] = max(differences[variant],
                                               float(np.abs(terms-reference).max())/scale)
        for variant in ("v1", "v2", "v3"):
            entry["variants"][variant] = {
                "wall_seconds": seconds[variant],
                "milliseconds_per_point": 1000.0*seconds[variant]/max(len(points), 1),
                "max_relative_difference_vs_v1": differences.get(variant, 0.0)}
        base = entry["variants"]["v1"]["wall_seconds"]
        for variant in ("v2", "v3"):
            entry["variants"][variant]["speedup_vs_v1"] = base/entry["variants"][variant]["wall_seconds"]
        payload["configs"][key] = entry
        print("%s v1 %.1f ms/pt  v2 %.1f ms/pt (%.2fx, d=%.2e)  v3 %.1f ms/pt (%.2fx, d=%.2e)" % (
            key, entry["variants"]["v1"]["milliseconds_per_point"],
            entry["variants"]["v2"]["milliseconds_per_point"],
            entry["variants"]["v2"]["speedup_vs_v1"],
            entry["variants"]["v2"]["max_relative_difference_vs_v1"],
            entry["variants"]["v3"]["milliseconds_per_point"],
            entry["variants"]["v3"]["speedup_vs_v1"],
            entry["variants"]["v3"]["max_relative_difference_vs_v1"]))
        sys.stdout.flush()
    payload["status"] = "ok"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
