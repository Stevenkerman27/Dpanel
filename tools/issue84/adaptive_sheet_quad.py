"""Item C3: distance-adaptive subdivision of the RM-P1 area quadrature.

The production rule ('panel/p1_regularized_sheet_velocity.py') integrates every
triangle with one fixed tensor-product Gauss-Legendre rule in a vertex-collapsed
Duffy map.  When the kernel peak width max(core, |x - x_s|) is smaller than the
triangle, that fixed rule aliases: order 32 was measured *worse* than order 8,
and near-sheet errors never decrease monotonically with order.

This module keeps the same kernel and the same edge term, and replaces only the
per-triangle rule by a midpoint 4-way subdivision whose level is chosen per
triangle so that the sub-triangle size is comparable to the peak width.  Small
sub-triangles then need only a low order, so the cost stays bounded.

Nothing here is wired into the solver; it is a controlled experiment whose
reference is the same scheme driven to a much finer level / order.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("PANEL_ROOT", Path.cwd()))
import json
import time

import numpy as np

from panel.p1_regularized_sheet_velocity import RegularizedP1SheetField


def _midpoint_split(tri):
    """4-way midpoint subdivision, vectorised over triangles."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, bc, ca = 0.5*(a+b), 0.5*(b+c), 0.5*(c+a)
    return np.stack([
        np.stack([a, ab, ca], axis=1),
        np.stack([ab, b, bc], axis=1),
        np.stack([ca, bc, c], axis=1),
        np.stack([ab, bc, ca], axis=1),
    ], axis=1).reshape(-1, 3, 3)


def subdivide(tri, level):
    out = tri
    for _ in range(int(level)):
        out = _midpoint_split(out)
    return out


def split_levels(field, point, *, max_level=3, ratio=1.0):
    """Per-triangle subdivision level from peak width against triangle size."""
    tri = field.triangles
    n = field.normals
    d = np.abs(np.einsum("fi,fi->f", point[None, :]-tri[:, 0], n))
    size = np.sqrt(field._triangle_area2)
    width = np.maximum(field.core_radius, d)
    raw = np.log2(np.maximum(size, 1e-300)/(ratio*np.maximum(width, 1e-300)))
    return np.clip(np.ceil(raw), 0, max_level).astype(int), d, size


def area_velocity_adaptive(field, point, *, order=48, sub_order=8, max_level=3,
                           ratio=1.0, block=256):
    levels, _d, _size = split_levels(field, point, max_level=max_level, ratio=ratio)
    total = np.zeros(3)
    for level in range(max_level+1):
        idx = np.flatnonzero(levels == level)
        if idx.size == 0:
            continue
        subs = subdivide(field.triangles[idx], level)
        gamma = np.repeat(field.gamma[idx], 4**level, axis=0)
        a2 = np.linalg.norm(np.cross(subs[:, 1]-subs[:, 0], subs[:, 2]-subs[:, 0]), axis=1)
        n_q = order if level == 0 else sub_order
        key = (n_q, level)
        if not hasattr(field, "_adaptive_quadrature"):
            field._adaptive_quadrature = {}
        if key not in field._adaptive_quadrature:
            q, w = np.polynomial.legendre.leggauss(n_q)
            q, w = (q+1.0)/2.0, w/2.0
            u, v = np.meshgrid(q, q, indexing="ij")
            # Vertex-collapsed Duffy map: |dA/d(u,v)| = area2 * (1-u), so the
            # quadrature weight must carry the (1-u) factor (same as the
            # production rule in panel/p1_regularized_sheet_velocity.py).
            weights = (w[:, None]*w[None, :]*(1.0-u)).reshape(-1)
            bary = np.stack((u, (1.0-u)*(1.0-v), (1.0-u)*v), axis=-1).reshape(-1, 3)
            field._adaptive_quadrature[key] = (bary, weights)
        bary, weights = field._adaptive_quadrature[key]
        for first in range(0, len(subs), block):
            sl = slice(first, first+block)
            src = np.einsum("qv,mvi->mqi", bary, subs[sl])
            r = point[None, None, :]-src
            den = (np.einsum("mqi,mqi->mq", r, r)+field.core_radius**2)**1.5
            integ = np.einsum("mq,mqi->mi", weights[None, :]/den, r)*a2[sl][:, None]
            total += np.sum(np.cross(gamma[sl], integ), axis=0)
    return total


def velocity_adaptive(field, point, **kw):
    _ = kw.pop("order", None)
    return field.freestream-(field._edge_velocity(point)+area_velocity_adaptive(field, point, **kw))/(4*np.pi)


def main():
    import argparse
    from tools.issue84.wake_node_rejection import build_case
    from panel.p1_lifting_g2 import assemble_multi, solve_multi
    from panel.p1_coupled_material_wake import TotalVelocity

    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="g2")
    ap.add_argument("--points", default="near")
    ap.add_argument("--levels", default="0,1,2,3")
    ap.add_argument("--orders", default="8,16,48")
    ap.add_argument("--out", default="runs/issue84/adaptive_quad.json")
    ap.add_argument("--nfaces", type=int, default=3)
    ap.add_argument("--ref-level", type=int, default=5)
    ap.add_argument("--ref-order", type=int, default=16)
    args = ap.parse_args()

    mesh, alpha = build_case([str(ROOT / "runs/half_cad_wingbody_v1/surface.json")])
    ops = assemble_multi(mesh)
    fresh = solve_multi(mesh, alpha, kutta="pressure", operators=ops)
    snap = np.load(str(ROOT / "runs/issue84/g2_ref/iteration_000.npz"))
    state = dict(fresh)
    state["mu"] = snap["mu"]; state["gamma"] = snap["gamma"]
    base = TotalVelocity(mesh, state, method="rm_regularized_p1", core_radius=0.01)
    field = base._regularized   # the very field the solver would use, core 0.01
    g = [gg for gg in mesh.groups if gg.name == "main_wing"][0]
    V = np.asarray(g.wake_xyz, float); F = np.asarray(g.wake_faces, int)
    tri = V[F]
    centers = tri.mean(axis=1)
    normals = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    face_idx = np.linspace(0, len(F)-1, args.nfaces).astype(int)
    pts, bands = [], []
    for h in (1e-2, 1e-3, 1e-4):
        for f in face_idx:
            pts.append(centers[f]+h*normals[f]); bands.append(h)
    pts = np.asarray(pts); bands = np.asarray(bands)

    t = time.time()
    ref = np.stack([velocity_adaptive(field, p, order=32, sub_order=args.ref_order,
                                      max_level=args.ref_level, ratio=1.0) for p in pts])
    ref_seconds = time.time()-t
    alt = np.stack([velocity_adaptive(field, p, order=32, sub_order=max(8, args.ref_order//2),
                                      max_level=args.ref_level-1, ratio=1.0) for p in pts])
    stability = float(np.max(np.linalg.norm(alt-ref, axis=1)/np.linalg.norm(ref, axis=1)))
    print("reference level %d order %d in %.1f s; cheaper variant differs by %.3e"
          % (args.ref_level, args.ref_order, ref_seconds, stability))
    res = {"points": int(len(pts)), "faces": int(len(field.triangles)),
           "core_radius": field.core_radius,
           "reference": "max_level=%d, sub_order=%d, order32" % (args.ref_level, args.ref_order),
           "reference_seconds": ref_seconds, "reference_stability": stability,
           "rows": {}}
    for order in [int(x) for x in args.orders.split(",")]:
        for level in [int(x) for x in args.levels.split(",")]:
            t = time.time()
            v = np.stack([velocity_adaptive(field, p, order=order, sub_order=order, max_level=level)
                          for p in pts])
            dt = time.time()-t
            err = np.linalg.norm(v-ref, axis=1)/np.linalg.norm(ref, axis=1)
            row = {"order": order, "max_level": level, "seconds": dt,
                   "max_rel": float(np.max(err)), "rel_l2": float(np.linalg.norm(v-ref)/np.linalg.norm(ref))}
            for h in (1e-2, 1e-3, 1e-4):
                m = bands == h
                row["h=%.0e" % h] = float(np.max(err[m]))
            res["rows"]["order%d_level%d" % (order, level)] = row
            print("order %2d level %d: %6.2f s | max rel %.3e | h1e-2 %.3e h1e-3 %.3e h1e-4 %.3e"
                  % (order, level, dt, row["max_rel"], row["h=1e-02"], row["h=1e-03"], row["h=1e-04"]))
    Path(ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(ROOT / args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()