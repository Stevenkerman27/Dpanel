"""Near/far split of the regularized P1 area kernel (issue #92, step 3).

The projection fan is the only quadrature that resolves the (|r|**2 + core**2)**-1.5
peak, and it costs one fan per triangle: 0.38 ms per face on the CAD wake, 292 ms for
the whole sheet at 48x1. Almost all of that work is wasted, because only the faces
within a few core radii of the target carry a peak: the median real sampler target has
six such faces out of 770 (docs/archive/issue/ISSUE92_PHASE1_WAKE_EVALUATOR.md, section 6b).

This module is that split, as a callable evaluator rather than a probe:

* faces whose exact distance to the target is within near_cores*core get the
  projection fan;
* the rest get the plain (non-projection) Duffy map at far_order.

Two deliberate differences from tools/issue84/far_field_split.py, which measured the
same idea and could not certify it:

* the near set is selected by the exact point-triangle distance, not by the distance
  to the centroid. A centroid test misclassifies every large face whose centroid is
  far while its surface is close, and the wake sheet has such faces;
* the radius is expressed in core radii, the only length that sets the width of the
  peak, instead of in multiples of the longest edge.

The far quadrature is not free: a low order has no error bound, and the acceptance
tool tools/issue92/hybrid_acceptance.py measures that error end to end against the
full fan before any of this is offered as a production default.
"""
from __future__ import annotations

import numpy as np

from .p1_projection_duffy import area_terms

__all__ = ["point_triangle_distance", "near_face_mask", "hybrid_area_terms",
           "hybrid_area_velocity", "hybrid_velocity"]


def point_triangle_distance(triangles, point):
    """Exact distance from one point to each triangle, shape (N,).

    Vectorised version of the closest-point test in Real-Time Collision Detection
    (Ericson, section 5.1.5): the Voronoi regions are tested in order and the closest
    point is built from the winning region, so a face is classified by its surface and
    not by its centroid.
    """
    tri = np.asarray(triangles, float)
    p = np.asarray(point, float)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError("triangles must be (N,3,3)")
    if p.shape != (3,):
        raise ValueError("point must be a (3,) vector")
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = va + vb + vc
        inv = np.where(denom != 0.0, 1.0 / denom, 0.0)
        v = vb * inv
        w = vc * inv
    closest = a + ab * v[:, None] + ac * w[:, None]

    def edge(u, vv, t):
        t = np.where(np.isfinite(t), t, 0.0)
        return u + (vv - u) * t[:, None]

    # vertex A
    mask_a = (d1 <= 0.0) & (d2 <= 0.0)
    closest = np.where(mask_a[:, None], a, closest)
    # vertex B
    mask_b = (d3 >= 0.0) & (d4 <= d3)
    closest = np.where(mask_b[:, None], b, closest)
    # edge AB
    mask = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0) & ~mask_a & ~mask_b
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(d1 - d3 != 0.0, d1 / (d1 - d3), 0.0)
    closest = np.where(mask[:, None], edge(a, b, t), closest)
    # vertex C
    mask_c = (d6 >= 0.0) & (d5 <= d6)
    closest = np.where(mask_c[:, None], c, closest)
    # edge AC
    mask = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0) & ~mask_a & ~mask_c
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(d2 - d6 != 0.0, d2 / (d2 - d6), 0.0)
    closest = np.where(mask[:, None], edge(a, c, t), closest)
    # edge BC
    mask = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0) & ~mask_b & ~mask_c
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where((d4 - d3) + (d5 - d6) != 0.0,
                     (d4 - d3) / ((d4 - d3) + (d5 - d6)), 0.0)
    closest = np.where(mask[:, None], edge(b, c, t), closest)
    return np.linalg.norm(p[None, :] - closest, axis=1)


def near_face_mask(triangles, point, core, *, near_cores=10.0):
    """Faces whose exact distance to the point is within near_cores core radii."""
    if not np.isfinite(core) or core <= 0.0:
        raise ValueError("core must be a positive finite radius")
    if not np.isfinite(near_cores) or near_cores <= 0.0:
        raise ValueError("near_cores must be a positive finite multiple of the core")
    return point_triangle_distance(triangles, point) < float(near_cores) * float(core)


def hybrid_area_terms(triangles, point, core, *, near_cores=10.0, fan_order=48,
                      fan_strips=1, far_order=16, fan_swap=False):
    """Per-face area kernel integrals with the fan near and a plain Duffy map far.

    Returns (N,3), the same array area_terms() returns, so a caller can take the face
    sum with its own strengths and compare against the full fan exactly.
    """
    tri = np.asarray(triangles, float)
    if type(fan_order) is not int or fan_order < 2:
        raise ValueError("fan_order must be an integer >= 2")
    if type(fan_strips) is not int or fan_strips < 1:
        raise ValueError("fan_strips must be an integer >= 1")
    if type(far_order) is not int or far_order < 4:
        raise ValueError("far_order must be an integer >= 4")
    near = near_face_mask(tri, point, core, near_cores=near_cores)
    out = np.zeros((len(tri), 3))
    if np.any(near):
        out[near] = area_terms(tri[near], point, core, order=fan_order,
                               strips=fan_strips, swap=fan_swap)
    if np.any(~near):
        out[~near] = area_terms(tri[~near], point, core, order=far_order, strips=1)
    return out


def hybrid_area_velocity(triangles, gamma, point, core, **kw):
    """Face sum of gamma x (hybrid area kernel integral)."""
    terms = hybrid_area_terms(triangles, point, core, **kw)
    return np.sum(np.cross(np.asarray(gamma, float), terms), axis=0)


def hybrid_velocity(field, point, **kw):
    """Freestream minus the induced velocity with the hybrid area kernel.

    field is a RegularizedP1SheetField; its closed-form regularized edge term is kept
    unchanged, because that part is already exact and cheap: only the area term is
    split. With near_cores=inf this reproduces field(point) up to rounding.
    """
    p = np.asarray(point, float)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError("field point must be a finite (3,) vector")
    near_cores = kw.pop("near_cores", 10.0)
    if np.isinf(near_cores):
        return field(p)
    area = hybrid_area_velocity(field.triangles, field.gamma, p, field.core_radius,
                                near_cores=near_cores, **kw)
    induced = field._edge_velocity(p) + area
    return field.freestream - induced / (4.0 * np.pi)
