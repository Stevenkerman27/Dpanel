"""Projection-aware Duffy quadrature for the regularized P1 area kernel.

Motivation (issue #84, PR #85). The production regularized area term integrates

    I_f(p) = int_{T_f} (p - x) / (|p - x|^2 + core^2)^{3/2} dS

with a single Duffy map that always collapses at the same vertex of each triangle.
When the target p is much closer to the triangle than the triangle is large, the
integrable peak sits in a corner of the parametric square, and when the triangle is
a sliver the collapse direction is nearly parallel to the long edge, so the peak is
smeared along the other direction: the error stops decreasing with order (measured in
docs/ISSUE84_RM_BENCH.md, main-wing wake triangles reach aspect 1405.7).

This module keeps the kernel identical and changes only the domain decomposition:

1. project p onto the triangle (interior / edge / vertex) and take the closest point q;
2. split the triangle into a fan of sub-triangles that all share q as a vertex
   (3 for an interior projection, 2 for an edge, 1 for a vertex), optionally cutting
   each fan sector into strips along its far edge (anisotropy control);
3. integrate every sub-triangle with a Duffy map collapsed at q, so the peak sits at a
   parametric corner with an exact Jacobian instead of being displaced.

Guarantees: the fan is a partition (areas sum to the parent area, sub-domains do not
overlap), the Jacobian is exact (no empirical weights), and a constant integrand is
integrated to the exact area. Everything here is pure numpy and opt-in; the default
solver stays on analytic_edges and this module is never imported by it.
"""
from __future__ import annotations

import numpy as np

from . import velocity_profile as _profile

EDGE_NAMES = ("01", "12", "20")


def _dot(a, b):
    return float(np.dot(a, b))


def closest_point_on_triangle(triangle, point, *, tol=1e-12):
    """Closest point to point on a triangle, with the region it landed in.

    Returns (q, region) with region {"kind": interior|edge|vertex, "index": ...}.
    The classification drives the fan, so it must be decisive at the boundaries: a
    projection that reaches a vertex reports vertex, never edge.
    """
    tri = np.asarray(triangle, float)
    p = np.asarray(point, float)
    if tri.shape != (3, 3) or p.shape != (3,):
        raise ValueError("triangle must be (3,3) and point (3,)")
    a, b, c = tri
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = _dot(ab, ap), _dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a.copy(), {"kind": "vertex", "index": 0}
    bp = p - b
    d3, d4 = _dot(ab, bp), _dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b.copy(), {"kind": "vertex", "index": 1}
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3) if (d1 - d3) != 0.0 else 0.0
        return a + v * ab, {"kind": "edge", "index": 0}
    cp = p - c
    d5, d6 = _dot(ab, cp), _dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c.copy(), {"kind": "vertex", "index": 2}
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6) if (d2 - d6) != 0.0 else 0.0
        return a + w * ac, {"kind": "edge", "index": 2}
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), {"kind": "edge", "index": 1}
    denom = va + vb + vc
    if denom == 0.0:
        raise ValueError("degenerate triangle: zero area")
    v = vb / denom
    w = vc / denom
    return a + v * ab + w * ac, {"kind": "interior", "index": None}


def _fan_sectors(tri, q, region):
    """Fan the triangle into sectors that all contain q as a vertex."""
    v = [tri[0], tri[1], tri[2]]
    kind = region["kind"]
    if kind == "vertex":
        i = region["index"]
        return [(q, v[(i + 1) % 3], v[(i + 2) % 3])]
    if kind == "edge":
        i = region["index"]
        j = (i + 1) % 3
        k = (i + 2) % 3
        return [(q, v[i], v[k]), (q, v[k], v[j])]
    if kind == "interior":
        return [(q, v[0], v[1]), (q, v[1], v[2]), (q, v[2], v[0])]
    raise ValueError("unknown projection region")


def _strip_sector(sector, strips):
    """Cut a sector (q, a, b) into strips along its far edge a-b, all sharing q."""
    if strips <= 1:
        return [sector]
    q, a, b = sector
    t = np.linspace(0.0, 1.0, strips + 1)[:, None]
    nodes = a[None, :] + t * (b - a)[None, :]
    return [(q, nodes[i], nodes[i + 1]) for i in range(strips)]


def projection_fan(triangle, point, *, strips=1, swap=False, tol=1e-12):
    """Partition a triangle into Duffy-friendly sub-triangles around the projection.

    Every returned sub-triangle has the closest point q as its first vertex, which is
    where the Duffy map collapses. strips>1 cuts each fan sector along its far edge;
    swap selects the other edge of the sector as the far edge, which changes the
    integration direction for slivers.
    """
    tri = np.asarray(triangle, float)
    if strips < 1:
        raise ValueError("strips must be >= 1")
    q, region = closest_point_on_triangle(tri, point, tol=tol)
    pieces = []
    for qv, a, b in _fan_sectors(tri, q, region):
        if swap:
            a, b = b, a
        pieces.extend(_strip_sector((qv, a, b), strips))
    sub = np.stack([np.stack(piece) for piece in pieces])
    if _profile.enabled():
        _profile.add("fan.calls")
        _profile.add("fan.region."+region["kind"])
        _profile.add("fan.subtriangles", len(pieces))
        _profile.add("fan.strips", strips)
    return sub, {"region": region, "n_sub": len(pieces)}


def _duffy_nodes(order):
    if type(order) is not int or order < 2:
        raise ValueError("order must be an integer >= 2")
    x, w = np.polynomial.legendre.leggauss(order)
    u = (x + 1.0) / 2.0
    wu = w / 2.0
    uu, vv = np.meshgrid(u, u, indexing="ij")
    ww = (wu[:, None] * wu[None, :] * uu).reshape(-1)
    return uu.reshape(-1), vv.reshape(-1), ww


def _sub_area2(subs):
    return np.linalg.norm(np.cross(subs[:, 1] - subs[:, 0], subs[:, 2] - subs[:, 0]), axis=1)


def area_terms(
    triangles, point, core, *, order=16, strips=1, swap=False, chunk=1024,
    face_chunk=64, return_sub_areas=False,
):
    """Area kernel integral per triangle: (N,3) int_T (p-x)/(|p-x|^2+core^2)^{3/2} dS.

    Memory is bounded by face_chunk sub-triangles times order**2 nodes, so a whole
    surface (a thousand faces) can be evaluated without a dense (faces, nodes) block.
    """
    tri = np.asarray(triangles, float)
    p = np.asarray(point, float)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError("triangles must be (N,3,3)")
    if not np.isfinite(core) or core <= 0.0:
        raise ValueError("core must be a positive finite radius")
    _profile.add("area_terms.calls")
    _profile.add("area_terms.triangles", len(tri))
    with _profile.stage("area_terms.nodes"):
        u, v, w = _duffy_nodes(order)
        _profile.add("area_terms.node_sets")
    uu = u[None, :, None]
    vv = v[None, :, None]
    out = np.zeros((len(tri), 3))
    sub_counts = np.zeros(len(tri), dtype=int)
    kept = []
    step = max(1, int(chunk))
    fstep = max(1, int(face_chunk))
    for f0 in range(0, len(tri), fstep):
        f1 = min(len(tri), f0 + fstep)
        subs = []
        owner = []
        _profile.begin("area_terms.projection")
        for f in range(f0, f1):
            piece, meta = projection_fan(tri[f], p, strips=strips, swap=swap)
            subs.append(piece)
            owner.append(np.full(len(piece), f, dtype=int))
            sub_counts[f] = meta["n_sub"]
        _profile.end("area_terms.projection")
        _profile.begin("area_terms.kernel")
        sub = np.concatenate(subs)
        own = np.concatenate(owner)
        _profile.add("area_terms.subtriangles", len(own))
        a2 = _sub_area2(sub)
        # Duffy: x = q + u*a + u*v*(b-a); |dS| = a2 * u du dv
        q = sub[:, 0][:, None, :]
        a = (sub[:, 1] - sub[:, 0])[:, None, :]
        ba = (sub[:, 2] - sub[:, 1])[:, None, :]
        source = q + uu * a + (uu * vv) * ba
        r = p[None, None, :] - source
        den = (np.einsum("sqi,sqi->sq", r, r) + core * core) ** 1.5
        quad = np.empty((len(sub), 3))
        for first in range(0, len(sub), step):
            sl = slice(first, first + step)
            quad[sl] = np.einsum("q,sqi->si", w, r[sl] / den[sl, :, None])
        np.add.at(out, own, quad * a2[:, None])
        _profile.end("area_terms.kernel")
        if return_sub_areas:
            kept.append((sub, own, a2))
    if return_sub_areas:
        return out, kept, sub_counts
    return out


def area_velocity(triangles, gamma, point, core, *, order=16, strips=1, swap=False, chunk=1024):
    """sum_f cross(gamma_f, area_terms_f): the regularized area velocity without the 1/4pi."""
    terms = area_terms(triangles, point, core, order=order, strips=strips, swap=swap, chunk=chunk)
    return np.sum(np.cross(np.asarray(gamma, float), terms), axis=0)


def field_area_velocity(field, point, *, order=16, strips=8, swap=False, chunk=1024,
                        face_chunk=64, return_face_terms=False):
    """Area velocity of a whole regularized field (wall plus wake parts) at one target."""
    tri = np.asarray(field.triangles, float)
    gamma = np.asarray(field.gamma, float)
    terms = area_terms(tri, point, float(field.core_radius), order=order, strips=strips,
                       swap=swap, chunk=chunk, face_chunk=face_chunk)
    value = np.sum(np.cross(gamma, terms), axis=0)
    if return_face_terms:
        return value, terms
    return value


def field_velocity(field, point, **kw):
    """Whole regularized P1 velocity, matching RegularizedP1SheetField.__call__."""
    p = np.asarray(point, float)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError("field point must be a finite (3,) vector")
    area = field_area_velocity(field, p, **kw)
    return field.freestream - (field._edge_velocity(p) + area) / (4.0 * np.pi)