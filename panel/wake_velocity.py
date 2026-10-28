"""Independent velocity of a piecewise-linear doublet (vortex) sheet.

Why this module exists
----------------------
The experimental P1 steady free wake moved wake nodes with

    dy/dx = Vy/Vx,  dz/dx = Vz/Vx

where the velocity came from a two-sided "probe" offset whose floor was tuned by
hand (docs/P1_FREE_WAKE_PROBE_FIX.md).  That is a sampling heuristic, not a
definition of the velocity of the sheet: it has no limit statement, no jump
condition, and it cannot represent recirculation (Vx <= 0).

This module defines the velocity of a triangulated doublet sheet with piecewise
linear strength, in a form that can be checked against independent quantities:

    u(p) = 1/(4*pi) * [ B(p) + S(p) ]

    B(p) = closed line integral of mu(x') * t x r / |r|^3 dl    (edge vortices)
    S(p) = gamma x ( closed integral of (1/|r|) * (t x n) dl  -  Omega * n )
    r    = p - x',  t = unit edge tangent,  n = triangle normal right-handed
           with the boundary direction,  gamma = grad_s(mu) x n

The surface integral of the vortex sheet reduces to a boundary integral through
the surface-gradient theorem  int_S grad_s f dS = closed_integral f (t x n) dl,
so the whole velocity costs O(#edges * quadrature), not O(area).

Sign conventions were calibrated once against an exact oracle, not assumed:
for constant mu this velocity equals the triangle vortex-ring velocity and
equals the finite-difference gradient of the exact solid-angle doublet
potential  phi = mu * Omega / (4*pi)  to 5e-11 (see test/test_wake_velocity.py).

What this module deliberately does NOT do
-----------------------------------------
* no vortex core, no cutoff, no probe floor: the kernel is the exact Biot-Savart
  kernel.  Near-sheet evaluation is reported as a one-sided limit, and the
  principal value is the average of the two sides.
* no claim that a steady solution exists; a non-vanishing jump here is physics,
  not an error.
* self-sheet evaluation on the sheet itself is a principal value; the caller
  must decide whether a PV is the right object for a material node update.
"""
from __future__ import annotations

import numpy as np

_EDGE_QUADRATURE = 48


def _gauss_legendre(order: int):
    x, w = np.polynomial.legendre.leggauss(int(order))
    return 0.5 * (x + 1.0), 0.5 * w


_NODES, _WEIGHTS = _gauss_legendre(_EDGE_QUADRATURE)


def triangle_normal(vertices: np.ndarray) -> np.ndarray:
    """Unit normal of one triangle, right-handed with (v0, v1, v2)."""
    v = np.asarray(vertices, dtype=float)
    if v.shape != (3, 3):
        raise ValueError("triangle_normal expects a (3, 3) vertex array")
    n = np.cross(v[1] - v[0], v[2] - v[0])
    length = np.linalg.norm(n)
    if length <= 0.0:
        raise ValueError("Degenerate triangle has no normal")
    return n / length


def triangle_area(vertices: np.ndarray) -> float:
    v = np.asarray(vertices, dtype=float)
    return 0.5 * float(np.linalg.norm(np.cross(v[1] - v[0], v[2] - v[0])))


def triangle_edges(vertices: np.ndarray):
    """Yield (start, end, delta) for the right-handed boundary v0 -> v1 -> v2."""
    v = np.asarray(vertices, dtype=float)
    for k in range(3):
        yield v[k], v[(k + 1) % 3], (v[(k + 1) % 3] - v[k])


def segment_velocity(p, a, b, weight=1.0) -> np.ndarray:
    """Biot-Savart velocity of a straight segment, unit circulation.

    Returns 1/(4*pi) * int_a^b dl x r / |r|^3 for a unit-strength line vortex.
    The exact closed form is used (no quadrature, no core):

        Omega = (r1 x r2) * (|r1| + |r2|) / (|r1| |r2| + r1 . r2)
    """
    r1 = a - p
    r2 = b - p
    n1 = np.linalg.norm(r1)
    n2 = np.linalg.norm(r2)
    den = n1 * n2 * (n1 * n2 + float(np.dot(r1, r2)))
    if den <= 1e-300:
        # collinear singularity: the exact limit is zero for a segment through p
        return np.zeros(3)
    omega = np.cross(r1, r2) * (n1 + n2) / den
    return weight * omega / (4.0 * np.pi)


def triangle_ring_velocity(p, vertices, circulation=1.0) -> np.ndarray:
    """Velocity of the constant-strength doublet triangle = vortex ring."""
    v = np.asarray(vertices, dtype=float)
    total = np.zeros(3)
    for k in range(3):
        total += segment_velocity(p, v[k], v[(k + 1) % 3], weight=circulation)
    return total


def solid_angle(p, vertices) -> float:
    """Exact signed solid angle of a flat triangle seen from p."""
    v = np.asarray(vertices, dtype=float)
    r = v - np.asarray(p, dtype=float)[None, :]
    la, lb, lc = (np.linalg.norm(r[i]) for i in range(3))
    num = float(np.dot(r[0], np.cross(r[1], r[2])))
    den = (la * lb * lc
           + float(np.dot(r[0], r[1])) * lc
           + float(np.dot(r[1], r[2])) * la
           + float(np.dot(r[2], r[0])) * lb)
    return float(2.0 * np.arctan2(num, den))


def doublet_potential(p, vertices, mu0: float) -> float:
    """Constant-strength doublet potential  mu0 * Omega / (4*pi) (exact)."""
    return mu0 * solid_angle(p, vertices) / (4.0 * np.pi)


def _edge_quadrature(a, b):
    """Quadrature nodes and arc-length weights on one edge."""
    length = float(np.linalg.norm(b - a))
    points = a[None, :] + _NODES[:, None] * (b - a)[None, :]
    return points, _WEIGHTS * length


def sheet_velocity(p, vertices, mu0: float, grad_mu) -> np.ndarray:
    """Velocity of one piecewise-linear doublet triangle at an off-sheet point.

    Parameters
    ----------
    p : (3,) field point.  Must not lie on the sheet (use sheet_velocity_pv).
    vertices : (3, 3) triangle vertices, right-handed boundary v0 -> v1 -> v2.
    mu0 : doublet strength at the triangle centroid.
    grad_mu : (3,) tangential gradient of the doublet strength.  The normal
        component is projected out, because only grad_s(mu) is physical.

    Returns
    -------
    (3,) velocity induced by this triangle, units of mu0 / length.
    """
    v = np.asarray(vertices, dtype=float)
    p = np.asarray(p, dtype=float)
    n = triangle_normal(v)
    centre = v.mean(axis=0)
    grad = np.asarray(grad_mu, dtype=float)
    grad_s = grad - n * float(np.dot(grad, n))
    gamma = np.cross(grad_s, n)          # calibrated sign, see module docstring

    boundary = np.zeros(3)               # int mu dl x r / |r|^3
    surface = np.zeros(3)                # int (1/|r|) (t x n) dl
    for k in range(3):
        a = v[k]
        b = v[(k + 1) % 3]
        dl = b - a
        length = float(np.linalg.norm(dl))
        if length <= 0.0:
            continue
        tangent = dl / length
        t_cross_n = np.cross(tangent, n)
        points, weights = _edge_quadrature(a, b)
        rel = p[None, :] - points
        radius = np.linalg.norm(rel, axis=1)
        mu_edge = mu0 + (points - centre) @ grad_s
        boundary += np.sum((weights * mu_edge / radius ** 3)[:, None]
                           * np.cross(np.broadcast_to(tangent, rel.shape), rel), axis=0)
        surface += t_cross_n * float(np.sum(weights / radius))

    omega = solid_angle(p, v)
    # int_S r/|r|^3 dS = closed_integral (1/|r|)(t x n) dl - Omega * n   (exact)
    sheet_integral = surface - omega * n
    return (boundary + np.cross(gamma, sheet_integral)) / (4.0 * np.pi)


def sheet_velocity_pv(p, vertices, mu0: float, grad_mu, offsets=None):
    """One-sided limits and principal value at a point on the sheet.

    The velocity of a vortex sheet jumps across it.  This evaluator never
    silently averages a singular kernel: it samples the off-sheet form at
    p +/- eps * n, returns both sides, their average (the Cauchy principal
    value) and their jump, plus the one-sided convergence of the estimate.

    Returns a dict with keys plus, minus, principal_value, jump, normal_jump,
    epsilon, convergence; convergence is the relative change of the principal
    value between the two finest offsets (a number, not a verdict).
    """
    v = np.asarray(vertices, dtype=float)
    p = np.asarray(p, dtype=float)
    n = triangle_normal(v)
    if offsets is None:
        offsets = (1e-2, 1e-3, 1e-4, 1e-5)
    plus = [sheet_velocity(p + eps * n, v, mu0, grad_mu) for eps in offsets]
    minus = [sheet_velocity(p - eps * n, v, mu0, grad_mu) for eps in offsets]
    pv = [0.5 * (a + b) for a, b in zip(plus, minus)]
    jump = [a - b for a, b in zip(plus, minus)]
    scale = max(float(np.linalg.norm(pv[-1])), 1e-30)
    convergence = float(np.linalg.norm(pv[-1] - pv[-2]) / scale)
    return {
        "plus": plus[-1],
        "minus": minus[-1],
        "principal_value": pv[-1],
        "jump": jump[-1],
        "normal_jump": float(np.dot(jump[-1], n)),
        "epsilon": float(offsets[-1]),
        "convergence": convergence,
    }


def sheet_velocity_sum(p, triangles, mu0, grad_mu, skip=None) -> np.ndarray:
    """Sum the velocity of a triangulated sheet.

    triangles : (n, 3, 3) array of right-handed vertex triples.
    mu0, grad_mu : per-triangle strength and tangential gradient, shape (n,) and
        (n, 3).
    skip : optional iterable of triangle indices to omit (a self face at a
        material node, whose PV has to be handled by the caller).
    """
    tris = np.asarray(triangles, dtype=float)
    mu0 = np.atleast_1d(np.asarray(mu0, dtype=float))
    grad = np.asarray(grad_mu, dtype=float)
    total = np.zeros(3)
    skipped = set() if skip is None else set(int(i) for i in skip)
    for index in range(tris.shape[0]):
        if index in skipped:
            continue
        total += sheet_velocity(p, tris[index], float(mu0[index]), grad[index])
    return total
