"""Exact logarithmic energy integral for a piecewise-linear full-span Gamma.

Research-only *ideal semi-infinite flat wake* diagnostic. No production
pressure, finite-wake force, geometry, Kutta equation or solver is changed.

For Gamma(a)=Gamma(b)=0, with constant piecewise slopes m_i:
    w(y) = -(1/(4 pi)) PV int Gamma'(s)/(y-s) ds
    Di = -rho int Gamma(y) w(y) dy
       = -rho/(4 pi) sum_ij m_i*m_j*J_ij
where J_ij = int_[ai,bi] int_[aj,bj] log|y-s| ds dy.
An exact primitive of log|t| is F(t)=t²*(log|t|/2-3/4), F(0)=0, giving
    J_ij = F(bi-aj) - F(ai-aj) - F(bi-bj) + F(ai-bj).

This removes the *midpoint-on-stations* quadrature bias from the existing
ideal Trefftz research estimator. It does not fix the physical model gap to
VSPAERO, nor claim actual near-field drag accuracy.
"""
from __future__ import annotations

import numpy as np


def _log_second_primitive(value):
    x = np.asarray(value, dtype=np.float64)
    result = np.zeros_like(x)
    nz = x != 0
    result[nz] = x[nz] ** 2 * (0.5 * np.log(np.abs(x[nz])) - 0.75)
    return result


def exact_trefftz_piecewise_linear(stations, gamma, *, U, density,
                                   reference_area, max_segments=512,
                                   chunk=64):
    """Return analytic ideal-wake CDi and KJ CL from a *full-span* Gamma(y).

    Assumes increasing physical y, finite piecewise-linear circulation and
    prescribed zero tip circulation at both ends. The passed stations must
    include both negative and positive half-spans for a symmetric aircraft;
    this function does NOT silently mirror half-span data. Computational
    complexity O(N²) and temporary memory O(chunk*N).
    """
    y, g = np.asarray(stations, dtype=np.float64), np.asarray(gamma, dtype=np.float64)
    refs = np.asarray([U, density, reference_area], dtype=np.float64)
    if (y.ndim != 1 or len(y) < 3 or g.shape != y.shape
            or not np.isfinite(y).all() or not np.isfinite(g).all()
            or not np.isfinite(refs).all() or np.any(refs <= 0)
            or np.any(np.diff(y) <= 0)
            or isinstance(max_segments, bool) or not isinstance(max_segments, int)
            or max_segments < 2 or len(y) - 1 > max_segments
            or isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1):
        raise ValueError("Invalid increasing full-span Gamma, positive references, or budget")
    scale = max(1., float(np.abs(g).max()))
    if max(abs(g[0]), abs(g[-1])) > 1e-12 * scale:
        raise ValueError("Gamma must vanish at both physical tips")

    a, b = y[:-1], y[1:]
    slopes = np.diff(g) / (b - a)
    energy = 0.
    for start in range(0, len(a), chunk):
        stop = min(start + chunk, len(a))
        ai, bi = a[start:stop, None], b[start:stop, None]
        aj, bj = a[None, :], b[None, :]
        kernel = (_log_second_primitive(bi - aj)
                  - _log_second_primitive(ai - aj)
                  - _log_second_primitive(bi - bj)
                  + _log_second_primitive(ai - bj))
        energy += float(np.sum(slopes[start:stop, None] * kernel * slopes[None, :]))

    force = -density * energy / (4 * np.pi)
    qS = 0.5 * density * U ** 2 * reference_area
    if force < -1e-10 * max(1., abs(energy) * density):
        raise ValueError("Negative energy from a zero-tip Gamma: check integration")
    circulation_integral = float(np.sum((g[:-1] + g[1:]) * np.diff(y) / 2))
    cl = density * U * circulation_integral / qS
    return dict(CDi_ideal=float(force / qS), CL_circulation=float(cl),
                induced_drag_N=float(force),
                circulation_lift_N=float(density * U * circulation_integral),
                station_count=int(len(y)),
                span=float(y[-1] - y[0]),
                method="analytic_logarithmic_energy_full_span_P1_gamma",
                physical_accuracy_accepted=False)
