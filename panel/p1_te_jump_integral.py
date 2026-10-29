"""Exact TE potential-jump compatibility audit; NOT a pressure-Kutta condition.

For piecewise-linear wall trace jump and piecewise-linear wake Gamma, the union
of both knot sets makes the residual linear on every subinterval. Its signed
integral, L2 norm, maximum, and P1 weak moments therefore have exact
closed-form segment integrals (no numerical quadrature).
"""
from __future__ import annotations
import numpy as np


def _vec(name, value):
    a = np.asarray(value, dtype=float).reshape(-1)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains nonfinite values")
    return a


def audit_te_jump_integral(te_y, mu_jump, gamma_y, gamma_values, *, atol=1e-10):
    """Compare Delta_mu(y)=mu_upper-mu_lower with Gamma(y) over the whole TE.

    te_y contains ALL wall TE trace stations, including tip endpoints.
    mu_jump is the signed upper-minus-lower potential at each TE station.
    gamma_y/gamma_values contains the full wake material interpolation,
    including prescribed tip values (normally zero).

    Reports both the signed integral (which can hide cancellation) and the
    exact L2/max residual on the common refinement of both knot sets.
    """
    y = _vec("te_y", te_y)
    jump = _vec("mu_jump", mu_jump)
    gy = _vec("gamma_y", gamma_y)
    gamma = _vec("gamma_values", gamma_values)
    atol = float(atol)
    if (len(y) < 2 or jump.shape != y.shape or len(gy) < 2 or gamma.shape != gy.shape
            or np.any(np.diff(y) <= 0) or np.any(np.diff(gy) <= 0)
            or gy[0] > y[0]+atol or gy[-1] < y[-1]-atol
            or not np.isfinite(atol) or atol <= 0):
        raise ValueError("invalid TE/Gamma knots or array sizes")
    interior_gamma_knots = gy[(gy > y[0]+atol) & (gy < y[-1]-atol)]
    aligned = all(np.min(np.abs(y-k)) <= atol for k in interior_gamma_knots)
    common = np.unique(np.r_[y, interior_gamma_knots])
    jump_common = np.interp(common, y, jump)
    gamma_common = np.interp(common, gy, gamma)
    residual = jump_common-gamma_common
    lengths = np.diff(common)
    signed_integral = float(np.sum(lengths*(residual[:-1]+residual[1:])/2))
    square_integral = float(np.sum(lengths*(residual[:-1]**2+
                                                residual[:-1]*residual[1:]+
                                                residual[1:]**2)/3))
    moments = np.zeros(len(common))
    for i, h in enumerate(lengths):
        r0, r1 = residual[i], residual[i+1]
        moments[i] += h*(2*r0+r1)/6
        moments[i+1] += h*(r0+2*r1)/6
    gamma_integral = float(np.sum(lengths*(gamma_common[:-1]+gamma_common[1:])/2))
    jump_integral = float(np.trapezoid(jump, y))
    scale = max(1., float(np.max(np.abs(jump))), float(np.max(np.abs(gamma))))
    edges = []
    for a,b in zip(y[:-1],y[1:]):
        mask = (common >= a-atol) & (common <= b+atol)
        yy = common[mask]
        rr = residual[mask]
        hh = np.diff(yy)
        ii = float(np.sum(hh*(rr[:-1]+rr[1:])/2))
        ss = float(np.sum(hh*(rr[:-1]**2+rr[:-1]*rr[1:]+rr[1:]**2)/3))
        edges.append(dict(y0=float(a),y1=float(b),integral_residual=ii,
                          l2_rms=float(np.sqrt(max(ss,0.)/(b-a))),
                          max_abs_residual=float(np.max(np.abs(rr))),
                          gamma_knot_inside=bool(np.any((interior_gamma_knots>a+atol)&
                                                        (interior_gamma_knots<b-atol)))))
    return dict(
        status="pass" if np.max(np.abs(residual)) <= atol*scale else "jump_mismatch",
        condition="potential_jump_compatibility_only_not_pressure_kutta",
        te_length=float(y[-1]-y[0]),
        te_station_count=int(len(y)),
        gamma_station_count=int(len(gy)),
        common_refinement_count=int(len(common)),
        gamma_knots_inside_te_edges=interior_gamma_knots.tolist(),
        partition_aligned=bool(aligned),
        nodal_residual=(jump-np.interp(y,gy,gamma)).tolist(),
        nodal_residual_max=float(np.max(np.abs(jump-np.interp(y,gy,gamma)))),
        jump_integral=jump_integral,
        gamma_integral=gamma_integral,
        integral_residual=signed_integral,
        integral_residual_normalized=signed_integral/(scale*(y[-1]-y[0])),
        l2_residual=float(np.sqrt(max(square_integral,0.))),
        l2_rms=float(np.sqrt(max(square_integral,0.)/(y[-1]-y[0]))),
        max_abs_residual=float(np.max(np.abs(residual))),
        weak_moment_linf=float(np.max(np.abs(moments))),
        scale=scale,
        edges=edges,
        integral_alone_can_hide_cancellation=(
            abs(signed_integral) <= atol*scale*(y[-1]-y[0]) and
            np.max(np.abs(residual)) > 10*atol*scale))


def audit_g1_te_jump(wing, mu, gamma, *, atol=1e-10):
    """Audit a solved G1 Wing, including shared geometric tip endpoints."""
    mu = _vec("mu", mu)
    gamma = _vec("gamma", gamma)
    upper = np.asarray(wing.te_upper, dtype=int)
    lower = np.asarray(wing.te_lower, dtype=int)
    y = np.asarray(wing.xyz, float)[np.asarray(wing.te_geo, dtype=int), 1]
    order = np.argsort(y)
    y, upper, lower = y[order], upper[order], lower[order]
    if len(mu) != int(wing.ndof_mu) or len(gamma) != len(wing.gamma_y)-2:
        raise ValueError("G1 mu/Gamma sizes do not match Wing")
    full_gamma = np.r_[0., gamma, 0.]
    return audit_te_jump_integral(y, mu[upper]-mu[lower], wing.gamma_y,
                                  full_gamma, atol=atol)


def audit_g2_group_te_jump(mesh, state, operators, group_name, *, atol=1e-10):
    """Audit one G2 lifting group; uses the exact group Gamma slice and TE pairs."""
    groups = {g.name:g for g in mesh.groups}
    if group_name not in groups:
        raise ValueError(f"unknown lifting group {group_name}")
    g = groups[group_name]
    pairs = np.asarray(g.te_pairs, int)
    mu = _vec("mu", state["mu"])
    gamma = _vec("gamma", state["gamma"])
    sl = operators["slices"][group_name]
    gfull = np.r_[0., gamma[sl], 0.]
    jumps = mu[pairs[:,0]]-mu[pairs[:,1]]
    y = np.r_[g.gamma_y[0], g.te_pair_y, g.gamma_y[-1]]
    dmu = np.r_[0., jumps, 0.]
    order = np.argsort(y)
    return audit_te_jump_integral(y[order], dmu[order], g.gamma_y, gfull, atol=atol)
