"""Core-regularized finite vortex-filament induction (research model).

The kernel is the Rosenhead–Moore regularization of Biot–Savart,

    u(P) = Gamma/(4*pi) * integral_A^B dl x (P-X)
                         / (|r|^2 + a^2)^(3/2),   a > 0.

The straight-segment integral is evaluated analytically. The core radius is an
explicit model parameter in the same length units as the geometry. It is not
an exact Cauchy principal value, a viscous solution, or automatically a
physically calibrated core. No default core, clipping, or hidden epsilon is
used.

This constant-circulation filament model is NOT generally equivalent to the
continuous P1 doublet sheet: P1 in-panel strength-gradient terms must not be
discarded when deciding how to couple the two representations.
"""
from __future__ import annotations

import numpy as np


def regularized_segment_velocity(point, start, end, circulation, core_radius):
    """Analytic induced velocity from one oriented finite straight filament.

    Uses the integral of dl x r / (|r|**2 + core_radius**2)**(3/2).
    A -> B sets the positive orientation by the right-hand rule.

    Parameters
    ----------
    point, start, end : array_like, shape (3,)
        Field point and segment endpoints in a common length unit.
    circulation : float
        Signed circulation (e.g. m**2/s if coordinates are metres).
    core_radius : float
        Explicit positive Rosenhead–Moore core radius in the geometry's units.

    Returns
    -------
    ndarray, shape (3,)
        Regularized velocity, in circulation/length units.
    """
    p = np.asarray(point, dtype=float)
    a = np.asarray(start, dtype=float)
    b = np.asarray(end, dtype=float)
    gamma = float(circulation)
    core = float(core_radius)
    if (p.shape != (3,) or a.shape != (3,) or b.shape != (3,)
            or not np.isfinite(p).all() or not np.isfinite(a).all()
            or not np.isfinite(b).all() or not np.isfinite(gamma)):
        raise ValueError("point, start, end must be finite 3-vectors and circulation finite")
    if not np.isfinite(core) or core <= 0.0:
        raise ValueError("core_radius must be finite and strictly positive")

    d = b - a
    length = float(np.linalg.norm(d))
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("segment must have nonzero finite length")
    tangent = d / length
    r = p - a
    axial = float(np.dot(r, tangent))
    rho = r - axial * tangent
    rho2 = float(np.dot(rho, rho))
    effective2 = rho2 + core * core

    # Integral_0^L ds / (effective2 + (axial-s)^2)^(3/2).
    end0 = axial / np.sqrt(effective2 + axial * axial)
    end1 = (axial - length) / np.sqrt(effective2 + (axial - length) ** 2)
    integral = (end0 - end1) / effective2
    velocity = (gamma / (4.0 * np.pi)) * np.cross(tangent, rho) * integral
    if not np.isfinite(velocity).all():
        raise FloatingPointError("nonfinite regularized filament velocity")
    return velocity


def regularized_polyline_velocity(point, vertices, circulation, core_radius, *,
                                  excluded_segments=()):
    """Sum regularized segment contributions for an oriented open polyline.

    The core is shared explicitly by every segment. excluded_segments is for
    controlled diagnostics only; it is not needed to suppress singularities
    because the kernel is regularized. Exclusion must never be used to imply
    that a physically correct curved-filament self-induction term was derived.
    """
    xyz = np.asarray(vertices, dtype=float)
    gamma = float(circulation)
    core = float(core_radius)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 2
            or not np.isfinite(xyz).all() or not np.isfinite(gamma)):
        raise ValueError("vertices must be a finite (N>=2, 3) array and circulation finite")
    if not np.isfinite(core) or core <= 0.0:
        raise ValueError("core_radius must be finite and strictly positive")
    excluded = {int(i) for i in excluded_segments}
    nseg = len(xyz) - 1
    if any(i < 0 or i >= nseg for i in excluded):
        raise ValueError("excluded segment index out of range")
    velocity = np.zeros(3, dtype=float)
    for i in range(nseg):
        if i not in excluded:
            velocity += regularized_segment_velocity(
                point, xyz[i], xyz[i + 1], gamma, core)
    return velocity
