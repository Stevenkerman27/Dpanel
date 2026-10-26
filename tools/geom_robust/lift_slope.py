"""Pure analytic helpers for the geometry-robustness report (issue #38).

Nothing in this module solves a flow field. It holds the two low-speed
lifting-line estimates, the finite-difference helpers and the aspect-ratio
convention, so that the report and the tests use exactly the same code.

Both estimates are empirical approximations for a single isolated unswept
wing. Neither is a physical truth value and neither is an acceptance
threshold.
"""
from __future__ import annotations

import math

DEG_PER_RAD = 180.0 / math.pi


def aspect_ratio(span: float, reference_area: float) -> float:
    """b^2 / S with the full (mirrored) span and the full reference area."""
    if span <= 0.0 or reference_area <= 0.0:
        raise ValueError("span and reference area must be positive")
    return span * span / reference_area


def prandtl_slope(ar: float) -> float:
    """Prandtl lifting-line style estimate 2*pi*AR/(AR+2)."""
    return 2.0 * math.pi * ar / (ar + 2.0)


def helmbold_slope(ar: float) -> float:
    """Helmbold thin-wing low-speed estimate 2*pi*AR/(2+sqrt(AR^2+4))."""
    return 2.0 * math.pi * ar / (2.0 + math.sqrt(ar * ar + 4.0))


def slope_single_point(cl: float, alpha_deg: float) -> float:
    """CL(alpha)/alpha_rad. Reported for compatibility only; not a derivative."""
    if alpha_deg == 0.0:
        raise ValueError("single-point slope needs a non-zero angle")
    return cl / math.radians(alpha_deg)


def slope_central(cl_plus: float, cl_minus: float, h_deg: float) -> float:
    """Central difference of CL in angle, returned per radian."""
    if h_deg == 0.0:
        raise ValueError("central difference needs a non-zero step")
    return (cl_plus - cl_minus) / (2.0 * math.radians(h_deg))


def even_part(cl_plus: float, cl_minus: float) -> float:
    """Symmetric component: camber, finite-mesh offset, reference tilt."""
    return 0.5 * (cl_plus + cl_minus)


def odd_part(cl_plus: float, cl_minus: float) -> float:
    """Antisymmetric component: carries the lift slope."""
    return 0.5 * (cl_plus - cl_minus)


def estimates(ar: float) -> dict:
    return dict(aspect_ratio=ar, prandtl=prandtl_slope(ar),
                helmbold=helmbold_slope(ar),
                note="neither estimate is a truth value or an acceptance gate")


def central_table(cl_of_alpha: dict, half_steps_deg) -> dict:
    """dCL/dalpha for every half step that has both +h and -h samples."""
    table = {}
    for h in half_steps_deg:
        plus = cl_of_alpha.get(float(h))
        minus = cl_of_alpha.get(-float(h))
        if plus is None or minus is None:
            table[float(h)] = None
            continue
        table[float(h)] = slope_central(plus, minus, h)
    finite = [value for value in table.values() if value is not None]
    return dict(slopes_per_rad=table,
                agreement=None if len(finite) < 2 else
                abs(finite[0] - finite[1]) / abs(finite[-1]))
