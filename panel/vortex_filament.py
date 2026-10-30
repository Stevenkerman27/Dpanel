"""Independent finite straight-vortex-filament Biot–Savart oracle.

This module is intentionally separate from the continuous-P1 doublet-sheet
field. It evaluates a single straight filament segment with constant
circulation; it is not a replacement for the P1 sheet's in-panel strength
gradient. No finite core or hidden softening is applied.

For a segment A -> B, circulation Gamma, and off-filament point P,

    u(P) = Gamma/(4*pi) * (r1 x r2)/|r1 x r2|^2
           * [ (B-A) dot (r1/|r1| - r2/|r2|) ]

where r1=P-A and r2=P-B. Reversing A/B reverses the velocity. Units are
circulation / length, i.e. velocity when Gamma has units length^2/time.

On the segment's interior centerline the integrand is identically zero away
from the singular point and its symmetric straight-segment principal value is
defined as zero. This convention applies ONLY to that straight segment's own
contribution. It is not a curvature/self-induction model for a polygonal
filament; evaluation at a segment endpoint remains singular and raises.
"""
from __future__ import annotations

import numpy as np


def segment_biot_savart(point, start, end, circulation=1.0, *,
                        on_segment="raise"):
    """Velocity induced by one finite, straight, constant-strength vortex segment.

    Parameters
    ----------
    point, start, end : array_like, shape (3,)
        Field point and oriented segment endpoints in a common length unit.
    circulation : float
        Signed circulation. Positive follows start -> end by the right-hand rule.
    on_segment : {"raise", "straight_pv_zero"}
        Policy for a point strictly inside the segment and exactly on its axis.
        "straight_pv_zero" returns the symmetric PV of this straight segment
        only. Endpoints remain singular under either policy.

    Returns
    -------
    ndarray, shape (3,)
        Induced velocity. No core radius, clipping, or softening is used.
    """
    p = np.asarray(point, dtype=float)
    a = np.asarray(start, dtype=float)
    b = np.asarray(end, dtype=float)
    gamma = float(circulation)
    if (p.shape != (3,) or a.shape != (3,) or b.shape != (3,)
            or not np.isfinite(p).all() or not np.isfinite(a).all()
            or not np.isfinite(b).all() or not np.isfinite(gamma)):
        raise ValueError("point, start, end must be finite 3-vectors and circulation finite")
    if on_segment not in ("raise", "straight_pv_zero"):
        raise ValueError("on_segment must be 'raise' or 'straight_pv_zero'")

    d = b - a
    length = float(np.linalg.norm(d))
    if not np.isfinite(length) or length == 0.0:
        raise ValueError("vortex segment must have nonzero finite length")
    r1 = p - a
    r2 = p - b
    r1n = float(np.linalg.norm(r1))
    r2n = float(np.linalg.norm(r2))
    if r1n == 0.0 or r2n == 0.0:
        raise FloatingPointError("field point is a singular filament endpoint")

    cross = np.cross(r1, r2)
    cross2 = float(np.dot(cross, cross))
    projection = float(np.dot(r1, d) / length)
    perpendicular = r1 - (projection / length) * d
    rho2 = float(np.dot(perpendicular, perpendicular))

    # Exact collinearity is treated explicitly; no epsilon/core is hidden here.
    if rho2 == 0.0:
        if 0.0 < projection < length:
            if on_segment == "straight_pv_zero":
                return np.zeros(3, dtype=float)
            raise FloatingPointError("field point lies on the singular vortex segment")
        # On the line extension, the finite segment induces exactly zero.
        return np.zeros(3, dtype=float)

    factor = float(np.dot(d, r1 / r1n - r2 / r2n))
    result = (gamma / (4.0 * np.pi)) * cross * (factor / cross2)
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite Biot–Savart segment velocity")
    return result


def polyline_biot_savart(point, vertices, circulation=1.0, *,
                         excluded_segments=(), on_segment="raise"):
    """Sum straight-segment contributions for an oriented open polygonal filament.

    excluded_segments is an explicit set of zero-based segment indices, e.g.
    to omit a segment's self contribution at an interior collocation point.
    Exclusion is bookkeeping, not a curvature correction: adjacent-segment
    endpoint singularities are not regularized and will raise.
    """
    xyz = np.asarray(vertices, dtype=float)
    gamma = float(circulation)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 2
            or not np.isfinite(xyz).all() or not np.isfinite(gamma)):
        raise ValueError("vertices must be a finite (N>=2, 3) array")
    excluded = {int(i) for i in excluded_segments}
    nseg = len(xyz) - 1
    if any(i < 0 or i >= nseg for i in excluded):
        raise ValueError("excluded segment index out of range")
    velocity = np.zeros(3, dtype=float)
    for i in range(nseg):
        if i not in excluded:
            velocity += segment_biot_savart(point, xyz[i], xyz[i+1], gamma,
                                            on_segment=on_segment)
    return velocity
