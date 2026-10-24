"""Analytic root-contact endpoint for a symmetric straight-TE planform."""
import numpy as np


def positive_root_contact(te_x, body_center_x, semi_axis_x, body_radius,
                          half_span, k, taper_ratio):
    values = np.array([
        te_x, body_center_x, semi_axis_x, body_radius, half_span, k, taper_ratio,
    ], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Planform and ellipsoid parameters must be finite")
    if semi_axis_x <= 0.0 or body_radius <= 0.0 or half_span <= 0.0:
        raise ValueError("Ellipsoid axes and wing half-span must be positive")
    if taper_ratio <= 0.0:
        raise ValueError("Tip/root chord ratio must be positive")

    d = te_x - body_center_x
    if d < 0.0 or d >= semi_axis_x:
        raise ValueError("Root TE crossing requires 0 <= d < semi_axis_x")
    minimum_te_x = te_x + min(k, 0.0) * half_span
    if body_center_x > minimum_te_x:
        raise ValueError("Ellipsoid center lies aft of the minimum-TE station")

    a = semi_axis_x
    r = body_radius
    quadratic_a = k**2 / a**2 + 1.0 / r**2
    quadratic_b = 2.0 * d * k / a**2
    quadratic_c = d**2 / a**2 - 1.0
    discriminant = quadratic_b**2 - 4.0 * quadratic_a * quadratic_c
    if quadratic_c >= 0.0 or discriminant <= 0.0:
        raise ValueError("The TE plane has no finite crossing under the aft-branch conditions")

    if quadratic_b >= 0.0:
        root_y = -2.0 * quadratic_c / (quadratic_b + np.sqrt(discriminant))
    else:
        root_y = (-quadratic_b + np.sqrt(discriminant)) / (2.0 * quadratic_a)
    if not 0.0 < root_y < half_span:
        raise ValueError("Body contact must leave exposed TE on both sides")
    if 1.0 + (taper_ratio - 1.0) <= 0.0:
        raise ValueError("Planform chord must remain positive to the tip")
    return root_y
