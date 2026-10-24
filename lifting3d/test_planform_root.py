"""Analytic regression checks for trapezoidal aft-body root geometry."""
import numpy as np

import planform_root


ncpu = 8
assert ncpu == 8
tolerance = 2.0e-13

te_x = 1.0
half_span = 2.0
semi_axis_x = 2.0
body_radius = 0.35
taper_ratio = 0.8

cases = [
    ("zero_sweep", 0.0, 0.5),
    ("positive_k", 0.12, 0.5),
    ("negative_k", -0.08, 0.5),
    ("minimum_te_k_positive", 0.12, 0.0),
    ("minimum_te_k_negative", -0.08, 0.16),
]
case_roots = {}
for name, k, d in cases:
    body_center_x = te_x - d
    root_y = planform_root.positive_root_contact(
        te_x, body_center_x, semi_axis_x, body_radius,
        half_span, k, taper_ratio)
    quadratic_a = k**2 / semi_axis_x**2 + 1.0 / body_radius**2
    quadratic_b = 2.0 * d * k / semi_axis_x**2
    quadratic_c = d**2 / semi_axis_x**2 - 1.0
    assert abs(quadratic_a * root_y**2 + quadratic_b * root_y + quadratic_c) < tolerance
    assert root_y > 0.0
    root_te_x = te_x + k * root_y
    root_body_x = body_center_x + semi_axis_x * np.sqrt(1.0 - (root_y / body_radius)**2)
    aft_branch_residual = root_body_x - root_te_x
    assert abs(aft_branch_residual) < tolerance

    contact_y = np.linspace(0.0, root_y, 101)
    contact_te_x = te_x + k * contact_y
    contact_body_x = body_center_x + semi_axis_x * np.sqrt(
        1.0 - (contact_y / body_radius)**2)
    assert np.all(contact_body_x >= contact_te_x - tolerance)

    outside_limit = min(body_radius, half_span)
    outside_y = np.linspace(root_y + (outside_limit-root_y)/101.0,
                            outside_limit, 100)
    outside_te_x = te_x + k * outside_y
    outside_body_x = body_center_x + semi_axis_x * np.sqrt(
        1.0 - (outside_y / body_radius)**2)
    assert np.all(outside_body_x < outside_te_x + tolerance)
    case_roots[name] = root_y

assert abs(case_roots["zero_sweep"] -
           body_radius * np.sqrt(1.0 - (0.5 / semi_axis_x)**2)) < tolerance
assert 0.0 < case_roots["minimum_te_k_positive"] < body_radius
assert case_roots["minimum_te_k_negative"] < half_span

invalid_cases = [
    ("d_negative", (te_x, te_x + 0.01, semi_axis_x, body_radius, half_span, 0.0, 1.0)),
    ("d_equal_a", (te_x, te_x - semi_axis_x, semi_axis_x, body_radius, half_span, 0.0, 1.0)),
    ("center_ahead_of_minimum_te",
     (te_x, te_x + 0.01, semi_axis_x, body_radius, half_span, -0.08, 1.0)),
    ("body_contact_covers_full_span",
     (te_x, te_x - 0.5, semi_axis_x, 3.0, half_span, 0.0, 1.0)),
    ("nonpositive_tip_chord",
     (te_x, te_x - 0.5, semi_axis_x, body_radius, half_span, 0.0, 0.0)),
]
for name, parameters in invalid_cases:
    rejected = False
    try:
        planform_root.positive_root_contact(*parameters)
    except ValueError:
        rejected = True
    assert rejected, f"Expected explicit geometry rejection: {name}"

print({
    "status": "passed",
    "scope": "Analytic geometry only; no mesh or flow solve",
    "ncpu_requested": ncpu,
    "positive_root_y": case_roots,
    "checks": [
        "k_zero_positive_negative",
        "xc_equals_minimum_TE_for_positive_and_negative_k",
        "d_zero_is_valid",
        "quadratic_residual",
        "invalid_crossings_and_planforms_rejected",
    ],
})
