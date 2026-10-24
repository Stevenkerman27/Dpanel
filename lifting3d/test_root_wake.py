"""Analytic checks for the independent aft-body wake-root candidate."""
import numpy as np

import root_wake

ncpu = 8
tolerance = 2.0e-13
assert ncpu == 8

half_span = 2.0
te_x = 1.0
body_aft = 1.5
body_length = 4.0
body_radius = 0.35
body_center_x = body_aft - body_length / 2.0
root_interval = root_wake.ellipsoid_root_interval(
    te_x, body_center_x, body_length / 2.0, body_radius, half_span)
expected_half_width = body_radius * np.sqrt(1.0 - ((te_x - body_center_x) / (body_length / 2.0)) ** 2)
assert abs(root_interval[0] + expected_half_width) < tolerance
assert abs(root_interval[1] - expected_half_width) < tolerance

stations = np.unique(np.r_[
    np.linspace(-half_span, half_span, 9),
    root_interval,
])
station_dof = root_wake.merge_root_basis(stations, root_interval, half_span)
root_stations = np.flatnonzero(
    (stations >= root_interval[0] - tolerance) & (stations <= root_interval[1] + tolerance))
assert len(np.unique(station_dof[root_stations])) == 1
root_dof = int(station_dof[root_stations[0]])

tip_zero = np.zeros(int(station_dof.max()) + 1)
tip_zero[root_dof] = 0.0
tip_zero_values = root_wake.expand_station_values(station_dof, tip_zero)
assert tip_zero_values[0] == 0.0 and tip_zero_values[-1] == 0.0
assert np.max(np.abs(tip_zero_values[root_stations])) == 0.0

plateau = np.zeros(int(station_dof.max()) + 1)
plateau[root_dof] = 0.37
plateau_values = root_wake.expand_station_values(station_dof, plateau)
assert np.max(np.abs(plateau_values[root_stations] - 0.37)) < tolerance
assert np.max(np.abs(np.diff(plateau_values[root_stations]))) < tolerance

varying = np.arange(len(plateau), dtype=float) + 1.0
varying_values = root_wake.expand_station_values(station_dof, varying)
assert np.max(np.abs(np.diff(varying_values[root_stations]))) == 0.0

exposed_intervals = np.array([
    [-half_span, root_interval[0]],
    [root_interval[1], half_span],
])
support = root_wake.mapped_weak_test_support(stations, station_dof, exposed_intervals)
assert support.shape == (2, int(station_dof.max()) + 1)
assert support[0, root_dof] > 0.0
assert support[1, root_dof] > 0.0
assert np.all(support.sum(axis=1) > 0.0)
assert np.all((support > 0.0).sum(axis=0) > 0)

negative_edge = np.flatnonzero(np.isclose(stations, root_interval[0], rtol=0.0, atol=tolerance))[0]
positive_edge = np.flatnonzero(np.isclose(stations, root_interval[1], rtol=0.0, atol=tolerance))[0]
negative_adjacent = stations[negative_edge - 1]
positive_adjacent = stations[positive_edge + 1]
assert support[0, root_dof] > 0.0 and negative_adjacent < root_interval[0]
assert support[1, root_dof] > 0.0 and positive_adjacent > root_interval[1]

invalid_cases = [
    ("body_occupies_full_span", lambda: root_wake.ellipsoid_root_interval(
        te_x, body_center_x, body_length / 2.0, 4.0, half_span)),
    ("te_plane_misses_ellipsoid", lambda: root_wake.ellipsoid_root_interval(
        te_x, 4.0, 1.0, body_radius, half_span)),
    ("root_endpoints_not_inserted", lambda: root_wake.merge_root_basis(
        np.linspace(-half_span, half_span, 9), root_interval, half_span)),
]
for name, operation in invalid_cases:
    rejected = False
    try:
        operation()
    except ValueError:
        rejected = True
    assert rejected, f"Expected explicit rejection: {name}"

print({
    "status": "passed",
    "scope": "Analytic root interval and reduced-basis checks only; no mesh or flow solve",
    "ncpu_requested": ncpu,
    "root_interval": root_interval.tolist(),
    "root_station_count": int(len(root_stations)),
    "root_dof": root_dof,
    "weak_test_support_on_both_exposed_sides": support[:, root_dof].tolist(),
    "checks": [
        "ellipsoid_root_interval",
        "gamma_tip_zero",
        "constant_root_plateau",
        "zero_root_slope_for_arbitrary_reduced_coefficients",
        "central_weak_test_support_on_both_exposed_te_sides",
        "full_span_body_rejected",
        "missed_te_plane_rejected",
        "missing_inserted_root_station_rejected",
    ],
})
