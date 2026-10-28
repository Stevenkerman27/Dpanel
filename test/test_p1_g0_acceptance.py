"""Acceptance contract for the G0 sphere discretisation (issue #56, part D)."""

from __future__ import annotations

import numpy as np
import pytest

from panel.p1_g0_acceptance import (
    analytic_potential,
    jump_check,
    layer_potential,
    load_check,
    offset_potential_check,
    penetration_check,
)


def test_analytic_potential_closed_form():
    points = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    values = analytic_potential(points, radius=1.0, freestream=(1.0, 0.0, 0.0))
    assert values[0] == pytest.approx(2.0 * (1.0 + 1.0 / 16.0))
    assert values[1] == pytest.approx(0.0, abs=1e-15)
    with pytest.raises(ValueError):
        analytic_potential(np.array([[0.1, 0.0, 0.0]]), radius=1.0)


def test_offset_potential_errors_converge_with_refinement():
    coarse = offset_potential_check(1)
    fine = offset_potential_check(2)
    near_coarse = coarse["rows"][0]["max_relative_error"]
    near_fine = fine["rows"][0]["max_relative_error"]
    assert 0.0 < near_fine < near_coarse < 5e-2
    for row in fine["rows"]:
        assert row["max_relative_error"] < 1e-2
    # the far field is easier than the near field at fixed mesh
    errors = [row["max_relative_error"] for row in fine["rows"]]
    assert errors[-1] < errors[0]


def test_layer_potential_is_negligible_far_away():
    from panel.p1_dirichlet import icosphere, solve_p1_dirichlet

    vertices, faces = icosphere(1)
    solution = solve_p1_dirichlet(vertices, faces, np.array([1.0, 0.0, 0.0]))
    point = np.array([[50.0, 0.0, 0.0]])
    layer = layer_potential(vertices, faces, solution["mu"], point)[0]
    assert abs(layer) < 1e-2


def test_load_check_recovers_zero_drag_and_the_area():
    report = load_check(1)
    assert report["drag_coefficient"] < 1e-12
    assert report["lift_coefficient"] < 1e-12
    ratio = report["area_total"] / report["area_exact"]
    assert 0.90 < ratio < 1.05


def test_jump_equals_the_doublet_density():
    coarse = jump_check(1, offset=1e-4)
    fine = jump_check(1, offset=1e-5)
    assert coarse["jump_rms_error"] < 2e-4
    # the jump itself is exact; the residual is O(offset) from the neighbouring panels
    ratio = coarse["jump_rms_error"] / fine["jump_rms_error"]
    assert 8.0 < ratio < 12.0


def test_penetration_velocity_error_converges():
    coarse = penetration_check(1, radii=(1.02, 1.05), max_targets=24)
    fine = penetration_check(2, radii=(1.02, 1.05), max_targets=24)
    near_coarse = coarse["rows"][0]["max_error_over_speed"]
    near_fine = fine["rows"][0]["max_error_over_speed"]
    assert near_fine < 0.6 * near_coarse
    assert near_coarse < 0.25
    # the metric is an absolute error against U; report it, do not hide it behind a ratio
    for row in coarse["rows"]:
        assert row["max_exact_over_speed"] > 0.0
