"""The lifting-line reference itself: known limits before it judges the solver."""

import numpy as np
import pytest

from tools.lifting_line import lifting_line

SPAN = 8.0
TWO_PI = 2.0 * np.pi


def _elliptic(root_chord=2.0):
    def chord(y):
        return root_chord * np.sqrt(np.clip(1.0 - (y / (0.5 * SPAN)) ** 2, 0.0, None))
    return chord


def test_elliptic_planform_is_the_unit_efficiency_case():
    result = lifting_line(chord_of_y=_elliptic(), span=SPAN, alpha_deg=4.0)
    assert abs(result["e"] - 1.0) < 2e-3, result["e"]
    ar = result["aspect_ratio"]
    expected = TWO_PI / (1.0 + TWO_PI / (np.pi * ar))
    assert abs(result["cl_alpha"] - expected) / expected < 0.01, (result["cl_alpha"], expected)
    assert abs(result["cl"] - result["cl_alpha"] * np.deg2rad(4.0)) < 1e-12


def test_rectangular_planform_lands_near_the_prandtl_slope():
    span = 6.0
    y = np.linspace(-0.5 * span, 0.5 * span, 401)
    result = lifting_line(y=y, chord=np.full(y.size, 1.0), alpha_deg=0.0, terms=127)
    prandtl = TWO_PI * 6.0 / (6.0 + 2.0)
    assert abs(result["cl_alpha"] - prandtl) / prandtl < 0.10
    assert 0.92 < result["e"] < 0.98, result["e"]


def test_drag_scales_with_the_square_of_the_lift():
    chord = _elliptic()
    low = lifting_line(chord_of_y=chord, span=SPAN, alpha_deg=2.0)
    high = lifting_line(chord_of_y=chord, span=SPAN, alpha_deg=4.0)
    assert abs(high["cdi"] / low["cdi"] - 4.0) < 1e-12
    assert abs(high["cl"] / low["cl"] - 2.0) < 1e-12


def test_a_chord_that_vanishes_inside_the_stations_is_rejected():
    y = np.linspace(-3.0, 3.0, 5)
    with pytest.raises(ValueError, match="non-negative"):
        lifting_line(y=y, chord=np.zeros(5))
    with pytest.raises(ValueError, match="vanishes"):
        lifting_line(y=y, chord=np.array([1.0, 0.0, 0.0, 0.0, 1.0]))


def test_gamma_coordinates_normalization_and_lift_integral():
    y = np.linspace(-4.0, 4.0, 4001)
    result = lifting_line(y, _elliptic()(y), alpha_deg=4.0)
    np.testing.assert_array_equal(result["y"], y)
    assert result["gamma"].shape == y.shape
    assert result["gamma"][0] == result["gamma"][-1] == 0.0
    cl_integral = 2.0 * np.trapezoid(result["gamma"], result["y"]) / result["area"]
    assert cl_integral == pytest.approx(result["cl"], rel=2e-5)
    callable_result = lifting_line(chord_of_y=_elliptic(), span=SPAN, alpha_deg=4.0)
    assert callable_result["y"].shape == callable_result["gamma"].shape
    assert np.all(np.diff(callable_result["y"]) > 0)
    assert callable_result["y"][[0, -1]].tolist() == [-4.0, 4.0]


def test_independent_circulation_and_downwash_integrals_match_modal_lift_and_drag():
    result = lifting_line(np.linspace(-3., 3., 801), np.ones(801),
                          alpha_deg=4.0, terms=63)
    nodes, weights = np.polynomial.legendre.leggauss(300)
    theta = (nodes + 1.0) * np.pi / 2.0
    sine = np.sin(np.outer(theta, result["modes"]))
    gamma = 2.0 * result["span"] * (sine @ result["coefficients"])
    induced_angle = (sine @ (result["modes"] * result["coefficients"])) / np.sin(theta)
    dy = weights * np.pi / 2.0 * result["span"] / 2.0 * np.sin(theta)
    cl = 2.0 * np.sum(gamma * dy) / result["area"]
    cdi = 2.0 * np.sum(gamma * induced_angle * dy) / result["area"]
    assert cl == pytest.approx(result["cl"], rel=1e-12)
    assert cdi == pytest.approx(result["cdi"], rel=1e-12)
    assert result["e"] == pytest.approx(cl * cl / (np.pi * result["aspect_ratio"] * cdi),
                                      rel=1e-12)


def test_odd_modes_refuse_asymmetric_chord():
    y = np.linspace(-3., 3., 101)
    with pytest.raises(ValueError, match="symmetric"):
        lifting_line(y, 1.0 + 0.1 * y)
