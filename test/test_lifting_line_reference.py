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
    # A sharp-tipped rectangle is a slowly converging case (the loading jumps at
    # the side edge); the module documents that instead of hiding it.
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