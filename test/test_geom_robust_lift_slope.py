"""Pure-math checks for the geometry-robustness lift-slope helpers (#38)."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "geom_robust"))
import lift_slope as ls  # noqa: E402


def test_both_estimates_match_their_closed_forms():
    ar = 5.0437
    assert ls.prandtl_slope(ar) == pytest.approx(2 * math.pi * ar / (ar + 2.0))
    assert ls.helmbold_slope(ar) == pytest.approx(
        2 * math.pi * ar / (2.0 + math.sqrt(ar * ar + 4.0)))


def test_prandtl_is_always_above_helmbold_and_both_are_below_two_pi():
    for ar in (0.5, 1.0, 2.0, 5.0437, 8.0, 20.0):
        prandtl, helmbold = ls.prandtl_slope(ar), ls.helmbold_slope(ar)
        assert 0.0 < helmbold < prandtl < 2 * math.pi


def test_estimates_vanish_at_zero_aspect_ratio_and_approach_two_pi():
    assert ls.prandtl_slope(1e-9) == pytest.approx(0.0, abs=1e-8)
    assert ls.helmbold_slope(1e-9) == pytest.approx(0.0, abs=1e-8)
    assert ls.prandtl_slope(1e7) == pytest.approx(2 * math.pi, rel=1e-6)
    assert ls.helmbold_slope(1e7) == pytest.approx(2 * math.pi, rel=1e-6)


def test_angle_units_convert_between_degrees_and_radians():
    # CL rises by exactly 0.1 per degree about zero: 0.1/deg = 5.7296/rad.
    assert ls.slope_single_point(0.5, 5.0) == pytest.approx(
        0.1 * 180.0 / math.pi)
    assert ls.slope_central(0.1, -0.1, 1.0) == pytest.approx(0.1 * 180.0 / math.pi)
    assert ls.slope_central(0.25, -0.25, 2.5) == pytest.approx(
        0.5 / (2.0 * math.radians(2.5)))
    assert ls.DEG_PER_RAD == pytest.approx(180.0 / math.pi)


def test_central_difference_is_exact_for_a_linear_law_with_nonzero_intercept():
    intercept, slope_per_deg = 3.73e-4, 0.0769

    def cl(alpha):
        return intercept + slope_per_deg * alpha

    for h in (2.5, 5.0):
        assert ls.slope_central(cl(h), cl(-h), h) == pytest.approx(
            slope_per_deg * 180.0 / math.pi, rel=1e-14)
    # The single-point quotient is biased by the intercept; the central
    # difference is not. That is exactly why the report may not call the old
    # number a slope.
    assert ls.slope_single_point(cl(5.0), 5.0) != pytest.approx(
        slope_per_deg * 180.0 / math.pi, rel=1e-6)


def test_even_and_odd_parts_split_offset_from_slope():
    intercept, slope_per_deg, h = 3.73e-4, 0.0769, 2.5
    plus = intercept + slope_per_deg * h
    minus = intercept - slope_per_deg * h
    assert ls.even_part(plus, minus) == pytest.approx(intercept)
    assert ls.odd_part(plus, minus) == pytest.approx(slope_per_deg * h)
    assert ls.even_part(plus, minus) + ls.odd_part(plus, minus) == pytest.approx(plus)


def test_central_table_reports_agreement_and_missing_steps():
    table = ls.central_table({2.5: 0.2, -2.5: -0.2, 5.0: 0.4, -5.0: -0.4}, (2.5, 5.0))
    assert table["slopes_per_rad"][2.5] == pytest.approx(table["slopes_per_rad"][5.0])
    assert table["agreement"] == pytest.approx(0.0, abs=1e-15)
    partial = ls.central_table({2.5: 0.2, -2.5: -0.2}, (2.5, 5.0))
    assert partial["slopes_per_rad"][5.0] is None
    assert partial["agreement"] is None


def test_degenerate_inputs_are_rejected():
    with pytest.raises(ValueError):
        ls.slope_single_point(0.3, 0.0)
    with pytest.raises(ValueError):
        ls.slope_central(0.3, -0.3, 0.0)
    with pytest.raises(ValueError):
        ls.aspect_ratio(0.0, 3.0)
    with pytest.raises(ValueError):
        ls.aspect_ratio(4.0, -1.0)
