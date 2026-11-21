"""End-to-end checks against independent inviscid references."""

import pytest

from panel.p1_lifting_g2 import combine_wings
from panel.p1_steady_free_wake import solve_steady_free_wake
from tools.calibration import _thin_wing, case_c02, case_c04, elliptic_row

pytestmark = pytest.mark.macro
DEFAULT_WING_MODE = "galerkin, pressure, area"


def test_sphere_pressure_converges_to_potential_flow():
    result = case_c02()
    coarse, medium, fine = result["levels"]
    assert medium["cp_l2"] < 0.7 * coarse["cp_l2"]
    assert fine["cp_l2"] < 0.7 * medium["cp_l2"]
    assert fine["cp_l2"] < 0.02
    assert result["differences"]["force_coefficient_norm"] < 1e-10


def test_default_thin_wing_approaches_lifting_line_with_refinement():
    coarse = case_c04(chord_panels=4, span_panels=16, alphas=(1., 3.))
    finer = case_c04(chord_panels=6, span_panels=24, alphas=(1., 3.))
    for report in (coarse, finer):
        assert report["solver_mode"] == DEFAULT_WING_MODE
        assert all(row["converged"] and row["kutta"] == "pressure" for row in report["levels"])
    coarse_error = abs(coarse["differences"]["lifting_line_relative"])
    finer_error = abs(finer["differences"]["lifting_line_relative"])
    assert finer_error < coarse_error
    assert finer_error < 0.12


def test_default_elliptic_wing_lift_loading_and_induced_drag():
    row = elliptic_row(6., 4., sections=5, chordwise=4, spanwise=2)
    assert row["solver_mode"] == DEFAULT_WING_MODE
    assert row["converged"]
    assert abs(row["cl_alpha_relative"]) < 0.06
    assert row["gamma_shape_rms"] < 0.09
    assert row["cdi_trefftz"] > 0.
    assert 0.94 < row["e"] <= 1. + 1e-10


def test_default_free_wake_respects_zero_incidence_symmetry():
    wing = _thin_wing(chord_panels=4, span_panels=16)
    result = solve_steady_free_wake(combine_wings([wing]), 0., iterations=4)
    state = result["state"]
    mode = ", ".join(state[key] for key in ("wall_formulation", "kutta", "pressure_recovery"))
    assert mode == DEFAULT_WING_MODE
    assert state["nonlinear"]["success"]
    assert result["status"] == "completed"
    assert result["converged"]
    final = result["history"][-1]
    wake, = final["wakes"].values()
    assert abs(final["CL"]) < 1e-3
    assert max(final["raw_map_residual"].values()) < 1e-3
    assert wake["normal_slip_rms"] < 5e-4
    assert wake["wake_cp_jump_rms"] < 5e-5
