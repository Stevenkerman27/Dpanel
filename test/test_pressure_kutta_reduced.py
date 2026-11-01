"""Tests for the structured pressure-Kutta Gauss-Newton solver."""
from __future__ import annotations

import numpy as np
import pytest

from panel.p1_pressure_kutta_reduced import (
    StructuredPressureKutta,
    solve_structured_pressure,
)


def _model(*, scale=None, quadratic=0.1, offset=0.0, seed=7):
    """Small synthetic reduced problem with a short nonlinear pressure block."""
    rng = np.random.default_rng(seed)
    n_full, n_free, n_wall, n_pressure = 6, 4, 9, 3
    q, _ = np.linalg.qr(rng.normal(size=(n_full, n_free)))
    B = rng.normal(size=(n_wall, n_full))
    if scale is not None:
        B = B @ np.diag(scale)
    A = rng.normal(size=(n_pressure, n_full))
    bw = rng.normal(size=n_wall)

    def pressure(v):
        out = A @ v + offset
        out = out.copy()
        out[0] += quadratic * v[0] ** 2
        out[1] += quadratic * v[2] ** 2
        return out

    def jacobian(v):
        out = A.copy()
        out[0, 0] += 2.0 * quadratic * v[0]
        out[1, 2] += 2.0 * quadratic * v[2]
        return out

    return dict(wall_operator=B, reduction=q, wall_rhs=bw, pressure=pressure,
                pressure_jacobian=jacobian, weight=5.)


def _stacked(solver, z):
    v = solver.N @ z
    wall = solver._M @ z - solver.bw
    p = np.asarray(solver.pressure(v), float)
    F = solver.weight * np.asarray(solver.pressure_jacobian(v), float) @ solver.N
    r = np.r_[wall, solver.weight * p]
    return np.vstack([solver._M, F]), r, F


def test_the_step_solves_the_same_stacked_least_squares_problem():
    solver = StructuredPressureKutta(**_model())
    rng = np.random.default_rng(11)
    z = rng.normal(size=solver.n_free)
    step, gradient, schur = solver.step(z)
    stacked, residual, _ = _stacked(solver, z)
    reference = np.linalg.lstsq(stacked, -residual, rcond=None)[0]
    assert np.allclose(step, reference, atol=1e-10, rtol=0.0)
    assert np.allclose(gradient, stacked.T @ residual, atol=1e-10, rtol=0.0)
    assert schur.shape == (solver.n_pressure, solver.n_pressure)


def _conditioned_solver(singular, *, weight=5.):
    """Solver whose wall block M has exactly the prescribed singular values."""
    rng = np.random.default_rng(13)
    n_wall = 9
    n_free = singular.size
    q1, _ = np.linalg.qr(rng.normal(size=(n_wall, n_free)))
    q2, _ = np.linalg.qr(rng.normal(size=(n_free, n_free)))
    M = q1 @ np.diag(singular) @ q2.T
    B = np.zeros((n_wall, 6))
    B[:, :n_free] = M
    model = _model()
    return StructuredPressureKutta(wall_operator=B, reduction=np.eye(6)[:, :n_free],
                                   wall_rhs=model["wall_rhs"], pressure=model["pressure"],
                                   pressure_jacobian=model["pressure_jacobian"], weight=weight)


def test_the_step_stays_accurate_on_a_moderately_ill_conditioned_wall_block():
    solver = _conditioned_solver(np.array([1., 1e-1, 1e-2, 1e-3]))
    assert solver.condition_number_M > 1e2
    rng = np.random.default_rng(21)
    z = rng.normal(size=solver.n_free)
    step, _, _ = solver.step(z)
    stacked, residual, _ = _stacked(solver, z)
    reference = np.linalg.lstsq(stacked, -residual, rcond=None)[0]
    error = float(np.linalg.norm(step - reference) / max(np.linalg.norm(reference), 1e-300))
    assert error < 1e-5, error
    # The Woodbury form loses digits as the wall block gets stiffer, because the
    # correction subtracts two large terms; the condition number is reported so a
    # caller can see when the step is no longer trustworthy.
    stiff = _conditioned_solver(np.array([1., 1e-3, 1e-4, 1e-5]))
    stiff_step, _, _ = stiff.step(z[: stiff.n_free])
    stiff_stacked, stiff_residual, _ = _stacked(stiff, z[: stiff.n_free])
    stiff_reference = np.linalg.lstsq(stiff_stacked, -stiff_residual, rcond=None)[0]
    stiff_error = float(np.linalg.norm(stiff_step - stiff_reference)
                        / max(np.linalg.norm(stiff_reference), 1e-300))
    assert stiff_error > error


def test_a_rank_deficient_wall_block_is_refused():
    model = _model()
    model["wall_operator"] = np.zeros((9, 6))
    with pytest.raises(ValueError, match="rank deficient"):
        StructuredPressureKutta(**model)


def test_malformed_inputs_are_refused():
    model = _model()
    with pytest.raises(ValueError, match="non-finite"):
        StructuredPressureKutta(**{**model, "wall_operator": np.full((9, 6), np.nan)})
    with pytest.raises(ValueError, match="disagree on the unknown count"):
        StructuredPressureKutta(**{**model, "reduction": np.ones((5, 3))})
    with pytest.raises(ValueError, match="one entry per wall row"):
        StructuredPressureKutta(**{**model, "wall_rhs": np.zeros(4)})
    with pytest.raises(ValueError, match="weight must be positive"):
        StructuredPressureKutta(**{**model, "weight": 0.})
    with pytest.raises(ValueError, match="wall_operator must be two-dimensional"):
        StructuredPressureKutta(**{**model, "wall_operator": np.zeros(9)})


def test_a_linear_problem_converges_to_the_least_squares_solution():
    solver = StructuredPressureKutta(**_model(quadratic=0.0))
    result = solve_structured_pressure(
        wall_operator=solver.B, reduction=solver.N, wall_rhs=solver.bw,
        pressure=solver.pressure, pressure_jacobian=solver.pressure_jacobian,
        x0=np.zeros(solver.n_free), weight=solver.weight)
    stacked, residual, _ = _stacked(solver, np.zeros(solver.n_free))
    reference = np.linalg.lstsq(stacked, -residual, rcond=None)[0]
    assert result.success
    assert np.allclose(result.x, reference, atol=1e-9, rtol=0.0)


def test_a_nonzero_residual_floor_terminates_on_the_relative_tolerance():
    solver = StructuredPressureKutta(**_model(offset=0.5))
    # Switch the cost-reduction and step tests off: with a residual floor the
    # absolute gradient never reaches gtol, so only the relative test can stop it.
    result = solver.solve(np.zeros(solver.n_free), ftol=0.0, xtol=0.0)
    assert result.status in (4, 5)
    assert result.nfev < 100
    assert result.relative_optimality is not None
    assert result.relative_optimality < 1e-6
    assert result.history
    last = result.history[-1]
    for key in ("iteration", "nfev", "cost", "gradient_inf", "gradient_relative",
                "wall_rms", "pressure_rms"):
        assert key in last


def test_the_evaluation_budget_is_respected():
    solver = StructuredPressureKutta(**_model())
    result = solver.solve(np.zeros(solver.n_free), max_nfev=1)
    assert result.status == 0
    assert "maximum number of function evaluations" in result.message
    assert result.nfev <= 1


def test_the_solver_reports_the_reference_gradient_and_cost():
    solver = StructuredPressureKutta(**_model())
    rng = np.random.default_rng(17)
    z = rng.normal(size=solver.n_free)
    stacked, residual, F = _stacked(solver, z)
    gradient, _ = solver.gradient(z)
    assert np.allclose(gradient, stacked.T @ residual, atol=1e-10, rtol=0.0)
    assert np.isclose(solver.cost(z), 0.5 * float(residual @ residual), rtol=1e-12)
    assert np.isclose(solver._relative_optimality(gradient, F, residual),
                      float(np.linalg.norm(gradient, np.inf))
                      / (np.sqrt(solver._M_frobenius ** 2 + float(np.sum(F * F)))
                         * float(np.linalg.norm(residual))), rtol=1e-12)


def test_the_production_entry_points_expose_the_solver():
    pytest.importorskip("scipy")
    from panel.p1_lifting_g1 import make_wing, solve
    wing = make_wing(chord_panels=4, span_panels=4, thickness=0.12, span=4.0,
                     chord=1.0, wake_panels=2)
    reference = solve(wing, 5.0, kutta="pressure", pressure_solver="reference")
    structured = solve(wing, 5.0, kutta="pressure", pressure_solver="structured")
    assert structured["pressure_solver"] == "structured"
    assert structured["pressure_nfev"] >= 1
    assert structured["pressure_history"]
    assert structured["CL"] == pytest.approx(reference["CL"], abs=1e-9)
    assert structured["CD_pressure"] == pytest.approx(reference["CD_pressure"], abs=1e-12)
    assert structured["CM_y"] == pytest.approx(reference["CM_y"], abs=1e-9)
    with pytest.raises(ValueError, match="pressure_solver must be"):
        solve(wing, 5.0, kutta="pressure", pressure_solver="cuda")

def test_the_multi_body_entry_point_exposes_the_solver():
    from panel.p1_lifting_g2 import make_conventional_fixture, solve_multi
    mesh = make_conventional_fixture()
    reference = solve_multi(mesh, 5.0, kutta="pressure", pressure_solver="reference")
    structured = solve_multi(mesh, 5.0, kutta="pressure", pressure_solver="structured")
    assert structured["pressure_solver"] == "structured"
    assert structured["pressure_nfev"] >= 1
    assert structured["CL"] == pytest.approx(reference["CL"], abs=1e-9)
    # The pressure drag is a difference of large mu terms, so it is amplified
    # relative to the state; the bench records the same effect at 1e-9 absolute.
    assert structured["CD_pressure"] == pytest.approx(reference["CD_pressure"], rel=1e-7)
    with pytest.raises(ValueError, match="pressure_solver must be"):
        solve_multi(mesh, 5.0, kutta="pressure", pressure_solver="cuda")
