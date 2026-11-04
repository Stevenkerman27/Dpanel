"""Tests for the implicit adjoints (Route B, slice S8)."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

TORCH = pytest.importorskip("torch", reason="the implicit adjoint is a torch module")

from panel.p1_implicit_adjoint import (  # noqa: E402
    AdjointProblem, finite_difference_angle, implicit_state)
from panel.p1_pressure_kutta_torch_driver import TorchPressureModel  # noqa: E402

MODULE = Path(__file__).resolve().parents[1]/"panel"/"p1_implicit_adjoint.py"


def _problem(seed=7, rows=40, columns=10, constraints=3, model=None, pairs=None):
    rng = np.random.default_rng(seed)
    operator = rng.normal(size=(rows, columns))
    constraint = rng.normal(size=(constraints, columns))
    centers = rng.normal(size=(rows, 3))*0.1
    weights = 0.5 + rng.random(rows)
    return AdjointProblem(wall_operator=operator, kutta_operator=constraint,
                          centers=centers, weights=weights, model=model,
                          te_face_pairs=pairs)


def _grads(problem, columns):
    rng = np.random.default_rng(11)
    return torch.as_tensor(rng.normal(size=columns), dtype=torch.float64)


import torch  # noqa: E402  (after the numpy problem builder, which does no LAPACK)


def test_linear_state_solves_the_projected_least_squares():
    problem = _problem()
    rhs = problem.rhs(35.0)
    reference = torch.linalg.lstsq(problem.wall_operator @ problem.basis, rhs,
                                   driver="gelsd").solution
    state = problem.linear_state(35.0)
    assert torch.allclose(state, problem.basis @ reference, atol=1e-12)


def test_adjoint_matches_autograd_through_lstsq():
    problem = _problem()
    weight = _grads(problem, int(problem.wall_operator.shape[1]))
    alpha = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
    rhs = problem.rhs(alpha)
    x = problem.basis @ torch.linalg.lstsq(problem.wall_operator @ problem.basis, rhs,
                                           driver="gelsd").solution
    (reference,) = torch.autograd.grad(x @ weight, alpha)
    state = problem.linear_state(alpha).detach()
    gradient = problem.angle_gradient(alpha.detach(), state, weight)
    assert abs(float(gradient) - float(reference)) <= 1e-9*max(1.0, abs(float(reference)))


def test_gradcheck_of_the_implicit_state():
    problem = _problem()
    state_size = int(problem.wall_operator.shape[1])
    weight = torch.as_tensor(np.random.default_rng(3).normal(size=state_size),
                             dtype=torch.float64)

    def objective(alpha):
        return torch.sum(implicit_state(alpha, problem) @ weight)

    alpha = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(objective, (alpha,), eps=1e-6, atol=1e-6, rtol=1e-4)


def test_directional_vjp_matches_a_central_difference():
    problem = _problem()
    weight = torch.as_tensor(np.random.default_rng(5).normal(size=int(problem.wall_operator.shape[1])),
                             dtype=torch.float64)
    alpha = 6.0
    state = problem.linear_state(alpha).detach()
    analytic = float(problem.angle_gradient(torch.tensor(alpha, dtype=torch.float64), state,
                                            weight))
    numeric = finite_difference_angle(
        lambda angle: float(problem.linear_state(angle).detach() @ weight), alpha)
    assert abs(analytic - numeric) <= 1e-7*max(1.0, abs(numeric))


def test_pressure_level_adjoint_matches_a_central_difference():
    rng = np.random.default_rng(13)
    rows, columns = 24, 8
    grad = rng.normal(size=(rows, columns, 3))*0.05
    pairs = [(np.array([0, 1]), np.array([2, 3])), (np.array([4]), np.array([5, 6]))]
    model = TorchPressureModel(torch.as_tensor(grad, dtype=torch.float64), pairs, columns)
    problem = _problem(seed=17, rows=rows, columns=columns, constraints=2, model=model,
                       pairs=pairs)
    weight = torch.as_tensor(rng.normal(size=columns), dtype=torch.float64)
    alpha = torch.tensor(4.0, dtype=torch.float64)
    state = problem.state(alpha)
    analytic = float(problem.angle_gradient(alpha, state, weight))
    numeric = finite_difference_angle(
        lambda angle: float(problem.state(angle) @ weight), 4.0, step=1e-4)
    residual = abs(float(problem.last_pressure_residual.max()))
    assert abs(analytic - numeric) <= 1e-3 + 10.0*residual


def test_rank_deficient_operator_is_refused():
    operator = np.zeros((6, 4))
    operator[0] = 1.0
    constraint = np.eye(1, 4)
    with pytest.raises(ValueError, match="rank deficient"):
        AdjointProblem(wall_operator=operator, kutta_operator=constraint,
                       centers=np.zeros((6, 3)), weights=np.ones(6))


def test_module_does_not_touch_numpy_linalg():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""])
            for name in names:
                assert not name.startswith("scipy"), name
                assert name != "numpy", name
        if isinstance(node, ast.Attribute) and node.attr == "linalg":
            base = node.value
            if isinstance(base, ast.Name):
                assert base.id != "np", "np.linalg in the adjoint module"

def _quartic_problem(seed=13, rows=24, columns=8):
    rng = np.random.default_rng(seed)
    grad = rng.normal(size=(rows, 3, columns))*0.05
    pairs = [(np.array([0, 1]), np.array([2, 3])), (np.array([4]), np.array([5, 6]))]
    model = TorchPressureModel(torch.as_tensor(grad, dtype=torch.float64), pairs, columns)
    problem = AdjointProblem(
        wall_operator=rng.normal(size=(rows, columns)),
        kutta_operator=rng.normal(size=(2, columns)),
        centers=rng.normal(size=(rows, 3))*0.1, weights=0.5+rng.random(rows),
        model=model, te_face_pairs=pairs)
    return problem, model, torch.tensor(4.0, dtype=torch.float64)


def test_reduced_hessian_is_the_autograd_jacobian_of_the_reduced_gradient():
    problem, model, alpha = _quartic_problem()
    state = problem.state(alpha).detach()
    hessian = problem.reduced_hessian(state)
    basis = problem.basis

    def potential(z):
        x = basis @ z
        residual = problem.wall_operator @ x-problem.rhs(alpha)
        pressure = model(x)
        return 0.5*(residual @ residual)+0.5*problem.weight**2*(pressure @ pressure)

    z = (basis.mT @ state).detach().clone().requires_grad_(True)
    gradient = torch.autograd.grad(potential(z), z, create_graph=True)[0]
    rows = [torch.autograd.grad(gradient[i], z, retain_graph=True)[0]
            for i in range(int(z.shape[0]))]
    numeric = torch.stack(rows)
    scale = float(numeric.abs().max())
    assert torch.allclose(numeric, hessian, rtol=1e-9, atol=1e-9*scale)


def test_the_gauss_newton_hessian_is_not_the_exact_one_when_pressure_is_nonzero():
    problem, model, alpha = _quartic_problem()
    state = problem.state(alpha).detach()
    exact = problem.reduced_hessian(state)
    residual = float(problem.last_pressure_residual.abs().max())
    approximate = AdjointProblem(
        wall_operator=problem.wall_operator, kutta_operator=problem.kutta_operator,
        centers=problem.centers, weights=problem.weights, model=model,
        te_face_pairs=problem.te_face_pairs, hessian="gauss_newton").reduced_hessian(state)
    difference = float((exact-approximate).abs().max())
    assert difference > 0.0
    residual = float(problem.last_pressure_residual.abs().max())
    assert residual > 0.0


def test_unknown_hessian_name_is_refused():
    with pytest.raises(ValueError, match="hessian must be"):
        AdjointProblem(wall_operator=np.eye(6), kutta_operator=np.eye(1, 6),
                       centers=np.zeros((6, 3)), weights=np.ones(6), hessian="newton")


def test_exact_adjoint_beats_gauss_newton_against_a_central_difference():
    rng = np.random.default_rng(29)
    problem, model, alpha = _quartic_problem(seed=31)
    weight = torch.as_tensor(rng.normal(size=int(problem.wall_operator.shape[1])),
                             dtype=torch.float64)
    state = problem.state(alpha).detach()
    exact = abs(float(problem.angle_gradient(alpha, state, weight))
                - finite_difference_angle(lambda a: float(problem.state(a) @ weight), 4.0,
                                          step=1e-4))
    approximate = AdjointProblem(
        wall_operator=problem.wall_operator, kutta_operator=problem.kutta_operator,
        centers=problem.centers, weights=problem.weights, model=model,
        te_face_pairs=problem.te_face_pairs, hessian="gauss_newton")
    residual = float(problem.last_pressure_residual.abs().max())
    error = abs(float(approximate.angle_gradient(alpha, state, weight))
                - finite_difference_angle(lambda a: float(problem.state(a) @ weight), 4.0,
                                          step=1e-4))
    assert exact <= 1e-3+10.0*residual or exact < error
