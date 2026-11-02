"""S3 tests: the torch driver for the nonlinear pressure-Kutta solve.

Two layers:

* a static syntax-tree guard, runnable in any interpreter, that keeps the torch
  path free of NumPy/SciPy (a LAPACK call after torch has executed in the same
  process dies with 0xC0000005, section 6f of the phase-1 doc);
* analytic checks of the trailing-edge pressure model and a frozen end-to-end
  case, which need torch and are skipped in the default .conda_env.

The frozen case (docs/issue92/s3_reference_g1_wing_rect.npz) holds the system,
the linear initial state and the NumPy structured result, all produced in a
separate process by tools.issue92.s3_pressure_parity bundle. It is gated by
PANEL_TORCH_PARITY=1 because a passing run leaves the interpreter unusable for
later LAPACK calls.

    PANEL_TORCH_PARITY=1 python -m pytest test/test_p1_pressure_kutta_torch_driver.py
"""
from __future__ import annotations

import ast
import os
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
BUNDLE = REPO / "docs" / "issue92" / "s3_reference_g1_wing_rect.npz"
TORCH_MODULES = ("panel/p1_pressure_kutta_torch.py",
                 "panel/p1_pressure_kutta_torch_driver.py",
                 "panel/p1_rank_aware_torch.py")

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _torch():
    return pytest.importorskip("torch")


@pytest.mark.parametrize("relative", TORCH_MODULES)
def test_torch_modules_have_no_numpy_or_scipy(relative):
    """Torch modules must not import or call NumPy/SciPy; docstrings may mention them."""
    tree = ast.parse((REPO / relative).read_text(encoding="utf-8"))
    banned = ("numpy", "scipy")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in banned, "%s imports %s" % (relative, alias.name)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            assert root not in banned, "%s imports from %s" % (relative, node.module)
        elif isinstance(node, ast.Attribute):
            chain = []
            current = node
            while isinstance(current, ast.Attribute):
                chain.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                chain.append(current.id)
            assert chain[-1] not in ("np", "numpy", "scipy"), \
                "%s calls %s" % (relative, ".".join(reversed(chain)))


def _reference_model(grad, pairs, state, n_unknown):
    """Plain NumPy transcription of the production pressure residual and Jacobian."""
    n_wall = grad.shape[2]
    velocity = np.einsum("fij,j->fi", grad, state[:n_wall])
    pressure = np.array([np.mean(np.sum(velocity[top] ** 2, axis=1))
                         - np.mean(np.sum(velocity[bot] ** 2, axis=1)) for top, bot in pairs])
    rows = np.array([2 * np.mean(np.einsum("fi,fij->fj", velocity[top], grad[top]), axis=0)
                     - 2 * np.mean(np.einsum("fi,fij->fj", velocity[bot], grad[bot]), axis=0)
                     for top, bot in pairs])
    jacobian = np.column_stack((rows, np.zeros((rows.shape[0], n_unknown - n_wall))))
    return pressure, jacobian


def _ragged_problem(seed=7, n_faces=9, n_wall=11, n_gamma=3):
    rng = np.random.default_rng(seed)
    grad = rng.normal(size=(n_faces, 3, n_wall))
    pairs = [([0, 1], [2]), ([3], [4, 5, 6]), ([7, 8], [0])]
    state = rng.normal(size=n_wall + n_gamma)
    return grad, pairs, state


def test_pressure_model_matches_the_numpy_definition():
    torch = _torch()
    from panel.p1_pressure_kutta_torch_driver import TorchPressureModel

    grad, pairs, state = _ragged_problem()
    model = TorchPressureModel(grad, pairs, state.size)
    reference, reference_jacobian = _reference_model(grad, pairs, state, state.size)
    pressure = model(torch.as_tensor(state))
    jacobian = model.jacobian(torch.as_tensor(state))
    assert pressure.shape == (len(pairs),)
    assert jacobian.shape == (len(pairs), state.size)
    np.testing.assert_allclose(pressure.numpy(), reference, rtol=0, atol=1e-12)
    np.testing.assert_allclose(jacobian.numpy(), reference_jacobian, rtol=0, atol=1e-12)
    assert np.all(jacobian.numpy()[:, model.n_wall:] == 0.0)


def test_pressure_jacobian_matches_finite_differences():
    torch = _torch()
    from panel.p1_pressure_kutta_torch_driver import TorchPressureModel

    grad, pairs, state = _ragged_problem(seed=11)
    model = TorchPressureModel(grad, pairs, state.size)
    step = 1e-6
    numeric = np.zeros((len(pairs), state.size))
    for j in range(state.size):
        bump = np.zeros_like(state)
        bump[j] = step
        numeric[:, j] = (model(torch.as_tensor(state + bump)).numpy()
                         - model(torch.as_tensor(state - bump)).numpy()) / (2 * step)
    analytic = model.jacobian(torch.as_tensor(state)).numpy()
    np.testing.assert_allclose(analytic, numeric, rtol=0, atol=1e-7)


def test_solver_matches_the_numpy_structured_backend():
    # Order matters: the NumPy/SciPy reference runs first, before any torch op has
    # executed in this process. A LAPACK call after torch has run dies with
    # 0xC0000005 (section 6f), so the comparison cannot be interleaved.
    from panel.p1_pressure_kutta_reduced import StructuredPressureKutta
    from scipy.linalg import null_space

    rng = np.random.default_rng(23)
    n_wall, n_gamma, n_rows, n_kutta = 14, 3, 40, 4
    n_unknown = n_wall + n_gamma
    B = rng.normal(size=(n_rows, n_unknown))
    bw = rng.normal(size=n_rows)
    kutta = rng.normal(size=(n_kutta, n_unknown))
    grad, pairs, _ = _ragged_problem(seed=5, n_faces=9, n_wall=n_wall, n_gamma=n_gamma)
    state0 = rng.normal(size=n_unknown)
    weight = 5.0

    reduction = null_space(kutta, rcond=1e-11)

    def pressure(state):
        return _reference_model(grad, pairs, state, n_unknown)[0]

    def jacobian(state):
        return _reference_model(grad, pairs, state, n_unknown)[1]

    structured = StructuredPressureKutta(wall_operator=B, reduction=reduction, wall_rhs=bw,
                                         pressure=pressure, pressure_jacobian=jacobian,
                                         weight=weight)
    expected = structured.solve(reduction.T @ state0, max_nfev=60)

    torch = _torch()
    from panel.p1_pressure_kutta_torch_driver import solve_pressure_kutta_torch

    out = solve_pressure_kutta_torch(wall_operator=B, wall_rhs=bw, kutta_operator=kutta,
                                     grad=grad, te_face_pairs=pairs, x0=state0,
                                     weight=weight, max_nfev=60)
    assert int(out.result.status) == int(expected.status)
    assert int(out.result.nfev) == int(expected.nfev)
    assert str(out.result.message) == str(expected.message)
    got = out.state.numpy()
    want = reduction @ expected.x
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-10 * max(1.0, np.abs(want).max()))


@pytest.mark.skipif(os.environ.get("PANEL_TORCH_PARITY") != "1",
                    reason="torch run leaves LAPACK unusable; set PANEL_TORCH_PARITY=1")
def test_frozen_case_matches_the_numpy_structured_result():
    torch = _torch()
    from panel.p1_pressure_kutta_torch_driver import solve_pressure_kutta_torch

    with np.load(BUNDLE) as data:
        B, bw, kutta, grad = (data[key] for key in ("B", "bw", "kutta", "grad"))
        x0, expected_state = data["x0"], data["expected"]
        top_flat, top_ptr = data["top_flat"], data["top_ptr"]
        bot_flat, bot_ptr = data["bot_flat"], data["bot_ptr"]
        status, nfev = int(data["status"]), int(data["nfev"])
        message = str(data["message"])
        max_nfev = int(data["max_nfev"])
        weight = float(data["weight"])
    tops = np.split(top_flat, top_ptr[1:-1])
    bots = np.split(bot_flat, bot_ptr[1:-1])
    pairs = [(list(map(int, top)), list(map(int, bot))) for top, bot in zip(tops, bots)]

    out = solve_pressure_kutta_torch(wall_operator=B, wall_rhs=bw, kutta_operator=kutta,
                                     grad=grad, te_face_pairs=pairs, x0=x0,
                                     weight=weight, max_nfev=max_nfev)
    assert int(out.result.status) == status
    assert int(out.result.nfev) == nfev
    assert str(out.result.message) == message
    np.testing.assert_allclose(out.state.numpy(), expected_state, rtol=0,
                               atol=1e-10 * max(1.0, np.abs(expected_state).max()))


def test_rejects_inconsistent_inputs():
    torch = _torch()
    from panel.p1_pressure_kutta_torch_driver import TorchPressureModel, solve_pressure_kutta_torch

    grad, pairs, state = _ragged_problem()
    with pytest.raises(ValueError):
        TorchPressureModel(grad, [], state.size)
    with pytest.raises(ValueError):
        TorchPressureModel(grad, [([0], [])], state.size)
    with pytest.raises(ValueError):
        TorchPressureModel(grad, pairs, grad.shape[2] - 1)
    kutta = np.zeros((2, state.size))
    with pytest.raises(ValueError):
        solve_pressure_kutta_torch(wall_operator=np.zeros((5, state.size)), wall_rhs=np.zeros(4),
                                   kutta_operator=kutta, grad=grad, te_face_pairs=pairs, x0=state)
