"""S3 tests: the torch driver for the nonlinear pressure-Kutta solve.

Two layers:

* a static syntax-tree guard, runnable in any interpreter, that keeps the torch
  path free of NumPy/SciPy (a LAPACK call after torch has executed in the same
  process dies with 0xC0000005, section 6f of the phase-1 doc);
* analytic checks of the trailing-edge pressure model and a frozen end-to-end
  case, which need torch and are skipped in the default .conda_env.

The frozen case (verification/issue92/s3_reference_g1_wing_rect.npz) holds the system,
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
BUNDLE = REPO / "verification" / "issue92" / "s3_reference_g1_wing_rect.npz"
TORCH_MODULES = ("panel/p1_pressure_kutta_torch.py",
                 "panel/p1_pressure_kutta_torch_driver.py",
                 "panel/p1_rank_aware_torch.py",
                 "panel/p1_assembly_torch.py",
                 "panel/p1_solve_torch.py")

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


def _reference_model(maps, state):
    top, bottom = maps
    n_wall = top.shape[2]
    vt, vb = top @ state[:n_wall], bottom @ state[:n_wall]
    pressure = (vt**2 - vb**2).sum(axis=1)
    rows = 2. * (vt[..., None] * top - vb[..., None] * bottom).sum(axis=1)
    jacobian = np.column_stack((rows, np.zeros((len(rows), state.size - n_wall))))
    return pressure, jacobian


def _recovered_problem(seed=7, stations=3, n_wall=11, n_gamma=3):
    rng = np.random.default_rng(seed)
    maps = rng.normal(size=(2, stations, 3, n_wall))
    state = rng.normal(size=n_wall + n_gamma)
    return maps, state


def test_pressure_model_matches_the_numpy_definition():
    torch = _torch()
    from panel.p1_pressure_kutta_torch_driver import TorchPressureModel

    maps, state = _recovered_problem()
    model = TorchPressureModel(maps, state.size)
    reference, reference_jacobian = _reference_model(maps, state)
    pressure = model(torch.as_tensor(state))
    jacobian = model.jacobian(torch.as_tensor(state))
    assert pressure.shape == (maps.shape[1],)
    assert jacobian.shape == (maps.shape[1], state.size)
    np.testing.assert_allclose(pressure.numpy(), reference, rtol=0, atol=1e-12)
    np.testing.assert_allclose(jacobian.numpy(), reference_jacobian, rtol=0, atol=1e-12)
    assert np.all(jacobian.numpy()[:, model.n_wall:] == 0.0)


def test_pressure_jacobian_matches_finite_differences():
    torch = _torch()
    from panel.p1_pressure_kutta_torch_driver import TorchPressureModel

    maps, state = _recovered_problem(seed=11)
    model = TorchPressureModel(maps, state.size)
    step = 1e-6
    numeric = np.zeros((maps.shape[1], state.size))
    for j in range(state.size):
        bump = np.zeros_like(state)
        bump[j] = step
        numeric[:, j] = (model(torch.as_tensor(state + bump)).numpy()
                         - model(torch.as_tensor(state - bump)).numpy()) / (2 * step)
    analytic = model.jacobian(torch.as_tensor(state)).numpy()
    np.testing.assert_allclose(analytic, numeric, rtol=0, atol=1e-7)


@pytest.mark.skipif(os.environ.get("PANEL_TORCH_PARITY") != "1",
                    reason="torch run leaves LAPACK unusable; set PANEL_TORCH_PARITY=1")
def test_frozen_case_matches_the_numpy_structured_result():
    torch = _torch()
    from panel.p1_pressure_kutta_torch_driver import solve_pressure_kutta_torch
    from tools.issue92.s3_pressure_parity import FrozenFacePressure

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
                                     pressure_model=FrozenFacePressure(grad, pairs, B.shape[1]), x0=x0,
                                     weight=weight, max_nfev=max_nfev)
    assert int(out.result.status) == status
    assert int(out.result.nfev) == nfev
    assert str(out.result.message) == message
    np.testing.assert_allclose(out.state.numpy(), expected_state, rtol=0,
                               atol=1e-10 * max(1.0, np.abs(expected_state).max()))


def test_rejects_inconsistent_inputs():
    torch = _torch()
    from panel.p1_pressure_kutta_torch_driver import TorchPressureModel, solve_pressure_kutta_torch

    maps, state = _recovered_problem()
    with pytest.raises(ValueError):
        TorchPressureModel((maps[0][:0], maps[1][:0]), state.size)
    with pytest.raises(ValueError):
        TorchPressureModel((maps[0], maps[1][:-1]), state.size)
    with pytest.raises(ValueError):
        TorchPressureModel(maps, maps.shape[-1] - 1)
    kutta = np.zeros((2, state.size))
    with pytest.raises(ValueError):
        solve_pressure_kutta_torch(wall_operator=np.zeros((5, state.size)), wall_rhs=np.zeros(4),
                                   kutta_operator=kutta, pressure_model=TorchPressureModel(maps, state.size),
                                   x0=state)
