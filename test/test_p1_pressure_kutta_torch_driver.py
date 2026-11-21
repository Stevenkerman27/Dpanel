"""Pressure model equations, finite-difference derivatives, and input validation."""
from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
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
