"""S1 parity tests: torch pressure-Kutta against the frozen NumPy reference.

The reference values in docs/issue92/s1_reference_<scenario>.npz were produced by
the NumPy/SciPy backend in a separate process (tools/issue92/s1_parity.py numpy
stage), because torch and NumPy LAPACK cannot share an interpreter: once torch
runs an op, the next LAPACK call dies with 0xC0000005
(docs/archive/issue/ISSUE92_PHASE1_WAKE_EVALUATOR.md section 6f).

Running these tests therefore needs two conditions:

* a torch interpreter (the repository's default .conda_env has no torch, so the
  tests skip there);
* PANEL_TORCH_PARITY=1 in the environment, because a passing run still leaves the
  interpreter in a state where later NumPy LAPACK calls can crash - the tests
  only load frozen .npz files, but the rest of the suite would run afterwards.

    PANEL_TORCH_PARITY=1 python -m pytest test/test_p1_pressure_kutta_torch.py

The static test (no NumPy/SciPy in the torch module) runs everywhere.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
REFERENCE = REPO / "docs" / "issue92"
SCENARIOS = ("base", "hard", "backtrack")

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_torch_module_has_no_numpy_or_scipy():
    """The torch path must not reach back into NumPy LAPACK or SciPy.

    Checked on the syntax tree, not the text: the module docstring lists the
    NumPy/SciPy calls it replaces, and those names must stay allowed there.
    """
    import ast

    source = (REPO / "panel" / "p1_pressure_kutta_torch.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned_modules = ("numpy", "scipy")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in banned_modules,                     "torch module imports %s" % alias.name
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            assert root not in banned_modules, "torch module imports from %s" % node.module
        elif isinstance(node, ast.Attribute):
            chain = []
            current = node
            while isinstance(current, ast.Attribute):
                chain.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                chain.append(current.id)
            root = chain[-1]
            assert root not in ("np", "numpy", "scipy"),                 "torch module calls %s" % ".".join(reversed(chain))


def _problem(scenario):
    """Build the shared problem before torch is allowed to run any op."""
    from tools.issue92.s1_problem import build

    return build(scenario=scenario)


def _reference(scenario):
    path = REFERENCE / ("s1_reference_%s.npz" % scenario)
    if not path.exists():
        pytest.skip("reference %s missing" % path.name)
    return np.load(path, allow_pickle=False)


def _solve_in_torch(problem, device="cpu"):
    import torch

    from panel.p1_pressure_kutta_torch import solve_structured_pressure_torch

    tensors = {k: torch.as_tensor(problem[k], dtype=torch.float64, device=device)
               for k in ("B", "N", "bw", "K", "target")}
    n_mu = problem["n_mu"]
    n_pressure = int(problem["K"].shape[0])
    K, c2, target = tensors["K"], problem["c2"], tensors["target"]

    def pressure(v):
        mu = v[:n_mu]
        return K @ mu + c2 * mu[:n_pressure] ** 2 - target

    def jacobian(v):
        mu = v[:n_mu]
        block = K.clone()
        idx = torch.arange(n_pressure, device=mu.device)
        block[idx, idx] = block[idx, idx] + 2.0 * c2 * mu[:n_pressure]
        P = torch.zeros((n_pressure, v.numel()), dtype=mu.dtype, device=mu.device)
        P[:, :n_mu] = block
        return P

    return solve_structured_pressure_torch(
        wall_operator=tensors["B"], reduction=tensors["N"], wall_rhs=tensors["bw"],
        pressure=pressure, pressure_jacobian=jacobian,
        x0=torch.as_tensor(problem["x0"], dtype=torch.float64, device=device),
        weight=5., device=device)


@pytest.mark.skipif(not os.environ.get("PANEL_TORCH_PARITY"), reason="set PANEL_TORCH_PARITY=1")
@pytest.mark.skipif(not pytest.importorskip("importlib").util.find_spec("torch"),
                    reason="torch is not installed")
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_torch_matches_frozen_reference(scenario):
    problem = _problem(scenario)
    reference = _reference(scenario)
    assert str(reference["scenario"]) == scenario

    import torch

    result = _solve_in_torch(problem)
    x_ref = torch.as_tensor(reference["x"], dtype=torch.float64)
    difference = torch.linalg.vector_norm(result.x - x_ref)
    relative = float(difference / torch.linalg.vector_norm(x_ref))

    assert result.status == int(reference["status"])
    assert result.nfev == int(reference["nfev"])
    assert result.nit == int(reference["nit"])
    assert result.backtracks == int(reference["backtracks"])
    assert result.message == str(reference["message"])
    assert relative <= 1e-11, "relative iterate difference %.3e" % relative
    cost_relative = abs(result.cost - float(reference["cost"])) / abs(float(reference["cost"]))
    assert cost_relative <= 1e-11, "relative cost difference %.3e" % cost_relative


@pytest.mark.skipif(not os.environ.get("PANEL_TORCH_PARITY"), reason="set PANEL_TORCH_PARITY=1")
def test_torch_rejects_a_bad_problem():
    problem = _problem("base")
    with pytest.raises(ValueError):
        _solve_in_torch(dict(problem, bw=problem["bw"][:-1]))
    with pytest.raises(ValueError):
        _solve_in_torch(dict(problem, x0=problem["x0"][:-1]))
