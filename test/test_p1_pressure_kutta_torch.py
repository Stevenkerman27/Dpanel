"""Structured Torch pressure solver dependencies and invalid input handling."""
from __future__ import annotations

import pathlib
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

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


def test_torch_rejects_a_bad_problem():
    pytest.importorskip("torch")
    problem = _problem("base")
    with pytest.raises(ValueError):
        _solve_in_torch(dict(problem, bw=problem["bw"][:-1]))
    with pytest.raises(ValueError):
        _solve_in_torch(dict(problem, x0=problem["x0"][:-1]))
