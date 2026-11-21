"""Rank-aware Torch solves on matrices with known null spaces and solutions."""
import ast
import importlib.util
from pathlib import Path

import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")

MODULE = Path(__file__).resolve().parents[1] / "panel" / "p1_rank_aware_torch.py"


def _api():
    from panel.p1_rank_aware_torch import constrained_solve, lstsq, null_space

    return constrained_solve, lstsq, null_space


def test_module_never_imports_numpy_or_scipy():
    """A NumPy LAPACK call after torch has run kills the process (0xC0000005)."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    banned = ("numpy", "np", "scipy", "sp")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            names = [node.value.id]
        else:
            continue
        for name in names:
            assert name not in banned, (name, node.lineno)


@requires_torch
def test_null_space_is_orthonormal_and_annihilates_the_matrix():
    import torch

    _, _, null_space = _api()
    generator = torch.Generator().manual_seed(7)
    matrix = torch.randn(20, 8, dtype=torch.float64, generator=generator)
    matrix[:, 6:] = 0.0                      # two exact null directions
    basis = null_space(matrix, rcond=1e-11)
    assert basis.shape == (8, 2)
    identity = torch.eye(2, dtype=torch.float64)
    assert torch.linalg.matrix_norm(basis.mT @ basis - identity) < 1e-14
    assert torch.linalg.matrix_norm(matrix @ basis) < 1e-14


@requires_torch
def test_null_space_truncates_at_rcond():
    import torch

    _, _, null_space = _api()
    matrix = torch.diag(torch.tensor([1.0, 1e-6, 1e-13], dtype=torch.float64))
    assert null_space(matrix, rcond=1e-11).shape == (3, 1)
    assert null_space(matrix, rcond=1e-4).shape == (3, 2)


@requires_torch
def test_null_space_reports_the_truncation_margin():
    import torch

    _, _, null_space = _api()
    matrix = torch.diag(torch.tensor([1.0, 1e-6, 1e-13], dtype=torch.float64))
    _, info = null_space(matrix, rcond=1e-11, report=True)
    assert (info.rank, info.null_dim) == (2, 1)
    assert info.largest_discarded == pytest.approx(1e-13)
    assert info.smallest_kept == pytest.approx(1e-6)
    # margin and gap are reported in units of the truncation threshold / the gap.
    assert info.margin == pytest.approx(1e-6 / 1e-11)
    assert info.gap_ratio == pytest.approx(1e-6 / 1e-13)


@requires_torch
def test_lstsq_is_minimum_norm_on_a_rank_deficient_system():
    import torch

    _, lstsq, _ = _api()
    matrix = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    result = lstsq(matrix, torch.tensor([2.0, 5.0], dtype=torch.float64))
    assert result.rank == 1
    assert torch.allclose(result.x, torch.tensor([2.0, 0.0], dtype=torch.float64))
    assert result.residual == pytest.approx(5.0)


@requires_torch
def test_constrained_solve_returns_the_projected_state():
    import torch

    constrained_solve, _, _ = _api()
    wall = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    rhs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    constraints = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    state, report = constrained_solve(wall, rhs, constraints)
    assert report["null_space"]["null_dim"] == 2
    assert torch.allclose(state, torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64), atol=1e-14)


@requires_torch
@pytest.mark.parametrize("which", ["empty", "one_dim", "nan", "rcond_zero", "rcond_inf",
                                   "rhs_mismatch", "fully_constrained"])
def test_invalid_inputs_are_rejected(which):
    import torch

    constrained_solve, lstsq, null_space = _api()
    eye = torch.eye(3, dtype=torch.float64)
    calls = {
        "empty": lambda: null_space(torch.zeros(0, 3, dtype=torch.float64)),
        "one_dim": lambda: null_space(torch.zeros(3, dtype=torch.float64)),
        "nan": lambda: null_space(torch.full((3, 3), float("nan"), dtype=torch.float64)),
        "rcond_zero": lambda: null_space(eye, rcond=0.0),
        "rcond_inf": lambda: null_space(eye, rcond=float("inf")),
        "rhs_mismatch": lambda: lstsq(eye, torch.zeros(2, dtype=torch.float64)),
        "fully_constrained": lambda: constrained_solve(eye, torch.zeros(3, dtype=torch.float64), eye),
    }
    with pytest.raises(ValueError):
        calls[which]()
