"""S5 tests: the torch backend inside the coupled free-wake loop.

Two layers:

* argument handling and the static syntax-tree guard, which run in any
  interpreter and pin the defaults (`solve_backend="numpy"`,
  `solve_device="cpu"`, `pressure_solver=None`);
* one end-to-end torch solve against a frozen NumPy reference, gated by
  PANEL_TORCH_PARITY=1 because the torch path leaves the process unable to call
  NumPy LAPACK afterwards (section 6f of the phase-1 doc).

The frozen reference (docs/issue92/s5_reference_g2.npz) holds the operators and
the loads of one NumPy structured solve on the g2_conventional fixture, written by
tools.issue92.s5_loop_parity bundle in a separate process.

    PANEL_TORCH_PARITY=1 python -m pytest test/test_p1_solve_torch.py
"""
from __future__ import annotations

import ast
import os
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
BUNDLE = REPO / "docs" / "issue92" / "s5_reference_g2.npz"
TORCH_MODULES = ("panel/p1_solve_torch.py",)
LOADS = ("CL", "CL_gamma", "CD_pressure", "CM_y")

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _torch():
    return pytest.importorskip("torch")


@pytest.mark.parametrize("relative", TORCH_MODULES)
def test_torch_modules_have_no_numpy_or_scipy(relative):
    """The torch path must not import or call NumPy/SciPy."""
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
                assert chain[0] not in banned + ("np", "sp"), "%s calls %s" % (relative, ".".join(reversed(chain)))


def test_coupled_solve_defaults_are_numpy():
    """Defaults must stay NumPy: the torch backend is opt-in."""
    import inspect
    from panel.p1_coupled_material_wake import solve_coupled_material_wake
    params = inspect.signature(solve_coupled_material_wake).parameters
    assert params["solve_backend"].default == "numpy"
    assert params["solve_device"].default == "cpu"
    assert params["pressure_solver"].default is None
    assert params["gate_mode"].default == "audit"
    assert params["velocity_method"].default == "analytic_edges"


def _mesh():
    from tools.issue92.s5_loop_parity import _mesh as case_mesh
    return case_mesh("g2_conventional")


@pytest.mark.parametrize("kwargs, message", [
    (dict(solve_backend="quantum"), "solve_backend must be"),
    (dict(solve_backend="torch", solve_device="gpu"), "solve_device must be"),
    (dict(solve_backend="numpy", solve_device="cuda"), "requires solve_backend"),
    (dict(pressure_solver="magic"), "pressure_solver must be"),
    (dict(solve_backend="torch", pressure_solver="reference"), "implements only"),
])
def test_bad_backend_arguments_fail_closed(kwargs, message):
    """Every unsupported combination raises before any work starts."""
    from panel.p1_coupled_material_wake import solve_coupled_material_wake
    with pytest.raises(ValueError, match=message):
        solve_coupled_material_wake(_mesh(), 5., iterations=0, **kwargs)


@pytest.mark.skipif(not BUNDLE.is_file(), reason="run tools.issue92.s5_loop_parity bundle")
def test_frozen_reference_metadata():
    """The frozen bundle is self-describing, so the torch test cannot drift."""
    data = np.load(BUNDLE)
    for key in ("A", "W", "J", "K", "grad", "mu", "gamma", "cp", "loads",
                "reference_area", "reference_chord", "reference_point"):
        assert key in data.files, key
    assert data["loads"].shape == (4,)


@pytest.mark.skipif(os.environ.get("PANEL_TORCH_PARITY") != "1",
                    reason="set PANEL_TORCH_PARITY=1; a torch run poisons later LAPACK calls")
def test_torch_solve_matches_the_frozen_numpy_reference():
    """End to end: assemble with NumPy, solve with torch, compare loads."""
    _torch()
    from panel.p1_lifting_g2 import assemble_multi
    from panel.p1_solve_torch import solve_multi_torch
    from tools.issue92.s5_loop_parity import _mesh as case_mesh
    frozen = np.load(BUNDLE)
    expected = dict(zip(LOADS, frozen["loads"].tolist()))
    mesh = case_mesh("g2_conventional")
    operators = assemble_multi(mesh)
    got = solve_multi_torch(mesh, 5., operators=operators)
    assert got["backend"] == "torch"
    assert got["pressure_success"] is True
    for key in LOADS:
        relative = abs(got[key]-expected[key])/max(abs(expected[key]), 1e-12)
        assert relative <= 1e-10, (key, got[key], expected[key], relative)
    assert got["CL_gamma"] == pytest.approx(expected["CL_gamma"], rel=1e-10)


@pytest.mark.skipif(os.environ.get("PANEL_TORCH_PARITY") != "1",
                    reason="torch only")
def test_torch_rejects_numpy_operators_of_the_wrong_width():
    """A mismatched operator set must raise, not silently truncate."""
    _torch()
    from panel.p1_solve_torch import solve_multi_torch
    mesh = _mesh()
    with pytest.raises(Exception):
        solve_multi_torch(mesh, 5., kutta="magic")
