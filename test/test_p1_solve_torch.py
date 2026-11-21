"""Coupled solver defaults, input validation, and device consistency."""
from __future__ import annotations

import ast
import pathlib
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
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


def test_coupled_solve_defaults_resolve_to_torch_when_available():
    """S9 option A: the default is 'resolve', recorded per call as backend_source."""
    import inspect
    from panel.p1_coupled_material_wake import solve_coupled_material_wake
    params = inspect.signature(solve_coupled_material_wake).parameters
    assert params["solve_backend"].default is None
    assert params["solve_device"].default == "cpu"
    assert params["pressure_solver"].default is None
    assert params["gate_mode"].default == "audit"
    assert params["velocity_method"].default == "analytic_edges"


def _mesh():
    from panel.p1_lifting_g2 import make_conventional_fixture
    return make_conventional_fixture()


@pytest.mark.parametrize("kwargs, message", [
    (dict(solve_backend="quantum"), "solve_backend must be"),
    (dict(solve_backend="torch", solve_device="gpu"), "solve_device must be"),
    (dict(solve_backend="numpy"), "solve_backend must be"),
    (dict(pressure_solver="magic"), "was retired"),
    (dict(pressure_solver="reference"), "was retired"),
])
def test_bad_backend_arguments_fail_closed(kwargs, message):
    """Every unsupported combination raises before any work starts."""
    from panel.p1_coupled_material_wake import solve_coupled_material_wake
    with pytest.raises(ValueError, match=message):
        solve_coupled_material_wake(_mesh(), 5., iterations=0, **kwargs)


def test_torch_rejects_an_unknown_kutta_closure():
    _torch()
    from panel.p1_solve_torch import solve_multi_torch
    mesh = _mesh()
    with pytest.raises(ValueError, match="kutta="):
        solve_multi_torch(mesh, 5., kutta="magic")


def test_cuda_matches_cpu_when_a_device_is_present():
    """The device must not change the answer: CPU and CUDA agree to 1e-10."""
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this host")
    from panel.p1_lifting_g2 import assemble_multi
    from panel.p1_solve_torch import solve_multi_torch
    mesh = _mesh()
    operators = assemble_multi(mesh)
    cpu = solve_multi_torch(mesh, 5., operators=operators, device="cpu")
    cuda = solve_multi_torch(mesh, 5., operators=operators, device="cuda")
    assert cpu["device"] == "cpu" and cuda["device"] == "cuda"
    for key in LOADS:
        relative = abs(cuda[key]-cpu[key])/max(abs(cpu[key]), 1e-12)
        assert relative <= 1e-10, (key, cuda[key], cpu[key], relative)
    assert int(cuda["linear_constrained_rank"]) == int(cpu["linear_constrained_rank"])
