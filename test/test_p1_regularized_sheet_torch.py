"""Tests for the torch regularized-sheet evaluator (Route B, slice S7)."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from panel.p1_regularized_sheet_velocity import RegularizedP1SheetField

TORCH = pytest.importorskip("torch", reason="torch backend tests need a torch build")

from panel.p1_regularized_sheet_velocity_torch import (  # noqa: E402
    CHUNK_BYTES, TorchRegularizedP1SheetField)

MODULE = Path(__file__).resolve().parents[1]/"panel"/"p1_regularized_sheet_velocity_torch.py"


def _sheet():
    """A 4x4 node patch split into 18 triangles with a smooth nodal strength."""
    xs = np.linspace(0.0, 0.4, 4)
    ys = np.linspace(-0.2, 0.2, 4)
    xyz = np.array([[x, y, 0.0] for x in xs for y in ys])
    faces = []
    for i in range(3):
        for j in range(3):
            a = i*4 + j
            faces.append([a, a + 1, a + 5])
            faces.append([a, a + 5, a + 4])
    faces = np.asarray(faces, dtype=int)
    values = 0.01 + 0.2*xyz[:, 0] - 0.05*xyz[:, 1]
    return [(xyz, faces, values[faces])]


def _points():
    return np.array([[0.13, -0.05, 0.02], [0.24, 0.07, -0.03],
                     [0.05, 0.11, 0.004], [0.37, -0.15, 0.05]])


@pytest.mark.parametrize("quadrature,keywords", [
    ("legacy_duffy", dict(quadrature_order=12)),
    ("projection_fan", dict(quadrature_order=48, fan_order=8, fan_strips=1)),
])
def test_torch_field_matches_numpy_field(quadrature, keywords):
    base = RegularizedP1SheetField(_sheet(), freestream=(0.99, 0.0, 0.1), core_radius=0.01,
                                   area_quadrature=quadrature, **keywords)
    ported = TorchRegularizedP1SheetField(_sheet(), freestream=(0.99, 0.0, 0.1),
                                          core_radius=0.01, area_quadrature=quadrature,
                                          **keywords)
    points = _points()
    reference = base.evaluate_many(points, chunk_size=2)
    value = ported.evaluate_many(points, chunk_size=2)
    scale = np.abs(reference).max()
    assert np.abs(value - reference).max()/scale < 1e-12
    single = np.stack([ported(point) for point in points])
    assert np.abs(single - reference).max()/scale < 1e-12


def test_batch_size_does_not_change_the_answer():
    ported = TorchRegularizedP1SheetField(_sheet(), freestream=(1.0, 0.0, 0.0),
                                          core_radius=0.01, quadrature_order=12)
    points = _points()
    one = ported.velocity(points, chunk=1)
    many = ported.velocity(points, chunk=16)
    assert np.abs(one - many).max() <= 1e-15*np.abs(many).max()


def test_adaptive_quadrature_is_refused_not_silently_downgraded():
    with pytest.raises(ValueError, match="projection_fan_adaptive"):
        TorchRegularizedP1SheetField(_sheet(), freestream=(1.0, 0.0, 0.0),
                                     core_radius=0.01,
                                     area_quadrature="projection_fan_adaptive")


def test_chunk_budget_bounds_temporary_storage():
    ported = TorchRegularizedP1SheetField(_sheet(), freestream=(1.0, 0.0, 0.0),
                                          core_radius=0.01, quadrature_order=12)
    chunk = ported._chunk(12, 4096, False)
    bytes_used = chunk*len(_sheet()[0][1])*12*12*3*8
    assert 1 <= chunk <= 4096
    assert bytes_used <= CHUNK_BYTES


def test_invalid_points_are_rejected():
    ported = TorchRegularizedP1SheetField(_sheet(), freestream=(1.0, 0.0, 0.0),
                                          core_radius=0.01)
    with pytest.raises(ValueError):
        ported.velocity(np.array([[0.0, 0.0, np.nan]]))
    with pytest.raises(ValueError):
        ported.velocity(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        ported.velocity(_points(), order=2)


def test_cuda_is_refused_when_unavailable():
    if TORCH.cuda.is_available():
        pytest.skip("CUDA is available on this host")
    with pytest.raises(RuntimeError):
        TorchRegularizedP1SheetField(_sheet(), freestream=(1.0, 0.0, 0.0),
                                     core_radius=0.01, device="cuda")


def test_coupled_loop_wiring_records_and_rejects():
    from panel.p1_coupled_material_wake import TotalVelocity
    from tools.issue92.s5_loop_parity import _mesh

    mesh = _mesh("g2_conventional")
    gamma_dofs = sum(group.gamma_hat.shape[1] for group in mesh.groups)
    state = dict(mu=np.zeros(mesh.dof_face.max() + 1), gamma=np.zeros(gamma_dofs),
                 alpha_deg=5.0)
    field = TotalVelocity(mesh, state, method="rm_regularized_p1", core_radius=0.01,
                          rm_backend="torch")
    assert field.rm_backend == "torch"
    assert field.rm_device == "cpu"
    with pytest.raises(ValueError, match="rm_backend applies only"):
        TotalVelocity(mesh, state, method="analytic_edges", rm_backend="torch")
    with pytest.raises(ValueError, match="rm_device"):
        TotalVelocity(mesh, state, method="rm_regularized_p1", core_radius=0.01,
                      rm_device="cuda")


def test_module_keeps_numpy_linalg_out_of_the_evaluator():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""])
            for name in names:
                assert not name.startswith("scipy"), name
        if isinstance(node, ast.Attribute):
            if node.attr in ("linalg", "least_squares", "lstsq", "svd", "qr"):
                base = node.value
                if isinstance(base, ast.Name):
                    assert base.id not in ("np", "numpy"), node.attr
