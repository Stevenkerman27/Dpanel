"""Tests for the torch assembly operators (issue #92, S4).

The static guard runs everywhere. The numerical tests need torch and skip without it.
Every NumPy reference is computed *before* the first torch kernel runs in that test:
a NumPy LAPACK call after torch has executed dies with 0xC0000005 (see
docs/ISSUE92_PHASE1_WAKE_EVALUATOR.md 6f).
"""
import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")

MODULE = Path(__file__).resolve().parents[1] / "panel" / "p1_assembly_torch.py"


def _api():
    from panel.p1_assembly_torch import (doublet_matrix, gradient_map, loads, mesh_kernel,
                                         wake_operator)

    return doublet_matrix, gradient_map, loads, mesh_kernel, wake_operator


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


def _two_triangles():
    triangles = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                          [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]]])
    targets = np.random.default_rng(7).normal(size=(13, 3))
    return triangles, targets


@requires_torch
def test_doublet_matrix_matches_the_numpy_kernel():
    from panel.p1_lifting_g1 import _doublet_matrix

    triangles, targets = _two_triangles()
    reference = _doublet_matrix(triangles, targets)
    doublet_matrix, _, _, _, _ = _api()
    got = doublet_matrix(triangles, targets).numpy()
    assert got.shape == reference.shape
    assert np.max(np.abs(got - reference)) < 1e-15


@requires_torch
def test_self_face_rows_use_the_principal_value_limit():
    triangles, _ = _two_triangles()
    centers = triangles.mean(axis=1)
    doublet_matrix, _, _, _, _ = _api()
    got = doublet_matrix(triangles, centers, self_faces=np.arange(len(triangles)))
    for j, triangle in enumerate(triangles):
        e1, e2 = triangle[1] - triangle[0], triangle[2] - triangle[0]
        raw = np.cross(e1, e2)
        scale = np.linalg.norm(raw)
        normal, g1 = raw / scale, np.cross(e2, raw / scale) / scale
        g2 = np.cross(raw / scale, e1) / scale
        basis = np.stack((-g1 - g2, g1, g2))
        offset = centers[j] - triangle.mean(axis=0)
        expected = -0.5 * (1.0 / 3.0 + offset @ basis.T)
        assert np.allclose(got[j].numpy()[3 * j:3 * j + 3], expected, atol=1e-15)


@requires_torch
def test_scatter_shares_dofs_between_faces_like_the_numpy_kernel():
    from panel.p1_lifting_g1 import _doublet_matrix

    triangles, targets = _two_triangles()
    face_dofs = np.array([[0, 1, 2], [1, 2, 3]])
    reference = _doublet_matrix(triangles, targets, face_dofs=face_dofs, ndof=4)
    doublet_matrix, _, _, _, _ = _api()
    got = doublet_matrix(triangles, targets, face_dofs=face_dofs, ndof=4).numpy()
    assert np.max(np.abs(got - reference)) < 1e-15


@requires_torch
def test_gradient_map_is_an_exact_scatter():
    triangles, _ = _two_triangles()
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    _, gradient_map, _, mesh_kernel, _ = _api()
    kernel = mesh_kernel(triangles.reshape(-1, 3), faces)
    basis = kernel["basis_gradients"].numpy()
    dof_face = np.array([[0, 1, 2], [2, 3, 4]])
    expected = np.zeros((2, 3, 5))
    for face in range(2):
        for k in range(3):
            expected[face, :, dof_face[face, k]] += basis[face, k]
    got = gradient_map(faces, dof_face, basis).numpy()
    assert np.array_equal(got, expected)


@requires_torch
def test_wake_operator_per_face_matches_the_g1_loop():
    from panel.p1_lifting_g1 import _doublet_matrix

    wake = np.array([[[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0]],
                     [[1.0, 0.0, -1.0], [2.0, 0.0, -1.0], [1.0, 1.0, -1.0]]])
    gamma = np.random.default_rng(3).normal(size=(2, 3, 4))
    targets = np.random.default_rng(4).normal(size=(9, 3))
    targets[:, 2] = 0.5
    expected = np.zeros((9, 4))
    for j in range(2):
        single = _doublet_matrix(wake[j][None], targets, face_dofs=np.arange(3)[None], ndof=3)
        expected += single @ gamma[j]
    _, _, _, _, wake_operator = _api()
    got = wake_operator(wake, targets, gamma.reshape(-1, 4), mode="per_face").numpy()
    assert np.max(np.abs(got - expected)) < 1e-15


@requires_torch
def test_wake_operator_nodal_matches_the_g2_contraction():
    from panel.p1_lifting_g1 import _doublet_matrix

    wake = np.array([[[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0]],
                     [[1.0, 0.0, -1.0], [2.0, 0.0, -1.0], [1.0, 1.0, -1.0]]])
    wake_faces = np.array([[0, 1, 2], [1, 3, 4]])
    gamma = np.random.default_rng(6).normal(size=(5, 2))
    targets = np.random.default_rng(8).normal(size=(7, 3))
    coefficients = _doublet_matrix(wake, targets, face_dofs=wake_faces, ndof=5)
    expected = coefficients @ gamma
    _, _, _, _, wake_operator = _api()
    got = wake_operator(wake, targets, gamma, mode="nodal", face_dofs=wake_faces,
                        ndof=5).numpy()
    assert np.max(np.abs(got - expected)) < 1e-15


@requires_torch
def test_loads_match_the_shared_numpy_map_on_a_small_wing():
    import panel.p1_lifting_g1 as g1
    from tools.issue92.s2_loads import loads_single

    wing = g1.make_wing(chord_panels=3, span_panels=4, thickness=0.12, camber=0.02)
    ops = g1.assemble(wing)
    nm = ops["A"].shape[1]
    state = np.random.default_rng(11).normal(size=nm + ops["W"].shape[1]) * 0.01
    expected = loads_single(ops, wing, 5.0, state)
    import torch

    _, _, loads, _, _ = _api()
    mesh = ops["mesh"]
    got = loads(grad=torch.from_numpy(ops["grad"]),
                mu=torch.from_numpy(state[:nm]),
                areas=torch.from_numpy(mesh.areas),
                normals=torch.from_numpy(mesh.normals),
                centers=torch.from_numpy(mesh.centers),
                reference_point=torch.from_numpy(np.asarray(wing.reference_point, float)),
                reference_area=float(wing.reference_area),
                reference_chord=float(wing.chord),
                alpha_deg=5.0,
                group_gamma=[(np.r_[0.0, state[nm:], 0.0],
                              np.asarray(wing.gamma_y, float))])
    for key in ("CL", "CL_gamma", "CD_pressure", "CM_y"):
        assert abs(got[key] - expected[key]) <= 1e-13 * max(1.0, abs(expected[key]))


@requires_torch
def test_degenerate_sources_and_bad_arguments_are_rejected():
    doublet_matrix, gradient_map, _, mesh_kernel, wake_operator = _api()
    flat = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    targets = np.array([[0.0, 0.0, 1.0]])
    with pytest.raises((ValueError, FloatingPointError)):
        doublet_matrix(flat, targets)
    with pytest.raises(ValueError):
        mesh_kernel(np.zeros((3, 3)), np.array([[0, 1, 2]]))
    with pytest.raises(ValueError):
        doublet_matrix(targets[None], targets)
    with pytest.raises(ValueError):
        wake_operator(flat, targets, np.zeros((3, 1)), mode="nope")
    with pytest.raises(ValueError):
        doublet_matrix(flat * 0 + np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
                       targets, face_chunk=0)
    with pytest.raises(ValueError):
        gradient_map(np.array([[0, 1, 2]]), np.array([[0, 1, 2]]), np.zeros((1, 2, 3)))
