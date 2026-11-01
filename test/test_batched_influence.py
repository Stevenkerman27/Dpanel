"""Parity tests for the batched P1 influence kernel (issue 92, Phase 1 step 1).

The kernel must reproduce the reference NumPy assembly, not merely approximate it: the
checks below compare the assembled A and W on a synthetic geometry and on a frozen
fixture, and confirm that the block sizes only change the traversal order.
"""
from __future__ import annotations

import numpy as np
import pytest

from panel.p1_batched_influence import influence_matrix, source_invariants, wake_matrix
from panel.p1_lifting_g1 import _doublet_matrix, _mesh_as_kernel, assemble, solve

torch = pytest.importorskip("torch")


def _synthetic():
    rng = np.random.default_rng(7)
    tri = rng.normal(size=(6, 3, 3))
    tri[:, :, 2] = tri[:, :, 2] * 0.2
    targets = rng.normal(size=(9, 3))
    targets[:, 2] = np.abs(targets[:, 2]) + 0.5
    dofs = np.arange(len(tri) * 3).reshape(-1, 3)
    return tri, targets, dofs


def test_synthetic_influence_matches_the_reference():
    tri, targets, dofs = _synthetic()
    reference = _doublet_matrix(tri, targets, face_dofs=dofs, ndof=dofs.max() + 1)
    batched = influence_matrix(tri, targets, face_dofs=dofs, ndof=dofs.max() + 1)
    assert batched.shape == reference.shape
    assert np.max(np.abs(batched - reference)) <= 1e-13 * max(1.0, np.max(np.abs(reference)))


def test_self_panel_rows_match_the_reference():
    tri, _targets, dofs = _synthetic()
    centers = tri.mean(axis=1)
    reference = _doublet_matrix(tri, centers, self_faces=np.arange(len(tri)), face_dofs=dofs,
                                ndof=dofs.max() + 1)
    batched = influence_matrix(tri, centers, self_faces=np.arange(len(tri)), face_dofs=dofs,
                               ndof=dofs.max() + 1)
    assert np.max(np.abs(batched - reference)) <= 1e-13 * max(1.0, np.max(np.abs(reference)))


def test_block_sizes_only_reorder_the_accumulation():
    tri, targets, dofs = _synthetic()
    reference = influence_matrix(tri, targets, face_dofs=dofs, ndof=dofs.max() + 1,
                                 target_block=64, source_block=64)
    for target_block, source_block in ((1, 1), (2, 3), (5, 2), (9, 6)):
        other = influence_matrix(tri, targets, face_dofs=dofs, ndof=dofs.max() + 1,
                                 target_block=target_block, source_block=source_block)
        assert np.max(np.abs(other - reference)) <= 1e-14 * max(1.0, np.max(np.abs(reference)))


def test_wake_matrix_matches_the_per_face_reference_loop():
    rng = np.random.default_rng(11)
    xyz = np.column_stack((rng.uniform(0, 2, size=8), rng.uniform(-1, 1, size=8),
                           np.zeros(8)))
    faces = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4], [3, 5, 4]])
    gamma = rng.normal(size=(len(xyz), 2))
    targets = np.column_stack((rng.uniform(0, 2, size=5), rng.uniform(-1, 1, size=5),
                              np.full(5, 0.3)))
    reference = np.zeros((len(targets), 2))
    for face in faces:
        D = _doublet_matrix(xyz[face][None], targets, face_dofs=np.arange(3)[None], ndof=3)
        reference += D @ gamma[face]
    batched = wake_matrix(xyz, faces, targets) @ gamma
    assert np.max(np.abs(batched - reference)) <= 1e-13 * max(1.0, np.max(np.abs(reference)))


def test_precomputed_source_invariants_reuse_the_same_arithmetic():
    tri, targets, dofs = _synthetic()
    src = source_invariants(tri)
    fresh = influence_matrix(tri, targets, face_dofs=dofs, ndof=dofs.max() + 1)
    reused = influence_matrix(tri, targets, face_dofs=dofs, ndof=dofs.max() + 1, src=src)
    assert np.array_equal(fresh, reused)
    with pytest.raises(ValueError, match="do not match tris"):
        influence_matrix(tri[:2], targets, face_dofs=dofs[:2], ndof=dofs.max() + 1, src=src)


def test_zero_area_source_is_rejected():
    tri = np.zeros((1, 3, 3))
    tri[0, 1, 0] = 1.0
    tri[0, 2, 0] = 2.0
    with pytest.raises(ValueError, match="Zero-area source triangle"):
        source_invariants(tri)


def test_target_on_a_source_vertex_is_rejected():
    tri, _targets, dofs = _synthetic()
    targets = tri[:1, 0]
    with pytest.raises(ValueError, match="coincides"):
        influence_matrix(tri, targets, face_dofs=dofs, ndof=dofs.max() + 1)


def test_face_dofs_are_validated():
    tri, targets, dofs = _synthetic()
    with pytest.raises(ValueError, match="face_dofs must be"):
        influence_matrix(tri, targets, face_dofs=dofs[:, :2], ndof=32)
    with pytest.raises(ValueError, match="non-negative"):
        influence_matrix(tri, targets, face_dofs=dofs - 1, ndof=32)
    with pytest.raises(ValueError, match="must index"):
        influence_matrix(tri, targets, face_dofs=dofs, ndof=4)
    with pytest.raises(ValueError, match="positive"):
        influence_matrix(tri, targets, face_dofs=dofs, target_block=0)


def _fixture_wing(name="g1_wing_rect"):
    from tools.issue92 import fixtures, profile_production

    if not fixtures.available(name):
        pytest.skip("frozen fixture not present")
    entry = fixtures.load_manifest()["fixtures"][name]
    return entry, profile_production._wing_object(name, entry)


def test_production_assemble_and_solve_accept_the_torch_backend():
    entry, wing = _fixture_wing()
    reference = assemble(wing)
    other = assemble(wing, backend="torch")
    for key in ("A", "W"):
        scale = max(1.0, float(np.max(np.abs(reference[key]))))
        assert float(np.max(np.abs(other[key] - reference[key]))) <= 1e-13 * scale, key
    base = solve(wing, entry["meta"]["alpha_deg"], kutta=entry["meta"]["kutta"])
    fast = solve(wing, entry["meta"]["alpha_deg"], kutta=entry["meta"]["kutta"], backend="torch")
    for key in ("CL", "CL_gamma", "CD_pressure", "CM_y"):
        assert abs(float(fast[key]) - float(base[key])) <= 1e-10 * max(1.0, abs(float(base[key]))), key
    with pytest.raises(ValueError, match="backend must be"):
        assemble(wing, backend="cuda-fast")


def test_production_multi_assembly_and_solve_accept_the_torch_backend():
    from panel.p1_lifting_g2 import assemble_multi, solve_multi
    from tools.issue92 import fixtures, profile_production

    if not fixtures.available("g2_conventional"):
        pytest.skip("frozen fixture not present")
    entry = fixtures.load_manifest()["fixtures"]["g2_conventional"]
    mesh = profile_production._multi_object(entry)
    reference = assemble_multi(mesh)
    other = assemble_multi(mesh, backend="torch")
    for key in ("A", "W"):
        scale = max(1.0, float(np.max(np.abs(reference[key]))))
        assert float(np.max(np.abs(other[key] - reference[key]))) <= 1e-13 * scale, key
    alpha, kutta = entry["meta"]["alpha_deg"], entry["meta"]["kutta"]
    base = solve_multi(mesh, alpha, kutta=kutta)
    fast = solve_multi(mesh, alpha, kutta=kutta, operators=other)
    for key in ("CL", "CL_gamma", "CD_pressure", "CM_y"):
        assert abs(float(fast[key]) - float(base[key])) <= 1e-10 * max(1.0, abs(float(base[key]))), key


def test_frozen_fixture_assembly_and_loads_agree_with_the_reference():
    from tools.issue92 import fixtures, profile_production

    if not fixtures.available("g1_wing_rect"):
        pytest.skip("frozen fixture not present")
    manifest = fixtures.load_manifest()
    entry = manifest["fixtures"]["g1_wing_rect"]
    wing = profile_production._wing_object("g1_wing_rect", entry)
    mesh = _mesh_as_kernel(wing)
    ops = assemble(wing)

    batched_A = influence_matrix(mesh.triangles, mesh.centers,
                                 self_faces=np.arange(len(wing.faces)),
                                 face_dofs=np.asarray(wing.dof_face), ndof=int(wing.ndof_mu))
    batched_W = wake_matrix(wing.wake_xyz, wing.wake_faces, mesh.centers) @ np.asarray(wing.gamma_hat)
    for name, reference, batched in (("A", ops["A"], batched_A), ("W", ops["W"], batched_W)):
        scale = max(1.0, float(np.max(np.abs(reference))))
        assert float(np.max(np.abs(batched - reference))) <= 1e-13 * scale, name

    patched = dict(ops)
    patched["A"] = batched_A
    patched["W"] = batched_W
    import panel.p1_lifting_g1 as lifting
    original = lifting.assemble
    try:
        lifting.assemble = lambda w, **kw: patched
        result = solve(wing, 5.0, kutta="signed_speed")
    finally:
        lifting.assemble = original
    reference = solve(wing, 5.0, kutta="signed_speed")
    for key in ("CL", "CL_gamma", "CD_pressure", "CM_y"):
        assert abs(float(result[key]) - float(reference[key])) <= 1e-10 * max(1.0, abs(float(reference[key]))), key