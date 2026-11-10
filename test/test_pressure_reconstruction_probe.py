"""Offline checks for the issue #9 three-arm reconstruction comparison."""
import numpy as np
import pytest

from panel.pressure_reconstruction_probe import (
    face_frames, two_ring_neighbors, fit_gradients, cp, tangential_gradient,
)


def plate(n=9, jitter=True):
    """Jittered plate split down the middle column; the seam is returned as cuts."""
    yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, n), np.linspace(-1.0, 1.0, n))
    points = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(n * n)))
    j0 = n // 2
    if jitter:
        inner = ((np.abs(points[:, 0]) < 1.0) & (np.abs(points[:, 1]) < 1.0)
                 & (np.abs(np.arange(n * n) % n - j0) > 0))
        rng = np.random.default_rng(19)
        points[inner, :2] += rng.uniform(-0.03, 0.03, (inner.sum(), 2))
    faces = []
    columns = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            columns.extend((j, j))
            faces.extend(([a, a + n, a + n + 1], [a, a + n + 1, a + 1]))
    cuts = np.array([[i * n + j0, (i + 1) * n + j0] for i in range(n - 1)], dtype=np.int64)
    return points, np.array(faces), cuts, np.array(columns)


FLOW = np.array([np.cos(np.deg2rad(4.0)), 0.0, np.sin(np.deg2rad(4.0))])


def test_affine_field_is_recovered_by_every_arm():
    points, faces, cuts, _ = plate()
    centers, _, _, _ = face_frames(points, faces)
    ids = two_ring_neighbors(faces, cuts)
    mu = 0.21 * centers[:, 0] - 0.37 * centers[:, 1] + 1.4
    for mode in ("normal", "qr", "weighted"):
        gradient, ratio, rejected = fit_gradients(points, faces, mu, ids, mode=mode)
        np.testing.assert_allclose(gradient, np.tile([0.21, -0.37, 0.0], (len(faces), 1)),
                                   atol=3e-13)
        assert rejected == 0
        assert np.min(ratio) > 1e-3


def test_qr_matches_the_normal_equations_and_the_weighted_arm_does_not():
    points, faces, cuts, _ = plate()
    centers, _, _, _ = face_frames(points, faces)
    ids = two_ring_neighbors(faces, cuts)
    mu = 0.21 * centers[:, 0] - 0.37 * centers[:, 1] + 0.4 * centers[:, 1] ** 3
    normal_gradient, _, _ = fit_gradients(points, faces, mu, ids, mode="normal")
    qr_gradient, _, _ = fit_gradients(points, faces, mu, ids, mode="qr")
    weighted, _, _ = fit_gradients(points, faces, mu, ids, mode="weighted")
    np.testing.assert_allclose(qr_gradient, normal_gradient, rtol=1e-10, atol=1e-13)
    assert np.max(np.abs(weighted - normal_gradient)) > 1e-4


def test_stencils_never_cross_the_declared_cut():
    points, faces, cuts, columns = plate()
    j0 = 4
    ids = two_ring_neighbors(faces, cuts)
    left = columns < j0
    right = ~left
    for i in np.flatnonzero(left):
        assert not (set(ids[i].tolist()) & set(np.flatnonzero(right).tolist()))


def test_rank_deficient_stencil_is_counted_not_repaired():
    points, faces, _, _ = plate(jitter=False)
    centers, _, _, _ = face_frames(points, faces)
    ids = [np.array([i, i, i], dtype=np.int64) for i in range(len(faces))]
    mu = centers[:, 0].copy()
    gradient, ratio, rejected = fit_gradients(points, faces, mu, ids, mode="normal")
    assert rejected == len(faces)
    assert np.allclose(gradient, 0.0)
    assert np.all(ratio == 0.0)


def test_cp_uses_the_surface_convention_and_projects_normal_components():
    points, faces, _, _ = plate()
    centers, normal, _, _ = face_frames(points, faces)
    gradient = np.tile([0.1, -0.05, 0.0], (len(faces), 1))
    values = cp(gradient, normal, FLOW)
    assert values.shape == (len(faces),)
    offset = gradient + np.array([0.0, 0.0, 0.3])
    np.testing.assert_allclose(tangential_gradient(offset, normal), gradient, atol=1e-15)
    np.testing.assert_allclose(cp(offset, normal, FLOW), values, atol=1e-15)
    with pytest.raises(ValueError):
        cp(gradient, normal * 2.0, FLOW)


def test_stencil_depth_is_bounded_by_the_requested_rings():
    points, faces, cuts, _ = plate()
    one = two_ring_neighbors(faces, cuts, rings=1)
    two = two_ring_neighbors(faces, cuts, rings=2)
    assert all(len(b) >= len(a) for a, b in zip(one, two))
    assert min(len(a) for a in one) >= 2
    assert max(len(a) for a in two) > max(len(b) for b in one)
