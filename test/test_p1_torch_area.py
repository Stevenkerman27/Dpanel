"""Parity of the torch float64 area kernel against the numpy reference (issue #92).

The torch backend only replaces the quadrature kernel inside area_terms; the fan and
the sub-triangle partition stay in numpy. These tests therefore compare the assembled
per-face integrals, which is what the wake evaluator consumes.

The whole module is skipped when torch is not importable, so the default suite keeps
running in the numpy-only environment. CUDA tests are skipped without a device.
"""
from __future__ import annotations

import numpy as np
import pytest

from panel.p1_projection_duffy import area_terms, clear_caches

torch = pytest.importorskip("torch")


def _sheet(seed=7, n=6):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-1.0, 1.0, size=(n + 2, 3))
    xyz[:, 2] = 0.0
    faces = np.array([[i, i + 1, i + 2] for i in range(n)])
    return xyz[faces]


def _relative(new, ref):
    scale = np.max(np.abs(ref))
    return float(np.max(np.abs(new - ref)) / (scale if scale > 0 else 1.0))


def test_numpy_backend_is_the_default_and_bitwise_stable():
    tri, point, core = _sheet(), np.array([0.13, -0.21, 0.55]), 0.02
    ref = area_terms(tri, point, core, order=6, strips=1)
    default = area_terms(tri, point, core, order=6, strips=1)
    explicit = area_terms(tri, point, core, order=6, strips=1, backend="numpy")
    np.testing.assert_array_equal(ref, default)
    np.testing.assert_array_equal(ref, explicit)


def test_torch_cpu_matches_numpy_on_a_random_patch():
    tri, point, core = _sheet(), np.array([0.13, -0.21, 0.55]), 0.02
    ref = area_terms(tri, point, core, order=6, strips=1)
    got = area_terms(tri, point, core, order=6, strips=1, backend="torch", device="cpu")
    assert got.shape == ref.shape
    assert not isinstance(got, torch.Tensor)
    assert _relative(got, ref) < 1e-13


def test_torch_cpu_matches_numpy_with_strips():
    tri, point, core = _sheet(seed=11), np.array([-0.4, 0.2, 0.31]), 0.05
    ref = area_terms(tri, point, core, order=5, strips=3)
    got = area_terms(tri, point, core, order=5, strips=3, backend="torch", device="cpu")
    assert _relative(got, ref) < 1e-13


def test_torch_cpu_matches_numpy_on_a_sliver_near_the_target():
    # A 200:1 sliver whose target sits one core radius off the surface: exactly the
    # configuration where the fan exists at all.
    base = np.array([0.0, 0.0, 0.0])
    long_edge = np.array([1.0, 0.0, 0.0])
    thin = np.array([0.0, 5e-3, 0.0])
    tri = np.stack([base, base + long_edge, base + thin])[None, :, :]
    point = np.array([0.4, 2.5e-3, 0.02])
    ref = area_terms(tri, point, 0.01, order=8, strips=2)
    got = area_terms(tri, point, 0.01, order=8, strips=2, backend="torch", device="cpu")
    assert _relative(got, ref) < 1e-13


@pytest.mark.parametrize("point", [
    [0.0, 0.0, 0.4],      # projects onto vertex 0
    [0.5, 0.0, 0.4],      # projects onto an edge
    [0.3, 0.2, 0.4],      # projects into the interior
])
def test_torch_cpu_matches_numpy_for_every_projection_region(point):
    tri = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    p = np.asarray(point, float)
    ref = area_terms(tri, p, 0.05, order=6, strips=1)
    got = area_terms(tri, p, 0.05, order=6, strips=1, backend="torch", device="cpu")
    assert _relative(got, ref) < 1e-13


def test_sub_areas_are_identical_across_backends():
    tri, point, core = _sheet(seed=3), np.array([0.0, 0.0, 0.3]), 0.02
    ref, ref_sub, ref_counts = area_terms(tri, point, core, order=4, strips=2,
                                          return_sub_areas=True)
    got, got_sub, got_counts = area_terms(tri, point, core, order=4, strips=2,
                                          return_sub_areas=True, backend="torch")
    assert len(ref_sub) == len(got_sub)
    assert np.array_equal(ref_counts, got_counts)
    for (sub_a, own_a, a2_a), (sub_b, own_b, a2_b) in zip(ref_sub, got_sub):
        np.testing.assert_array_equal(sub_a, sub_b)
        np.testing.assert_array_equal(own_a, own_b)
        np.testing.assert_array_equal(a2_a, a2_b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available")
def test_torch_cuda_matches_numpy():
    tri, point, core = _sheet(seed=5), np.array([0.05, 0.1, 0.4]), 0.02
    ref = area_terms(tri, point, core, order=8, strips=2)
    got = area_terms(tri, point, core, order=8, strips=2, backend="torch", device="cuda")
    assert _relative(got, ref) < 1e-13


def test_unknown_backend_is_rejected():
    tri, point = _sheet(), np.array([0.0, 0.0, 0.3])
    with pytest.raises(ValueError):
        area_terms(tri, point, 0.01, backend="cuda_kernel")


def test_cuda_request_without_a_device_raises_instead_of_falling_back(monkeypatch):
    tri, point = _sheet(), np.array([0.0, 0.0, 0.3])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        area_terms(tri, point, 0.01, backend="torch", device="cuda")


@pytest.mark.parametrize("device", [3, None, "gpu", torch.device("meta")])
def test_invalid_devices_are_rejected(device):
    tri, point = _sheet(), np.array([0.0, 0.0, 0.3])
    with pytest.raises((ValueError, RuntimeError)):
        area_terms(tri, point, 0.01, backend="torch", device=device)


def test_cache_is_not_keyed_on_the_backend_or_the_target():
    tri = _sheet()
    a = area_terms(tri, np.array([0.0, 0.0, 0.3]), 0.02, order=6, node_chunk=3)
    b = area_terms(tri, np.array([0.4, -0.2, 0.1]), 0.02, order=6, node_chunk=3,
                   backend="torch")
    c = area_terms(tri, np.array([0.4, -0.2, 0.1]), 0.02, order=6, node_chunk=3,
                   backend="torch", device="cpu")
    np.testing.assert_array_equal(b, c)
    assert not np.array_equal(a, c)
    clear_caches()
