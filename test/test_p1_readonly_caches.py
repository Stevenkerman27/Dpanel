"""Read-only caches added for issue #84 P1 (nodes, per-triangle geometry)."""
from __future__ import annotations

import numpy as np
import pytest

from panel import p1_projection_duffy as duffy
from panel.p1_regularized_sheet_velocity import RegularizedP1SheetField


def tiny_sheet():
    xyz = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    values = np.array([[0.5, 1.0, 1.5], [1.0, 1.5, 2.0]])
    return xyz, faces, values


def field(**kw):
    xyz, faces, values = tiny_sheet()
    return RegularizedP1SheetField([(xyz, faces, values)], core_radius=0.01,
                                  area_quadrature="projection_fan", fan_order=8,
                                  fan_strips=2, **kw)


def test_duffy_nodes_cached_and_read_only():
    duffy.clear_caches()
    first = duffy._duffy_nodes(6)
    hit = duffy._duffy_nodes(6)
    assert all(a.flags.writeable is False for a in hit)
    for a, b in zip(first, hit):
        np.testing.assert_array_equal(a, b)
        assert a is b
    assert duffy.cache_info()["duffy_nodes"]["hits"] >= 1
    other = duffy._duffy_nodes(8)
    assert other[0].shape == (64,)
    assert not np.array_equal(other[0], first[0])
    duffy.clear_caches()
    assert duffy.cache_info()["duffy_nodes"]["currsize"] == 0


def test_duffy_nodes_rejects_bad_order():
    for bad in (1, 0, -3, 2.0, "4"):
        with pytest.raises(ValueError):
            duffy._duffy_nodes(bad)


def test_area_terms_value_unchanged_by_cache():
    duffy.clear_caches()
    xyz, faces, values = tiny_sheet()
    tri = xyz[faces]
    point = np.array([0.4, 0.3, 0.05])
    cold = duffy.area_terms(tri, point, 0.01, order=8, strips=2)
    warm = duffy.area_terms(tri, point, 0.01, order=8, strips=2)
    np.testing.assert_array_equal(cold, warm)


def test_field_geometry_is_read_only_and_matches_direct_computation():
    f = field()
    tri = f.triangles
    edges = np.linalg.norm(np.stack((tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 1],
                                     tri[:, 0]-tri[:, 2]), axis=1), axis=2)
    np.testing.assert_allclose(f.triangle_edge_lengths, edges)
    np.testing.assert_allclose(f.triangle_centroid, tri.mean(axis=1))
    np.testing.assert_allclose(f.triangle_longest, edges.max(axis=1))
    np.testing.assert_allclose(f.triangle_shortest, edges.min(axis=1))
    area2 = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
    np.testing.assert_allclose(f.triangle_circumradius,
                               np.prod(edges, axis=1)/(2.0*area2))
    for name in ("triangle_edge_lengths", "triangle_centroid", "triangle_longest",
                 "triangle_shortest", "triangle_aspect", "triangle_circumradius"):
        assert getattr(f, name).flags.writeable is False, name


def test_field_geometry_does_not_alter_the_velocity():
    xyz, faces, values = tiny_sheet()
    f = field()
    point = np.array([0.4, 0.3, 0.2])
    direct = duffy.field_velocity(f, point, order=8, strips=2)
    np.testing.assert_allclose(f(point, order=8), direct, rtol=0, atol=0)
    assert np.isfinite(f(point, order=8)).all()
