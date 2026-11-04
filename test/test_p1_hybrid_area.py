import numpy as np
import pytest

from panel.p1_hybrid_area import (hybrid_area_terms, hybrid_area_velocity,
                                  hybrid_velocity, near_face_mask,
                                  point_triangle_distance)
from panel.p1_projection_duffy import area_terms
from panel.p1_regularized_sheet_velocity import RegularizedP1SheetField


def fixture():
    """Two-triangle strip, shared internal edge, non-planar strengths."""
    xyz = np.array([[0., 0., 0.], [1.0, 0., 0.1], [1.0, 1.0, 0.],
                    [0., 1.0, -0.1], [2.0, 0.5, 0.2]])
    faces = np.array([[0, 1, 2], [0, 2, 3], [1, 4, 2]])
    values = np.array([[0.3, -0.2, 0.8], [0.3, 0.8, 0.1], [-0.2, 1.1, 0.8]])
    return xyz, faces, values


def field_with(**kw):
    xyz, faces, values = fixture()
    return RegularizedP1SheetField([(xyz, faces, values)], **kw)


def test_point_triangle_distance_is_exact_in_each_voronoi_region():
    tri = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
    # above the interior: the distance is the plane distance
    np.testing.assert_allclose(
        point_triangle_distance(tri, np.array([0.25, 0.25, 0.7])), [0.7], rtol=0,
        atol=0)
    # beyond vertex B: the distance is the distance to that vertex
    np.testing.assert_allclose(
        point_triangle_distance(tri, np.array([2.0, -1.0, 0.0])),
        [np.sqrt(2.0)], rtol=1e-12)
    # beyond edge AC: perpendicular distance to the line x = 0
    np.testing.assert_allclose(
        point_triangle_distance(tri, np.array([-0.4, 0.5, 0.0])), [0.4], rtol=1e-12)


def test_point_triangle_distance_bounds_dense_barycentric_sampling():
    rng = np.random.default_rng(7)
    tri = rng.normal(size=(6, 3, 3))
    u = np.linspace(0.0, 1.0, 120)
    grid = np.array([(a, b) for a in u for b in u if a + b <= 1.0])
    bary = np.stack((1.0-grid[:, 0]-grid[:, 1], grid[:, 0], grid[:, 1]), axis=1)
    for point in rng.normal(size=(4, 3)):
        exact = point_triangle_distance(tri, point)
        sampled = np.linalg.norm(
            point[None, None, :]-np.einsum("qf,nfi->nqi", bary, tri), axis=2).min(axis=1)
        assert np.all(exact <= sampled+1e-12)
        assert np.all(exact >= sampled-3e-2)


def test_hybrid_with_every_face_near_is_the_full_fan_exactly():
    xyz, faces, values = fixture()
    point = np.array([0.4, 0.3, 0.6])
    core = 0.05
    got = hybrid_area_terms(xyz[faces], point, core, near_cores=1e9,
                            fan_order=8, fan_strips=1)
    expected = area_terms(xyz[faces], point, core, order=8, strips=1)
    np.testing.assert_array_equal(got, expected)


def test_hybrid_with_no_face_near_is_the_plain_duffy_map():
    xyz, faces, values = fixture()
    point = np.array([0.4, 0.3, 0.6])
    core = 0.05
    got = hybrid_area_terms(xyz[faces], point, core, near_cores=1e-9,
                            far_order=12)
    expected = area_terms(xyz[faces], point, core, order=12, strips=1)
    np.testing.assert_array_equal(got, expected)


def test_far_order_reduces_the_error_against_the_full_fan():
    xyz, faces, values = fixture()
    point = np.array([0.45, 0.35, 0.20])
    core = 0.05
    reference = area_terms(xyz[faces], point, core, order=48, strips=1)
    errors = []
    for far_order in (4, 8, 16):
        got = hybrid_area_terms(xyz[faces], point, core, near_cores=3.0,
                                fan_order=48, fan_strips=1, far_order=far_order)
        errors.append(float(np.linalg.norm(got-reference)))
    assert errors[0] > errors[1] > errors[2]


def test_mask_agrees_with_the_distance_test():
    xyz, faces, values = fixture()
    core = 0.05
    for point in (np.array([0.4, 0.3, 0.02]), np.array([4.0, -3.0, 2.0])):
        distance = point_triangle_distance(xyz[faces], point)
        mask = near_face_mask(xyz[faces], point, core, near_cores=10.0)
        np.testing.assert_array_equal(mask, distance < 10.0*core)


@pytest.mark.parametrize("kw", [
    {"core": 0.0}, {"core": -1.0}, {"core": np.nan}, {"core": 0.05, "near_cores": 0.0},
    {"core": 0.05, "near_cores": -2.0}, {"core": 0.05, "near_cores": np.inf},
    {"core": 0.05, "fan_order": 1}, {"core": 0.05, "fan_strips": 0},
    {"core": 0.05, "far_order": 3}])
def test_invalid_arguments_are_rejected(kw):
    xyz, faces, values = fixture()
    args = dict(kw)
    core = args.pop("core")
    with pytest.raises(ValueError):
        hybrid_area_terms(xyz[faces], np.array([0.4, 0.3, 0.6]), core, **args)


def test_hybrid_velocity_reproduces_the_production_fan_field():
    field = field_with(core_radius=0.06, area_quadrature="projection_fan",
                       fan_order=8, fan_strips=1, fan_refinement_order=16)
    point = np.array([0.42, 0.33, 0.51])
    np.testing.assert_array_equal(
        hybrid_velocity(field, point, near_cores=np.inf), field(point))
    np.testing.assert_allclose(
        hybrid_velocity(field, point, near_cores=1e9, fan_order=8, fan_strips=1),
        field(point), rtol=0, atol=0)
