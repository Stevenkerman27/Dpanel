import numpy as np
import pytest

from panel.vortex_filament import segment_biot_savart, polyline_biot_savart


def direct_line_integral(point, start, end, circulation=1.0, order=160):
    """Independent Gauss-Legendre integration of dl x (P-X) / |P-X|^3."""
    p = np.asarray(point, float)
    a, b = np.asarray(start, float), np.asarray(end, float)
    d = b - a
    length = np.linalg.norm(d)
    tangent = d / length
    q, w = np.polynomial.legendre.leggauss(order)
    s = (q + 1.0) * length / 2.0
    weights = w * length / 2.0
    source = a[None, :] + s[:, None] * tangent
    r = p[None, :] - source
    rn = np.linalg.norm(r, axis=1)
    integrand = np.cross(np.broadcast_to(tangent, r.shape), r) / rn[:, None]**3
    return circulation / (4.0 * np.pi) * np.sum(weights[:, None] * integrand, axis=0)


def rotation():
    # Proper orthogonal rotation, composed from z and y rotations.
    az, ay = 0.73, -0.41
    rz = np.array([[np.cos(az), -np.sin(az), 0.],
                   [np.sin(az), np.cos(az), 0.],
                   [0., 0., 1.]])
    ry = np.array([[np.cos(ay), 0., np.sin(ay)],
                   [0., 1., 0.],
                   [-np.sin(ay), 0., np.cos(ay)]])
    return rz @ ry


def test_matches_independent_line_quadrature_off_axis():
    a = np.array([-0.7, 0.2, -0.1])
    b = np.array([1.1, 0.6, 0.3])
    p = np.array([0.2, -0.4, 0.9])
    got = segment_biot_savart(p, a, b, circulation=1.7)
    expected = direct_line_integral(p, a, b, circulation=1.7)
    np.testing.assert_allclose(got, expected, rtol=2e-12, atol=2e-13)


def test_orientation_and_circulation_sign():
    p = np.array([0.4, 0.3, 0.8])
    a, b = np.array([-1., 0., 0.]), np.array([1., 0., 0.])
    forward = segment_biot_savart(p, a, b, 2.0)
    reverse = segment_biot_savart(p, b, a, 2.0)
    negative = segment_biot_savart(p, a, b, -2.0)
    np.testing.assert_allclose(reverse, -forward, atol=1e-15)
    np.testing.assert_allclose(negative, -forward, atol=1e-15)


def test_translation_and_rotation_covariance():
    p = np.array([0.2, -0.7, 0.8])
    a, b = np.array([-0.8, 0.1, 0.2]), np.array([0.9, 0.5, -0.1])
    base = segment_biot_savart(p, a, b, 0.6)
    shift = np.array([13.0, -4.5, 2.2])
    moved = segment_biot_savart(p+shift, a+shift, b+shift, 0.6)
    np.testing.assert_allclose(moved, base, rtol=1e-13, atol=1e-14)
    q = rotation()
    rotated = segment_biot_savart(q@p, q@a, q@b, 0.6)
    np.testing.assert_allclose(rotated, q@base, rtol=1e-13, atol=1e-14)


def test_inverse_length_scaling_and_far_field_decay():
    a, b = np.array([-0.5, 0., 0.]), np.array([0.5, 0., 0.])
    p = np.array([0., 0.2, 0.])
    v = segment_biot_savart(p, a, b, 1.0)
    scale = 3.7
    vs = segment_biot_savart(scale*p, scale*a, scale*b, 1.0)
    np.testing.assert_allclose(vs, v/scale, rtol=2e-14, atol=1e-14)

    p1, p2 = np.array([0., 0., 8.]), np.array([0., 0., 16.])
    f1 = np.linalg.norm(segment_biot_savart(p1, a, b))
    f2 = np.linalg.norm(segment_biot_savart(p2, a, b))
    assert f1/f2 == pytest.approx(4.0, rel=0.04)


def test_near_segment_offset_has_expected_inverse_distance_signature():
    a, b = np.array([-1., 0., 0.]), np.array([1., 0., 0.])
    p1, p2 = np.array([0., 0.02, 0.]), np.array([0., 0.01, 0.])
    v1 = np.linalg.norm(segment_biot_savart(p1, a, b))
    v2 = np.linalg.norm(segment_biot_savart(p2, a, b))
    assert v2/v1 == pytest.approx(2.0, rel=0.01)


def test_collinear_self_policy_is_limited_to_segment_interior():
    a, b = np.array([0., 0., 0.]), np.array([2., 0., 0.])
    with pytest.raises(FloatingPointError, match="singular vortex segment"):
        segment_biot_savart([1., 0., 0.], a, b)
    np.testing.assert_array_equal(
        segment_biot_savart([1., 0., 0.], a, b, on_segment="straight_pv_zero"),
        np.zeros(3))
    np.testing.assert_array_equal(segment_biot_savart([-1., 0., 0.], a, b), np.zeros(3))
    with pytest.raises(FloatingPointError, match="endpoint"):
        segment_biot_savart(a, a, b, on_segment="straight_pv_zero")


def test_polyline_sums_segments_and_explicit_exclusion():
    vertices = np.array([[-1., 0., 0.], [0., 0.5, 0.], [1., 0., 0.]])
    p = np.array([0., -0.4, 0.8])
    v0 = segment_biot_savart(p, vertices[0], vertices[1], 0.7)
    v1 = segment_biot_savart(p, vertices[1], vertices[2], 0.7)
    np.testing.assert_allclose(polyline_biot_savart(p, vertices, 0.7), v0+v1)
    np.testing.assert_allclose(polyline_biot_savart(
        p, vertices, 0.7, excluded_segments=(0,)), v1)
    with pytest.raises(ValueError, match="out of range"):
        polyline_biot_savart(p, vertices, excluded_segments=(2,))


def test_invalid_inputs_fail_closed():
    with pytest.raises(ValueError):
        segment_biot_savart([0., 0.], [0., 0., 0.], [1., 0., 0.])
    with pytest.raises(ValueError, match="nonzero"):
        segment_biot_savart([0., 1., 0.], [0., 0., 0.], [0., 0., 0.])
    with pytest.raises(FloatingPointError, match="singular vortex segment"):
        segment_biot_savart([0.5, 0., 0.], [0., 0., 0.], [1., 0., 0.])


def test_curved_polyline_pv_of_one_straight_self_segment_is_not_total_self_induction():
    # The query lies inside segment 0 but not on segment 1. The straight
    # segment's symmetric PV is zero, while the rest of the bent filament
    # still induces a nonzero velocity. This intentionally does NOT claim an
    # analytic curvature correction for the complete filament.
    vertices = np.array([[-1., 0., 0.], [0., 0., 0.], [0.5, 0.8, 0.2]])
    point = np.array([-0.4, 0., 0.])
    remaining = segment_biot_savart(point, vertices[1], vertices[2], 1.0)
    total = polyline_biot_savart(point, vertices, 1.0, on_segment="straight_pv_zero")
    np.testing.assert_allclose(total, remaining, rtol=1e-14, atol=1e-14)
    assert np.linalg.norm(total) > 0.0
