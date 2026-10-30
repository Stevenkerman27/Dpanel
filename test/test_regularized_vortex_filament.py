import numpy as np
import pytest

from panel.regularized_vortex_filament import (
    regularized_segment_velocity,
    regularized_polyline_velocity,
)


def independent_regularized_line_integral(point, start, end, circulation, core_radius,
                                          order=240, subintervals=1):
    """Composite Gauss-Legendre integral of dl x r / (r^2+a^2)^(3/2).

    Subdivision is essential near the core: one high-order panel can still
    underresolve the narrow peak when the offset is much smaller than a.
    """
    p = np.asarray(point, float)
    a = np.asarray(start, float)
    b = np.asarray(end, float)
    d = b-a
    length = np.linalg.norm(d)
    tangent = d/length
    q, w = np.polynomial.legendre.leggauss(order)
    xi = (q+1.0)/2.0
    s = (np.arange(subintervals)[:, None]+xi[None, :])*(length/subintervals)
    weights = np.broadcast_to(w[None, :]*(length/(2*subintervals)), s.shape)
    source = a[None, None, :] + s[:, :, None]*tangent[None, None, :]
    r = p[None, None, :] - source
    den = (np.einsum("sqi,sqi->sq", r, r) + core_radius**2)**1.5
    integrand = np.cross(np.broadcast_to(tangent, r.shape), r)/den[:, :, None]
    return circulation/(4*np.pi)*np.sum(weights[:, :, None]*integrand, axis=(0, 1))


def independent_rm_circular_ring_speed(radius, circulation, core_radius,
                                       order=24, subintervals=256):
    """Direct azimuthal quadrature of the continuous RM circular-ring integral.

    The field point lies on the circular filament. This is independent of the
    polygonal segment implementation and is used as its continuous-loop oracle.
    """
    q, w = np.polynomial.legendre.leggauss(order)
    theta = (np.arange(subintervals)[:, None] + (q[None, :]+1.0)/2.0) * (
        2*np.pi/subintervals)
    weights = np.broadcast_to(w[None, :]*(np.pi/subintervals), theta.shape)
    numerator = radius**2*(1.0-np.cos(theta))
    denominator = (core_radius**2 + 2*radius**2*(1.0-np.cos(theta)))**1.5
    return circulation/(4*np.pi)*np.sum(weights*numerator/denominator)


def rotation():
    az, ay = 0.71, -0.37
    rz = np.array([[np.cos(az), -np.sin(az), 0.],
                   [np.sin(az), np.cos(az), 0.],
                   [0., 0., 1.]])
    ry = np.array([[np.cos(ay), 0., np.sin(ay)],
                   [0., 1., 0.],
                   [-np.sin(ay), 0., np.cos(ay)]])
    return rz @ ry


def test_regularized_segment_matches_independent_quadrature():
    a = np.array([-0.8, 0.1, -0.2])
    b = np.array([1.2, 0.4, 0.5])
    p = np.array([0.15, -0.12, 0.23])
    for core in (0.03, 0.2, 0.7):
        got = regularized_segment_velocity(p, a, b, 1.4, core)
        expected = independent_regularized_line_integral(p, a, b, 1.4, core)
        np.testing.assert_allclose(got, expected, rtol=2e-12, atol=2e-13)


@pytest.mark.parametrize("offset_factor", [0.1, 0.05, 0.01])
def test_regularized_segment_matches_composite_quadrature_near_core(offset_factor):
    start = np.array([-1., 0., 0.])
    end = np.array([1., 0., 0.])
    core = 0.01
    point = np.array([0., offset_factor*core, 0.])
    got = regularized_segment_velocity(point, start, end, 1.0, core)
    expected = independent_regularized_line_integral(
        point, start, end, 1.0, core, order=24, subintervals=64)
    np.testing.assert_allclose(got, expected, rtol=2e-12, atol=2e-13)


def test_orientation_and_circulation_sign():
    p = np.array([0.2, 0.4, 0.7])
    a, b = np.array([-1., 0., 0.]), np.array([1., 0.2, 0.])
    forward = regularized_segment_velocity(p, a, b, 0.8, 0.1)
    reverse = regularized_segment_velocity(p, b, a, 0.8, 0.1)
    negative = regularized_segment_velocity(p, a, b, -0.8, 0.1)
    np.testing.assert_allclose(reverse, -forward, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(negative, -forward, rtol=1e-14, atol=1e-15)


def test_translation_rotation_and_scaled_geometry_covariance():
    p = np.array([0.2, -0.7, 0.8])
    a, b = np.array([-0.8, 0.1, 0.2]), np.array([0.9, 0.5, -0.1])
    core, gamma = 0.12, 0.6
    base = regularized_segment_velocity(p, a, b, gamma, core)
    shift = np.array([13., -4.5, 2.2])
    np.testing.assert_allclose(
        regularized_segment_velocity(p+shift, a+shift, b+shift, gamma, core),
        base, rtol=1e-13, atol=1e-14)
    q = rotation()
    np.testing.assert_allclose(
        regularized_segment_velocity(q@p, q@a, q@b, gamma, core),
        q@base, rtol=1e-13, atol=1e-14)
    scale = 3.4
    scaled = regularized_segment_velocity(scale*p, scale*a, scale*b,
                                          gamma, scale*core)
    np.testing.assert_allclose(scaled, base/scale, rtol=2e-13, atol=1e-14)


def test_core_keeps_on_axis_and_endpoint_evaluations_finite():
    a, b = np.array([0., 0., 0.]), np.array([2., 0., 0.])
    for point in ([1., 0., 0.], a, b, [1., 1e-12, 0.]):
        got = regularized_segment_velocity(point, a, b, 1., 0.05)
        assert np.isfinite(got).all()
    np.testing.assert_array_equal(
        regularized_segment_velocity([1., 0., 0.], a, b, 1., 0.05),
        np.zeros(3))


def test_velocity_is_linear_in_circulation_and_core_is_explicit():
    p, a, b = np.array([0., 0.2, 0.3]), np.array([-1., 0., 0.]), np.array([1., 0., 0.])
    u1 = regularized_segment_velocity(p, a, b, 1., 0.1)
    u3 = regularized_segment_velocity(p, a, b, 3., 0.1)
    np.testing.assert_allclose(u3, 3*u1)
    u_small = regularized_segment_velocity(p, a, b, 1., 0.05)
    u_large = regularized_segment_velocity(p, a, b, 1., 0.2)
    assert not np.allclose(u_small, u_large)


def test_polyline_sums_regularized_segments_and_supports_diagnostic_exclusion():
    vertices = np.array([[-1., 0., 0.], [0., 0.5, 0.], [1., 0., 0.]])
    p = np.array([0., -0.4, 0.8])
    core, gamma = 0.07, 0.9
    v0 = regularized_segment_velocity(p, vertices[0], vertices[1], gamma, core)
    v1 = regularized_segment_velocity(p, vertices[1], vertices[2], gamma, core)
    np.testing.assert_allclose(
        regularized_polyline_velocity(p, vertices, gamma, core), v0+v1)
    np.testing.assert_allclose(
        regularized_polyline_velocity(p, vertices, gamma, core, excluded_segments=(0,)), v1)
    with pytest.raises(ValueError, match="out of range"):
        regularized_polyline_velocity(p, vertices, gamma, core, excluded_segments=(2,))


def test_fixed_core_curved_loop_refinement_matches_continuous_rm_oracle():
    radius, circulation = 1.0, 1.0
    speeds = {}
    for core in (0.05, 0.1, 0.2):
        exact = independent_rm_circular_ring_speed(radius, circulation, core)
        for n in (64, 128, 256, 512, 4096):
            theta = 2*np.pi*np.arange(n)/n
            ring = np.column_stack((radius*np.cos(theta),
                                    radius*np.sin(theta), np.zeros(n)))
            ring = np.vstack((ring, ring[0]))
            point = 0.5*(ring[0]+ring[1])
            u = regularized_polyline_velocity(point, ring, circulation, core)
            speed = np.linalg.norm(u)
            assert np.isfinite(u).all()
            speeds[(core, n)] = speed
            if n == 4096:
                np.testing.assert_allclose(speed, exact, rtol=5e-5, atol=1e-10)
        errors = [abs(speeds[(core, n)]-exact) for n in (64, 128, 256, 512, 4096)]
        assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
    assert len({round(speeds[(core, 4096)], 5) for core in (0.05, 0.1, 0.2)}) == 3


def test_rm_circular_ring_thin_core_self_induction_law():
    radius, circulation, core = 1.0, 1.0, 0.05
    n = 4096
    theta = 2*np.pi*np.arange(n)/n
    ring = np.column_stack((radius*np.cos(theta),
                            radius*np.sin(theta), np.zeros(n)))
    ring = np.vstack((ring, ring[0]))
    point = 0.5*(ring[0]+ring[1])
    numerical = np.linalg.norm(
        regularized_polyline_velocity(point, ring, circulation, core))
    asymptotic = (circulation/(4*np.pi*radius)) * (
        np.log(8*radius/core)-1.0)
    # The -1 constant belongs to this Rosenhead–Moore core convention.
    # Finite-core corrections are O((a/R)^2), so this is an asymptotic,
    # not exact finite-a, equality.
    np.testing.assert_allclose(numerical, asymptotic, rtol=5e-4, atol=1e-10)


@pytest.mark.parametrize("core", [0., -0.1, np.nan, np.inf])
def test_invalid_core_radius_rejected(core):
    with pytest.raises(ValueError, match="core_radius"):
        regularized_segment_velocity([0., 1., 0.], [0., 0., 0.],
                                      [1., 0., 0.], 1., core)


def test_degenerate_geometry_and_invalid_polyline_rejected():
    with pytest.raises(ValueError, match="nonzero"):
        regularized_segment_velocity([0., 1., 0.], [0., 0., 0.],
                                     [0., 0., 0.], 1., 0.1)
    with pytest.raises(ValueError, match="vertices"):
        regularized_polyline_velocity([0., 0., 0.], [[0., 0.], [1., 0.]],
                                      1., 0.1)
