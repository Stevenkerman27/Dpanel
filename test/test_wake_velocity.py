"""Phase A acceptance for the independent wake-sheet velocity kernel.

The kernel has to be checkable against quantities that were not derived from it:

* the exact solid-angle doublet potential  phi = mu * Omega / (4*pi)  (constant mu)
* the exact segment Biot-Savart closed form  (vortex ring of one triangle)
* a brute-force area quadrature of the doublet potential  (linear mu)
* the production P1 kernel panel.kernel.triangle_potential  (linear_y=True)
* the analytic vortex-sheet jump condition  jump = gamma x n

Every tolerance below is a measured number, not a guess (see
docs/ISSUE65_PHASE_A_WAKE_VELOCITY.md for the recorded values).
"""
import numpy as np
import pytest

from panel.wake_velocity import (
    doublet_potential,
    segment_velocity,
    sheet_velocity,
    sheet_velocity_pv,
    sheet_velocity_sum,
    solid_angle,
    triangle_area,
    triangle_normal,
    triangle_ring_velocity,
)

TRI = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
NORMAL = triangle_normal(TRI)
MU_GRAD = np.array([0.0, 0.7, 0.0])
MU_CENTRE = 0.3 + 0.7 / 3.0          # value of 0.3 + 0.7*y at the centroid


def _fd(field, point, step=1e-6):
    gradient = np.zeros(3)
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = step
        gradient[axis] = (field(point + delta) - field(point - delta)) / (2 * step)
    return gradient


def _midpoints(n):
    """Midpoint rule on a triangle, total weight = area (0.5)."""
    bary = []
    weights = []
    for i in range(n):
        for j in range(n - i):
            bary.append(((i + 1 / 3) / n, (j + 1 / 3) / n))
            weights.append(0.5 / n / n)
            if i + j < n - 1:
                bary.append(((i + 2 / 3) / n, (j + 2 / 3) / n))
                weights.append(0.5 / n / n)
    b = np.array(bary)
    a = b[:, 0]
    c = b[:, 1]
    return np.stack([1 - a - c, a, c], axis=1) @ TRI, np.array(weights)


def _area_potential(points, weights, point):
    rel = point[None, :] - points
    radius = np.linalg.norm(rel, axis=1)
    strength = 0.3 + 0.7 * points[:, 1]
    return -(np.sum(weights * strength * (rel @ NORMAL) / radius ** 3)) / (4 * np.pi)


def test_segment_closed_form_matches_quadrature():
    """The closed form used for edges must equal direct Biot-Savart integration."""
    t, w = np.polynomial.legendre.leggauss(400)
    s = 0.5 * (t + 1.0)
    ws = 0.5 * w
    a = np.array([0.1, 0.2, 0.3])
    b = np.array([0.9, -0.1, 0.2])
    for point in (np.array([0.3, 0.9, 0.4]), np.array([-0.5, 0.1, 0.0]), np.array([0.5, 0.5, 2.0])):
        nodes = a[None, :] + s[:, None] * (b - a)[None, :]
        rel = point[None, :] - nodes
        radius = np.linalg.norm(rel, axis=1)
        reference = np.sum((ws / radius ** 3)[:, None]
                           * np.cross(np.broadcast_to(b - a, rel.shape), rel), axis=0) / (4 * np.pi)
        assert np.linalg.norm(segment_velocity(point, a, b) - reference) / np.linalg.norm(reference) < 1e-9


def test_constant_strength_equals_solid_angle_gradient():
    """Constant mu: velocity must be the exact gradient of mu*Omega/(4*pi)."""
    for point in (np.array([0.3, 0.3, 0.8]), np.array([-0.4, 0.2, 0.3]), np.array([0.5, 0.5, 2.0])):
        velocity = sheet_velocity(point, TRI, 1.0, np.zeros(3))
        reference = _fd(lambda p: doublet_potential(p, TRI, 1.0), point)
        assert np.linalg.norm(velocity - reference) / np.linalg.norm(reference) < 1e-8


def test_constant_strength_equals_vortex_ring():
    """The doublet sheet must reproduce the single vortex ring of the boundary."""
    point = np.array([0.3, 0.3, 0.8])
    velocity = sheet_velocity(point, TRI, 1.0, np.zeros(3))
    ring = triangle_ring_velocity(point, TRI, 1.0)
    assert np.linalg.norm(velocity - ring) / np.linalg.norm(ring) < 1e-10


def test_linear_strength_matches_area_quadrature():
    """Linear mu: the edge-only reduction must match brute-force area quadrature."""
    points, weights = _midpoints(40)
    for point in (np.array([0.3, 0.3, 0.8]), np.array([-0.4, 0.2, 0.3]), np.array([0.2, 0.7, 0.12])):
        velocity = sheet_velocity(point, TRI, MU_CENTRE, MU_GRAD)
        reference = _fd(lambda p: _area_potential(points, weights, p), point, 1e-5)
        assert np.linalg.norm(velocity - reference) / np.linalg.norm(reference) < 2e-3


def test_matches_production_kernel_sign_and_value():
    """Cross-check against panel.kernel.triangle_potential (linear_y=True)."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from panel.kernel import triangle_potential

    def kernel_pair(point):
        doublet, moment = triangle_potential(
            jnp.asarray(point[jnp.newaxis, :]), jnp.asarray(TRI[jnp.newaxis, :]), linear_y=True)
        return float(doublet[0, 0]), float(moment[0, 0])

    def kernel_potential(point):
        doublet, moment = kernel_pair(point)
        return MU_CENTRE * doublet + 0.7 * moment

    for point in (np.array([0.3, 0.3, 0.8]), np.array([-0.4, 0.2, 0.3]), np.array([0.5, 0.5, 2.0])):
        velocity = sheet_velocity(point, TRI, MU_CENTRE, MU_GRAD)
        reference = _fd(kernel_potential, point, 1e-5)
        # the production kernel stores -Omega/(4*pi); the velocity kernel stores
        # +Omega/(4*pi).  The relation is a single global sign, asserted here.
        residual = np.linalg.norm(velocity + reference) / np.linalg.norm(velocity)
        assert residual < 1e-8, residual


def test_principal_value_jump_is_surface_vorticity():
    """Across a sheet the velocity must jump by gamma x n, not by a tuned number."""
    centre = TRI.mean(axis=0)
    gamma = np.cross(MU_GRAD, NORMAL)
    result = sheet_velocity_pv(centre, TRI, MU_CENTRE, MU_GRAD)
    expected = np.cross(gamma, NORMAL)
    assert np.linalg.norm(result["jump"] - expected) / np.linalg.norm(expected) < 2e-3
    assert abs(result["normal_jump"]) < 1e-9
    assert result["convergence"] < 1e-4


def test_principal_value_is_independent_of_offset():
    """The principal value must be the limit of the two-sided average."""
    centre = TRI.mean(axis=0)
    coarse = sheet_velocity_pv(centre, TRI, MU_CENTRE, MU_GRAD, offsets=(1e-2, 1e-3))
    fine = sheet_velocity_pv(centre, TRI, MU_CENTRE, MU_GRAD, offsets=(1e-3, 1e-4))
    scale = np.linalg.norm(fine["principal_value"])
    assert np.linalg.norm(coarse["principal_value"] - fine["principal_value"]) / scale < 1e-3


def test_dimensions_and_scale_invariance():
    """u scales like strength / length; the kernel must respect that exactly."""
    point = np.array([0.3, 0.3, 0.8])
    base = sheet_velocity(point, TRI, 2.0, np.zeros(3))
    assert np.linalg.norm(sheet_velocity(point, TRI, 1.0, np.zeros(3)) * 2.0 - base) < 1e-14
    scaled = TRI * 3.0
    assert np.linalg.norm(sheet_velocity(point * 3.0, scaled, 2.0, np.zeros(3)) - base / 3.0) < 1e-12


def test_area_and_normal_helpers():
    assert abs(triangle_area(TRI) - 0.5) < 1e-15
    assert np.allclose(NORMAL, np.array([0.0, 0.0, 1.0]))
    assert abs(solid_angle(np.array([0.3, 0.3, 0.8]), TRI) + 0.6305832949727995) < 1e-12


def test_sheet_sum_matches_area_oracle():
    """A two-strip wake must sum triangle by triangle, with self faces skippable."""
    strip = np.array([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
    ])
    strengths = np.array([0.4, 0.6])
    gradients = np.array([[0.0, 0.2, 0.0], [0.0, -0.1, 0.0]])
    point = np.array([0.4, 0.4, 0.5])
    total = sheet_velocity_sum(point, strip, strengths, gradients)
    manual = sum(sheet_velocity(point, strip[i], strengths[i], gradients[i]) for i in range(2))
    assert np.linalg.norm(total - manual) < 1e-15
    without = sheet_velocity_sum(point, strip, strengths, gradients, skip=[1])
    assert np.linalg.norm(without - sheet_velocity(point, strip[0], strengths[0], gradients[0])) < 1e-15


def test_empty_sheet_returns_zero():
    point = np.array([0.0, 0.0, 1.0])
    assert np.allclose(sheet_velocity_sum(point, np.zeros((0, 3, 3)), [], np.zeros((0, 3))), 0.0)
