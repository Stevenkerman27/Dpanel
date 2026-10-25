"""Small analytic checks for saved-field pressure reconstruction diagnostics."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

module_path = Path(__file__).resolve().parents[1] / "panel" / "probe_pressure_reconstruction.py"
spec = importlib.util.spec_from_file_location("panel_pressure_reconstruction_probe", module_path)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def _flat_grid():
    x, y = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
    center = np.column_stack((x.ravel(), y.ravel(), np.zeros(9)))
    tangent_l = np.tile([1., 0., 0.], (9, 1))
    tangent_m = np.tile([0., 1., 0.], (9, 1))
    neighbors = np.tile(np.arange(9), (9, 1))
    mu = 0.7 + 2 * center[:, 0] - 3 * center[:, 1]
    return center, mu, neighbors, tangent_l, tangent_m


@pytest.mark.parametrize("method", ["affine_qr", "weighted_affine_qr"])
def test_linear_potential_exactly_recovers_surface_gradient(method):
    centers, mu, ids, l, m = _flat_grid()
    grad, stats = probe.reconstruct_gradients(
        centers, mu, ids, l, m, method=method
    )
    np.testing.assert_allclose(grad, np.tile([2., -3.], (9, 1)), atol=1e-12, rtol=1e-12)
    assert stats["minimum_singular_ratio"] > 0
    assert stats["maximum_fit_relative_residual"] < 1e-12

    cp = probe.pressure_from_tangent_gradient(grad, l, m, 0.)
    # Freestream x velocity = 1; tangent perturbation = (2,-3).
    np.testing.assert_allclose(cp, -17 * np.ones(9), atol=1e-12, rtol=0)


def test_rank_deficient_stencil_is_rejected_not_regularized():
    c = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    mu = np.array([1., 2., 3.])
    neighbors = np.tile(np.arange(3), (3, 1))
    l = np.tile([1., 0., 0.], (3, 1))
    m = np.tile([0., 1., 0.], (3, 1))
    with pytest.raises(ValueError, match="rank-deficient"):
        probe.reconstruct_gradients(c, mu, neighbors, l, m, method="affine_qr")


def test_load_reference_point_and_circulation_integral():
    cp = np.array([1.])
    center = np.array([[2., 0., 0.]])
    normal = np.array([[0., 0., 1.]])
    area = np.array([1.])
    loads = probe.normalized_pressure_loads(
        cp, center, normal, area, 0., np.zeros(3), 1.
    )
    assert loads["CL"] == pytest.approx(-1.)
    assert loads["CM_y"] == pytest.approx(2.)
    translated_reference = probe.normalized_pressure_loads(
        cp, center, normal, area, 0., np.array([1., 0., 0.]), 1.
    )
    assert translated_reference["CM_y"] == pytest.approx(1.)
    fixture = {
        "gamma_stations": [0., 1., 2.],
        "gamma_station_dof": [-1, 0, -1],
        "reference_chord": 1.,
        "reference_area": 2.,
        "origin": [0., 0., 0.],
    }
    assert probe.circulation_cl(fixture, np.array([2.])) == pytest.approx(2.)
    assert probe.circulation_cl(fixture, np.array([2.]), span_factor=1.5) == pytest.approx(3.)


def test_opt_in_quadratic_fit_recovers_gradient_of_exact_quadratic_potential():
    x, y = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
    centers = np.column_stack((x.ravel(), y.ravel(), np.zeros(25)))
    l = np.tile([1., 0., 0.], (25, 1))
    m = np.tile([0., 1., 0.], (25, 1))
    neighbors = np.tile(np.arange(25), (25, 1))
    mu = 1. + 2*centers[:, 0] - 3*centers[:, 1] + (
        2*centers[:, 0]**2 - 2*centers[:, 0]*centers[:, 1] + 3*centers[:, 1]**2
    )
    expected = np.column_stack((
        2. + 4*centers[:, 0] - 2*centers[:, 1],
        -3. - 2*centers[:, 0] + 6*centers[:, 1],
    ))
    gradient, diagnostic = probe.reconstruct_gradients(
        centers, mu, neighbors, l, m, method="quadratic_qr"
    )
    np.testing.assert_allclose(gradient, expected, rtol=1e-12, atol=1e-12)
    assert diagnostic["minimum_singular_ratio"] > 0
