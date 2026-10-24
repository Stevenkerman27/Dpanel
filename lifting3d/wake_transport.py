"""Experimental material-coordinate wake displacement transport."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def integrate_displacement(x, eta, velocity):
    x = np.asarray(x, dtype=float)
    eta = np.asarray(eta, dtype=float)
    velocity = np.asarray(velocity, dtype=float)

    if x.ndim != 1 or eta.ndim != 1:
        raise ValueError("x and eta must be one-dimensional grids")
    if len(x) < 2 or len(eta) < 2:
        raise ValueError("x and eta grids must each contain at least two points")
    if not np.isfinite(x).all() or not np.isfinite(eta).all():
        raise ValueError("x and eta grids must be finite")
    if np.any(np.diff(x) <= 0.0) or np.any(np.diff(eta) <= 0.0):
        raise ValueError("x and eta grids must be strictly increasing")
    if velocity.shape != (len(x), len(eta), 3):
        raise ValueError("velocity must have shape (len(x), len(eta), 3)")
    if not np.isfinite(velocity).all():
        raise ValueError("velocity must be finite")
    if np.any(velocity[:, :, 0] <= 0.0):
        raise ValueError("wake transport rejected: qx must be positive everywhere")

    slope_y = velocity[:, :, 1] / velocity[:, :, 0]
    slope_z = velocity[:, :, 2] / velocity[:, :, 0]
    displacement = np.zeros((len(x), len(eta), 2), dtype=float)
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        displacement[i, :, 0] = displacement[i - 1, :, 0] + 0.5 * dx * (
            slope_y[i - 1, :] + slope_y[i, :])
        displacement[i, :, 1] = displacement[i - 1, :, 1] + 0.5 * dx * (
            slope_z[i - 1, :] + slope_z[i, :])
    return displacement


def extend_displacement(reference_points, x, eta, displacement, radius, span):
    reference_points = np.asarray(reference_points, dtype=float)
    x = np.asarray(x, dtype=float)
    eta = np.asarray(eta, dtype=float)
    displacement = np.asarray(displacement, dtype=float)

    if reference_points.ndim != 2 or reference_points.shape[1] != 3:
        raise ValueError("reference_points must have shape (npoints, 3)")
    if x.ndim != 1 or eta.ndim != 1 or len(x) < 2 or len(eta) < 2:
        raise ValueError("x and eta must be one-dimensional grids with at least two points")
    if not np.isfinite(x).all() or not np.isfinite(eta).all():
        raise ValueError("x and eta grids must be finite")
    if np.any(np.diff(x) <= 0.0) or np.any(np.diff(eta) <= 0.0):
        raise ValueError("x and eta grids must be strictly increasing")
    if displacement.shape != (len(x), len(eta), 2):
        raise ValueError("displacement must have shape (len(x), len(eta), 2)")
    if not np.isfinite(reference_points).all() or not np.isfinite(displacement).all():
        raise ValueError("reference points and displacement must be finite")
    if not np.isfinite([radius, span]).all() or radius <= span / 2.0 or span <= 0.0:
        raise ValueError("require finite radius > span/2 > 0")
    if abs(x[0] - 1.0) > 1.0e-12 or x[-1] <= 1.0:
        raise ValueError("wake x grid must start at TE x=1 and extend downstream")
    te_tolerance = 1.0e-12 * max(1.0, float(np.max(np.abs(displacement))))
    if np.max(np.abs(displacement[0])) > te_tolerance:
        raise ValueError("wake displacement must be zero at TE x=1")
    if abs(eta[0] + span / 2.0) > 1.0e-10 or abs(eta[-1] - span / 2.0) > 1.0e-10:
        raise ValueError("eta grid must span exactly [-span/2, span/2]")

    deformed_points = reference_points.copy()
    downstream_nodes = np.flatnonzero(reference_points[:, 0] > 1.0)
    downstream_x = reference_points[downstream_nodes, 0]
    if np.any(downstream_x > x[-1] + 1.0e-12):
        raise ValueError("wake x grid does not cover all downstream reference nodes")
    if np.any(downstream_x < x[0] - 1.0e-12):
        raise ValueError("wake x grid must cover downstream nodes from the TE")

    interpolator = RegularGridInterpolator(
        (x, eta), displacement, method="linear", bounds_error=True)
    material_eta = reference_points[downstream_nodes, 1]
    clamped_eta = np.clip(material_eta, eta[0], eta[-1])
    query = np.column_stack([downstream_x, clamped_eta])
    wake_displacement = interpolator(query)

    span_distance = np.maximum(np.abs(material_eta) - span / 2.0, 0.0)
    span_weight = np.maximum(1.0 - span_distance / (radius - span / 2.0), 0.0) ** 2
    reference_z = reference_points[downstream_nodes, 2]
    z_weight = np.maximum(1.0 - np.abs(reference_z) / radius, 0.0) ** 2
    total_weight = span_weight * z_weight

    deformed_points[downstream_nodes, 1] += total_weight * wake_displacement[:, 0]
    deformed_points[downstream_nodes, 2] += total_weight * wake_displacement[:, 1]
    return deformed_points
