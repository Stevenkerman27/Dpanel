"""Analytic checks for the experimental wake transport kernel."""
import hashlib
import json
from pathlib import Path

import numpy as np

import wake_transport

root = Path(__file__).resolve().parent
out = root / "results" / "wake_transport_v1"
out.mkdir(parents=True, exist_ok=True)
ncpu = 8
tolerance = 2.0e-13
checks = []

x = np.array([1.0, 1.2, 1.7, 2.4, 3.0])
eta = np.array([-1.0, -0.4, 0.25, 1.0])
velocity = np.zeros((len(x), len(eta), 3))
velocity[:, :, 0] = 2.0
velocity[:, :, 1] = 1.0
velocity[:, :, 2] = -0.5
displacement = wake_transport.integrate_displacement(x, eta, velocity)
expected_y = 0.5 * (x - 1.0)
expected_z = -0.25 * (x - 1.0)
error = max(float(np.max(np.abs(displacement[:, :, 0] - expected_y[:, None]))),
            float(np.max(np.abs(displacement[:, :, 1] - expected_z[:, None]))))
assert error < tolerance
assert np.max(np.abs(displacement[0])) == 0.0
checks.append({"name": "uniform_oblique_flow", "passed": True, "max_error": error})

velocity[:, :, 1:] = 0.0
displacement = wake_transport.integrate_displacement(x, eta, velocity)
error = float(np.max(np.abs(displacement)))
assert error == 0.0
checks.append({"name": "zero_side_flow", "passed": True, "max_error": error})

x = np.array([1.0, 1.4, 2.0, 3.0])
eta = np.array([-1.0, -0.3, 0.4, 1.0])
velocity = np.zeros((len(x), len(eta), 3))
velocity[:, :, 0] = 2.0
velocity[:, :, 1] = 0.8 + 0.4 * x[:, None] + 0.2 * eta[None, :]
velocity[:, :, 2] = -0.2 + 0.6 * x[:, None] - 0.1 * eta[None, :]
displacement = wake_transport.integrate_displacement(x, eta, velocity)
exact_y = (0.8 * (x - 1.0) + 0.2 * (x**2 - 1.0))[:, None] / 2.0
exact_y = exact_y + (0.2 * eta / 2.0)[None, :] * (x - 1.0)[:, None]
exact_z = (-0.2 * (x - 1.0) + 0.3 * (x**2 - 1.0))[:, None] / 2.0
exact_z = exact_z - (0.1 * eta / 2.0)[None, :] * (x - 1.0)[:, None]
error_y = float(np.max(np.abs(displacement[:, :, 0] - exact_y)))
error_z = float(np.max(np.abs(displacement[:, :, 1] - exact_z)))
assert max(error_y, error_z) < tolerance
checks.append({"name": "x_eta_coupled_linear_velocity", "passed": True,
               "max_y_error": error_y, "max_z_error": error_z})

x = np.array([1.0, 2.0, 3.5])
eta = np.array([-1.0, 0.0, 1.0])
displacement = np.zeros((len(x), len(eta), 2))
displacement[:, :, 0] = 0.1 * (x[:, None] - 1.0) * (1.0 + eta[None, :])
displacement[:, :, 1] = 0.05 * (x[:, None] - 1.0) * (1.0 - eta[None, :])
reference_points = np.array([
    [0.5, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [2.0, 0.0, 0.0],
    [2.0, 1.5, 0.0],
    [2.0, 0.0, 3.0],
    [2.0, 3.0, 0.0],
    [3.5, 0.0, 0.0],
])
deformed_points = wake_transport.extend_displacement(
    reference_points, x, eta, displacement, radius=3.0, span=2.0)
assert np.array_equal(deformed_points[:2], reference_points[:2])
assert np.allclose(deformed_points[2], [2.0, 0.1, 0.05], atol=tolerance, rtol=0.0)
assert np.allclose(deformed_points[3], [2.0, 1.5 + 0.1125, 0.0], atol=tolerance, rtol=0.0)
assert np.array_equal(deformed_points[4], reference_points[4])
assert np.array_equal(deformed_points[5], reference_points[5])
assert np.allclose(deformed_points[6], [3.5, 0.25, 0.125], atol=tolerance, rtol=0.0)
assert np.array_equal(deformed_points[:, 0], reference_points[:, 0])
checks.append({
    "name": "reference_reconstruction_tapers_and_boundaries",
    "passed": True,
    "upstream_and_TE_fixed": True,
    "spanwise_clamp_then_quadratic_taper": True,
    "z_farfield_fixed": True,
    "y_farfield_fixed": True,
    "downstream_x_farfield_plane_may_move": True,
})

bad_velocity = np.ones((len(x), len(eta), 3))
bad_velocity[1, 2, 0] = 0.0
zero_qx_rejected = False
try:
    wake_transport.integrate_displacement(x, eta, bad_velocity)
except ValueError as error:
    zero_qx_rejected = "qx must be positive" in str(error)
assert zero_qx_rejected
checks.append({"name": "zero_qx_rejected_without_clipping", "passed": True})

bad_velocity[1, 2, 0] = -0.25
negative_qx_rejected = False
try:
    wake_transport.integrate_displacement(x, eta, bad_velocity)
except ValueError as error:
    negative_qx_rejected = "qx must be positive" in str(error)
assert negative_qx_rejected
checks.append({"name": "negative_qx_rejected_without_clipping", "passed": True})

summary = {
    "status": "passed",
    "scope": "Unit verification of a graph-based wake-line transport kernel; not a free-vortex-sheet or Euler validation",
    "ncpu_requested": ncpu,
    "integration": "Cumulative trapezoid in x of qy/qx and qz/qx; initial displacement zero at x[0]",
    "checks": checks,
    "wake_transport_sha256": hashlib.sha256((root / "wake_transport.py").read_bytes()).hexdigest(),
    "test_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
