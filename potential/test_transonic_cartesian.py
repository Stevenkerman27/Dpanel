"""Preregistered 2D potential-residual and oblique-flux diagnostics."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from transonic_cartesian import (
    analytic_potential_residual,
    cartesian_potential_residual,
    discrete_curl,
    normal_mass_flux,
    normal_sonic_speed,
    oblique_projected_eo_h,
    potential_grid_gradient,
    shared_face_flux,
)
from transonic_flux import density_from_speed_squared, eo_h


gamma = 1.4
mach_numbers = (0.5, 0.8)
mesh_sizes = (32, 64, 128, 256)
epsilon = 0.03
results_dir = Path(__file__).with_name("results") / "transonic_cartesian_v1"
results_dir.mkdir(parents=True, exist_ok=True)
test_results = []

for mach_inf in mach_numbers:
    q_star = np.sqrt(
        (mach_inf**-2 + 0.5 * (gamma - 1.0))
        / (0.5 * (gamma + 1.0))
    )
    spacing = 2.0 * np.pi / 64
    zero_potential = np.zeros((64, 64))
    freestream = np.array([1.0, 0.23])
    free_gradient = potential_grid_gradient(
        zero_potential, freestream, spacing, spacing
    )
    free_residual, free_x_flux, free_y_flux = cartesian_potential_residual(
        free_gradient, spacing, spacing, mach_inf, gamma
    )
    free_stream_residual = float(np.max(np.abs(free_residual)))
    free_density = float(
        density_from_speed_squared(np.dot(freestream, freestream), mach_inf, gamma)
    )
    free_scale = max(1.0, free_density * np.linalg.norm(freestream))
    free_stream_relative = free_stream_residual / free_scale
    curl_error = float(
        np.max(np.abs(discrete_curl(free_gradient, spacing, spacing)))
    )

    angle = 0.643
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    rotated_gradient = np.einsum("ij,...j->...i", rotation, free_gradient)
    residual_rotated, x_flux_rotated, y_flux_rotated = (
        cartesian_potential_residual(
            rotated_gradient,
            spacing,
            spacing,
            mach_inf,
            gamma,
            rotation @ np.array([1.0, 0.0]),
            rotation @ np.array([0.0, 1.0]),
        )
    )
    rotation_error = float(
        np.max(np.abs(residual_rotated - free_residual))
    )
    rotation_flux_error = float(
        max(
            np.max(np.abs(x_flux_rotated - free_x_flux)),
            np.max(np.abs(y_flux_rotated - free_y_flux)),
        )
    )

    one_d_cell_count = 128
    one_d_dx = 2.0 * np.pi / one_d_cell_count
    one_d_x = np.arange(one_d_cell_count) * one_d_dx
    one_d_phi = 0.08 * np.sin(one_d_x)[:, None] * np.ones(
        (1, one_d_cell_count)
    )
    one_d_base = np.array([1.0, 0.0])
    one_d_gradient = potential_grid_gradient(
        one_d_phi, one_d_base, one_d_dx, one_d_dx
    )
    _, _, one_d_y_flux = cartesian_potential_residual(
        one_d_gradient, one_d_dx, one_d_dx, mach_inf, gamma
    )
    transverse_flux_error = float(np.max(np.abs(one_d_y_flux)))
    one_d_left = one_d_gradient
    one_d_right = np.roll(one_d_gradient, -1, axis=0)
    one_d_face = shared_face_flux(
        one_d_left,
        one_d_right,
        np.array([one_d_dx, 0.0]),
        mach_inf,
        gamma,
    )
    one_d_reference = np.empty_like(one_d_face)
    for i in range(one_d_cell_count):
        for j in range(one_d_cell_count):
            q_left = np.linalg.norm(one_d_left[i, j])
            q_right = np.linalg.norm(one_d_right[i, j])
            one_d_reference[i, j] = -eo_h(
                q_right, q_left, mach_inf, gamma
            ) * one_d_dx
    one_d_error = float(np.max(np.abs(one_d_face - one_d_reference)))

    reverse_gradient = potential_grid_gradient(
        one_d_phi, np.array([-1.0, 0.0]), one_d_dx, one_d_dx
    )
    reverse_flux = shared_face_flux(
        reverse_gradient,
        np.roll(reverse_gradient, -1, axis=0),
        np.array([one_d_dx, 0.0]),
        mach_inf,
        gamma,
    )
    reverse_reference = np.empty_like(reverse_flux)
    for i in range(one_d_cell_count):
        for j in range(one_d_cell_count):
            q_left = np.linalg.norm(reverse_gradient[i, j])
            q_right = np.linalg.norm(np.roll(reverse_gradient, -1, axis=0)[i, j])
            reverse_reference[i, j] = -eo_h(
                q_left, q_right, mach_inf, gamma
            ) * (-1.0) * one_d_dx
    reverse_error = float(np.max(np.abs(reverse_flux - reverse_reference)))

    smooth_errors = []
    smooth_rotation_error = 0.0
    smooth_rotation_flux_error = 0.0
    smooth_curl_error = 0.0
    for cell_count in mesh_sizes:
        dx = 2.0 * np.pi / cell_count
        dy = dx
        x = np.arange(cell_count) * dx
        y = np.arange(cell_count) * dy
        x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
        phi_periodic = epsilon * np.sin(x_grid) * np.sin(y_grid)
        base_gradient = np.array([1.0, 0.12])
        gradient = potential_grid_gradient(
            phi_periodic, base_gradient, dx, dy
        )
        numerical_residual, numerical_x_flux, numerical_y_flux = (
            cartesian_potential_residual(
            gradient, dx, dy, mach_inf, gamma
            )
        )
        exact_residual = analytic_potential_residual(
            x_grid,
            y_grid,
            mach_inf,
            gamma,
            base_gradient,
            epsilon,
        )
        residual_error = float(
            np.max(np.abs(numerical_residual - exact_residual))
        )
        smooth_curl_error = max(
            smooth_curl_error,
            float(np.max(np.abs(discrete_curl(gradient, dx, dy)))),
        )
        rotated_gradient = np.einsum(
            "ij,...j->...i", rotation, gradient
        )
        rotated_residual, rotated_x_flux, rotated_y_flux = (
            cartesian_potential_residual(
                rotated_gradient,
                dx,
                dy,
                mach_inf,
                gamma,
                rotation @ np.array([1.0, 0.0]),
                rotation @ np.array([0.0, 1.0]),
            )
        )
        smooth_rotation_error = max(
            smooth_rotation_error,
            float(np.max(np.abs(rotated_residual - numerical_residual))),
        )
        smooth_rotation_flux_error = max(
            smooth_rotation_flux_error,
            float(
                max(
                    np.max(np.abs(rotated_x_flux - numerical_x_flux)),
                    np.max(np.abs(rotated_y_flux - numerical_y_flux)),
                )
            ),
        )
        smooth_errors.append(
            {"cells_per_axis": cell_count, "residual_linf_error": residual_error}
        )
    smooth_orders = []
    for coarse, fine in zip(smooth_errors[:-1], smooth_errors[1:]):
        smooth_orders.append(
            float(
                np.log2(
                    coarse["residual_linf_error"]
                    / fine["residual_linf_error"]
                )
            )
        )
    smooth_pass = bool(
        all(
            fine["residual_linf_error"] < coarse["residual_linf_error"]
            for coarse, fine in zip(smooth_errors[:-1], smooth_errors[1:])
        )
        and min(smooth_orders[-2:]) >= 0.7
    )

    tangential_speed = 0.35
    normal = np.array(
        [np.cos(np.deg2rad(37.0)), np.sin(np.deg2rad(37.0))]
    )
    q_total_super = 1.1 * q_star
    beta_super = 1.0 - 0.5 * (gamma - 1.0) * mach_inf**2 * (
        q_total_super**2 - 1.0
    )
    normal_speed_oblique = q_total_super * np.cos(np.deg2rad(60.0))
    total_mach_squared = mach_inf**2 * q_total_super**2 / beta_super
    normal_mach_squared = (
        mach_inf**2 * normal_speed_oblique**2 / beta_super
    )
    total_sensor_active = total_mach_squared > 1.0
    normal_sensor_active = normal_mach_squared > 1.0

    normal_limit = np.sqrt(
        1.0
        + 1.0 / (0.5 * (gamma - 1.0) * mach_inf**2)
        - tangential_speed**2
    )
    qn_states = np.linspace(1.0e-6, normal_limit * (1.0 - 1.0e-8), 121)
    intermediate_fractions = np.linspace(0.0, 1.0, 101)
    projected_positive_violation = -np.inf
    normal_eo_positive_violation = -np.inf
    projected_counterexample = None
    for qn_right in qn_states:
        for qn_left in qn_states:
            qn_middle = qn_left + (
                qn_right - qn_left
            ) * intermediate_fractions
            physical_flux_middle = -np.array(
                [
                    normal_mass_flux(
                        qn_value, tangential_speed, mach_inf, gamma
                    )
                    for qn_value in qn_middle
                ]
            )
            projected_h = oblique_projected_eo_h(
                qn_right,
                qn_left,
                tangential_speed,
                normal,
                mach_inf,
                gamma,
            )
            normal_h = float(
                eo_h(
                    qn_right,
                    qn_left,
                    mach_inf,
                    gamma,
                    tangential_speed,
                )
            )
            signed_projected = np.sign(qn_right - qn_left) * (
                projected_h - physical_flux_middle
            )
            signed_normal = np.sign(qn_right - qn_left) * (
                normal_h - physical_flux_middle
            )
            projected_index = int(np.argmax(signed_projected))
            normal_violation = float(np.max(signed_normal))
            if signed_projected[projected_index] > projected_positive_violation:
                projected_positive_violation = float(
                    signed_projected[projected_index]
                )
                projected_counterexample = {
                    "qn_right_downstream": float(qn_right),
                    "qn_left_upstream": float(qn_left),
                    "qn_intermediate": float(
                        qn_middle[projected_index]
                    ),
                    "positive_violation": projected_positive_violation,
                }
            normal_eo_positive_violation = max(
                normal_eo_positive_violation, normal_violation
            )

    oblique_sonic = normal_sonic_speed(
        tangential_speed, mach_inf, gamma
    )
    qn_compression_left = 1.2 * oblique_sonic
    compression_mass = normal_mass_flux(
        qn_compression_left, tangential_speed, mach_inf, gamma
    )
    qn_compression_right = brentq(
        lambda qn: normal_mass_flux(
            qn, tangential_speed, mach_inf, gamma
        )
        - compression_mass,
        1.0e-10,
        oblique_sonic,
        xtol=1.0e-13,
        rtol=1.0e-13,
    )
    rh_mass_error = abs(
        normal_mass_flux(
            qn_compression_right, tangential_speed, mach_inf, gamma
        )
        - compression_mass
    )
    shock_numerical_flux = oblique_projected_eo_h(
        qn_compression_right,
        qn_compression_left,
        tangential_speed,
        normal,
        mach_inf,
        gamma,
    )

    free_pass = free_stream_relative <= 2.0e-12
    curl_pass = curl_error <= 2.0e-12 * max(
        1.0, float(np.max(np.linalg.norm(free_gradient, axis=-1)))
    ) and smooth_curl_error <= 2.0e-12
    rotation_pass = (
        rotation_error <= 2.0e-12 * free_scale
        and rotation_flux_error <= 2.0e-12 * free_scale
        and smooth_rotation_error <= 2.0e-12 * free_scale
        and smooth_rotation_flux_error <= 2.0e-12 * free_scale
    )
    degeneracy_pass = (
        one_d_error <= 2.0e-12
        and reverse_error <= 2.0e-12
        and transverse_flux_error <= 2.0e-12
    )
    oblique_e_pass = (
        projected_positive_violation <= 2.0e-10
        and normal_eo_positive_violation <= 2.0e-10
    )
    run_pass = (
        free_pass
        and curl_pass
        and rotation_pass
        and degeneracy_pass
        and smooth_pass
        and oblique_e_pass
    )

    result = {
        "mach_inf": mach_inf,
        "gamma": gamma,
        "free_stream_max_residual": free_stream_residual,
        "free_stream_relative_residual": free_stream_relative,
        "potential_discrete_curl_max": curl_error,
        "rotation_residual_max_error": rotation_error,
        "rotation_face_flux_max": rotation_flux_error,
        "smooth_rotation_residual_max_error": smooth_rotation_error,
        "smooth_rotation_face_flux_max_error": smooth_rotation_flux_error,
        "smooth_potential_discrete_curl_max": smooth_curl_error,
        "positive_1d_degeneracy_max_error": one_d_error,
        "reverse_1d_degeneracy_max_error": reverse_error,
        "1d_transverse_flux_max": transverse_flux_error,
        "smooth_manufactured_errors": smooth_errors,
        "smooth_observed_orders": smooth_orders,
        "smooth_gate": "PASS" if smooth_pass else "FAIL",
        "oblique_sensor_diagnostic": {
            "total_speed": q_total_super,
            "normal_speed_at_60deg": normal_speed_oblique,
            "total_mach_squared": total_mach_squared,
            "normal_mach_squared": normal_mach_squared,
            "section6_total_speed_switch_active": bool(total_sensor_active),
            "normal_only_switch_active": bool(normal_sensor_active),
            "classification": "DIAGNOSTIC",
        },
        "oblique_frozen_tangent_e_flux": {
            "projected_total_speed_flux_max_positive_violation": (
                projected_positive_violation
            ),
            "projected_flux_counterexample": projected_counterexample,
            "normal_frozen_tangent_eo_max_positive_violation": (
                normal_eo_positive_violation
            ),
            "gate": "PASS" if oblique_e_pass else "FAIL",
        },
        "planar_compression_rh_diagnostic": {
            "tangential_speed": tangential_speed,
            "qn_left_super": qn_compression_left,
            "qn_right_sub": qn_compression_right,
            "rh_normal_mass_flux": compression_mass,
            "rh_mass_mismatch": rh_mass_error,
            "candidate_h_at_discontinuous_face": shock_numerical_flux,
            "classification": "DIAGNOSTIC_NOT_A_STEADY_SHOCK_SOLUTION",
        },
        "status": "PASS" if run_pass else "FAIL",
        "certification": "NOT_CERTIFIED_FOR_2D_TRANSONIC_FLOW",
    }
    test_results.append(result)
    print(
        f"M_inf={mach_inf:.1f} free={free_stream_relative:.3e} "
        f"curl={curl_error:.3e} rotation={rotation_error:.3e} "
        f"1D=[{one_d_error:.3e},{reverse_error:.3e}] "
        f"smooth_orders={smooth_orders} "
        f"total_switch={total_sensor_active} "
        f"normal_switch={normal_sensor_active} "
        f"oblique_E=[{projected_positive_violation:.3e},"
        f"{normal_eo_positive_violation:.3e}] "
        f"RH_error={rh_mass_error:.3e} status={result['status']}"
    )

source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
kernel_hash = hashlib.sha256(
    Path(__file__).with_name("transonic_cartesian.py").read_bytes()
).hexdigest()
flux_hash = hashlib.sha256(
    Path(__file__).with_name("transonic_flux.py").read_bytes()
).hexdigest()
protocol_hash = hashlib.sha256(
    Path(__file__).with_name("TRANSONIC_CARTESIAN_PROTOCOL.md").read_bytes()
).hexdigest()
summary = {
    "scope": (
        "Independent Cartesian potential residual tests only; "
        "not a transonic airfoil solver or certification."
    ),
    "gamma": gamma,
    "mach_numbers": list(mach_numbers),
    "mesh_sizes": list(mesh_sizes),
    "cpu_thread_environment": {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "unset"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", "unset"),
    },
    "runs": test_results,
    "overall_status": (
        "PASS"
        if all(row["status"] == "PASS" for row in test_results)
        else "FAIL"
    ),
    "source_sha256": source_hash,
    "kernel_sha256": kernel_hash,
    "shared_flux_sha256": flux_hash,
    "protocol_sha256": protocol_hash,
}
with (results_dir / "summary.json").open("w", encoding="utf-8") as output_file:
    json.dump(summary, output_file, indent=2, allow_nan=False)
print(
    f"overall_status={summary['overall_status']} "
    f"source_sha256={source_hash} kernel_sha256={kernel_hash} "
    f"protocol_sha256={protocol_hash}"
)

if summary["overall_status"] != "PASS":
    raise SystemExit(1)
