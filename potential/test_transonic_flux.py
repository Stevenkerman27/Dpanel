"""Independent density-bias experiment and Cartesian E-O face-kernel tests."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from transonic_flux import (
    density_biased_h,
    density_from_speed_squared,
    eo_cartesian_face_flux,
    eo_h,
)


gamma = 1.4
mach_numbers = (0.5, 0.8)
mesh_sizes = (64, 128, 256, 512)
tolerance = 2.0e-10
results_dir = Path(__file__).with_name("results") / "transonic_flux_v1"
results_dir.mkdir(parents=True, exist_ok=True)

candidate_results = []
kernel_results = []
overall_kernel_pass = True

for mach_inf in mach_numbers:
    coefficient = 0.5 * (gamma - 1.0) * mach_inf**2
    q_star = np.sqrt(
        (mach_inf**-2 + 0.5 * (gamma - 1.0))
        / (0.5 * (gamma + 1.0))
    )
    q_limit = np.sqrt(1.0 + 1.0 / coefficient)
    positive_states = np.linspace(1.0e-6, q_limit * (1.0 - 1.0e-9), 321)
    signed_states = np.concatenate((-positive_states[::-1], positive_states))

    q1 = signed_states[:, None]
    q0 = signed_states[None, :]
    candidate_h = density_biased_h(q1, q0, mach_inf, gamma)
    first_argument_increase = np.diff(candidate_h, axis=0)
    second_argument_increase = np.diff(candidate_h, axis=1)
    first_argument_violation = float(
        np.max(np.maximum(first_argument_increase, 0.0))
    )
    second_argument_violation = float(
        np.max(np.maximum(-second_argument_increase, 0.0))
    )
    beta_middle = 1.0 - coefficient * ((0.5 * (q1 + q0)) ** 2 - 1.0)
    if np.any(beta_middle <= 0.0):
        valid_pair = beta_middle > 0.0
    else:
        valid_pair = np.ones_like(beta_middle, dtype=bool)
    candidate_consistency = np.diag(
        density_biased_h(signed_states, signed_states, mach_inf, gamma)
        + density_from_speed_squared(signed_states**2, mach_inf, gamma)
        * signed_states
    )
    candidate_interior_violation = -np.inf
    candidate_counterexample = None
    interior_fractions = np.linspace(0.0, 1.0, 101)
    for left_index in range(len(signed_states)):
        for right_index in range(len(signed_states)):
            if not valid_pair[left_index, right_index]:
                continue
            middle = (
                signed_states[left_index]
                + interior_fractions
                * (signed_states[right_index] - signed_states[left_index])
            )
            beta_middle_values = 1.0 - coefficient * (middle**2 - 1.0)
            valid_middle = beta_middle_values > 0.0
            rho_middle = beta_middle_values[valid_middle] ** (
                1.0 / (gamma - 1.0)
            )
            f_middle = -rho_middle * middle[valid_middle]
            signed_e_flux = (
                np.sign(signed_states[left_index] - signed_states[right_index])
                * (
                    candidate_h[left_index, right_index]
                    - f_middle
                )
            )
            violation_index = int(np.argmax(signed_e_flux))
            violation = float(signed_e_flux[violation_index])
            if violation > candidate_interior_violation:
                candidate_interior_violation = violation
                candidate_counterexample = {
                    "q1_downstream": float(signed_states[left_index]),
                    "q0_upstream": float(signed_states[right_index]),
                    "q_intermediate": float(
                        middle[valid_middle][violation_index]
                    ),
                    "positive_e_flux_violation": violation,
                }

    candidate_consistency_error = float(np.max(np.abs(candidate_consistency)))
    candidate_e_pass = candidate_interior_violation <= tolerance
    candidate_monotonicity_pass = bool(
        first_argument_violation <= tolerance
        and second_argument_violation <= tolerance
    )

    smooth_errors = []
    for cell_count in mesh_sizes:
        dx = 2.0 * np.pi / cell_count
        centers = (np.arange(cell_count) + 0.5) * dx
        q_center = 1.0 + 0.08 * np.sin(centers)
        q_exact_face = 1.0 + 0.08 * np.sin(np.arange(cell_count) * dx)
        q_left = np.roll(q_center, 1)
        q_right = q_center
        candidate_face_h = density_biased_h(q_right, q_left, mach_inf, gamma)
        exact_face_h = -density_from_speed_squared(
            q_exact_face**2, mach_inf, gamma
        ) * q_exact_face
        numerical_divergence = (
            np.roll(candidate_face_h, -1) - candidate_face_h
        ) / dx
        beta_center = 1.0 - coefficient * (q_center**2 - 1.0)
        rho_center = beta_center ** (1.0 / (gamma - 1.0))
        exact_flux_derivative = (
            rho_center
            * (mach_inf**2 * q_center**2 / beta_center - 1.0)
            * 0.08
            * np.cos(centers)
        )
        smooth_errors.append(
            {
                "cells": cell_count,
                "max_face_flux_error": float(
                    np.max(np.abs(candidate_face_h - exact_face_h))
                ),
                "manufactured_residual_linf_error": float(
                    np.max(
                        np.abs(
                            numerical_divergence - exact_flux_derivative
                        )
                    )
                ),
            }
        )

    smooth_orders = []
    smooth_residual_orders = []
    for coarse, fine in zip(smooth_errors[:-1], smooth_errors[1:]):
        ratio = coarse["max_face_flux_error"] / fine["max_face_flux_error"]
        smooth_orders.append(float(np.log2(ratio)))
        residual_ratio = (
            coarse["manufactured_residual_linf_error"]
            / fine["manufactured_residual_linf_error"]
        )
        smooth_residual_orders.append(float(np.log2(residual_ratio)))
    smooth_pass = bool(
        min(smooth_orders) >= np.log2(3.2)
        and min(smooth_residual_orders) >= np.log2(3.0)
    )
    candidate_status = (
        "PASS"
        if candidate_e_pass
        and candidate_monotonicity_pass
        and smooth_pass
        and candidate_consistency_error <= tolerance
        else "REJECTED"
    )
    candidate_results.append(
        {
            "mach_inf": mach_inf,
            "gamma": gamma,
            "q_star": float(q_star),
            "consistency_error": candidate_consistency_error,
            "first_argument_monotonicity_violation": (
                first_argument_violation
            ),
            "second_argument_monotonicity_violation": (
                second_argument_violation
            ),
            "max_dense_interval_e_flux_positive_violation": (
                candidate_interior_violation
            ),
            "counterexample": candidate_counterexample,
            "smooth_face_errors": smooth_errors,
            "smooth_observed_orders": smooth_orders,
            "smooth_residual_observed_orders": smooth_residual_orders,
            "smooth_gate": "PASS" if smooth_pass else "FAIL",
            "monotonicity_gate": (
                "PASS" if candidate_monotonicity_pass else "FAIL"
            ),
            "e_flux_gate": "PASS" if candidate_e_pass else "FAIL",
            "status": (
                "PASS"
                if candidate_e_pass
                and candidate_monotonicity_pass
                and smooth_pass
                else "REJECTED"
            ),
        }
    )

    positive_lo = 1.0e-8
    positive_hi = q_limit * (1.0 - 1.0e-8)
    q_dense = np.linspace(positive_lo, positive_hi, 241)
    q1_dense = q_dense[:, None]
    q0_dense = q_dense[None, :]
    eo_dense_h = eo_h(q1_dense, q0_dense, mach_inf, gamma)
    q_intermediate = (
        q0_dense[:, :, None]
        + (q1_dense - q0_dense)[:, :, None]
        * np.linspace(0.0, 1.0, 81)
    )
    beta_intermediate = 1.0 - coefficient * (q_intermediate**2 - 1.0)
    rho_intermediate = beta_intermediate ** (1.0 / (gamma - 1.0))
    f_intermediate = -rho_intermediate * q_intermediate
    eo_violation = float(
        np.max(
            np.maximum(
                np.sign(q1_dense - q0_dense)[:, :, None]
                * (eo_dense_h[:, :, None] - f_intermediate),
                0.0,
            )
        )
    )
    eo_consistency = float(
        np.max(
            np.abs(
                eo_h(q_dense, q_dense, mach_inf, gamma)
                + density_from_speed_squared(q_dense**2, mach_inf, gamma)
                * q_dense
            )
        )
    )

    uniform_gradient = np.array([0.37, -0.21])
    square_area_vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    free_stream_fluxes = np.array(
        [
            eo_cartesian_face_flux(
                uniform_gradient,
                uniform_gradient,
                area_vector,
                mach_inf,
                gamma,
            )
            for area_vector in square_area_vectors
        ]
    )
    free_stream_residual = float(np.sum(free_stream_fluxes))

    angle = 0.731
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    gradient_left = np.array([1.36, 0.17])
    gradient_right = np.array([1.13, -0.12])
    area_vector = np.array([0.8, -0.6])
    original_flux = eo_cartesian_face_flux(
        gradient_left, gradient_right, area_vector, mach_inf, gamma
    )
    rotated_flux = eo_cartesian_face_flux(
        rotation @ gradient_left,
        rotation @ gradient_right,
        rotation @ area_vector,
        mach_inf,
        gamma,
    )
    rotation_error = float(abs(original_flux - rotated_flux))

    one_d_left = np.array([1.2 * q_star, 0.0])
    one_d_right = np.array([0.91 * q_star, 0.0])
    one_d_kernel = eo_cartesian_face_flux(
        one_d_left, one_d_right, np.array([1.0, 0.0]), mach_inf, gamma
    )
    one_d_reference = -float(
        eo_h(one_d_right[0], one_d_left[0], mach_inf, gamma)
    )
    one_d_error = abs(one_d_kernel - one_d_reference)
    reverse_kernel = eo_cartesian_face_flux(
        -one_d_right, -one_d_left, np.array([1.0, 0.0]), mach_inf, gamma
    )
    reverse_reference = float(
        eo_h(one_d_right[0], one_d_left[0], mach_inf, gamma)
    )
    reverse_error = abs(reverse_kernel - reverse_reference)

    kernel_pass = bool(
        eo_violation <= tolerance
        and eo_consistency <= tolerance
        and abs(free_stream_residual) <= 2.0e-12
        and rotation_error <= 2.0e-12
        and one_d_error <= 2.0e-12
        and reverse_error <= 2.0e-12
    )
    overall_kernel_pass = overall_kernel_pass and kernel_pass
    kernel_results.append(
        {
            "mach_inf": mach_inf,
            "gamma": gamma,
            "eo_dense_positive_state_e_flux_violation": eo_violation,
            "eo_consistency_error": eo_consistency,
            "cartesian_free_stream_closed_polygon_residual": free_stream_residual,
            "rotation_covariance_absolute_error": rotation_error,
            "positive_1d_degeneracy_error": one_d_error,
            "reverse_1d_degeneracy_error": reverse_error,
            "status": "PASS" if kernel_pass else "FAIL",
        }
    )

    print(
        f"M_inf={mach_inf:.1f} density_candidate={candidate_status} "
        f"E_violation={candidate_interior_violation:.6e} "
        f"monotonicity=[{first_argument_violation:.3e},"
        f"{second_argument_violation:.3e}] "
        f"smooth_gate={candidate_results[-1]['smooth_gate']} "
        f"EO_kernel={'PASS' if kernel_pass else 'FAIL'} "
        f"EO_E_violation={eo_violation:.3e} "
        f"free_stream={free_stream_residual:.3e} "
        f"rotation={rotation_error:.3e} "
        f"1D={one_d_error:.3e} reverse_1D={reverse_error:.3e}"
    )

source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
kernel_hash = hashlib.sha256(
    Path(__file__).with_name("transonic_flux.py").read_bytes()
).hexdigest()
protocol_hash = hashlib.sha256(
    Path(__file__).with_name("TRANSONIC_FLUX_PROTOCOL.md").read_bytes()
).hexdigest()
candidate_accepted = all(
    row["status"] == "PASS" for row in candidate_results
)
summary = {
    "scope": (
        "Independent flux unit tests only; no 2D transonic solve or "
        "certification."
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
    "candidate_density_bias": {
        "overall_status": (
            "ACCEPTED_FOR_NEXT_GATE"
            if candidate_accepted
            else "REJECTED"
        ),
        "runs": candidate_results,
    },
    "independent_eo_cartesian_face_kernel": {
        "overall_status": "PASS" if overall_kernel_pass else "FAIL",
        "runs": kernel_results,
    },
    "source_sha256": source_hash,
    "kernel_sha256": kernel_hash,
    "protocol_sha256": protocol_hash,
}
with (results_dir / "summary.json").open("w", encoding="utf-8") as output_file:
    json.dump(summary, output_file, indent=2, allow_nan=False)

with (results_dir / "candidate_violations.json").open(
    "w", encoding="utf-8"
) as output_file:
    json.dump(candidate_results, output_file, indent=2, allow_nan=False)

print(
    f"candidate_status={summary['candidate_density_bias']['overall_status']} "
    f"kernel_status={summary['independent_eo_cartesian_face_kernel']['overall_status']} "
    f"source_sha256={source_hash} kernel_sha256={kernel_hash} "
    f"protocol_sha256={protocol_hash}"
)

if not overall_kernel_pass:
    raise SystemExit(1)
