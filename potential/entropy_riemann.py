"""One-dimensional Riemann evolution for the E-O full-potential flux."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


def f_prime(q, mach_inf, gamma):
    coefficient = (gamma - 1.0) * mach_inf**2 / 2.0
    exponent = 1.0 / (gamma - 1.0)
    beta = 1.0 - coefficient * (q**2 - 1.0)
    derivative_factor = beta - 2.0 * coefficient * exponent * q**2
    return -(beta ** (exponent - 1.0)) * derivative_factor


def f_second(q, mach_inf, gamma):
    coefficient = (gamma - 1.0) * mach_inf**2 / 2.0
    exponent = 1.0 / (gamma - 1.0)
    beta = 1.0 - coefficient * (q**2 - 1.0)
    derivative_factor = beta - 2.0 * coefficient * exponent * q**2
    return (
        2.0
        * coefficient
        * q
        * beta ** (exponent - 2.0)
        * ((exponent - 1.0) * derivative_factor + (1.0 + 2.0 * exponent) * beta)
    )


gamma = 1.4
mesh_sizes = (64, 128, 256, 512)
cfl_target = 0.4
entropy_tolerance = 2.0e-12
mass_tolerance = 5.0e-12
bound_tolerance = 5.0e-13
output_dir = Path(__file__).with_name("results") / "entropy_v1"
output_dir.mkdir(parents=True, exist_ok=True)

solution_rows = []
history_rows = []
run_results = []
all_runs_pass = True

for mach_inf in (0.5, 0.8):
    coefficient = (gamma - 1.0) * mach_inf**2 / 2.0
    exponent = 1.0 / (gamma - 1.0)
    entropy_exponent = gamma / (gamma - 1.0)
    q_star = np.sqrt(
        (mach_inf**-2 + (gamma - 1.0) / 2.0) / ((gamma + 1.0) / 2.0)
    )
    beta_star = 1.0 - coefficient * (q_star**2 - 1.0)
    rho_star = beta_star**exponent
    mass_star = rho_star * q_star
    q_left_super = 1.2 * q_star
    beta_left_super = 1.0 - coefficient * (q_left_super**2 - 1.0)
    rho_left_super = beta_left_super**exponent
    mass_left_super = rho_left_super * q_left_super
    q_right_sub = brentq(
        lambda q: (1.0 - coefficient * (q**2 - 1.0)) ** exponent * q
        - mass_left_super,
        1.0e-12,
        q_star,
        xtol=1.0e-14,
        rtol=1.0e-14,
    )

    q_convexity = np.linspace(q_right_sub, q_left_super, 20001)
    f_second_values = f_second(q_convexity, mach_inf, gamma)
    minimum_f_second = float(np.min(f_second_values))
    if minimum_f_second <= 0.0:
        print(
            f"M_inf={mach_inf:.1f} precheck=FAIL "
            f"minimum_f_second={minimum_f_second:.12g}"
        )
        raise SystemExit(1)

    beta_right_sub = 1.0 - coefficient * (q_right_sub**2 - 1.0)
    rho_right_sub = beta_right_sub**exponent
    mass_right_sub = rho_right_sub * q_right_sub
    f_left_super = -mass_left_super
    f_right_sub = -mass_right_sub
    rh_speed = (f_right_sub - f_left_super) / (q_right_sub - q_left_super)
    mass_match_error = abs(mass_right_sub - mass_left_super)

    q_speed_samples = np.linspace(q_right_sub, q_left_super, 20001)
    f_prime_samples = f_prime(q_speed_samples, mach_inf, gamma)
    max_positive_speed = float(np.max(np.maximum(f_prime_samples, 0.0)))
    max_negative_speed = float(np.max(np.maximum(-f_prime_samples, 0.0)))
    speed_sum = max_positive_speed + max_negative_speed
    speed_left = float(f_prime(q_right_sub, mach_inf, gamma))
    speed_right = float(f_prime(q_left_super, mach_inf, gamma))
    final_time = 0.25 / max(abs(speed_left), abs(speed_right))
    beta_domain_limit = np.sqrt(1.0 + 1.0 / coefficient)
    if beta_left_super <= 0.0 or q_left_super >= beta_domain_limit:
        print(f"M_inf={mach_inf:.1f} precheck=FAIL nonpositive Bernoulli domain")
        raise SystemExit(1)

    for case_name in ("compression", "expansion"):
        if case_name == "compression":
            q_left_state = q_left_super
            q_right_state = q_right_sub
            left_inflow = True
            right_inflow = True
        else:
            q_left_state = q_right_sub
            q_right_state = q_left_super
            left_inflow = False
            right_inflow = False

        case_results = []

        for cell_count in mesh_sizes:
            edges = np.linspace(-1.0, 1.0, cell_count + 1)
            dx = float(edges[1] - edges[0])
            if edges[cell_count // 2] != 0.0:
                raise RuntimeError("The initial jump must lie exactly on a cell face.")
            centers = 0.5 * (edges[:-1] + edges[1:])
            q_values = np.where(centers < 0.0, q_left_state, q_right_state)

            initial_mass = float(np.sum(q_values) * dx)
            boundary_mass_change = 0.0
            max_entropy_violation = 0.0
            entropy_violation_count = 0
            cell_entropy_max_violation = np.zeros(cell_count)
            minimum_state = float(np.min(q_values))
            maximum_state = float(np.max(q_values))
            maximum_cfl = 0.0
            time_value = 0.0
            step_count = 0
            steps = []

            while time_value < final_time:
                dt_limit = cfl_target * dx / speed_sum
                dt = min(dt_limit, final_time - time_value)
                cfl_actual = dt / dx * speed_sum
                maximum_cfl = max(maximum_cfl, cfl_actual)

                if left_inflow:
                    left_ghost = q_left_state
                else:
                    left_ghost = float(q_values[0])
                if right_inflow:
                    right_ghost = q_right_state
                else:
                    right_ghost = float(q_values[-1])

                face_left = np.concatenate(([left_ghost], q_values))
                face_right = np.concatenate((q_values, [right_ghost]))
                beta_left = 1.0 - coefficient * (face_left**2 - 1.0)
                beta_right = 1.0 - coefficient * (face_right**2 - 1.0)
                rho_left = beta_left**exponent
                rho_right = beta_right**exponent
                mass_left = rho_left * face_left
                mass_right = rho_right * face_right
                split_left = np.where(
                    face_left <= q_star, 0.0, mass_left - mass_star
                )
                split_right = np.where(
                    face_right <= q_star, 0.0, mass_right - mass_star
                )

                # F(q_left,q_right) = h(q_right,q_left), with h = -mass flux.
                numerical_flux = -mass_right + split_right - split_left

                q_left_entropy = face_left
                q_right_entropy = face_right
                beta_left_entropy = beta_left
                beta_right_entropy = beta_right
                mass_left_entropy = mass_left
                mass_right_entropy = mass_right
                continuous_q_left = (
                    -q_left_entropy * mass_left_entropy
                    - beta_left_entropy**entropy_exponent
                    / (gamma * mach_inf**2)
                )
                continuous_q_right = (
                    -q_right_entropy * mass_right_entropy
                    - beta_right_entropy**entropy_exponent
                    / (gamma * mach_inf**2)
                )
                h_integral_left = np.where(
                    q_left_entropy <= q_star,
                    0.0,
                    q_left_entropy * mass_left_entropy
                    - q_star * mass_star
                    - (
                        beta_star**entropy_exponent
                        - beta_left_entropy**entropy_exponent
                    )
                    / (gamma * mach_inf**2),
                )
                h_integral_right = np.where(
                    q_right_entropy <= q_star,
                    0.0,
                    q_right_entropy * mass_right_entropy
                    - q_star * mass_star
                    - (
                        beta_star**entropy_exponent
                        - beta_right_entropy**entropy_exponent
                    )
                    / (gamma * mach_inf**2),
                )
                numerical_entropy_flux = (
                    -h_integral_left
                    + continuous_q_right
                    + h_integral_right
                )

                entropy_old = 0.5 * q_values**2
                q_next = q_values - (dt / dx) * (
                    numerical_flux[1:] - numerical_flux[:-1]
                )
                entropy_new = 0.5 * q_next**2
                entropy_change = entropy_new - entropy_old
                entropy_flux_divergence = (dt / dx) * (
                    numerical_entropy_flux[1:] - numerical_entropy_flux[:-1]
                )
                discrete_entropy_residual = (
                    entropy_change + entropy_flux_divergence
                )
                step_entropy_violation = float(
                    np.max(np.maximum(discrete_entropy_residual, 0.0))
                )
                max_entropy_violation = max(
                    max_entropy_violation, step_entropy_violation
                )
                entropy_violation_count += int(
                    np.count_nonzero(
                        discrete_entropy_residual > entropy_tolerance
                    )
                )
                cell_entropy_max_violation = np.maximum(
                    cell_entropy_max_violation,
                    np.maximum(discrete_entropy_residual, 0.0),
                )

                boundary_mass_change += dt * (
                    numerical_flux[0] - numerical_flux[-1]
                )
                time_value += dt
                step_count += 1
                minimum_state = min(minimum_state, float(np.min(q_next)))
                maximum_state = max(maximum_state, float(np.max(q_next)))
                q_values = q_next

                step_mass = float(np.sum(q_values) * dx)
                mass_balance = step_mass - initial_mass - boundary_mass_change
                steps.append(
                    {
                        "mach_inf": mach_inf,
                        "case": case_name,
                        "cells": cell_count,
                        "step": step_count,
                        "time": time_value,
                        "dt": dt,
                        "cfl": cfl_actual,
                        "mass_balance": mass_balance,
                        "max_entropy_positive_violation": step_entropy_violation,
                        "q_min": float(np.min(q_values)),
                        "q_max": float(np.max(q_values)),
                    }
                )

            quadrature_nodes, quadrature_weights = np.polynomial.legendre.leggauss(8)
            exact_cell_average = np.empty(cell_count)
            if case_name == "compression":
                exact_cell_average[:] = np.where(
                    centers < 0.0, q_left_super, q_right_sub
                )
            else:
                fan_left_speed = f_prime(q_right_sub, mach_inf, gamma)
                fan_right_speed = f_prime(q_left_super, mach_inf, gamma)
                for cell_index in range(cell_count):
                    x_quadrature = centers[cell_index] + 0.5 * dx * quadrature_nodes
                    q_quadrature = np.empty(len(quadrature_nodes))
                    for quadrature_index in range(len(quadrature_nodes)):
                        similarity_speed = x_quadrature[quadrature_index] / final_time
                        if similarity_speed <= fan_left_speed:
                            q_quadrature[quadrature_index] = q_right_sub
                        elif similarity_speed >= fan_right_speed:
                            q_quadrature[quadrature_index] = q_left_super
                        else:
                            q_quadrature[quadrature_index] = brentq(
                                lambda q: f_prime(q, mach_inf, gamma)
                                - similarity_speed,
                                q_right_sub,
                                q_left_super,
                                xtol=1.0e-13,
                                rtol=1.0e-13,
                            )
                    exact_cell_average[cell_index] = (
                        0.5 * np.dot(quadrature_weights, q_quadrature)
                    )

            absolute_error = np.abs(q_values - exact_cell_average)
            l1_error = float(np.sum(absolute_error) * dx)
            linf_error = float(np.max(absolute_error))
            state_jump = q_left_super - q_right_sub
            lower_transition = q_right_sub + 0.01 * state_jump
            upper_transition = q_right_sub + 0.99 * state_jump

            if case_name == "compression":
                transition_cells = (
                    (q_values > lower_transition)
                    & (q_values < upper_transition)
                )
                wave_width = float(np.count_nonzero(transition_cells) * dx)
                strongest_jump = int(np.argmax(np.abs(np.diff(q_values))))
                wave_position = float(edges[strongest_jump + 1])
                exact_wave_width = 0.0
            else:
                transition_cells = (
                    (q_values > lower_transition)
                    & (q_values < upper_transition)
                )
                if np.any(transition_cells):
                    transition_centers = centers[transition_cells]
                    wave_width = float(
                        transition_centers[-1]
                        - transition_centers[0]
                        + dx
                    )
                else:
                    wave_width = 0.0
                exact_wave_width = float(
                    final_time
                    * (
                        f_prime(upper_transition, mach_inf, gamma)
                        - f_prime(lower_transition, mach_inf, gamma)
                    )
                )
                sonic_crossings = np.where(
                    (q_values[:-1] - q_star) * (q_values[1:] - q_star) <= 0.0
                )[0]
                if len(sonic_crossings) == 0:
                    wave_position = float("nan")
                else:
                    crossing = int(sonic_crossings[0])
                    q0 = q_values[crossing]
                    q1 = q_values[crossing + 1]
                    x0 = centers[crossing]
                    x1 = centers[crossing + 1]
                    if q1 == q0:
                        wave_position = float(0.5 * (x0 + x1))
                    else:
                        wave_position = float(
                            x0 + (q_star - q0) * (x1 - x0) / (q1 - q0)
                        )

            final_mass = float(np.sum(q_values) * dx)
            mass_balance = final_mass - initial_mass - boundary_mass_change
            bound_violation = max(
                0.0,
                float(q_right_sub - minimum_state),
                float(maximum_state - q_left_super),
            )
            entropy_pass = bool(max_entropy_violation <= entropy_tolerance)
            mass_pass = bool(abs(mass_balance) <= mass_tolerance)
            bounded_pass = bool(bound_violation <= bound_tolerance)
            if case_name == "compression":
                wave_pass = bool(
                    abs(wave_position) <= 0.5 * dx
                    and wave_width <= 2.0 * dx
                )
            else:
                wave_pass = bool(abs(wave_position) <= 2.0 * dx)

            run_pass = bool(
                entropy_pass and mass_pass and bounded_pass and wave_pass
            )
            all_runs_pass = bool(all_runs_pass and run_pass)
            result = {
                "mach_inf": mach_inf,
                "case": case_name,
                "cells": cell_count,
                "gamma": gamma,
                "q_star": q_star,
                "q_left_super": q_left_super,
                "q_right_sub": q_right_sub,
                "mass_left": mass_left_super,
                "mass_right": mass_right_sub,
                "mass_match_error": mass_match_error,
                "rh_speed": rh_speed,
                "minimum_f_second": minimum_f_second,
                "fprime_left_state": (
                    f_prime(q_left_state, mach_inf, gamma)
                ),
                "fprime_right_state": (
                    f_prime(q_right_state, mach_inf, gamma)
                ),
                "max_positive_fprime": max_positive_speed,
                "max_negative_abs_fprime": max_negative_speed,
                "cfl_target": cfl_target,
                "maximum_cfl": maximum_cfl,
                "final_time": final_time,
                "exact_fan_left_edge": final_time * speed_left,
                "exact_fan_right_edge": final_time * speed_right,
                "exact_fan_inside_domain": bool(
                    final_time * speed_left > edges[0]
                    and final_time * speed_right < edges[-1]
                ),
                "steps": step_count,
                "l1_error": l1_error,
                "linf_error": linf_error,
                "wave_width_1pct_to_99pct": wave_width,
                "exact_wave_width_1pct_to_99pct": exact_wave_width,
                "wave_position": wave_position,
                "mass_initial": initial_mass,
                "mass_final": final_mass,
                "boundary_mass_change": boundary_mass_change,
                "mass_balance_residual": mass_balance,
                "minimum_state_seen": minimum_state,
                "maximum_state_seen": maximum_state,
                "bound_violation": bound_violation,
                "max_fully_discrete_entropy_positive_violation": (
                    max_entropy_violation
                ),
                "fully_discrete_entropy_violation_count": (
                    entropy_violation_count
                ),
                "entropy_pass": entropy_pass,
                "mass_pass": mass_pass,
                "bounded_pass": bounded_pass,
                "wave_pass": wave_pass,
                "status": "PASS" if run_pass else "FAIL",
            }
            run_results.append(result)
            case_results.append(result)
            history_rows.extend(steps)

            for cell_index in range(cell_count):
                solution_rows.append(
                    {
                        "mach_inf": mach_inf,
                        "case": case_name,
                        "cells": cell_count,
                        "x_center": centers[cell_index],
                        "q_initial": (
                            q_left_state if centers[cell_index] < 0.0 else q_right_state
                        ),
                        "q_numerical": q_values[cell_index],
                        "q_exact_cell_average": exact_cell_average[cell_index],
                        "cell_max_entropy_positive_violation": (
                            cell_entropy_max_violation[cell_index]
                        ),
                    }
                )

        if case_name == "expansion":
            previous_l1 = None
            for result in case_results:
                refinement_pass = bool(
                    previous_l1 is None
                    or result["l1_error"] <= previous_l1 + 1.0e-12
                )
                result["l1_refinement_pass"] = refinement_pass
                if not refinement_pass:
                    result["status"] = "FAIL"
                    all_runs_pass = bool(False)
                previous_l1 = result["l1_error"]
        else:
            for result in case_results:
                result["l1_refinement_pass"] = True

source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
protocol_sha256 = hashlib.sha256(
    Path(__file__).with_name("ENTROPY_PROTOCOL.md").read_bytes()
).hexdigest()
flux_test_sha256 = hashlib.sha256(
    Path(__file__).with_name("test_entropy_flux.py").read_bytes()
).hexdigest()

summary = {
    "model": "q_t + f(q)_x = 0; f(q) = -rho(q) q",
    "scope": (
        "One-dimensional auxiliary evolution only; not a 2D transonic-flow "
        "solver or certification."
    ),
    "gamma": gamma,
    "mach_numbers": [0.5, 0.8],
    "mesh_sizes": list(mesh_sizes),
    "cfl_target": cfl_target,
    "fully_discrete_entropy_tolerance": entropy_tolerance,
    "mass_balance_tolerance": mass_tolerance,
    "bound_tolerance": bound_tolerance,
    "artificial_clipping": False,
    "additional_artificial_dissipation": False,
    "source_sha256": source_sha256,
    "protocol_sha256": protocol_sha256,
    "flux_test_sha256": flux_test_sha256,
    "runs": run_results,
    "status": "PASS" if all_runs_pass else "FAIL",
}

with (output_dir / "summary.json").open("w", encoding="utf-8") as output_file:
    json.dump(summary, output_file, indent=2, allow_nan=False)

with (output_dir / "solutions.csv").open(
    "w", newline="", encoding="utf-8"
) as output_file:
    fieldnames = list(solution_rows[0])
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(solution_rows)

with (output_dir / "history.csv").open(
    "w", newline="", encoding="utf-8"
) as output_file:
    fieldnames = list(history_rows[0])
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(history_rows)

for result in run_results:
    print(
        f"M_inf={result['mach_inf']:.1f} case={result['case']} "
        f"N={result['cells']} steps={result['steps']} "
        f"L1={result['l1_error']:.6e} Linf={result['linf_error']:.6e} "
        f"width={result['wave_width_1pct_to_99pct']:.6e} "
        f"position={result['wave_position']:.6e} "
        f"mass_balance={result['mass_balance_residual']:.3e} "
        f"entropy_positive_violation="
        f"{result['max_fully_discrete_entropy_positive_violation']:.3e} "
        f"entropy_violations={result['fully_discrete_entropy_violation_count']} "
        f"bounded=[{result['minimum_state_seen']:.9g},"
        f"{result['maximum_state_seen']:.9g}] status={result['status']}"
    )

print(
    f"overall_status={summary['status']} "
    f"source_sha256={source_sha256} protocol_sha256={protocol_sha256}"
)

if not all_runs_pass:
    raise SystemExit(1)
