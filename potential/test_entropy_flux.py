"""Dense one-dimensional algebra checks for the E-O full-potential flux."""
import numpy as np

gamma = 1.4
endpoint_count = 257
interior_count = 65
tolerance = 2.0e-11
overall_pass = True

for mach_inf in (0.5, 0.8):
    q_star = np.sqrt((mach_inf**-2 + (gamma - 1.0) / 2.0) / ((gamma + 1.0) / 2.0))
    rho_star = (
        1.0 - (gamma - 1.0) / 2.0 * mach_inf**2 * (q_star**2 - 1.0)
    ) ** (1.0 / (gamma - 1.0))
    q_limit = np.sqrt(1.0 + 2.0 / ((gamma - 1.0) * mach_inf**2))
    q = np.sort(
        np.concatenate(
            (np.linspace(1.0e-8, q_limit * (1.0 - 1.0e-10), endpoint_count), [q_star])
        )
    )
    bernoulli = 1.0 - (gamma - 1.0) / 2.0 * mach_inf**2 * (q**2 - 1.0)
    rho = bernoulli ** (1.0 / (gamma - 1.0))
    mass_flux = rho * q
    split_flux = np.where(q <= q_star, 0.0, mass_flux - rho_star * q_star)

    q1 = q[:, None]
    q0 = q[None, :]
    mass1 = mass_flux[:, None]
    split1 = split_flux[:, None]
    split0 = split_flux[None, :]
    h = -mass1 + split1 - split0

    consistency_error = float(np.max(np.abs(np.diag(h) + mass_flux)))
    first_argument_increase = np.diff(h, axis=0)
    second_argument_increase = np.diff(h, axis=1)
    first_argument_violation = float(np.max(np.maximum(first_argument_increase, 0.0)))
    second_argument_violation = float(np.max(np.maximum(-second_argument_increase, 0.0)))

    intermediate_fraction = np.linspace(0.0, 1.0, interior_count)
    q_between = q0[:, :, None] + (q1 - q0)[:, :, None] * intermediate_fraction
    bernoulli_between = (
        1.0 - (gamma - 1.0) / 2.0 * mach_inf**2 * (q_between**2 - 1.0)
    )
    rho_between = bernoulli_between ** (1.0 / (gamma - 1.0))
    mass_between = rho_between * q_between
    signed_e_flux = np.sign(q1 - q0)[:, :, None] * (h[:, :, None] + mass_between)
    e_flux_violation = float(np.max(np.maximum(signed_e_flux, 0.0)))

    consistency_pass = consistency_error <= tolerance
    monotonicity_pass = (
        first_argument_violation <= tolerance
        and second_argument_violation <= tolerance
    )
    e_flux_pass = e_flux_violation <= tolerance
    mach_pass = consistency_pass and monotonicity_pass and e_flux_pass
    overall_pass = overall_pass and mach_pass

    print(
        f"M_inf={mach_inf:.1f} q_star={q_star:.12g} q_limit={q_limit:.12g} "
        f"consistency_error={consistency_error:.3e} "
        f"first_argument_monotonicity_violation={first_argument_violation:.3e} "
        f"second_argument_monotonicity_violation={second_argument_violation:.3e} "
        f"e_flux_positive_violation={e_flux_violation:.3e} "
        f"status={'PASS' if mach_pass else 'FAIL'}"
    )

if not overall_pass:
    raise SystemExit(1)

print("These one-dimensional algebra checks do not validate a transonic flow solution.")
