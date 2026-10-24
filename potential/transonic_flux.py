"""Independent Cartesian E-O face kernel and density-bias candidate."""
import numpy as np


def beta_from_speed_squared(speed_squared, mach_inf, gamma):
    return 1.0 - 0.5 * (gamma - 1.0) * mach_inf**2 * (speed_squared - 1.0)


def density_from_speed_squared(speed_squared, mach_inf, gamma):
    beta = beta_from_speed_squared(speed_squared, mach_inf, gamma)
    if np.any(beta <= 0.0):
        raise ValueError("Nonpositive Bernoulli factor.")
    return beta ** (1.0 / (gamma - 1.0))


def density_biased_h(q_right, q_left, mach_inf, gamma):
    qbar = 0.5 * (q_left + q_right)
    rho_left = density_from_speed_squared(q_left**2, mach_inf, gamma)
    rho_right = density_from_speed_squared(q_right**2, mach_inf, gamma)
    rho_bar = 0.5 * (rho_left + rho_right)
    beta_bar = beta_from_speed_squared(qbar**2, mach_inf, gamma)
    local_mach_squared = mach_inf**2 * qbar**2 / beta_bar
    inverse_local_mach_squared = np.divide(
        1.0,
        local_mach_squared,
        out=np.full_like(local_mach_squared, np.inf, dtype=float),
        where=local_mach_squared > 0.0,
    )
    nu = np.maximum(0.0, 1.0 - inverse_local_mach_squared)
    rho_face = rho_bar - 0.5 * nu * np.sign(qbar) * (rho_right - rho_left)
    return -qbar * rho_face


def eo_h(q1, q0, mach_inf, gamma, tangential_speed=0.0):
    q1 = np.asarray(q1, dtype=float)
    q0 = np.asarray(q0, dtype=float)
    tangential_squared = tangential_speed**2
    coefficient = 0.5 * (gamma - 1.0) * mach_inf**2
    beta_at_zero = 1.0 + coefficient * (1.0 - tangential_squared)
    if beta_at_zero <= 0.0:
        raise ValueError("Frozen tangential state is outside the Bernoulli domain.")
    qn_star_squared = beta_at_zero / (mach_inf**2 + coefficient)
    qn_star = np.sqrt(qn_star_squared)
    beta_star = beta_from_speed_squared(
        qn_star_squared + tangential_squared, mach_inf, gamma
    )
    rho_star = beta_star ** (1.0 / (gamma - 1.0))
    m_star = rho_star * qn_star
    beta1 = beta_from_speed_squared(q1**2 + tangential_squared, mach_inf, gamma)
    beta0 = beta_from_speed_squared(q0**2 + tangential_squared, mach_inf, gamma)
    if np.any(beta1 <= 0.0) or np.any(beta0 <= 0.0):
        raise ValueError("Face state has a nonpositive Bernoulli factor.")
    rho1 = beta1 ** (1.0 / (gamma - 1.0))
    rho0 = beta0 ** (1.0 / (gamma - 1.0))
    m1 = rho1 * q1
    m0 = rho0 * q0
    s1 = np.where(q1 > qn_star, m1 - m_star, 0.0)
    s0 = np.where(q0 > qn_star, m0 - m_star, 0.0)
    return -m1 + s1 - s0


def eo_cartesian_face_flux(gradient_left, gradient_right, area_vector, mach_inf, gamma):
    gradient_left = np.asarray(gradient_left, dtype=float)
    gradient_right = np.asarray(gradient_right, dtype=float)
    area_vector = np.asarray(area_vector, dtype=float)
    area = float(np.linalg.norm(area_vector))
    if area <= 0.0:
        raise ValueError("Face area must be positive.")
    normal = area_vector / area
    q_left = float(np.dot(gradient_left, normal))
    q_right = float(np.dot(gradient_right, normal))
    average_gradient = 0.5 * (gradient_left + gradient_right)
    q_average = 0.5 * (q_left + q_right)
    tangent = average_gradient - q_average * normal
    tangent_speed = float(np.linalg.norm(tangent))

    if q_average >= 0.0:
        h_value = eo_h(q_right, q_left, mach_inf, gamma, tangent_speed)
        return -float(h_value) * area

    h_reversed = eo_h(-q_left, -q_right, mach_inf, gamma, tangent_speed)
    return float(h_reversed) * area
