"""Cartesian finite-volume residual prototype for a scalar potential."""
import numpy as np

from transonic_flux import density_from_speed_squared, eo_h


def shared_face_flux(gradient_left, gradient_right, area_vector, mach_inf, gamma):
    gradient_left = np.asarray(gradient_left, dtype=float)
    gradient_right = np.asarray(gradient_right, dtype=float)
    area_vector = np.asarray(area_vector, dtype=float)
    face_area = float(np.linalg.norm(area_vector))
    normal = area_vector / face_area
    speed_left = np.linalg.norm(gradient_left, axis=-1)
    speed_right = np.linalg.norm(gradient_right, axis=-1)
    gradient_face = 0.5 * (gradient_left + gradient_right)
    speed_face = np.linalg.norm(gradient_face, axis=-1)
    normal_speed = np.einsum("...i,i->...", gradient_face, normal)
    direction_cosine = np.divide(
        normal_speed,
        speed_face,
        out=np.zeros_like(speed_face, dtype=float),
        where=speed_face > 0.0,
    )

    forward_h = eo_h(speed_right, speed_left, mach_inf, gamma)
    reverse_h = eo_h(speed_left, speed_right, mach_inf, gamma)
    face_flux = np.where(
        direction_cosine >= 0.0,
        -direction_cosine * forward_h,
        -direction_cosine * reverse_h,
    )
    return face_area * face_flux


def potential_grid_gradient(phi_periodic, freestream_gradient, dx, dy):
    phi_periodic = np.asarray(phi_periodic, dtype=float)
    freestream_gradient = np.asarray(freestream_gradient, dtype=float)
    gradient = np.empty(phi_periodic.shape + (2,), dtype=float)
    gradient[..., 0] = (
        (np.roll(phi_periodic, -1, axis=0) - np.roll(phi_periodic, 1, axis=0))
        / (2.0 * dx)
        + freestream_gradient[0]
    )
    gradient[..., 1] = (
        (np.roll(phi_periodic, -1, axis=1) - np.roll(phi_periodic, 1, axis=1))
        / (2.0 * dy)
        + freestream_gradient[1]
    )
    return gradient


def cartesian_potential_residual(
    gradient,
    dx,
    dy,
    mach_inf,
    gamma,
    basis_x=np.array([1.0, 0.0]),
    basis_y=np.array([0.0, 1.0]),
):
    basis_x = np.asarray(basis_x, dtype=float)
    basis_y = np.asarray(basis_y, dtype=float)
    left_gradient = np.asarray(gradient, dtype=float)
    right_x_gradient = np.roll(left_gradient, -1, axis=0)
    right_y_gradient = np.roll(left_gradient, -1, axis=1)
    x_flux = shared_face_flux(
        left_gradient,
        right_x_gradient,
        basis_x * dy,
        mach_inf,
        gamma,
    )
    y_flux = shared_face_flux(
        left_gradient,
        right_y_gradient,
        basis_y * dx,
        mach_inf,
        gamma,
    )
    cell_area = dx * dy
    residual = (
        x_flux
        - np.roll(x_flux, 1, axis=0)
        + y_flux
        - np.roll(y_flux, 1, axis=1)
    ) / cell_area
    return residual, x_flux, y_flux


def discrete_curl(gradient, dx, dy):
    gradient = np.asarray(gradient, dtype=float)
    derivative_y_of_x = (
        np.roll(gradient[..., 0], -1, axis=1)
        - np.roll(gradient[..., 0], 1, axis=1)
    ) / (2.0 * dy)
    derivative_x_of_y = (
        np.roll(gradient[..., 1], -1, axis=0)
        - np.roll(gradient[..., 1], 1, axis=0)
    ) / (2.0 * dx)
    return derivative_x_of_y - derivative_y_of_x


def analytic_potential_residual(x, y, mach_inf, gamma, base_gradient, epsilon):
    u = base_gradient[0] + epsilon * np.cos(x) * np.sin(y)
    v = base_gradient[1] + epsilon * np.sin(x) * np.cos(y)
    phi_xx = -epsilon * np.sin(x) * np.sin(y)
    phi_yy = -epsilon * np.sin(x) * np.sin(y)
    phi_xy = epsilon * np.cos(x) * np.cos(y)
    speed_squared = u**2 + v**2
    beta = 1.0 - 0.5 * (gamma - 1.0) * mach_inf**2 * (
        speed_squared - 1.0
    )
    rho = beta ** (1.0 / (gamma - 1.0))
    hessian_quadratic = (
        u**2 * phi_xx + 2.0 * u * v * phi_xy + v**2 * phi_yy
    )
    return (
        rho * (phi_xx + phi_yy)
        - mach_inf**2 * rho / beta * hessian_quadratic
    )


def oblique_projected_eo_h(
    qn_right, qn_left, tangential_speed, normal, mach_inf, gamma
):
    normal = np.asarray(normal, dtype=float)
    right_gradient = qn_right * normal
    left_gradient = qn_left * normal
    tangent = np.array([-normal[1], normal[0]]) * tangential_speed
    right_gradient = right_gradient + tangent
    left_gradient = left_gradient + tangent
    speed_right = float(np.linalg.norm(right_gradient))
    speed_left = float(np.linalg.norm(left_gradient))
    average_gradient = 0.5 * (right_gradient + left_gradient)
    speed_average = float(np.linalg.norm(average_gradient))
    cosine = float(np.dot(average_gradient, normal) / speed_average)
    if cosine >= 0.0:
        return cosine * float(eo_h(speed_right, speed_left, mach_inf, gamma))
    return cosine * float(eo_h(speed_left, speed_right, mach_inf, gamma))


def normal_sonic_speed(tangential_speed, mach_inf, gamma):
    coefficient = 0.5 * (gamma - 1.0) * mach_inf**2
    beta_at_zero_normal = 1.0 + coefficient * (1.0 - tangential_speed**2)
    return np.sqrt(beta_at_zero_normal / (mach_inf**2 + coefficient))


def normal_mass_flux(qn, tangential_speed, mach_inf, gamma):
    speed_squared = qn**2 + tangential_speed**2
    return density_from_speed_squared(speed_squared, mach_inf, gamma) * qn
