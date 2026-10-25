"""Analytic constant-triangle potentials, following core/triangle_kernel.py.

No collocation displacement, vortex core, cutoff, or small denominator is used.
The diagonal double-layer trace is the analytic interior value -1/2.
"""
import jax.numpy as jnp


def triangle_potential(targets, triangles, interior_diagonal=False, source=True, linear_y=False):
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    area_vector = jnp.cross(edge1, edge2)
    normals = area_vector / jnp.linalg.norm(area_vector, axis=1)[:, None]
    r = triangles[None] - targets[:, None, None, :]
    radius = jnp.linalg.norm(r, axis=3)
    r0, r1, r2 = r[:, :, 0], r[:, :, 1], r[:, :, 2]
    determinant = jnp.sum(r0 * jnp.cross(r1, r2), axis=2)
    denominator = (
        radius[:, :, 0] * radius[:, :, 1] * radius[:, :, 2]
        + jnp.sum(r0 * r1, axis=2) * radius[:, :, 2]
        + jnp.sum(r1 * r2, axis=2) * radius[:, :, 0]
        + jnp.sum(r2 * r0, axis=2) * radius[:, :, 1]
    )
    if interior_diagonal:
        diagonal = jnp.eye(len(targets), dtype=bool)
        determinant = jnp.where(diagonal, 0., determinant)
        denominator = jnp.where(diagonal, 1., denominator)
    omega = -2 * jnp.arctan2(determinant, denominator)
    doublet = omega / (4 * jnp.pi)
    if interior_diagonal:
        doublet = jnp.where(diagonal, -.5, doublet)
    if not source and not linear_y:
        return doublet
    height = -jnp.sum(r0 * normals[None], axis=2)
    edge_sum = jnp.zeros_like(height)
    gradient_y = jnp.zeros_like(height)
    for k, q in ((0, 1), (1, 2), (2, 0)):
        edge = triangles[:, q] - triangles[:, k]
        length = jnp.linalg.norm(edge, axis=1)
        outward = jnp.cross(edge / length[:, None], normals)
        cross = jnp.cross(r[:, :, k], r[:, :, q])
        dot = jnp.sum(r[:, :, k] * r[:, :, q], axis=2)
        product = radius[:, :, k] * radius[:, :, q]
        opposite = dot < 0
        # The inactive division uses 1 exactly; no small number changes the kernel.
        plus = jnp.where(opposite, jnp.sum(cross * cross, axis=2)
                         / jnp.where(opposite, product - dot, 1.), product + dot)
        gap = 2 * plus / (radius[:, :, k] + radius[:, :, q] + length[None])
        line_integral = jnp.log1p(2 * length[None] / gap)
        edge_sum += jnp.sum(r[:, :, k] * outward[None], axis=2) * line_integral
        gradient_y -= outward[None, :, 1] * line_integral
    if linear_y:
        # Integral (y_source-y_centroid)*dG/dn = (y_target-y_centroid)*D + h*dS/dy.
        # Wake normals are +z, so dS/dy is purely tangential.
        moment_y = (targets[:, None, 1] - triangles.mean(axis=1)[None, :, 1]) * doublet
        moment_y += height * gradient_y / (4 * jnp.pi)
        return doublet, moment_y
    single = (edge_sum - height * omega) / (4 * jnp.pi)
    return single, doublet
