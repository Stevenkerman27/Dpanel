"""Analytic constant-triangle potentials, following core/triangle_kernel.py.

No collocation displacement, vortex core, cutoff, or small denominator is used.
The diagonal double-layer trace is the analytic interior value -1/2.

Targets are evaluated in fixed-size blocks (PANEL_KERNEL_BLOCK, default 64)
so the temporary r array is O(block*panels*3*3) instead of O(panels^2*9).
The arithmetic per target is unchanged; only the loop structure differs.
"""
import os

import jax
import jax.numpy as jnp

_BLOCK = int(os.environ.get("PANEL_KERNEL_BLOCK", "64"))
_CHECKPOINT = os.environ.get("PANEL_KERNEL_CHECKPOINT", "1") != "0"


def triangle_potential(targets, triangles, interior_diagonal=False, source=True,
                       linear_y=False, block_size=None, linear_direction=None):
    """Return single- and double-layer potentials of triangles at targets.

    block_size defaults to PANEL_KERNEL_BLOCK (64); 0 disables blocking.
    linear_direction is a per-triangle material-coordinate gradient.
    """
    block = _BLOCK if block_size is None else int(block_size)
    count = targets.shape[0]
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    area_vector = jnp.cross(edge1, edge2)
    normals = area_vector / jnp.linalg.norm(area_vector, axis=1)[:, None]
    triangle_index = jnp.arange(triangles.shape[0])
    if linear_direction is not None:
        linear_y = True
    direction = jnp.array([0., 1., 0.]) if linear_direction is None else linear_direction

    def evaluate(block_targets, rows):
        r = triangles[None] - block_targets[:, None, None, :]
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
            diagonal = rows[:, None] == triangle_index[None, :]
            determinant = jnp.where(diagonal, 0., determinant)
            denominator = jnp.where(diagonal, 1., denominator)
        omega = -2 * jnp.arctan2(determinant, denominator)
        doublet = omega / (4 * jnp.pi)
        if interior_diagonal:
            doublet = jnp.where(diagonal, -.5, doublet)
        if not source and not linear_y:
            return (doublet,)
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
            gradient_y -= jnp.sum(outward * direction, axis=1)[None] * line_integral
        if linear_y:
            # M_eta = ((target-centroid).grad_eta)*D + h*(grad_S.grad_eta).
            relative = block_targets[:, None] - triangles.mean(axis=1)[None]
            moment_y = jnp.sum(relative * direction, axis=2) * doublet
            moment_y += height * (gradient_y / (4 * jnp.pi) - jnp.sum(normals * direction, axis=1)[None] * doublet)
            return (doublet, moment_y)
        single = (edge_sum - height * omega) / (4 * jnp.pi)
        return (single, doublet)

    if block <= 0 or count <= block:
        values = evaluate(targets, jnp.arange(count))
    else:
        padding = (-count) % block
        padded = targets if padding == 0 else jnp.concatenate(
            [targets, jnp.repeat(targets[-1:], padding, axis=0)], axis=0)
        padded_count = padded.shape[0]

        def one_block(start):
            rows = start + jnp.arange(block)
            return evaluate(jax.lax.dynamic_slice_in_dim(padded, start, block, axis=0), rows)

        body = jax.checkpoint(one_block) if _CHECKPOINT else one_block
        stacked = jax.lax.map(body, jnp.arange(0, padded_count, block))
        values = tuple(
            value.reshape((padded_count,) + value.shape[2:])[:count] for value in stacked)
    return values[0] if len(values) == 1 else (values[0], values[1])
