# Extracted from Stevenkerman27/GRASS-plane/panel3d_kernels.py
# Source commit: aa6d81faf03c556db475d9edec6c025f90eb5fac
# SPDX-License-Identifier: Apache-2.0
# Modifications: package path / relative imports only.
"""Analytic constant-source and constant-doublet triangle influences.

This is the independent kernel for a differentiable thick-body 3D panel solver.
All vertices, field coordinates, and influence values remain PyTorch tensors.
Conventions:
  S = (1/4pi) int_T 1/|x-y| dS
  D = (1/4pi) int_T n(y) dot (x-y)/|x-y|**3 dS
Exterior limits are +1/2 for D when approaching an outward-facing panel
from the outer halfspace. Interior limit is -1/2.

The self boundary integral at an interior collocation point must use the
interior -1/2 jump, not the undefined pointwise limit of atan2(0, 0).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


def triangle_influence(
    field_points: Tensor,
    triangles: Tensor,
    *,
    interior_diagonal: bool = False,
) -> tuple[Tensor, Tensor]:
    """Return unit source/doublet potential matrices [P,T].

    field_points: [P,3]; triangles: [T,3,3] with consistent winding.
    Assumes strictly nondegenerate triangles, and no field point lies on
    an influencing triangle except aligned diagonal self interactions.
    For interior_diagonal=True, P must equal T and field points must
    be triangle centroids in the matching order. Compute all nonsingular
    off-diagonal interactions analytically and set self D=-1/2.
    """
    if field_points.ndim != 2 or field_points.shape[-1] != 3:
        raise ValueError("field_points must have shape [P,3]")
    if triangles.ndim != 3 or triangles.shape[-2:] != (3, 3):
        raise ValueError("triangles must have shape [T,3,3]")
    if field_points.dtype != triangles.dtype or field_points.device != triangles.device:
        raise ValueError("field_points and triangles require matching dtype/device")
    if interior_diagonal and field_points.shape[0] != triangles.shape[0]:
        raise ValueError("interior diagonal requires one matching field point per panel")

    a, b, c = triangles.unbind(1)
    raw_normal = torch.cross(b - a, c - a, dim=-1)
    twice_area = torch.linalg.vector_norm(raw_normal, dim=-1)
    if torch.any(twice_area <= 0) or not torch.isfinite(twice_area).all():
        raise ValueError("triangle surface must be finite and nondegenerate")
    n = raw_normal / twice_area[:, None]
    p = field_points[:, None, :]
    r0, r1, r2 = (a[None] - p, b[None] - p, c[None] - p)

    def length(x: Tensor) -> Tensor:
        return torch.linalg.vector_norm(x, dim=-1)

    L0, L1, L2 = length(r0), length(r1), length(r2)
    numerator = torch.sum(r0 * torch.cross(r1, r2, dim=-1), dim=-1)
    denominator = (
        L0 * L1 * L2
        + torch.sum(r0 * r1, dim=-1) * L2
        + torch.sum(r1 * r2, dim=-1) * L0
        + torch.sum(r2 * r0, dim=-1) * L1
    )
    solid_angle = -2 * torch.atan2(numerator, denominator)
    signed_normal_distance = -torch.sum(r0 * n[None], dim=-1)

    edge_sum = torch.zeros_like(solid_angle)
    for u, v in ((a, b), (b, c), (c, a)):
        edge = v - u
        edge_length = length(edge)
        tangent = edge / edge_length[:, None]
        outward = torch.cross(tangent, n, dim=-1)
        relative = u[None] - p
        t0 = torch.sum(relative * tangent[None], dim=-1)
        t1 = t0 + edge_length[None]
        perpendicular_squared = torch.sum(relative.square(), dim=-1) - t0.square()
        perpendicular_distance = torch.sqrt(perpendicular_squared.clamp_min(1e-26))
        line_integral = torch.asinh(t1 / perpendicular_distance) - torch.asinh(t0 / perpendicular_distance)
        edge_sum = edge_sum + torch.sum(relative * outward[None], dim=-1) * line_integral

    source = (edge_sum - signed_normal_distance * solid_angle) / (4 * math.pi)
    doublet = solid_angle / (4 * math.pi)
    if interior_diagonal:
        # No autograd-breaking in-place assignment on the original tensor.
        diagonal = torch.eye(len(field_points), device=doublet.device, dtype=doublet.dtype)
        doublet = doublet * (1 - diagonal) - 0.5 * diagonal
    return source, doublet
