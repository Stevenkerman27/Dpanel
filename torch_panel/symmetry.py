# Extracted from Stevenkerman27/GRASS-plane/panel3d_symmetry.py
# Source commit: aa6d81faf03c556db475d9edec6c025f90eb5fac
# SPDX-License-Identifier: Apache-2.0
# Modifications: package path / relative imports only.
"""Exact XZ-symmetry influence assembly for the 3D source-doublet panels.

For a flow symmetric about y=0, only the y>=0 panels are unknowns.
Every real panel has an equal-strength reflected source and doublet.
Reflection reverses handedness, so the mirrored triangle winding MUST be
reversed to obtain the correctly reflected outward normal.

For body panels: B_half = B_real + B_mirror,
                 D_half = D_real + D_mirror.
The interior -1/2 jump belongs ONLY to the real panel diagonal.
For lifting bodies, wake matrices must undergo the same reflection before
the trailing-edge Kutta jump map is applied.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .kernels import triangle_influence


def mirror_xz_triangles(triangles: Tensor) -> Tensor:
    """Reflect [N,3,3] triangles over y=0, preserving outward orientation.

    Input and output are differentiable in all vertex coordinates.
    Do NOT simply negate y without reversing vertex winding: the reflected
    doublet potential would then have the wrong sign.
    """
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("triangles must have shape [N,3,3]")
    reflected = torch.stack(
        (triangles[..., 0], -triangles[..., 1], triangles[..., 2]), dim=-1
    )
    return reflected.flip(dims=(1,))


def symmetric_triangle_influence(
    field_points: Tensor,
    half_triangles: Tensor,
    *,
    interior_diagonal: bool = False,
) -> tuple[Tensor, Tensor]:
    """Source and doublet influence [P,N] including mirrored source panels.

    `half_triangles` contains only positive-y surface panels (seam vertices
    may lie on y=0). For interior_diagonal, field_points must correspond to
    the real panel centroids, in the same order, and must have y>0.
    Evaluation field points may be anywhere off both physical surfaces.
    """
    if torch.any(half_triangles[..., 1] < -1e-12):
        raise ValueError("half_triangles must lie on y >= 0")
    if torch.any(half_triangles[..., 1].amax(dim=1) <= 0):
        raise ValueError("each half-model panel must have at least one vertex at y > 0; symmetry-plane caps are not physical panels")
    if interior_diagonal and torch.any(field_points[:, 1] <= 0):
        raise ValueError("interior collocation points must lie in y > 0")
    direct_source, direct_doublet = triangle_influence(
        field_points, half_triangles, interior_diagonal=interior_diagonal
    )
    mirror_source, mirror_doublet = triangle_influence(
        field_points, mirror_xz_triangles(half_triangles)
    )
    return direct_source + mirror_source, direct_doublet + mirror_doublet
