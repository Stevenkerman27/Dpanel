# Extracted from Stevenkerman27/GRASS-plane/panel3d_nonlifting.py
# Source commit: aa6d81faf03c556db475d9edec6c025f90eb5fac
# SPDX-License-Identifier: Apache-2.0
# Modifications: package path / relative imports only.
"""Differentiable low-order closed-body Dirichlet panel solve.

Non-lifting *validation primitive* only; finite-wing wake/Kutta and Cp are
separate, not-yet-implemented stages. Input faces must form an outward-oriented
closed triangular surface, with collocation points at triangle centroids.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from torch import Tensor

from .kernels import triangle_influence


def _face_components(vertex_count: int, faces: Tensor) -> Tensor:
    """Label each face with the closed component it belongs to (shared vertex)."""
    mesh = faces.detach().cpu().numpy()
    graph = coo_matrix(
        (np.ones(mesh.size, dtype=np.int8), (mesh.ravel(), mesh[:, [1, 2, 0]].ravel())),
        shape=(vertex_count, vertex_count),
    )
    _, vertex_labels = connected_components(graph, directed=False)
    _, inverse = np.unique(vertex_labels[mesh[:, 0]], return_inverse=True)
    return torch.from_numpy(inverse).to(faces.device)


def validate_closed_outward(vertices: Tensor, faces: Tensor) -> None:
    """Reject inputs the Dirichlet solve would otherwise accept silently.

    The formulation needs every face to be shared by exactly two faces running in
    opposite directions, and every closed component to be wound outward: an
    inward-wound or open surface still solves, but the doublet strengths are then
    the exterior problem or a meaningless system, so the potential is wrong with
    no residual signature.  Checks are non-differentiable by design.
    """
    start = torch.cat((faces[:, 0], faces[:, 1], faces[:, 2]))
    end = torch.cat((faces[:, 1], faces[:, 2], faces[:, 0]))
    key = torch.sort(torch.stack((start, end), dim=1), dim=1).values
    unique, inverse, counts = torch.unique(key, dim=0, return_inverse=True, return_counts=True)
    if bool((counts != 2).any()):
        bad = int(torch.nonzero(counts != 2)[0])
        raise ValueError(
            f"surface is not closed: edge {unique[bad].tolist()} is shared by "
            f"{int(counts[bad])} faces, a closed manifold needs exactly 2"
        )
    net_direction = torch.zeros(unique.shape[0], dtype=torch.long, device=faces.device)
    net_direction.index_add_(0, inverse, end - start)
    if bool((net_direction != 0).any()):
        bad = int(torch.nonzero(net_direction != 0)[0])
        raise ValueError(
            f"surface is inconsistently oriented: edge {unique[bad].tolist()} is traversed "
            "in the same direction by both of its faces"
        )

    points = vertices.detach()
    first, second, third = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    signed_six_volume = (first * torch.linalg.cross(second, third, dim=-1)).sum(-1)
    labels = _face_components(points.shape[0], faces)
    component_volume = torch.zeros(int(labels.max()) + 1, dtype=points.dtype, device=points.device)
    component_volume.index_add_(0, labels, signed_six_volume / 6)
    if bool((component_volume <= 0).any()):
        bad = int(torch.nonzero(component_volume <= 0)[0])
        raise ValueError(
            f"closed component {bad} is oriented inward (signed volume "
            f"{float(component_volume[bad]):.6g}); face winding must be outward"
        )


def closed_body_potential(
    vertices: Tensor,
    faces: Tensor,
    freestream: Tensor,
    evaluation_points: Tensor | None = None,
) -> dict[str, Tensor]:
    """Solve body doublet strengths from a uniform free stream.

    Uses unit source strength sigma=-Vinf dot n and internal perturbation
    Dirichlet potential zero: D_internal @ mu = B @ sigma.
    For an exterior point, perturbation potential is D @ mu - B @ sigma.
    This is NOT a lifting-wing solver, and does not enforce a Kutta condition.
    """
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must be [V,3]")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype != torch.long:
        raise ValueError("faces must be int64 [N,3]")
    if freestream.shape != (3,):
        raise ValueError("freestream must be [3]")
    validate_closed_outward(vertices, faces)
    triangles = vertices[faces]
    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    normal_area = torch.cross(edges_a, edges_b, dim=-1)
    double_area = torch.linalg.vector_norm(normal_area, dim=-1)
    if torch.any(double_area <= 0):
        raise ValueError("degenerate triangle")
    normals = normal_area / double_area[:, None]
    centers = triangles.mean(dim=1)
    sigma = -(normals * freestream).sum(-1)
    B, D = triangle_influence(centers, triangles, interior_diagonal=True)
    rhs = B @ sigma
    mu = torch.linalg.solve(D, rhs)
    result = {
        "centers": centers,
        "normals": normals,
        "areas": 0.5 * double_area,
        "sigma": sigma,
        "mu": mu,
        "residual": D @ mu - rhs,
    }
    if evaluation_points is not None:
        SB, DB = triangle_influence(evaluation_points, triangles)
        result["perturbation_potential"] = DB @ mu - SB @ sigma
    return result
