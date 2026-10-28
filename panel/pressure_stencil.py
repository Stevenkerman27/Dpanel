"""Rank-aware, cut-preserving neighborhood selection for raw panel pressure.

This runs once on the host BEFORE automatic differentiation. It changes only
the frozen neighbor index arrays for faces whose existing two-ring stencil
fails the ORIGINAL rank check. The pressure recovery formula is unchanged.
No cross-cut edges, pseudo-inverse, regularization or rank tolerance relaxation.
"""
from __future__ import annotations

import numpy as np


def adaptive_pressure_neighbors(centers, triangles, adjacency, row_count, *,
                                initial_rings=2, max_rings=6,
                                min_singular_ratio=1e-6):
    """Expand BFS pressure stencils only until each local 2D fit is full rank.

    adjacency is the frozen edge graph *after* all TE and wall-contact wake
    cuts have been removed. Right-side rows may contain mirrored ghost faces.
    The first valid BFS shell wins; neighboring cut banks are never connected
    by an artificial edge. If a cut-isolated strip is still 1D at the allowed
    maximum hop depth, the input is rejected with the affected face ID.
    """
    centers = np.asarray(centers, dtype=np.float64)
    tri = np.asarray(triangles, dtype=np.float64)
    if (centers.ndim != 2 or centers.shape[1] != 3
            or tri.shape != (len(centers), 3, 3)
            or len(adjacency) != len(centers)
            or not 0 < row_count <= len(centers)
            or not 1 <= initial_rings <= max_rings
            or not 0 < min_singular_ratio < 1):
        raise ValueError("Invalid cut-aware stencil configuration")
    if not np.isfinite(centers).all() or not np.isfinite(tri).all():
        raise ValueError("Nonfinite pressure geometry")

    edges = tri[:, 1]-tri[:, 0]
    lens = np.linalg.norm(edges, axis=1)
    cross = np.cross(edges, tri[:, 2]-tri[:, 0])
    normal_len = np.linalg.norm(cross, axis=1)
    if np.any(lens <= 0) or np.any(normal_len <= 0):
        raise ValueError("Degenerate pressure faces")
    tangent = edges/lens[:, None]
    secondary = np.cross(cross/normal_len[:, None], tangent)

    selected = []
    depths = []
    initial_ratios = []
    final_ratios = []
    initial_bad = []
    expanded = []
    for i in range(row_count):
        seen = {i}
        frontier = {i}
        start_ratio = None
        for depth in range(1, max_rings+1):
            next_layer = set()
            for k in frontier:
                next_layer.update(adjacency[k])
            frontier = next_layer-seen
            seen.update(frontier)
            if depth < initial_rings:
                continue
            ids = sorted(seen-{i})
            if len(ids) >= 2:
                delta = centers[ids]-centers[i]
                design = np.column_stack(
                    (delta@tangent[i], delta@secondary[i]))
                singular = np.linalg.svd(design, compute_uv=False)
                ratio = (float(singular[-1]/singular[0])
                         if singular[0] > 0 else 0.)
            else:
                ratio = 0.
            if start_ratio is None:
                start_ratio = ratio
            if ratio >= min_singular_ratio:
                selected.append(ids)
                depths.append(depth)
                initial_ratios.append(start_ratio)
                final_ratios.append(ratio)
                if start_ratio < min_singular_ratio:
                    initial_bad.append(i)
                    expanded.append(i)
                break
            if not frontier or depth == max_rings:
                raise ValueError(
                    "Rank-deficient cut-aware pressure stencil "
                    f"face={i}, initial_ratio={start_ratio:.3e}, "
                    f"last_ratio={ratio:.3e}, neighbors={len(ids)}, "
                    f"last_depth={depth}, max_rings={max_rings}; "
                    "do not cross true TE/contact cut or relax the rank gate")
    width = max(map(len, selected))
    indices = np.asarray(
        [neighbors+[i]*(width-len(neighbors))
         for i, neighbors in enumerate(selected)],
        dtype=np.int32)
    return indices, dict(
        initial_rank_deficient_faces=initial_bad,
        expanded_faces=expanded,
        selected_ring_depth=depths,
        initial_singular_ratios=initial_ratios,
        final_singular_ratios=final_ratios,
        pressure_neighbor_width=width,
        initial_stencil_rings=initial_rings,
        maximum_stencil_rings=max_rings,
        minimum_required_singular_ratio=min_singular_ratio,
    )
