"""Offline comparison of three face-gradient reconstructions (issue #9).

Identical geometry, field and stencil; nothing is truncated, clipped, smoothed,
averaged or regularized. Three arms:

  normal    the production unweighted normal equations, (DtD) c = Dt dm
  qr        the same unweighted fit by untruncated QR (mathematically equal)
  weighted  untruncated QR on rows scaled by inverse distance to the face center

Rank is diagnosed from the singular values of the design matrix and reported,
not repaired: a face whose stencil is rank deficient is counted and skipped.
This is a postprocessing comparison, never a flow solve and never a
physical-accuracy claim.
"""
from __future__ import annotations

import numpy as np

from .pressure_recovery_ab import mesh_geometry, surface_cp


def face_frames(vertices, faces):
    """Triangle centers, unit normals and the two in-plane tangent directions."""
    _, f, tri, centers, _, _, normal = mesh_geometry(vertices, faces)
    tangent = tri[:, 1]-tri[:, 0]
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    second = np.cross(normal, tangent)
    return centers, normal, tangent, second


def two_ring_neighbors(faces, cut_edges, rings=2):
    """Face stencils on the cut-free edge graph; cuts never connect faces."""
    f = np.asarray(faces, dtype=np.int64)
    cuts = {tuple(sorted(map(int, e))) for e in np.asarray(cut_edges).reshape(-1, 2)}
    edges = {}
    for i, face in enumerate(f):
        for a, b in zip(face, np.roll(face, -1)):
            edges.setdefault(tuple(sorted((int(a), int(b)))), []).append(i)
    adjacency = [set() for _ in f]
    for edge, owners in edges.items():
        if len(owners) == 2 and edge not in cuts:
            i, j = owners
            adjacency[i].add(j)
            adjacency[j].add(i)
    ids = []
    for i in range(len(f)):
        shell, frontier, seen = {i}, {i}, {i}
        for _ in range(rings):
            frontier = set().union(*(adjacency[k] for k in frontier))-seen
            seen |= frontier
            shell |= frontier
        ids.append(np.array(sorted(shell), dtype=np.int64))
    return ids


def _design(centers, tangent, second, i, js, weights):
    d = centers[js]-centers[i]
    design = np.column_stack((d@tangent[i], d@second[i]))
    if weights is None:
        return design
    return design*weights[:, None]


def fit_gradients(vertices, faces, mu, neighbor_ids, *, mode="normal",
                  min_ratio=1e-6):
    """Return (gradient, singular_ratio, rejected) for one reconstruction arm."""
    centers, normal, tangent, second = face_frames(vertices, faces)
    mu = np.asarray(mu, dtype=np.float64)
    gradient = np.zeros((len(faces), 3))
    ratio = np.full(len(faces), np.nan)
    rejected = 0
    for i, js in enumerate(neighbor_ids):
        d = centers[js]-centers[i]
        distance = np.linalg.norm(d, axis=1)
        if mode == "weighted":
            floor = 1e-3*np.median(distance[distance > 0]) if np.any(distance > 0) else 1.0
            weights = 1.0/np.maximum(distance, floor)
            values = (mu[js]-mu[i])*weights
        else:
            weights, values = None, mu[js]-mu[i]
        design = _design(centers, tangent, second, i, js, weights)
        if len(design) < design.shape[1]:
            rejected += 1
            continue
        s = np.linalg.svd(design, compute_uv=False)
        ratio[i] = float(s[-1]/s[0]) if s[0] > 0 else 0.0
        if not ratio[i] >= min_ratio:
            rejected += 1
            continue
        if mode == "normal":
            coeff = np.linalg.solve(design.T@design, design.T@values)
        else:
            q, r = np.linalg.qr(design, mode="reduced")
            coeff = np.linalg.solve(r, q.T@values)
        gradient[i] = coeff[0]*tangent[i]+coeff[1]*second[i]
    return gradient, ratio, rejected


def tangential_gradient(gradient, normal):
    """Remove any normal component so the Cp convention sees a surface gradient."""
    g = np.asarray(gradient, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    return g-n*(np.sum(g*n, axis=1))[:, None]


def cp(gradient, normal, freestream):
    return surface_cp(tangential_gradient(gradient, normal), normal, freestream)
