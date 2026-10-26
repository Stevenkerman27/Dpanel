"""Research-only, cut-aware nodal potential recovery on a frozen panel field.

No flow solve, Cp clipping, pressure averaging, pseudoinverse or regularization.
Vertex values are separate on each incident face fan divided by true cut edges.
A full-rank affine QR fit recovers each value, followed by a P1 surface gradient
(equivalently the triangle Green-Gauss edge integral). This is NOT the analytic
trace of the piecewise-constant panel field and not a physical-accuracy claim.
"""
from __future__ import annotations

import numpy as np


def mesh_geometry(vertices, faces):
    x = np.asarray(vertices, dtype=np.float64)
    raw = np.asarray(faces)
    if (x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all()
            or raw.ndim != 2 or raw.shape[1] != 3 or len(raw) == 0
            or not np.issubdtype(raw.dtype, np.integer)):
        raise ValueError("Finite (V,3) coordinates and integer (F,3) faces required")
    f = raw.astype(np.int64, copy=False)
    if f.min() < 0 or f.max() >= len(x):
        raise ValueError("Face index out of range")
    tri = x[f]
    av = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])/2
    area = np.linalg.norm(av, axis=1)
    if np.any(area <= 0) or not np.isfinite(area).all():
        raise ValueError("Degenerate triangle")
    return x, f, tri, tri.mean(axis=1), av, area, av/area[:, None]


def _values(mu, nf):
    v = np.asarray(mu, dtype=np.float64)
    if v.shape != (nf,) or not np.isfinite(v).all():
        raise ValueError("One finite scalar potential per face required")
    return v


def _qr(design, values, min_ratio):
    """SVD diagnoses rank only. All coefficients are solved by untruncated QR."""
    if len(design) < design.shape[1]:
        return None, 0.
    s = np.linalg.svd(design, compute_uv=False)
    ratio = float(s[-1]/s[0]) if s[0] > 0 else 0.
    if ratio < min_ratio:
        return None, ratio
    q, r = np.linalg.qr(design, mode="reduced")
    return np.linalg.solve(r, q.T@values), ratio


def affine_face_gradient(vertices, faces, mu, neighbor_ids, *, min_ratio=1e-6):
    """Equivalent QR control for the original unweighted 2-column face fit.

    neighbor_ids may have fewer rows than faces, e.g. right rows in a mirrored
    full geometry. Repeated self IDs retain the original zero contributions.
    """
    _, f, tri, centers, _, _, normal = mesh_geometry(vertices, faces)
    mu = _values(mu, len(f))
    ids = np.asarray(neighbor_ids)
    if (ids.ndim != 2 or not np.issubdtype(ids.dtype, np.integer)
            or len(ids) > len(f) or ids.size == 0
            or ids.min() < 0 or ids.max() >= len(f)
            or not 0 < min_ratio < 1):
        raise ValueError("Invalid frozen neighbor indices or rank threshold")
    tangent = tri[:, 1]-tri[:, 0]
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    second = np.cross(normal, tangent)
    grad = np.empty((len(ids), 3))
    for i, js in enumerate(ids):
        d = centers[js]-centers[i]
        design = np.column_stack((d@tangent[i], d@second[i]))
        coeff, ratio = _qr(design, mu[js]-mu[i], min_ratio)
        if coeff is None:
            raise ValueError(f"Face {i}: rank-deficient baseline, ratio={ratio}")
        grad[i] = coeff[0]*tangent[i]+coeff[1]*second[i]
    return grad


def cut_vertex_gradient(vertices, faces, mu, cut_edges, *, max_rings=4,
                        min_ratio=1e-6):
    """Recover face-corner potentials without crossing the declared cuts.

    Fits use only geometry to choose patches. Start at an incident vertex fan;
    expand on the cut-free face graph only on rank failure, never enter another
    fan incident at that vertex. Return every patch's depth/rank/support audit.
    Curved patches have truncation error; planar affine reproduction is exact.
    """
    x, f, tri, centers, av, area, normal = mesh_geometry(vertices, faces)
    mu = _values(mu, len(f))
    if (not isinstance(max_rings, (int, np.integer)) or max_rings < 0
            or not 0 < min_ratio < 1):
        raise ValueError("Nonnegative integer max_rings and 0<min_ratio<1 required")
    raw_cuts = np.asarray(cut_edges)
    if raw_cuts.size == 0:
        raw_cuts = np.empty((0, 2), dtype=np.int64)
    if (raw_cuts.ndim != 2 or raw_cuts.shape[1] != 2
            or not np.issubdtype(raw_cuts.dtype, np.integer)):
        raise ValueError("cut_edges must be integer pairs")
    cuts = {tuple(sorted(map(int, e))) for e in raw_cuts}
    edges, incident = {}, [set() for _ in x]
    for i, face in enumerate(f):
        for vertex in face:
            incident[int(vertex)].add(i)
        for a, b in zip(face, np.roll(face, -1)):
            edges.setdefault(tuple(sorted((int(a), int(b)))), []).append(i)
    if not cuts.issubset(edges):
        raise ValueError("Cut is not a real wall edge")
    if any(len(v) > 2 for v in edges.values()):
        raise ValueError("Non-manifold wall edge")
    adjacency = [set() for _ in f]
    # Per-vertex adjacency uses only the two faces meeting at that vertex edge.
    fans_graph = [{i: set() for i in fs} for fs in incident]
    for edge, fs in edges.items():
        if len(fs) == 2 and edge not in cuts:
            i, j = fs
            adjacency[i].add(j)
            adjacency[j].add(i)
            for vertex in edge:
                fans_graph[vertex][i].add(j)
                fans_graph[vertex][j].add(i)
    corner_mu = np.empty((len(f), 3))
    audit = []
    for vertex, fs in enumerate(incident):
        remaining = set(fs)
        while remaining:
            root = min(remaining)
            fan, front = {root}, {root}
            while front:
                front = set().union(*(fans_graph[vertex][i] for i in front))-fan
                fan.update(front)
            remaining.difference_update(fan)
            initial = np.array(sorted(fan), dtype=int)
            n = av[initial].sum(axis=0)
            length = np.linalg.norm(n)
            if length <= 1e-12*area[initial].sum():
                raise ValueError(f"Vertex {vertex}: ambiguous fan normal")
            n /= length
            axis = np.eye(3)[np.argmin(np.abs(n))]
            t1 = np.cross(n, axis)
            t1 /= np.linalg.norm(t1)
            t2 = np.cross(n, t1)
            forbidden = fs-fan
            support, front = set(fan), set(fan)
            coeff, ratio = None, 0.
            for depth in range(max_rings+1):
                js = np.array(sorted(support), dtype=int)
                d = centers[js]-x[vertex]
                h = float(np.linalg.norm(d, axis=1).max())
                if h > 0:
                    design = np.column_stack((np.ones(len(js)), d@t1/h, d@t2/h))
                    # Subtract a local value to preserve potential-gauge accuracy.
                    offset = mu[js[0]]
                    coeff, ratio = _qr(design, mu[js]-offset, min_ratio)
                if coeff is not None:
                    break
                if depth == max_rings:
                    break
                front = set().union(*(adjacency[i] for i in front))-support-forbidden
                if not front:
                    break
                support.update(front)
            if coeff is None:
                raise ValueError(f"Vertex {vertex}, fan {sorted(fan)}: rank deficient; "
                                 f"ratio={ratio}, support={len(support)}, depth={depth}")
            value = float(offset+coeff[0])
            for i in fan:
                corner_mu[i, np.flatnonzero(f[i] == vertex)[0]] = value
            audit.append(dict(vertex=int(vertex), fan_faces=sorted(fan),
                              support_faces=sorted(support), depth=int(depth),
                              singular_ratio=ratio))
    # P1 triangle gradient; constant gauge cancels before edge summation.
    g = np.zeros((len(f), 3))
    shifted = corner_mu-corner_mu[:, :1]
    for k in range(3):
        edge = tri[:, (k+2) % 3]-tri[:, (k+1) % 3]
        g += shifted[:, k, None]*np.cross(normal, edge)/(2*area[:, None])
    return g, corner_mu, audit


def surface_cp(gradient_mu, normals, freestream):
    """Same tangential surface-potential convention as production, not raw field."""
    g, n, flow = (np.asarray(a, dtype=float) for a in
                  (gradient_mu, normals, freestream))
    if (g.shape != n.shape or g.ndim != 2 or g.shape[1] != 3
            or flow.shape != (3,) or not np.isfinite(g).all()
            or not np.isfinite(n).all() or not np.isfinite(flow).all()
            or np.linalg.norm(flow) <= 0
            or not np.allclose(np.linalg.norm(n, axis=1), 1., atol=1e-12, rtol=1e-12)):
        raise ValueError("Finite tangential gradient, unit normals and nonzero flow required")
    if np.max(np.abs(np.sum(g*n, axis=1))) > 1e-10*max(1., float(np.max(np.abs(g)))):
        raise ValueError("Recovery returned a nontangential surface gradient")
    velocity = flow-n*(n@flow)[:, None]+g
    return 1-np.sum(velocity**2, axis=1)/np.dot(flow, flow)


def mirror_half_wall(vertices, faces, cut_edges):
    """Diagnostic O(N) exact reflection; no node welding or left state solve."""
    x, f, *_ = mesh_geometry(vertices, faces)
    # Accepted mesher explicitly snaps the seam. Do not guess or weld near nodes.
    if np.any(x[:, 1] < 0) or not np.any(x[f, 1] == 0):
        raise ValueError("Need nonnegative half coordinates and exact y=0 seam")
    other = np.flatnonzero(x[:, 1] > 0)
    mapping = np.arange(len(x))
    mapping[other] = len(x)+np.arange(len(other))
    xx = np.vstack((x, x[other]*[1., -1., 1.]))
    ff = np.vstack((f, mapping[f[:, [0, 2, 1]]]))
    e = np.asarray(cut_edges, dtype=np.int64).reshape(-1, 2)
    if e.size and (e.min() < 0 or e.max() >= len(x)):
        raise ValueError("Invalid half cut index")
    ee = np.vstack((e, mapping[e]))
    return xx, ff, ee
