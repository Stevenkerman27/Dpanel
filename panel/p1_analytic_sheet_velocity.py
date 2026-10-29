"""Closed-form straight-edge integration for a triangular P1 doublet sheet.

For a planar triangular linear-doublet density mu, its off-sheet velocity is

  (B + sum_faces gamma x (S - Omega*n))/(4*pi),
  B = sum_edges integral_edge mu (t x r)/|r|**3 dl,
  S = sum_face_edges (t x n) integral_edge 1/|r| dl,
  gamma = grad_s(mu) x n.

Both straight-edge integrals, INCLUDING the linear-strength first moment, have
elementary antiderivatives. We combine opposite-oriented traces on the same
topological edge BEFORE evaluating B. Smooth P1 traces cancel exactly; physical
cuts retain their strength discontinuity. Different sheets are not mixed.

No viscous core/cutoff, no finite Gauss-edge quadrature, no claimed pointwise
velocity on a singular filament, and no proof of a PV limit at a free edge.
"""
from __future__ import annotations

import numpy as np


def segment_moments(point, starts, ends):
    """Exact segment (I1, I3, Il3, tangent, transverse_r).

    I1 = int_0^L dl/R, I3 = int_0^L dl/R**3,
    Il3 = int_0^L l*dl/R**3, R=|point - start - l*t|.
    Collinear points outside a segment are legal; points ON it are singular.
    """
    p = np.asarray(point, float)
    a, b = np.asarray(starts, float), np.asarray(ends, float)
    if (p.shape != (3,) or a.ndim != 2 or a.shape[1] != 3 or b.shape != a.shape
            or not np.isfinite(p).all() or not np.isfinite(a).all()
            or not np.isfinite(b).all()):
        raise ValueError("invalid point or segment geometry")
    d = b - a
    L = np.linalg.norm(d, axis=1)
    if np.any(L <= 0):
        raise ValueError("zero-length source segment")
    t = d / L[:, None]
    r = p - a
    s = np.einsum("ij,ij->i", r, t)
    rho = r - s[:, None]*t
    rho2 = np.einsum("ij,ij->i", rho, rho)
    q = s-L
    if np.any((rho2 == 0) & (s >= 0) & (s <= L)):
        raise FloatingPointError("field point lies on a singular source edge")
    R0 = np.hypot(s, np.sqrt(rho2))
    R1 = np.hypot(q, np.sqrt(rho2))
    nonzero = rho2 > 0
    I1 = np.empty_like(s)
    I1[nonzero] = (np.arcsinh(s[nonzero]/np.sqrt(rho2[nonzero]))
                   - np.arcsinh(q[nonzero]/np.sqrt(rho2[nonzero])))
    zero = ~nonzero
    I1[zero] = np.abs(np.log(np.abs(s[zero]/q[zero])))
    I3 = np.empty_like(s)
    same = s*q >= 0
    # Stable when the perpendicular foot is outside the segment.
    I3[same] = (L[same]*(R0[same]+R1[same])
                / (R0[same]*R1[same]
                   * (R0[same]*R1[same] + s[same]*q[same] + rho2[same])))
    mixed = ~same
    I3[mixed] = (s[mixed]/R0[mixed] - q[mixed]/R1[mixed])/rho2[mixed]
    Il3 = s*I3 + 1/R0 - 1/R1
    return I1, I3, Il3, t, rho


def _solid_angles(point, triangles):
    r = triangles - point
    lengths = np.linalg.norm(r, axis=2)
    if np.any(lengths == 0):
        raise FloatingPointError("field point on a source vertex")
    num = np.einsum("fi,fi->f", r[:, 0], np.cross(r[:, 1], r[:, 2]))
    den = np.prod(lengths, axis=1)
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        den += np.einsum("fi,fi->f", r[:, i], r[:, j])*lengths[:, k]
    return 2*np.arctan2(num, den)


class AnalyticSheetField:
    """Off-sheet total velocity with the sign used by DPanel's D=-Omega/4pi.

    Parts are tuples (xyz, faces, per_face_P1_vertex_strengths).
    Shared geometric vertex IDs may have different strengths across a true cut.
    Only edges within the SAME part are combined; wall and wakes stay distinct.
    """

    def __init__(self, parts, freestream=(1., 0., 0.)):
        self.freestream = np.asarray(freestream, float).copy()
        if self.freestream.shape != (3,) or not np.isfinite(self.freestream).all():
            raise ValueError("invalid freestream")
        all_tri, all_norm, all_gamma, face_start, face_end = [], [], [], [], []
        edge_start, edge_end, edge_mu = [], [], []
        for xyz, faces, face_values in parts:
            xyz = np.asarray(xyz, float)
            faces = np.asarray(faces, int)
            values = np.asarray(face_values, float)
            if (xyz.ndim != 2 or xyz.shape[1] != 3 or faces.ndim != 2
                    or faces.shape[1] != 3 or not len(faces)
                    or np.any(faces < 0) or np.any(faces >= len(xyz))
                    or values.shape != faces.shape or not np.isfinite(xyz).all()
                    or not np.isfinite(values).all()):
                raise ValueError("invalid sheet data or P1 trace")
            tri = xyz[faces]
            av = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
            a2 = np.linalg.norm(av, axis=1)
            if np.any(a2 <= 0):
                raise ValueError("degenerate sheet triangle")
            n = av/a2[:, None]
            g1 = np.cross(tri[:, 2]-tri[:, 0], n)/a2[:, None]
            g2 = np.cross(n, tri[:, 1]-tri[:, 0])/a2[:, None]
            shape = np.stack((-g1-g2, g1, g2), axis=1)
            grad = np.einsum("fa,fai->fi", values, shape)
            grad_s = grad - n*np.einsum("fi,fi->f", grad, n)[:, None]
            gamma = np.cross(grad_s, n)
            all_tri.append(tri.copy())
            all_norm.append(n)
            all_gamma.append(gamma)
            face_start.append(tri.copy())
            face_end.append(np.roll(tri, -1, axis=1).copy())
            traces = {}
            for f, mu in zip(faces, values):
                for i, j in ((0, 1), (1, 2), (2, 0)):
                    ai, bj = int(f[i]), int(f[j])
                    key = (min(ai, bj), max(ai, bj))
                    positive = ai < bj
                    sign = 1. if positive else -1.
                    pair = (mu[i], mu[j]) if positive else (mu[j], mu[i])
                    if key not in traces:
                        traces[key] = np.zeros(2)
                    traces[key] += sign*np.asarray(pair)
            keys = [key for key, pair in traces.items() if np.any(pair != 0)]
            if keys:
                ids = np.asarray(keys, int)
                edge_start.append(xyz[ids[:, 0]].copy())
                edge_end.append(xyz[ids[:, 1]].copy())
                edge_mu.append(np.stack([traces[key] for key in keys]))
        self.triangles = np.concatenate(all_tri)
        self.normals = np.concatenate(all_norm)
        self.gamma = np.concatenate(all_gamma)
        self.face_edge_a = np.concatenate(face_start).reshape(-1, 3)
        self.face_edge_b = np.concatenate(face_end).reshape(-1, 3)
        self.edge_a = np.concatenate(edge_start) if edge_start else np.empty((0, 3))
        self.edge_b = np.concatenate(edge_end) if edge_end else np.empty((0, 3))
        self.edge_mu = np.concatenate(edge_mu) if edge_mu else np.empty((0, 2))
        self.n_boundary_edges = len(self.edge_a)
        self.n_face_edges = len(self.face_edge_a)
        self.face_edge_n = np.repeat(self.normals, 3, axis=0)
        tangents = self.face_edge_b - self.face_edge_a
        tangents /= np.linalg.norm(tangents, axis=1)[:, None]
        self.t_cross_n = np.cross(tangents, self.face_edge_n)

    def __call__(self, point):
        p = np.asarray(point, float)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError("field point must be (3,) and finite")
        I1, _i3, _il3, _t, _rho = segment_moments(
            p, self.face_edge_a, self.face_edge_b)
        surface = (I1[:, None]*self.t_cross_n).reshape(-1, 3, 3).sum(axis=1)
        omega = _solid_angles(p, self.triangles)
        area_term = np.sum(np.cross(
            self.gamma, surface - omega[:, None]*self.normals), axis=0)
        if self.n_boundary_edges:
            _i1, i3, il3, t, rho = segment_moments(
                p, self.edge_a, self.edge_b)
            L = np.linalg.norm(self.edge_b-self.edge_a, axis=1)
            slope = (self.edge_mu[:, 1]-self.edge_mu[:, 0])/L
            weighted = self.edge_mu[:, 0]*i3 + slope*il3
            boundary_term = np.einsum("ij,i->j", np.cross(t, rho), weighted)
        else:
            boundary_term = np.zeros(3)
        result = self.freestream-(boundary_term+area_term)/(4*np.pi)
        if not np.isfinite(result).all():
            raise FloatingPointError("nonfinite analytic doublet-sheet velocity")
        return result
