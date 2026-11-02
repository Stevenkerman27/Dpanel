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

    @classmethod
    def from_arrays(cls, arrays, *, freestream=(1.0, 0.0, 0.0)):
        """Rebuild the field from geometry arrays of an existing field.

        This replays a frozen sheet without redoing the topological edge combination:
        it is what a dumped snapshot and a device port consume, so that every backend
        evaluates exactly the same edges, signs and tangents.
        """
        self = cls.__new__(cls)
        for key in ("triangles", "normals", "gamma", "face_edge_a", "face_edge_b",
                    "edge_a", "edge_b", "edge_mu"):
            setattr(self, key, np.ascontiguousarray(np.asarray(arrays[key], float)))
        self.freestream = np.asarray(freestream, float).copy()
        if self.freestream.shape != (3,) or not np.isfinite(self.freestream).all():
            raise ValueError("invalid freestream")
        self.n_face_edges = int(len(self.face_edge_a))
        self.n_boundary_edges = int(len(self.edge_a))
        self.face_edge_n = np.repeat(self.normals, 3, axis=0)
        tangents = self.face_edge_b - self.face_edge_a
        norm = np.linalg.norm(tangents, axis=1)
        if np.any(norm <= 0):
            raise ValueError("zero-length source edge")
        tangents = tangents/norm[:, None]
        self.t_cross_n = np.cross(tangents, self.face_edge_n)
        return self



    def _batch_geometry(self):
        """Cached target-independent edge geometry for the batched path."""
        if not hasattr(self, "_face_len"):
            self._face_len, self._face_tan = _unit_and_length(self.face_edge_a,
                                                             self.face_edge_b)
            self._edge_len, self._edge_tan = _unit_and_length(self.edge_a, self.edge_b)
        return (self._face_len, self._face_tan, self._edge_len, self._edge_tan)

    def _evaluate_rows(self, rows):
        """The closed form evaluated for a (P,3) block of points at once."""
        face_len, face_tan, edge_len, edge_tan = self._batch_geometry()
        I1, _i3, _il3, _rho = _batch_moments(rows, self.face_edge_a, face_tan, face_len)
        surface = (I1[:, :, None]*self.t_cross_n[None, :, :]).reshape(
            len(rows), -1, 3, 3).sum(axis=2)
        omega = _batch_solid_angles(rows, self.triangles)
        area_term = np.sum(np.cross(
            self.gamma[None, :, :], surface - omega[:, :, None]*self.normals[None, :, :]),
            axis=1)
        if self.n_boundary_edges:
            _i1, i3, il3, rho = _batch_moments(rows, self.edge_a, edge_tan, edge_len,
                                               moments=True)
            slope = (self.edge_mu[:, 1]-self.edge_mu[:, 0])/edge_len
            weighted = self.edge_mu[None, :, 0]*i3 + slope[None, :]*il3
            crossed = np.cross(edge_tan[None, :, :], rho)
            boundary_term = np.einsum("pej,pe->pj", crossed, weighted)
        else:
            boundary_term = np.zeros((len(rows), 3))
        result = self.freestream[None, :] - (boundary_term + area_term)/(4*np.pi)
        if not np.isfinite(result).all():
            raise FloatingPointError("nonfinite analytic doublet-sheet velocity")
        return result

    def evaluate_many(self, points, *, chunk=256):
        """Freestream plus induced velocity at (N,3) points, in target blocks.

        Same closed form as __call__, evaluated for a block of targets instead of one:
        the per-target Python and allocation overhead of the scalar path is amortised
        across chunk targets, and the target-independent edge geometry is hoisted out
        of the loop. Memory is bounded by chunk times the number of source edges, never
        by targets times faces, so a whole sampling ladder can be evaluated at once.
        """
        pts = np.asarray(points, float)
        if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
            raise ValueError("points must be a (N,3) finite array")
        if type(chunk) is not int or chunk < 1:
            raise ValueError("chunk must be a positive integer")
        out = np.empty((len(pts), 3))
        for first in range(0, len(pts), chunk):
            out[first:first+chunk] = self._evaluate_rows(pts[first:first+chunk])
        return out

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


def _unit_and_length(starts, ends):
    """Unit tangent and length of every source segment (target-independent)."""
    d = np.asarray(ends, float) - np.asarray(starts, float)
    length = np.linalg.norm(d, axis=1)
    if np.any(length <= 0):
        raise ValueError("zero-length source segment")
    return length, np.ascontiguousarray(d/length[:, None])


def _batch_moments(point_rows, starts, tangents, lengths, *, moments=False):
    """segment_moments for a batch of points: (P,E) arrays instead of one (E,) call.

    The expressions are term-for-term those of segment_moments; only the shape is
    different, so the batch reproduces the loop. I1 is always returned; I3/Il3 and the
    perpendicular residual are only built when the caller needs them (the boundary
    trace), because the face-edge pass discards them.
    """
    p = np.asarray(point_rows, float)
    a = np.asarray(starts, float)
    r = p[:, None, :] - a[None, :, :]
    s = np.einsum("pej,ej->pe", r, tangents)
    rho = r - s[..., None]*tangents[None, :, :]
    rho2 = np.einsum("pej,pej->pe", rho, rho)
    q = s - lengths[None, :]
    if np.any((rho2 == 0) & (s >= 0) & (s <= lengths[None, :])):
        raise FloatingPointError("field point lies on a singular source edge")
    root = np.sqrt(rho2)
    nonzero = rho2 > 0
    I1 = np.empty_like(s)
    I1[nonzero] = (np.arcsinh(s[nonzero]/root[nonzero])
                   - np.arcsinh(q[nonzero]/root[nonzero]))
    zero = ~nonzero
    I1[zero] = np.abs(np.log(np.abs(s[zero]/q[zero])))
    if not moments:
        return I1, None, None, rho
    R0 = np.hypot(s, root)
    R1 = np.hypot(q, root)
    I3 = np.empty_like(s)
    same = s*q >= 0
    length_rows = np.broadcast_to(lengths[None, :], s.shape)
    I3[same] = (length_rows[same]*(R0[same]+R1[same])
                / (R0[same]*R1[same]
                   * (R0[same]*R1[same] + s[same]*q[same] + rho2[same])))
    mixed = ~same
    I3[mixed] = (s[mixed]/R0[mixed] - q[mixed]/R1[mixed])/rho2[mixed]
    Il3 = s*I3 + 1/R0 - 1/R1
    return I1, I3, Il3, rho


def _batch_solid_angles(point_rows, triangles):
    """_solid_angles for a batch of points: (P,F) signed solid angles."""
    p = np.asarray(point_rows, float)
    r = np.asarray(triangles, float)[None, :, :, :] - p[:, None, None, :]
    lengths = np.linalg.norm(r, axis=3)
    if np.any(lengths == 0):
        raise FloatingPointError("field point on a source vertex")
    num = np.einsum("pfi,pfi->pf", r[:, :, 0], np.cross(r[:, :, 1], r[:, :, 2]))
    den = np.prod(lengths, axis=2)
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        den += np.einsum("pfk,pfk->pf", r[:, :, i], r[:, :, j])*lengths[:, :, k]
    return 2*np.arctan2(num, den)
