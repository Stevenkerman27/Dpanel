"""Rosenhead–Moore regularization of the continuous-P1 doublet-sheet velocity.

This is deliberately NOT a constant-strength-filament replacement. The
continuous P1 decomposition contains both:
  (1) topological boundary/cut vortex-filament terms with linearly varying
      P1 trace, and
  (2) an in-panel distributed-vorticity term gamma_f = grad_s(mu) x n.

Both terms are regularized with the same explicit Rosenhead–Moore core radius.
The edge line integral is analytic, including the linear-strength first moment.
The in-panel area integral uses a Duffy-mapped Gauss–Legendre rule.

The result is an opt-in regularized VELOCITY reconstruction for wake kinematics.
It does not regularize the BEM potential influence matrices or re-solve mu/Gamma
under a modified kernel. It is a controlled model comparison, not a fully
regularized boundary-element formulation or a calibrated viscous-core model.
"""
from __future__ import annotations

import numpy as np

from . import velocity_profile as _profile


def _triangle_geometry(xyz, faces, values):
    xyz = np.asarray(xyz, float)
    faces = np.asarray(faces, int)
    values = np.asarray(values, float)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or faces.ndim != 2
            or faces.shape[1] != 3 or len(faces) == 0
            or np.any(faces < 0) or np.any(faces >= len(xyz))
            or values.shape != faces.shape or not np.isfinite(xyz).all()
            or not np.isfinite(values).all()):
        raise ValueError("invalid P1 triangle geometry or nodal strengths")
    tri = xyz[faces]
    av = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    a2 = np.linalg.norm(av, axis=1)
    if np.any(a2 <= 0) or not np.isfinite(a2).all():
        raise ValueError("degenerate P1 triangle")
    normal = av/a2[:, None]
    g1 = np.cross(tri[:, 2]-tri[:, 0], normal)/a2[:, None]
    g2 = np.cross(normal, tri[:, 1]-tri[:, 0])/a2[:, None]
    shape = np.stack((-g1-g2, g1, g2), axis=1)
    grad = np.einsum("fa,fai->fi", values, shape)
    grad -= normal*np.einsum("fi,fi->f", grad, normal)[:, None]
    gamma = np.cross(grad, normal)
    return tri, normal, gamma, av, grad


def _edge_traces(xyz, faces, values):
    """Combine opposite-oriented traces on each actual topological edge."""
    traces = {}
    for face, nodal in zip(np.asarray(faces, int), np.asarray(values, float)):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            ai, bj = int(face[i]), int(face[j])
            key = (min(ai, bj), max(ai, bj))
            if ai < bj:
                sign, pair = 1.0, (nodal[i], nodal[j])
            else:
                sign, pair = -1.0, (nodal[j], nodal[i])
            traces.setdefault(key, np.zeros(2))
            traces[key] += sign*np.asarray(pair)
    keys = [key for key, pair in traces.items() if np.any(pair != 0.0)]
    if not keys:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 2))
    ids = np.asarray(keys, int)
    return (np.asarray(xyz, float)[ids[:, 0]],
            np.asarray(xyz, float)[ids[:, 1]],
            np.stack([traces[key] for key in keys]))


class RegularizedP1SheetField:
    """Off-sheet and on-sheet velocity for one or more continuous P1 sheets.

    Each part is (xyz, faces, face_nodal_strengths). Geometric vertices are
    shared by integer IDs within a part; distinct parts are never topologically
    merged. core_radius is mandatory and uses the geometry's length units.
    """

    def __init__(self, parts, *, freestream=(1.0, 0.0, 0.0),
                 core_radius, quadrature_order=48,
                 area_quadrature="legacy_duffy", fan_order=16, fan_strips=8,
                 fan_swap=False):
        self.freestream = np.asarray(freestream, float).copy()
        self.core_radius = float(core_radius)
        if (self.freestream.shape != (3,) or not np.isfinite(self.freestream).all()
                or not np.isfinite(self.core_radius) or self.core_radius <= 0):
            raise ValueError("finite freestream and strictly positive core_radius required")
        if type(quadrature_order) is not int or quadrature_order < 4:
            raise ValueError("quadrature_order must be an integer >= 4")
        self.quadrature_order = quadrature_order
        if area_quadrature not in ("legacy_duffy", "projection_fan"):
            raise ValueError("area_quadrature must be legacy_duffy or projection_fan")
        if type(fan_order) is not int or fan_order < 2:
            raise ValueError("fan_order must be an integer >= 2")
        if type(fan_strips) is not int or fan_strips < 1:
            raise ValueError("fan_strips must be an integer >= 1")
        self.area_quadrature = area_quadrature
        self.fan_order = fan_order
        self.fan_strips = fan_strips
        self.fan_swap = bool(fan_swap)
        self._quadrature = {}
        tri_parts, normal_parts, gamma_parts, edge_a, edge_b, edge_mu = [], [], [], [], [], []
        for xyz, faces, values in parts:
            tri, normal, gamma, _av, _grad = _triangle_geometry(xyz, faces, values)
            a, b, mu = _edge_traces(xyz, faces, values)
            tri_parts.append(tri)
            normal_parts.append(normal)
            gamma_parts.append(gamma)
            if len(a):
                edge_a.append(a)
                edge_b.append(b)
                edge_mu.append(mu)
        if not tri_parts:
            raise ValueError("at least one nonempty P1 sheet part is required")
        self.triangles = np.concatenate(tri_parts)
        # Cache the doubled area in the same order as self.triangles: compute it
        # from the concatenated array so that the per-part lists no longer have
        # to be kept alive after construction.
        self._triangle_area2 = np.linalg.norm(np.cross(
            self.triangles[:, 1]-self.triangles[:, 0],
            self.triangles[:, 2]-self.triangles[:, 0]), axis=1)
        self.normals = np.concatenate(normal_parts)
        self.gamma = np.concatenate(gamma_parts)
        self.edge_a = np.concatenate(edge_a) if edge_a else np.empty((0, 3))
        self.edge_b = np.concatenate(edge_b) if edge_b else np.empty((0, 3))
        self.edge_mu = np.concatenate(edge_mu) if edge_mu else np.empty((0, 2))
        self.n_boundary_edges = len(self.edge_a)

    def _edge_velocity(self, point):
        _profile.add("field.edge_velocity.calls")
        if not self.n_boundary_edges:
            return np.zeros(3)
        _profile.begin("field.edge_velocity")
        d = self.edge_b-self.edge_a
        length = np.linalg.norm(d, axis=1)
        tangent = d/length[:, None]
        r = point[None, :]-self.edge_a
        axial = np.einsum("ei,ei->e", r, tangent)
        rho = r-axial[:, None]*tangent
        effective2 = np.einsum("ei,ei->e", rho, rho)+self.core_radius**2
        root0 = np.sqrt(effective2+axial**2)
        root1 = np.sqrt(effective2+(axial-length)**2)
        i0 = (axial/root0-(axial-length)/root1)/effective2
        # I1 = integral_0^L s ds / (effective2 + (q-s)^2)^(3/2).
        i1 = axial*i0 + 1.0/root0-1.0/root1
        slope = (self.edge_mu[:, 1]-self.edge_mu[:, 0])/length
        weighted = self.edge_mu[:, 0]*i0+slope*i1
        velocity = np.einsum("ei,e->i", np.cross(tangent, rho), weighted)
        _profile.end("field.edge_velocity")
        return velocity

    def _area_velocity(self, point, order):
        _profile.add("field.area_velocity.calls")
        if self.area_quadrature == "projection_fan":
            _profile.add("field.area_velocity.fan_calls")
            from .p1_projection_duffy import area_terms
            terms = area_terms(self.triangles, np.asarray(point, float),
                               self.core_radius, order=self.fan_order,
                               strips=self.fan_strips, swap=self.fan_swap)
            return np.sum(np.cross(self.gamma, terms), axis=0)
        _profile.add("field.area_velocity.legacy_calls")
        if order not in self._quadrature:
            q, w = np.polynomial.legendre.leggauss(order)
            q, w = (q+1.0)/2.0, w/2.0
            u, v = np.meshgrid(q, q, indexing="ij")
            weights = (w[:, None]*w[None, :]*(1.0-u)).reshape(-1)
            bary = np.stack(((1.0-u)*(1.0-v), u, (1.0-u)*v), axis=-1).reshape(-1, 3)
            self._quadrature[order] = (bary, weights)
        bary, weights = self._quadrature[order]
        source = np.einsum("qv,fvi->fqi", bary, self.triangles)
        r = point[None, None, :]-source
        den = (np.einsum("fqi,fqi->fq", r, r)+self.core_radius**2)**1.5
        # Duffy map Jacobian is |(x1-x0) x (x2-x0)| * (1-u).
        area_integral = np.einsum(
            "fq,fqi->fi", weights[None, :]/den, r)*self._triangle_area2[:, None]
        return np.sum(np.cross(self.gamma, area_integral), axis=0)



    def evaluate_many(self, points, *, order=None, chunk_size=2):
        """Evaluate a batch of field points with bounded-memory vectorization.

        The expensive triangle/quadrature geometry is shared across targets.
        Chunking caps temporary storage at O(chunk_size * n_faces * order**2).
        """
        pts = np.asarray(points, float)
        if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
            raise ValueError("points must be a finite (N,3) array")
        n = self.quadrature_order if order is None else order
        if type(n) is not int or n < 4 or type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("invalid quadrature order or chunk_size")
        _profile.add("field.evaluate_many.calls")
        _profile.add("field.evaluate_many.targets", len(pts))
        if self.area_quadrature == "projection_fan":
            _profile.add("field.evaluate_many.fan_targets", len(pts))
            _profile.add("field.evaluate_many.fan_chunks",
                         -(-len(pts)//max(1, int(chunk_size))) if chunk_size else len(pts))
        if n not in self._quadrature:
            q, w = np.polynomial.legendre.leggauss(n)
            q, w = (q+1.0)/2.0, w/2.0
            u, v = np.meshgrid(q, q, indexing="ij")
            weights = (w[:, None]*w[None, :]*(1.0-u)).reshape(-1)
            bary = np.stack(((1.0-u)*(1.0-v), u, (1.0-u)*v), axis=-1).reshape(-1, 3)
            self._quadrature[n] = (bary, weights)
        if self.area_quadrature == "projection_fan":
            out = np.stack([self.freestream-(self._edge_velocity(point)
                +self._area_velocity(point, n))/(4*np.pi) for point in pts])
            if not np.isfinite(out).all():
                raise FloatingPointError("nonfinite projection-fan batch velocity")
            return out
        bary, weights = self._quadrature[n]
        source = np.einsum("qv,fvi->fqi", bary, self.triangles)
        out = np.empty_like(pts)
        for first in range(0, len(pts), chunk_size):
            p = pts[first:first+chunk_size]
            r = p[:, None, None, :]-source[None, :, :, :]
            den = (np.einsum("cfqi,cfqi->cfq", r, r)+self.core_radius**2)**1.5
            area_integral = np.einsum(
                "cfq,cfqi->cfi", weights[None, None, :]/den, r)
            area_integral *= self._triangle_area2[None, :, None]
            area = np.sum(np.cross(self.gamma[None, :, :], area_integral), axis=1)
            edge = np.stack([self._edge_velocity(point) for point in p])
            out[first:first+len(p)] = self.freestream-(edge+area)/(4*np.pi)
        if not np.isfinite(out).all():
            raise FloatingPointError("nonfinite regularized P1 batch velocity")
        return out

    def __call__(self, point, *, order=None):
        p = np.asarray(point, float)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError("field point must be a finite (3,) vector")
        n = self.quadrature_order if order is None else order
        if type(n) is not int or n < 4:
            raise ValueError("quadrature order must be an integer >= 4")
        _profile.add("field.call.count")
        induced = self._edge_velocity(p)+self._area_velocity(p, n)
        velocity = self.freestream-induced/(4.0*np.pi)
        if not np.isfinite(velocity).all():
            raise FloatingPointError("nonfinite regularized P1 sheet velocity")
        return velocity
