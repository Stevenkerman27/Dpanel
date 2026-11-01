"""Torch float64 port of the closed-form analytic doublet-sheet velocity (issue #92).

The numpy reference is panel/p1_analytic_sheet_velocity.py: for a planar triangular
linear-doublet density the off-sheet velocity is

    u = freestream - (B + sum_f gamma_f x (S_f - Omega_f n_f)) / (4 pi)

with B the boundary-edge integral (only true cuts contribute), S the Stokes edge sum
and Omega the solid angle. Every term is a closed-form expression of the target point,
so the port moves the same formulae onto a device: no quadrature, no tolerance and no
approximation changes hands. This is the production wake evaluator (method
'analytic_edges'), the one a batched or GPU run actually needs.

Two things stay target-independent and are precomputed once per sheet, because they
are pure geometry:

  * unit edge tangents and edge lengths (face edges and boundary edges),
  * t_cross_n = t x n for the face edges, and the boundary slope (mu1 - mu0)/L.

The target-dependent work (projection of the point on each edge, the moments, the
solid angles) is recomputed for every point and is never cached across targets.

Geometry is built by constructing the numpy field first (or replaying its dumped
arrays through AnalyticSheetField.from_arrays), so the edge combination on topological
cuts - which decides where the trace has a real jump - is performed by exactly one
implementation.
"""
from __future__ import annotations

import numpy as np

from .p1_analytic_sheet_velocity import AnalyticSheetField
from .p1_projection_duffy import _require_torch, resolve_device, torch_available  # noqa: F401

GEOMETRY_KEYS = ("triangles", "normals", "gamma", "face_edge_a", "face_edge_b",
                 "edge_a", "edge_b", "edge_mu", "t_cross_n")


def _moments(p, a, t, length, torcf):
    """Straight-segment moments (I1, I3, Il3, rho) for a batch of points.

    p is (B,1,3), a and t are (1,E,3), length is (1,E). Mirrors segment_moments in
    panel/p1_analytic_sheet_velocity.py term by term, including the two stable
    branches (collinear footing, and the perpendicular foot outside the segment).
    """
    r = p - a
    s = (r*t).sum(-1)
    rho = r - s[..., None]*t
    rho2 = (rho*rho).sum(-1)
    rho_len = torcf.sqrt(rho2)
    safe_rho = torcf.where(rho2 > 0, rho_len, torcf.ones_like(rho_len))
    q = s - length
    R0 = torcf.sqrt(s*s + rho2)
    R1 = torcf.sqrt(q*q + rho2)
    arc = torcf.arcsinh(s/safe_rho) - torcf.arcsinh(q/safe_rho)
    log_form = torcf.abs(torcf.log(torcf.abs(s)/torcf.abs(q)))
    I1 = torcf.where(rho2 > 0, arc, log_form)
    same = (s*q) >= 0
    stable = length*(R0 + R1)/(R0*R1*(R0*R1 + s*q + rho2))
    mixed = (s/R0 - q/R1)/safe_rho**2
    I3 = torcf.where(same, stable, mixed)
    Il3 = s*I3 + 1.0/R0 - 1.0/R1
    return I1, I3, Il3, rho


def _solid_angles(p, triangles, torcf):
    """Solid angle of every triangle seen from every point: (B,F)."""
    r = triangles[None, :, :, :] - p[:, None, None, :]
    lengths = torcf.sqrt((r*r).sum(-1))
    num = (r[:, :, 0]*(torcf.linalg.cross(r[:, :, 1], r[:, :, 2]))).sum(-1)
    den = lengths.prod(dim=-1)
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        den = den + (r[:, :, i]*r[:, :, j]).sum(-1)*lengths[..., k]
    return 2.0*torcf.atan2(num, den)


class TorchAnalyticSheetField:
    """Batched analytic doublet-sheet velocity on torch float64 (CPU or CUDA).

    device is explicit: 'cuda' raises when no device is available instead of falling
    back to the CPU. The numpy field stays reachable as .base when it was built from
    parts, and every backend evaluates the same dumped arrays.
    """

    def __init__(self, parts, *, freestream=(1.0, 0.0, 0.0), device="cpu"):
        self.base = AnalyticSheetField(parts, freestream=freestream)
        arrays = {key: getattr(self.base, key) for key in GEOMETRY_KEYS}
        self._setup(arrays, freestream=self.base.freestream, device=device)

    @classmethod
    def from_arrays(cls, arrays, *, freestream=(1.0, 0.0, 0.0), device="cpu"):
        self = cls.__new__(cls)
        self.base = None
        self._setup(arrays, freestream=freestream, device=device)
        return self

    def _setup(self, arrays, *, freestream, device):
        torcf = _require_torch()
        self._torch = torcf
        self.device = resolve_device(device)
        dev = self.device

        def as_tensor(value):
            array = np.ascontiguousarray(np.asarray(value, float))
            return torcf.as_tensor(array, dtype=torcf.float64, device=dev)

        self._g = {key: as_tensor(arrays[key]) for key in GEOMETRY_KEYS}
        # Target-independent edge geometry, hoisted out of the sampling loop.
        face_start = arrays["face_edge_a"]
        face_end = arrays["face_edge_b"]
        self._face_t, self._face_L = self._unit_and_length(face_start, face_end)
        self._edge_t, self._edge_L = self._unit_and_length(arrays["edge_a"], arrays["edge_b"])
        self.freestream = as_tensor(np.asarray(freestream, float))
        self._n_faces = int(len(self._g["triangles"]))
        self._n_boundary_edges = int(len(self._g["edge_a"]))
        mu = np.asarray(arrays["edge_mu"], float)
        if self._n_boundary_edges:
            length = np.linalg.norm(np.asarray(arrays["edge_b"], float)
                                    - np.asarray(arrays["edge_a"], float), axis=1)
            if np.any(length <= 0):
                raise ValueError("zero-length boundary edge")
            self._edge_slope = as_tensor((mu[:, 1] - mu[:, 0])/length)
            self._edge_mu0 = as_tensor(mu[:, 0])
        else:
            self._edge_slope = self._edge_mu0 = None

    def _unit_and_length(self, starts, ends):
        """Device tensors for unit tangent and length of each source edge."""
        torcf = self._torch
        d = np.asarray(ends, float) - np.asarray(starts, float)
        if not len(d):
            empty3 = torcf.empty((0, 3), dtype=torcf.float64, device=self.device)
            return empty3, torcf.empty((0,), dtype=torcf.float64, device=self.device)
        length = np.linalg.norm(d, axis=1)
        if np.any(length <= 0):
            raise ValueError("zero-length source edge")
        tangent = np.ascontiguousarray(d/length[:, None])
        length = np.ascontiguousarray(length)
        return (torcf.as_tensor(tangent, dtype=torcf.float64, device=self.device),
                torcf.as_tensor(length, dtype=torcf.float64, device=self.device))

    def velocity(self, points, *, chunk=256, return_tensor=False):
        """Freestream plus induced velocity at (3,) or (N,3) points, chunked."""
        torcf = self._torch
        pts = np.asarray(points, float)
        single = pts.ndim == 1
        if single:
            pts = pts[None, :]
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points must be (3,) or (N,3)")
        if not np.isfinite(pts).all():
            raise ValueError("points must be finite")
        out = torcf.empty((len(pts), 3), dtype=torcf.float64, device=self.device)
        step = max(1, int(chunk))
        for first in range(0, len(pts), step):
            block = torcf.as_tensor(np.ascontiguousarray(pts[first:first + step]),
                                    dtype=torcf.float64, device=self.device)
            out[first:first + step] = self._block_velocity(block)
        if not bool(torcf.isfinite(out).all()):
            raise FloatingPointError("nonfinite analytic doublet-sheet velocity")
        if return_tensor:
            return out[0] if single else out
        value = out.detach().cpu().numpy()
        return value[0] if single else value

    def _block_velocity(self, p):
        torcf = self._torch
        p3 = p[:, None, :]
        g = self._g
        I1, _i3, _il3, _rho = _moments(p3, g["face_edge_a"][None, :, :],
                                       self._face_t[None, :, :],
                                       self._face_L[None, :], torcf)
        surface_edges = I1[:, :, None]*g["t_cross_n"][None, :, :]
        surface = surface_edges.reshape(len(p), self._n_faces, 3, 3).sum(dim=2)
        omega = _solid_angles(p, g["triangles"], torcf)
        area_term = torcf.linalg.cross(
            g["gamma"][None, :, :],
            surface - omega[:, :, None]*g["normals"][None, :, :]).sum(dim=1)
        if self._n_boundary_edges:
            _j1, i3, il3, rho = _moments(p3, g["edge_a"][None, :, :],
                                         self._edge_t[None, :, :],
                                         self._edge_L[None, :], torcf)
            weighted = self._edge_mu0[None, :]*i3 + self._edge_slope[None, :]*il3
            boundary = (weighted[:, :, None]
                        *torcf.linalg.cross(self._edge_t[None, :, :], rho)).sum(dim=1)
        else:
            boundary = torcf.zeros((len(p), 3), dtype=torcf.float64, device=self.device)
        return self.freestream[None, :] - (boundary + area_term)/(4.0*np.pi)

    def __call__(self, point):
        return self.velocity(np.asarray(point, float))
