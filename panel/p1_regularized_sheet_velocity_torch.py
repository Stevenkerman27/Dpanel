"""Torch evaluator for the regularized P1 wake sheet (Route B, slice S7).

panel/p1_regularized_sheet_velocity.py evaluates the Rosenhead-Moore regularized
doublet sheet one target at a time: a closed-form edge sum (edge_velocity) plus a
numerical area term (area_velocity, legacy Duffy or projection fan). This module keeps
the same physics and the same geometry preparation - it subclasses that field, so every
attribute and every gate helper (quadrature_arguments, quadrature_plan,
pop_adaptive_estimates) is inherited unchanged - and moves the arithmetic to torch:

* the edge sum is batched over targets, so one kernel evaluates all edges for a block of
  targets instead of rebuilding the tangent, the perpendicular radius and the two roots
  once per target;
* the legacy Duffy area term is batched over (target, face, quadrature node) on the
  device, with the chunk size bounded by a byte budget;
* the projection fan keeps its per-target Python construction (the fan is a geometric
  decomposition, not a kernel), but its quadrature kernel runs through
  panel/p1_projection_duffy.py with backend='torch'.

The adaptive area quadrature (projection_fan_adaptive) has no torch backend yet and is
refused here rather than silently falling back, because a silent fallback would make a
device timing a CPU one.

Nothing here calls NumPy linalg, so it is safe to use after torch has run - the constraint
that shaped the rest of Route B (section 6f of docs/ISSUE92_TORCH_PORT_MAP.md).
"""
from __future__ import annotations

import numpy as np
import torch

from .p1_projection_duffy import area_terms, resolve_device
from .p1_regularized_sheet_velocity import RegularizedP1SheetField

# Bytes of temporary (target, face, node, 3) storage one chunk may allocate.
CHUNK_BYTES = 2.0e8


class TorchRegularizedP1SheetField(RegularizedP1SheetField):
    """Regularized P1 sheet with batched torch evaluation (float64 on device)."""

    def __init__(self, parts, *, device="cpu", dtype=torch.float64, **keywords):
        super().__init__(parts, **keywords)
        if self.area_quadrature == "projection_fan_adaptive":
            raise ValueError("area_quadrature='projection_fan_adaptive' has no torch "
                             "backend; use legacy_duffy or projection_fan")
        self.device = resolve_device(device)
        self.dtype = torch.float64 if dtype is None else dtype
        self._quadrature_torch = {}
        self._edge_a = self._tensor(self.edge_a)
        self._edge_b = self._tensor(self.edge_b)
        delta = self._edge_b - self._edge_a
        self._edge_length = delta.norm(dim=1)
        self._edge_tangent = delta / self._edge_length[:, None]
        self._edge_mu0 = self._tensor(self.edge_mu[:, 0])
        self._edge_slope = ((self._tensor(self.edge_mu[:, 1]) - self._edge_mu0)
                            / self._edge_length)
        self._core2 = float(self.core_radius)**2
        self._triangles = self._tensor(self.triangles)
        self._triangle_area2 = self._tensor(self._triangle_area2)
        self._gamma = self._tensor(self.gamma)
        self._freestream = self._tensor(self.freestream)

    # -- plumbing ---------------------------------------------------------
    def _tensor(self, value):
        data = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
        return torch.as_tensor(data, dtype=self.dtype, device=self.device)

    def _quadrature_pair(self, order):
        if order not in self._quadrature_torch:
            quad, weight = np.polynomial.legendre.leggauss(order)
            quad, weight = (quad + 1.0) / 2.0, weight / 2.0
            u, v = np.meshgrid(quad, quad, indexing="ij")
            weights = (weight[:, None] * weight[None, :] * (1.0 - u)).reshape(-1)
            bary = np.stack(((1.0 - u) * (1.0 - v), u, (1.0 - u) * v),
                            axis=-1).reshape(-1, 3)
            self._quadrature_torch[order] = (self._tensor(bary), self._tensor(weights))
        return self._quadrature_torch[order]

    def _chunk(self, order, chunk, fan_mode):
        if fan_mode:
            return max(1, int(chunk))
        nodes = order * order
        per_target = max(1, int(self.triangles.shape[0])) * nodes * 3 * 8
        budget = max(1, int(CHUNK_BYTES // per_target))
        return max(1, min(int(chunk), budget))

    # -- kernels ----------------------------------------------------------
    def edge_velocity(self, points):
        """Batched closed-form edge sum for a (B,3) tensor, returns (B,3)."""
        if not self.n_boundary_edges:
            return torch.zeros((points.shape[0], 3), dtype=self.dtype, device=self.device)
        r = points[:, None, :] - self._edge_a[None, :, :]
        axial = (r * self._edge_tangent[None]).sum(dim=-1)
        rho = r - axial[:, :, None] * self._edge_tangent[None]
        effective2 = (rho * rho).sum(dim=-1) + self._core2
        root0 = torch.sqrt(effective2 + axial * axial)
        root1 = torch.sqrt(effective2 + (axial - self._edge_length) ** 2)
        i0 = (axial / root0 - (axial - self._edge_length) / root1) / effective2
        i1 = axial * i0 + 1.0 / root0 - 1.0 / root1
        weighted = self._edge_mu0 * i0 + self._edge_slope * i1
        cross = torch.linalg.cross(self._edge_tangent[None].expand_as(rho), rho, dim=-1)
        return (cross * weighted[:, :, None]).sum(dim=1)

    def area_velocity(self, points, order, *, fan_order=None, fan_strips=None):
        """Batched area term for a (B,3) tensor, returns (B,3)."""
        if self.area_quadrature == "projection_fan":
            fan_order_used = self.fan_order if fan_order is None else fan_order
            fan_strips_used = self.fan_strips if fan_strips is None else fan_strips
            if type(fan_order_used) is not int or fan_order_used < 2:
                raise ValueError("invalid fan order")
            if type(fan_strips_used) is not int or fan_strips_used < 1:
                raise ValueError("invalid fan strips")
            # Measured on the G2 CAD sheet (794 faces, 8 targets, fan 8x1): moving the
            # fan kernel to torch costs more than it saves on CPU (0.151 s against
            # 0.123 s, the transfer dominates the elementwise work). The kernel formula
            # is the same either way, so CPU keeps the NumPy kernel and CUDA - where the
            # transfer is paid once per target against a much faster kernel - uses torch.
            kernel = "torch" if self.device.type == "cuda" else "numpy"
            rows = []
            for point in points.detach().cpu().numpy():
                terms = area_terms(self.triangles, np.asarray(point, float),
                                   self.core_radius, order=fan_order_used,
                                   strips=fan_strips_used, swap=self.fan_swap,
                                   backend=kernel, device=str(self.device))
                term = self._tensor(terms)
                rows.append(torch.linalg.cross(self._gamma, term, dim=-1).sum(dim=0))
            return torch.stack(rows)
        bary, weights = self._quadrature_pair(order)
        source = torch.einsum("qv,fvi->fqi", bary, self._triangles)
        step = self._chunk(order, 1, False)
        faces = int(self.triangles.shape[0])
        integral = torch.empty((points.shape[0], faces, 3), dtype=self.dtype,
                               device=self.device)
        for first in range(0, points.shape[0], step):
            block = points[first:first + step]
            r = block[:, None, None, :] - source[None, :, :, :]
            den = ((r * r).sum(dim=-1) + self._core2).pow(1.5)
            integral[first:first + block.shape[0]] = torch.einsum(
                "q,bfq,bfqi->bfi", weights, 1.0 / den, r)
        integral = integral * self._triangle_area2[None, :, None]
        cross = torch.linalg.cross(self._gamma[None].expand_as(integral), integral, dim=-1)
        return cross.sum(dim=1)

    # -- public surface ---------------------------------------------------
    def velocity(self, points, *, order=None, chunk=16, fan_order=None, fan_strips=None,
                 return_tensor=False):
        """Velocity at a batch of targets; NumPy out unless return_tensor."""
        pts = np.asarray(points, float)
        if pts.ndim == 1:
            pts = pts[None, :]
        if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
            raise ValueError("points must be a finite (N,3) array")
        n = self.quadrature_order if order is None else order
        if type(n) is not int or n < 4:
            raise ValueError("quadrature order must be an integer >= 4")
        fan_mode = self.area_quadrature == "projection_fan"
        step = self._chunk(n, chunk, fan_mode)
        out = np.empty_like(pts)
        for first in range(0, len(pts), step):
            block = pts[first:first + step]
            target = self._tensor(block)
            induced = (self.edge_velocity(target)
                       + self.area_velocity(target, n, fan_order=fan_order,
                                            fan_strips=fan_strips))
            value = self._freestream[None, :] - induced / (4.0 * np.pi)
            if not torch.isfinite(value).all():
                raise FloatingPointError("nonfinite regularized P1 batch velocity")
            out[first:first + block.shape[0]] = value.detach().cpu().numpy()
        if return_tensor:
            return self._tensor(out)
        return out

    def evaluate_many(self, points, *, order=None, chunk_size=2, fan_order=None,
                      fan_strips=None):
        pts = np.asarray(points, float)
        if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
            raise ValueError("points must be a finite (N,3) array")
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        return self.velocity(pts, order=order, chunk=max(chunk_size, 16),
                             fan_order=fan_order, fan_strips=fan_strips)

    def __call__(self, point, *, order=None, fan_order=None, fan_strips=None):
        pts = np.asarray(point, float)
        if pts.shape != (3,) or not np.isfinite(pts).all():
            raise ValueError("field point must be a finite (3,) vector")
        return self.velocity(pts, order=order, chunk=1, fan_order=fan_order,
                             fan_strips=fan_strips)[0]

    # Base-class kernels, kept consistent so inherited code paths still work.
    def _edge_velocity(self, point):
        target = self._tensor(np.asarray(point, float))[None, :]
        return self.edge_velocity(target)[0].detach().cpu().numpy()

    def _area_velocity(self, point, order, *, fan_order=None, fan_strips=None):
        target = self._tensor(np.asarray(point, float))[None, :]
        return self.area_velocity(target, order, fan_order=fan_order,
                                  fan_strips=fan_strips)[0].detach().cpu().numpy()
