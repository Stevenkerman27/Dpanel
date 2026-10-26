"""Explainable PyTorch half-native lifting source/doublet panel solver.

A frozen, conforming right-side topology is supplied by CAD/mesh preprocessing.
All continuous geometry, half wake/TE bases, matrix entries and local raw
pressure loads are PyTorch tensors on one CPU/CUDA device in float64.
The analytic mirror represents the physical left side without left DOFs.

Dense LU is intentional for this reference. This module does not claim
physical accuracy or a CUDA throughput advantage, and differentiates neither
CAD Boolean operations nor changes of stencil/face connectivity.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint


class TorchHalfTopology(NamedTuple):
    faces: torch.Tensor
    wake_faces: torch.Tensor
    te_edges: torch.Tensor
    upper_te: torch.Tensor
    lower_te: torch.Tensor
    hat_map: torch.Tensor
    wake_interval: torch.Tensor
    te_interval: torch.Tensor
    neighbor_ghost_ids: torch.Tensor
    component_ids: torch.Tensor


def topology_to_torch(topology, *, device="cpu") -> TorchHalfTopology:
    """Move FROZEN index/basis data outside the differentiated solve.

    topology may be a NumPy or an existing JAX-generated HalfTopology;
    no JAX is imported or needed by the PyTorch solve itself. The mesh
    producer must perform cut/seam, pressure-rank and contact validation.
    """
    result = []
    for name in TorchHalfTopology._fields:
        array = np.asarray(getattr(topology, name))
        dtype = torch.float64 if name == "hat_map" else torch.int64
        result.append(torch.as_tensor(array.copy(), dtype=dtype, device=device))
    return TorchHalfTopology(*result)


def _check_float64_pair(targets, triangles):
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("targets must be [P,3]")
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("triangles must be [T,3,3]")
    if (targets.device != triangles.device or targets.dtype != torch.float64
            or triangles.dtype != torch.float64):
        raise TypeError("Float64 targets and triangles must be on the same device")


def _triangle_block(targets, triangles, *, source, linear_y,
                    target_ids, source_ids):
    """Analytic constant-triangle single/doublet potential and linear-y moment.

    D=omega/(4*pi), omega=-2*atan2(...); D_ii(interior)=-1/2.
    The single-layer potential uses exact edge-log integration, not a core.
    Reflected triangle winding and wake y-moment sign matter.
    """
    edge01 = triangles[:, 1]-triangles[:, 0]
    edge02 = triangles[:, 2]-triangles[:, 0]
    normal_raw = torch.cross(edge01, edge02, dim=-1)
    normal = normal_raw / torch.linalg.vector_norm(normal_raw, dim=-1)[:, None]
    rel = triangles[None]-targets[:, None, None, :]
    rad = torch.linalg.vector_norm(rel, dim=-1)
    r0, r1, r2 = rel.unbind(2)
    det = (r0*torch.cross(r1, r2, dim=-1)).sum(-1)
    den = (rad[:, :, 0]*rad[:, :, 1]*rad[:, :, 2]
           +(r0*r1).sum(-1)*rad[:, :, 2]
           +(r1*r2).sum(-1)*rad[:, :, 0]
           +(r2*r0).sum(-1)*rad[:, :, 1])
    diagonal = None
    if target_ids is not None:
        diagonal = target_ids[:, None].eq(source_ids[None, :])
        det = torch.where(diagonal, torch.zeros_like(det), det)
        den = torch.where(diagonal, torch.ones_like(den), den)
    omega = -2*torch.atan2(det, den)
    doublet = omega/(4*math.pi)
    if diagonal is not None:
        doublet = torch.where(diagonal, -0.5*torch.ones_like(doublet), doublet)
    if not source and not linear_y:
        return doublet

    height = -(r0*normal[None]).sum(-1)
    edge_sum = torch.zeros_like(height) if source else None
    grad_y = torch.zeros_like(height) if linear_y else None
    for k, q in ((0, 1), (1, 2), (2, 0)):
        edge = triangles[:, q]-triangles[:, k]
        length = torch.linalg.vector_norm(edge, dim=-1)
        outward = torch.cross(edge/length[:, None], normal, dim=-1)
        r_k, r_q = rel[:, :, k], rel[:, :, q]
        cross = torch.cross(r_k, r_q, dim=-1)
        dot = (r_k*r_q).sum(-1)
        product = rad[:, :, k]*rad[:, :, q]
        opposite = dot < 0
        alternate = (cross*cross).sum(-1) / torch.where(
            opposite, product-dot, torch.ones_like(dot))
        plus = torch.where(opposite, alternate, product+dot)
        gap = 2*plus/(rad[:, :, k]+rad[:, :, q]+length[None])
        log = torch.log1p(2*length[None]/gap)
        if source:
            edge_sum = edge_sum + (r_k*outward[None]).sum(-1)*log
        if linear_y:
            grad_y = grad_y-outward[None, :, 1]*log
    if linear_y:
        moment_y = ((targets[:, None, 1]-triangles.mean(dim=1)[None, :, 1])
                    *doublet + height*grad_y/(4*math.pi))
        return doublet, moment_y
    single = (edge_sum-height*omega)/(4*math.pi)
    return single, doublet


def triangle_potential_torch(targets, triangles, *, interior_diagonal=False,
                             source=True, linear_y=False, block_size=32,
                             target_panel_ids=None, source_panel_ids=None):
    """Blockwise differentiable triangle influence, returned AIC still dense."""
    _check_float64_pair(targets, triangles)
    if (target_panel_ids is None) != (source_panel_ids is None):
        raise ValueError("Supply matching target/source panel IDs")
    if interior_diagonal and target_panel_ids is None and len(targets) != len(triangles):
        raise ValueError("Self interior trace requires matching targets and panels")
    if target_panel_ids is not None:
        row_ids = target_panel_ids.to(device=targets.device, dtype=torch.int64)
        col_ids = source_panel_ids.to(device=targets.device, dtype=torch.int64)
    elif interior_diagonal:
        row_ids = torch.arange(len(targets), device=targets.device)
        col_ids = torch.arange(len(triangles), device=targets.device)
    else:
        row_ids = col_ids = None
    count = len(targets)
    if count == 0:
        raise ValueError("Empty target array")
    chunk = count if block_size <= 0 else block_size
    chunks = []
    for start in range(0, count, chunk):
        end = min(start+chunk, count)
        chunks.append(_triangle_block(
            targets[start:end], triangles, source=source, linear_y=linear_y,
            target_ids=None if row_ids is None else row_ids[start:end],
            source_ids=col_ids))
    if isinstance(chunks[0], tuple):
        return tuple(torch.cat([item[k] for item in chunks], dim=0)
                     for k in range(len(chunks[0])))
    return torch.cat(chunks, dim=0)


def mirror_xz(triangles):
    """Polar reflection plus winding reversal preserves outward normals."""
    return triangles[:, [0, 2, 1], :]*triangles.new_tensor([1., -1., 1.])


class ImplicitLUSolve(torch.autograd.Function):
    """First-order implicit adjoint (pivots have no derivative).

    A.T lambda = dL/dstate, dL/dA=-lambda outer state, dL/db=lambda.
    """

    @staticmethod
    def forward(ctx, A, b):
        lu, pivots = torch.linalg.lu_factor(A)
        state = torch.linalg.lu_solve(lu, pivots, b.unsqueeze(-1)).squeeze(-1)
        ctx.save_for_backward(lu, pivots, state)
        return state

    @staticmethod
    def backward(ctx, cotangent):
        lu, pivots, state = ctx.saved_tensors
        adjoint = torch.linalg.lu_solve(
            lu, pivots, cotangent.contiguous().unsqueeze(-1),
            adjoint=True).squeeze(-1)
        dA = -adjoint[:, None]*state[None, :]
        return dA, adjoint


def solve_lu(A, rhs, *, implicit=True):
    if A.ndim != 2 or A.shape[0] != A.shape[1] or rhs.shape != (A.shape[0],):
        raise ValueError("Require square A and matching single RHS")
    return (ImplicitLUSolve.apply(A, rhs)
            if implicit else torch.linalg.solve(A, rhs))


def _wake_te_geometry(x, y, t):
    wake = x[t.wake_faces]
    wi = t.wake_interval
    dy = y[wi+1]-y[wi]
    fraction = (wake.mean(1)[:, 1]-y[wi])/dy
    basis = ((1-fraction[:, None])*t.hat_map[wi]
             +fraction[:, None]*t.hat_map[wi+1])
    slope = (t.hat_map[wi+1]-t.hat_map[wi])/dy[:, None]
    gauss = y.new_tensor([.5-math.sqrt(3.)/6., .5+math.sqrt(3.)/6.])
    ends = x[t.te_edges]
    te_y = ends[:, 0, 1, None]*(1-gauss)+ends[:, 1, 1, None]*gauss
    ti = t.te_interval
    tf = (te_y-y[ti])/(y[ti+1]-y[ti])
    te_basis = (1-tf[..., None])*t.hat_map[ti]+tf[..., None]*t.hat_map[ti+1]
    length = torch.linalg.vector_norm(ends[:, 1]-ends[:, 0], dim=-1)
    weight = (length[:, None]/2).expand_as(te_y)
    mass = torch.einsum("eq,eqi,eqj->ij", weight, te_basis, te_basis)
    integrals = torch.einsum("eq,eqi->ei", weight, te_basis)
    measure = integrals.sum(0)
    nf = t.faces.shape[0]
    incidence = x.new_zeros((len(t.te_edges), nf))
    rows = torch.arange(len(t.te_edges), dtype=torch.int64, device=x.device)
    incidence.index_put_((rows, t.upper_te), x.new_ones(len(rows)), accumulate=True)
    incidence.index_put_((rows, t.lower_te), -x.new_ones(len(rows)), accumulate=True)
    T = (integrals.T@incidence)/measure[:, None]
    M = mass/measure[:, None]
    return wake, basis, slope, T, M, te_basis


def _raw_half_loads(x, t, flow, density, area_ref, chord_ref, point, mu):
    tri = x[t.faces]
    av = torch.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0], dim=-1)/2
    normal = av/torch.linalg.vector_norm(av, dim=-1)[:, None]
    centers = tri.mean(1)
    reflected = x.new_tensor([1., -1., 1.])
    axial = x.new_tensor([-1., 1., -1.])
    virtual_centers = torch.cat((centers, centers*reflected), dim=0)
    virtual_mu = torch.cat((mu, mu), dim=0)
    tangent = tri[:, 1]-tri[:, 0]
    tangent = tangent/torch.linalg.vector_norm(tangent, dim=-1)[:, None]
    second = torch.cross(normal, tangent, dim=-1)
    delta = virtual_centers[t.neighbor_ghost_ids]-centers[:, None]
    dl = (delta*tangent[:, None]).sum(-1)
    dm = (delta*second[:, None]).sum(-1)
    dmu = virtual_mu[t.neighbor_ghost_ids]-mu[:, None]
    ll, mm, lm = (v.sum(1) for v in (dl*dl, dm*dm, dl*dm))
    bl, bm = (dl*dmu).sum(1), (dm*dmu).sum(1)
    determinant = ll*mm-lm*lm
    gl = (bl*mm-bm*lm)/determinant
    gm = (bm*ll-bl*lm)/determinant
    velocity = ((tangent@flow+gl)[:, None]*tangent
                +(second@flow+gm)[:, None]*second)
    speed2 = flow@flow
    cp = 1-(velocity*velocity).sum(-1)/speed2
    q = .5*density*speed2
    forces = -q*cp[:, None]*av
    moments = torch.cross(centers-point, forces, dim=-1)
    one_f, one_m = forces.sum(0), moments.sum(0)
    all_f = one_f+one_f*reflected
    all_m = one_m+one_m*axial
    drag_axis = flow/torch.sqrt(speed2)
    lift_axis = torch.cross(drag_axis, flow.new_tensor([0., 1., 0.]), dim=-1)
    lift_axis = lift_axis/torch.linalg.vector_norm(lift_axis)
    cf = torch.stack([forces[t.component_ids.eq(i)].sum(0) for i in (10, 20, 30)])
    cm = torch.stack([moments[t.component_ids.eq(i)].sum(0) for i in (10, 20, 30)])
    return dict(Cp_raw_half=cp, panel_forces_half=forces,
                panel_moments_half=moments, panel_centers_half=centers,
                panel_areas_half=torch.linalg.vector_norm(av, dim=-1),
                panel_normals_half=normal, surface_velocity_half=velocity,
                tangent_determinant=determinant, force=all_f, moment=all_m,
                component_forces=cf+cf*reflected,
                component_moments=cm+cm*axial,
                CL=all_f@lift_axis/(q*area_ref),
                CD_pressure=all_f@drag_axis/(q*area_ref),
                CM=all_m/(q*area_ref*chord_ref))


def solve_half_torch(vertices, station_y, topology, freestream, density,
                     reference_area, reference_chord, reference_point,
                     *, block_size=32, implicit_adjoint=True):
    """Torch CPU/CUDA Float64 end-to-end half lifting state and raw loads.

    Contract: frozen topology and pressure stencils prevalidated on host;
    physical length, SI density and speed, full-aircraft Sref; sideslip=0,
    reference_point.y=0; no CAD remeshing within the AD graph.
    """
    x, y, flow = vertices, station_y, freestream
    if x.dtype != torch.float64 or y.dtype != torch.float64 or flow.dtype != torch.float64:
        raise TypeError("Half solver requires Float64 geometry, station_y and freestream")
    if x.device != y.device or x.device != flow.device:
        raise ValueError("Inputs must reside on the same CPU/CUDA device")
    if any(getattr(topology, name).device != x.device for name in topology._fields):
        raise ValueError("Frozen half topology must reside on the same device")
    if (not bool(torch.isfinite(x).all().item())
            or not bool(torch.isfinite(y).all().item())
            or not bool(torch.isfinite(flow).all().item())
            or not bool(torch.all(y[1:] > y[:-1]).item())
            or bool((torch.abs(y[0]) > 1e-10).item())
            or bool((x[:, 1] < -1e-10).any().item())
            or bool((torch.abs(flow[1]) > 1e-12*torch.linalg.vector_norm(flow)).item())):
        raise ValueError("Half solver requires valid symmetric geometry, stations and zero sideslip")
    pt = torch.as_tensor(reference_point, dtype=x.dtype, device=x.device)
    if (pt.shape != (3,) or not bool(torch.isfinite(pt).all().item())
            or bool((pt[1].abs() > 1e-10).item())):
        raise ValueError("Half moment reference must lie on symmetry plane")
    scalar = lambda v: torch.as_tensor(v, dtype=x.dtype, device=x.device)
    tri = x[topology.faces]
    av = torch.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0], dim=-1)/2
    normal = av/torch.linalg.vector_norm(av, dim=-1)[:, None]
    sigma = -(normal@flow)
    center = tri.mean(1)
    direct_S, direct_D = triangle_potential_torch(
        center, tri, interior_diagonal=True, block_size=block_size)
    image_S, image_D = triangle_potential_torch(
        center, mirror_xz(tri), block_size=block_size)
    wake, basis, slope, T, M, te_basis = _wake_te_geometry(x, y, topology)
    Dw, My = triangle_potential_torch(
        center, wake, source=False, linear_y=True, block_size=block_size)
    Wi, Mi = triangle_potential_torch(
        center, mirror_xz(wake), source=False, linear_y=True,
        block_size=block_size)
    W = (Dw+Wi)@basis+(My-Mi)@slope
    D = direct_D+image_D
    S = direct_S+image_S
    A = torch.cat((torch.cat((D, W), dim=1),
                   torch.cat((-T, M), dim=1)), dim=0)
    rhs = torch.cat((S@sigma, x.new_zeros((M.shape[0],))))
    state = solve_lu(A, rhs, implicit=implicit_adjoint)
    nf = len(tri)
    mu, gamma = state[:nf], state[nf:]
    fields = _raw_half_loads(
        x, topology, flow, scalar(density), scalar(reference_area),
        scalar(reference_chord), scalar(reference_point), mu)
    fields.update(mu_half=mu, gamma_half=gamma, sigma_half=sigma,
                  matrix_half=A, rhs_half=rhs, wake_basis_half=basis,
                  wake_slope_half=slope, te_basis_half=te_basis,
                  linear_residual_half=A@state-rhs)
    return fields
