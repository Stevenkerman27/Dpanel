"""Fixed-topology CST wing and differentiable P1 Galerkin assembly experiment."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import torch

from .p1_assembly_torch import doublet_matrix, gradient_map, mesh_kernel
from .p1_galerkin_wall import _duffy_rule
from .p1_lifting_g1 import make_wing
from .p1_lifting_g2 import assemble_multi, combine_wings
from .p1_pressure_kutta_torch_driver import TorchPressureModel, solve_pressure_kutta_torch
from .p1_potential_jump_torch import affine_state, potential_jump_newton
from .p1_rank_aware_torch import null_space


def cst_geometry(code, chord_panels=4, span_panels=4, span=4., wake_panels=2):
    """GRASS 18-value Kulfan code on the fixed rectangular make_wing topology."""
    grass = Path(__file__).resolve().parents[2] / "GRASS-plane"
    if str(grass) not in sys.path:
        sys.path.insert(0, str(grass))
    from cst_airfoil_codec import cst_surfaces_at_x

    code = torch.as_tensor(code, dtype=torch.float64)
    if code.shape != (18,) or float(code[17].detach()) != 0.:
        raise ValueError("Expected a sharp-TE GRASS Kulfan code of shape (18,)")
    code = torch.cat((code[:17], torch.zeros_like(code[17:])))
    wing = make_wing(chord_panels=chord_panels, span_panels=span_panels,
                     span=span, wake_panels=wake_panels)
    xyz = torch.as_tensor(wing.xyz, dtype=code.dtype)
    x = torch.as_tensor(.5 * (1 - np.cos(np.linspace(0, np.pi, chord_panels + 1))),
                        dtype=code.dtype)
    upper, lower = cst_surfaces_at_x(code, x)
    upper, lower = upper[0], lower[0]
    z = torch.zeros(len(wing.xyz), dtype=code.dtype)
    for i in range(1, chord_panels):
        at_x = np.isclose(wing.xyz[:, 0], float(x[i]), atol=1e-12)
        z = torch.where(torch.as_tensor(at_x & (wing.xyz[:, 2] > 0)), upper[i], z)
        z = torch.where(torch.as_tensor(at_x & (wing.xyz[:, 2] < 0)), lower[i], z)
    # make_wing's two cap centres lie on the interpolated section camber line.
    k = int(np.searchsorted(x.numpy(), .42) - 1)
    fraction = (.42 - x[k]) / (x[k + 1] - x[k])
    camber = (upper + lower) / 2
    center_z = camber[k] * (1 - fraction) + camber[k + 1] * fraction
    z = torch.where(torch.as_tensor(np.isclose(wing.xyz[:, 0], .42)), center_z, z)
    return wing, torch.stack((xyz[:, 0], xyz[:, 1], z), dim=1)


def grass_section_geometry(geometry, chord_panels=4, span_panels_per_segment=2,
                           wake_panels=2, wake_length=10.):
    """GRASS section frames to a fixed symmetric P1 solid and attached wake."""
    grass = Path(__file__).resolve().parents[2] / "GRASS-plane"
    if str(grass) not in sys.path:
        sys.path.insert(0, str(grass))
    from cst_airfoil_codec import cst_surfaces_at_x
    from wing_geometry_encoder import wing_section_points

    geometry = {name: torch.as_tensor(geometry[name], dtype=torch.float64)
                for name in ("leading_edge_xyz", "chord", "twist_radians", "cst")}
    xyz0, chord, twist, code = (geometry[name] for name in
                                ("leading_edge_xyz", "chord", "twist_radians", "cst"))
    n = len(chord)
    if (n < 2 or xyz0.shape != (n, 3) or twist.shape != (n,) or code.shape != (n, 18)
            or any(not bool(torch.isfinite(value).all()) for value in geometry.values())
            or bool((xyz0[0] != 0).any()) or bool((xyz0[1:, 1] <= xyz0[:-1, 1]).any())
            or bool((chord <= 0).any()) or bool((code[:, 17] != 0).any())
            or not isinstance(span_panels_per_segment, int) or span_panels_per_segment < 1):
        raise ValueError("Require ordered GRASS sections at root origin with positive chord and sharp TE")
    code = torch.cat((code[:, :17], torch.zeros_like(code[:, 17:])), dim=1)
    geometry = dict(geometry, cst=code)
    half_panels = (n - 1) * span_panels_per_segment
    wing = make_wing(chord_panels=chord_panels, span_panels=2 * half_panels,
                     span=2., wake_panels=wake_panels, wake_length=wake_length)
    xi = torch.as_tensor(.5 * (1 - np.cos(np.linspace(0, np.pi, chord_panels + 1))))
    upper, lower = cst_surfaces_at_x(code, xi)
    placement = torch.zeros(3, dtype=chord.dtype)
    up = wing_section_points(geometry, chord[:, None] * xi, chord[:, None] * upper, placement)
    lo = wing_section_points(geometry, chord[:, None] * xi, chord[:, None] * lower, placement)
    rows = []
    for j in range(half_panels + 1):
        segment = min(j // span_panels_per_segment, n - 2)
        fraction = (j - segment * span_panels_per_segment) / span_panels_per_segment
        rows.append(((1 - fraction) * up[segment] + fraction * up[segment + 1],
                     (1 - fraction) * lo[segment] + fraction * lo[segment + 1]))
    nodes = []
    for point in wing.xyz[:-2]:
        j = int(round((point[1] + 1) * half_panels))
        i = int(np.argmin(np.abs(xi.numpy() - point[0])))
        pos = rows[abs(j - half_panels)][0 if point[2] >= 0 else 1][i]
        nodes.append(pos * torch.tensor([1., -1. if j < half_panels else 1., 1.]))
    k = int(np.searchsorted(xi.numpy(), .42) - 1)
    fraction = (.42 - xi[k]) / (xi[k + 1] - xi[k])
    camber = (rows[-1][0] + rows[-1][1]) / 2
    cap = camber[k] * (1 - fraction) + camber[k + 1] * fraction
    nodes.extend((cap * torch.tensor([1., -1., 1.]), cap))
    xyz = torch.stack(nodes)
    te = xyz[wing.te_geo]
    wake = torch.cat([te + torch.stack((chord[0] * (wake_length * k / wake_panels),
                                       chord[0] * 0, chord[0] * 0))
                      for k in range(wake_panels + 1)])
    area = 2 * torch.trapezoid(chord, xyz0[:, 1])
    values = xyz.detach().numpy()
    te_y = values[wing.te_geo, 1]
    if np.any(np.diff(te_y) <= 0):
        raise ValueError("Trailing-edge span coordinates must be strictly increasing")
    wing = replace(wing, xyz=values, wake_xyz=wake.detach().numpy(),
                   te_pair_y=te_y[1:-1], gamma_y=te_y,
                   chord=float(chord[0].detach()), span=float(te_y[-1] - te_y[0]),
                   reference_area=float(area.detach()))
    return wing, xyz, wake, area


def _hessian_rows(stationarity, z):
    """Hessian with one vector-Jacobian product per row, detached immediately.

    torch.autograd.functional.hessian keeps every row's graph alive until the last
    one finishes; at ~900 reduced unknowns that costs about 14 GB on the 1720-face
    meshes (measured), while the Hessian itself is only 6.5 MB. Row by row it is the
    same arithmetic with a single graph in memory.
    """
    n = int(z.numel())
    out = torch.empty((n, n), dtype=z.dtype)
    for i in range(n):
        seed = torch.zeros(n, dtype=z.dtype)
        seed[i] = 1.0
        out[i] = torch.autograd.grad(stationarity, z, grad_outputs=seed,
                                     retain_graph=True)[0].detach()
    return out


def _recompute(fn, *args):
    """Run an assembly so that its intermediates are recomputed in backward.

    The doublet assembler stores about ten intermediate tensors of shape
    (targets, face_chunk, 3, 3) per source block, so its graph grows like
    O(n_panels * ndof * face_chunk) and reaches several gigabytes at ~1e3
    unknowns.  Wrapping the call in a checkpoint keeps only the assembled
    (targets, ndof) result; the backward pass re-runs the assembly block by block.
    Values are unchanged (same code, same order), only the saved tensors differ.
    """
    if not torch.is_grad_enabled() or not any(
            torch.is_tensor(arg) and arg.requires_grad for arg in args):
        return fn(*args)
    return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)


def galerkin_operators(wing, xyz, order=3, wake_xyz=None, recompute=None):
    """Assemble the production weak operators with differentiable coordinates.

    `recompute` defaults to on whenever the coordinates carry gradients."""
    model = combine_wings([wing])
    base = assemble_multi(model)
    faces = torch.as_tensor(wing.faces, dtype=torch.long)
    dofs = torch.as_tensor(wing.dof_face, dtype=torch.long)
    mesh = mesh_kernel(xyz, faces)
    bary, weights = _duffy_rule(order)
    bary = torch.as_tensor(bary)
    weights = torch.as_tensor(weights)
    points = torch.einsum("qk,fkj->fqj", bary, mesh["triangles"])
    quadrature = 2 * mesh["areas"][:, None] * weights[None, :]
    flat = points.reshape(-1, 3)
    self_faces = torch.arange(len(faces)).repeat_interleave(len(weights))
    use_recompute = xyz.requires_grad if recompute is None else bool(recompute)
    assemble = _recompute if use_recompute else (lambda fn, *args: fn(*args))
    wall = assemble(lambda tri, target: doublet_matrix(
        tri, target, self_faces=self_faces, face_dofs=dofs, ndof=wing.ndof_mu),
        mesh["triangles"], flat)
    wake_xyz = torch.as_tensor(wing.wake_xyz, dtype=xyz.dtype) if wake_xyz is None else wake_xyz
    wake_faces = torch.as_tensor(wing.wake_faces, dtype=torch.long)
    wake = assemble(lambda tri, target: doublet_matrix(
        tri, target, face_dofs=wake_faces, ndof=len(wake_xyz)),
        wake_xyz[wake_faces], flat) @ torch.as_tensor(wing.gamma_hat)
    wall = wall.reshape(len(faces), len(weights), -1)
    wake = wake.reshape(len(faces), len(weights), -1)
    A = torch.zeros((wing.ndof_mu, wing.ndof_mu), dtype=xyz.dtype)
    W = torch.zeros((wing.ndof_mu, wake.shape[-1]), dtype=xyz.dtype)
    b = torch.zeros((wing.ndof_mu, 3), dtype=xyz.dtype)
    for k in range(3):
        w = quadrature * bary[None, :, k]
        A = A.index_add(0, dofs[:, k], torch.einsum("fq,fqj->fj", w, wall))
        W = W.index_add(0, dofs[:, k], torch.einsum("fq,fqj->fj", w, wake))
        b = b.index_add(0, dofs[:, k], -torch.einsum("fq,fqj->fj", w, points))
    mass = torch.zeros_like(A)
    local = torch.tensor([[2., 1., 1.], [1., 2., 1.], [1., 1., 2.]])
    for i in range(3):
        for j in range(3):
            indices = dofs[:, i] * wing.ndof_mu + dofs[:, j]
            mass = mass.reshape(-1).index_add(
                0, indices, mesh["areas"] * local[i, j] / 12).reshape_as(mass)
    grad = gradient_map(faces, dofs, mesh["basis_gradients"], ndof=wing.ndof_mu)
    return model, base, mesh, A, W, b, mass, grad


def fixed_wake_cst(code, alpha_deg=4., chord_panels=4, span_panels=4,
                   span=4., wake_panels=2, max_nfev=160, kutta="pressure"):
    """Small sharp-TE wing, exact stationary implicit gradient.

    The default closure stays "pressure" because the production solver
    (panel/p1_lifting_g2.py:solve_multi, kutta="pressure") is what this path is verified
    against: on the 3x3 CST wing the two agree to 5e-15 in CL, while "potential_jump"
    differs by 2.9e-4.  See docs/KUTTA_CONDITION.md section 7.
    """
    wing, xyz = cst_geometry(code, chord_panels, span_panels, span, wake_panels)
    return fixed_wake_geometry(wing, xyz, None, wing.reference_area, alpha_deg, max_nfev,
                               kutta=kutta)


def fixed_wake_sections(geometry, alpha_deg=4., chord_panels=4,
                        span_panels_per_segment=2, wake_panels=2, max_nfev=160,
                        kutta="pressure"):
    wing, xyz, wake, area = grass_section_geometry(
        geometry, chord_panels, span_panels_per_segment, wake_panels)
    return fixed_wake_geometry(wing, xyz, wake, area, alpha_deg, max_nfev, kutta=kutta)


def fixed_wake_surface(path, wall_xz, wake_xz, reference_area, alpha_deg=4., max_nfev=160):
    """Differentiate imported P1 wall/wake x,z coordinates at fixed y."""
    from .p1_lifting_g1 import from_dpanel_half_surface

    wing = from_dpanel_half_surface(path)
    wall_xz = torch.as_tensor(wall_xz, dtype=torch.float64)
    wake_xz = torch.as_tensor(wake_xz, dtype=torch.float64)
    if wall_xz.shape != (len(wing.xyz), 2) or wake_xz.shape != (len(wing.wake_xyz), 2):
        raise ValueError("Coordinates must match the imported P1 wall and wake topology")
    xyz = torch.stack((wall_xz[:, 0], torch.as_tensor(wing.xyz[:, 1]), wall_xz[:, 1]), dim=1)
    wake = torch.stack((wake_xz[:, 0], torch.as_tensor(wing.wake_xyz[:, 1]), wake_xz[:, 1]), dim=1)
    if wing.contact_pairs is not None and len(wing.contact_pairs):
        raise ValueError("Wake contact constraints are not supported by this derivative path")
    return fixed_wake_geometry(wing, xyz, wake, reference_area, alpha_deg, max_nfev)


def _coefficients(state, grad, areas, normals, reference_area, angle, wing, xyz,
                  include_ideal_drag):
    """Lift/drag coefficients of one reduced solution (explicit in x, quadratic in state)."""
    U = torch.stack((torch.cos(angle), torch.zeros_like(angle), torch.sin(angle)))
    lift = torch.stack((-torch.sin(angle), torch.zeros_like(angle), torch.cos(angle)))
    velocity = torch.einsum("fij,j->fi", grad, state[:wing.ndof_mu])
    cp = 1 - (velocity * velocity).sum(dim=1)
    force = (-(cp * areas)[:, None] * normals).sum(dim=0)
    coefficients = torch.stack((force @ lift, force @ U)) / reference_area
    if include_ideal_drag:
        y = xyz[torch.as_tensor(wing.te_geo), 1]
        gamma = torch.cat((state.new_zeros(1), state[wing.ndof_mu:], state.new_zeros(1)))
        left, right = y[:-1], y[1:]
        width = right - left
        centers = (left + right) / 2
        slope = torch.diff(gamma) / width
        logs = torch.log(torch.abs((centers[:, None] - left[None, :]) /
                                   (centers[:, None] - right[None, :])))
        downwash = -(logs @ slope) / (4 * torch.pi)
        drag = -((gamma[:-1] + gamma[1:]) * downwash * width).sum() / reference_area
        coefficients = torch.cat((coefficients, drag.reshape(1)))
    return coefficients


def _stationarity(state, B, rhs, reduction, pressure):
    """d/dstate of (|B s - rhs|^2 + 25 |p(s)|^2)/2, written out so it stays first order in x."""
    residual = B @ state - rhs
    p = pressure(state)
    return reduction.t() @ (B.t() @ residual + 25. * (pressure.jacobian(state).t() @ p))


class _FixedWakeImplicit(torch.autograd.Function):
    """Pressure-Kutta coefficients with an exact implicit gradient in one assembly pass.

    The forward solve runs under no_grad and stores only the reduced solution, the
    null-space basis and the (small) Kutta Hessian.  The backward rebuilds the
    assembly once with gradients and evaluates the adjoint identity

        dJ/dx = dJ/dx|_s - lambda^T dF/dx|_s,   lambda = H^-T (dJ/dz),

    where F is the stationarity residual written in closed form.  Both terms are
    first order in the coordinates, so no graph of a graph is ever built.  This is
    necessary for a cheap backward but not sufficient for a small one: the 15 GB
    peak at 1720 faces came from the assembly checkpoint recomputing every source
    block at once (see panel/p1_assembly_torch.py:_checkpointed), not from the
    double graph.
    """

    @staticmethod
    def forward(ctx, xyz, wake, reference_area, meta):
        wing = meta["wing"]
        alpha_deg = float(meta["alpha_deg"])
        include_ideal_drag = bool(meta["include_ideal_drag"])
        max_nfev = int(meta["max_nfev"])
        recompute = meta["recompute"]
        wake_xyz = wake if wake.numel() else None
        with torch.no_grad():
            model, base, mesh, A, W, b, mass, grad = galerkin_operators(
                wing, xyz, wake_xyz=wake_xyz, recompute=recompute)
            angle = torch.tensor(alpha_deg * np.pi / 180., dtype=xyz.dtype)
            U = torch.stack((torch.cos(angle), torch.zeros_like(angle), torch.sin(angle)))
            L = torch.linalg.cholesky(mass)
            B = torch.linalg.solve_triangular(L, torch.cat((A, W), dim=1), upper=False)
            rhs = torch.linalg.solve_triangular(L, (b @ U)[:, None], upper=False)[:, 0]
            J = torch.as_tensor(base["J"])
            pairs = [p for group in base["te_face_pairs"] for p in group]
            K = torch.stack([grad[top, 0].mean(dim=0) - grad[bottom, 0].mean(dim=0)
                             for top, bottom in pairs])
            K = torch.cat((K, torch.zeros((len(pairs), W.shape[1]), dtype=xyz.dtype)), dim=1)
            linear_basis = null_space(torch.cat((J, K), dim=0), rcond=1e-11)
            start = linear_basis @ torch.linalg.lstsq(B @ linear_basis, rhs).solution
            result = solve_pressure_kutta_torch(
                wall_operator=B, wall_rhs=rhs, kutta_operator=J, grad=grad,
                te_face_pairs=pairs, x0=start, weight=5., max_nfev=max_nfev)
            reduction = result.reduction
            z = result.reduced
            state = reduction @ z
            pressure = TorchPressureModel(grad, pairs, B.shape[1])
            stationarity = _stationarity(state, B, rhs, reduction, pressure)
            with torch.enable_grad():
                zt = z.detach().clone().requires_grad_(True)
                hessian = _hessian_rows(
                    _stationarity(reduction @ zt, B, rhs, reduction, pressure), zt)
            stationarity_inf = float(torch.linalg.vector_norm(stationarity, ord=float("inf")))
            factor, hessian_status = torch.linalg.cholesky_ex(hessian)
            # Scale-free gate.  |F| alone says nothing about convergence: the reduced
            # Hessian grows like O(faces^2) (8.9e3 at 132 faces, 4.6e7 at 6640), so the
            # same residual implies wildly different Newton steps.  Compare the step.
            if int(hessian_status) == 0 and bool(torch.isfinite(stationarity).all()):
                step = torch.cholesky_solve(stationarity[:, None], factor)[:, 0]
                relative_step = float(step.abs().max()) / max(float(z.abs().max()), 1e-30)
            else:
                relative_step = float("inf")
            if (not bool(torch.isfinite(stationarity).all())
                    or not bool(torch.isfinite(hessian).all())
                    or relative_step > 1e-8 or int(hessian_status) != 0):
                raise RuntimeError("pressure Kutta is not stationary; implicit gradient is not valid "
                                   "(status=%s, stationarity_inf=%.3g, relative_step=%.3g, "
                                   "hessian_status=%d)"
                                   % (result.result.status, stationarity_inf, relative_step,
                                      int(hessian_status)))
            coefficients = _coefficients(state, grad, mesh["areas"], mesh["normals"],
                                         reference_area, angle, wing, xyz, include_ideal_drag)
        ctx.save_for_backward(xyz, wake, reference_area, z, reduction, hessian)
        ctx.meta = meta
        meta["sink"]["info"] = dict(
            status=result.result.status, success=result.result.success,
            stationarity=float(torch.linalg.vector_norm(stationarity)),
            relative_step=relative_step,
            stationary_verified=True, evaluations=result.result.nfev,
            physical_accuracy_accepted=False)
        return coefficients

    @staticmethod
    def backward(ctx, grad_coef):
        xyz, wake, reference_area, z, reduction, hessian = ctx.saved_tensors
        meta = ctx.meta
        wing = meta["wing"]
        alpha_deg = float(meta["alpha_deg"])
        include_ideal_drag = bool(meta["include_ideal_drag"])
        recompute = meta["recompute"]
        with torch.enable_grad():
            xg = xyz.detach().clone().requires_grad_(True)
            wg = wake.detach().clone().requires_grad_(True) if wake.numel() else wake
            ag = reference_area.detach().clone().requires_grad_(True)
            model, base, mesh, A, W, b, mass, grad = galerkin_operators(
                wing, xg, wake_xyz=(wg if wg.numel() else None), recompute=recompute)
            angle = torch.tensor(alpha_deg * np.pi / 180., dtype=xyz.dtype)
            U = torch.stack((torch.cos(angle), torch.zeros_like(angle), torch.sin(angle)))
            L = torch.linalg.cholesky(mass)
            B = torch.linalg.solve_triangular(L, torch.cat((A, W), dim=1), upper=False)
            rhs = torch.linalg.solve_triangular(L, (b @ U)[:, None], upper=False)[:, 0]
            pairs = [p for group in base["te_face_pairs"] for p in group]
            pressure = TorchPressureModel(grad, pairs, B.shape[1])
            state = reduction @ z
            coef_x = _coefficients(state, grad, mesh["areas"], mesh["normals"], ag,
                                   angle, wing, xg, include_ideal_drag)
            residual_x = _stationarity(state, B, rhs, reduction, pressure)
            zt = z.detach().clone().requires_grad_(True)
            coef_z = _coefficients(reduction @ zt, grad.detach(), mesh["areas"].detach(),
                                   mesh["normals"].detach(), ag.detach(), angle, wing,
                                   xg.detach(), include_ideal_drag)
            dz = torch.autograd.grad((coef_z * grad_coef).sum(), zt)[0]
            lam = torch.linalg.solve(hessian.t(), dz).detach()
            total = (coef_x * grad_coef).sum() - (lam * residual_x).sum()
            targets = [xg, ag] + ([wg] if wg.numel() else [])
            grads = torch.autograd.grad(total, targets, allow_unused=True)
        grad_wake = grads[2] if wg.numel() else None
        return grads[0], grad_wake, grads[1], None



def _affine_state(B, rhs, J, nt, ng):
    """Affine map state(gamma) = [mu_base + mapping @ gamma; gamma]; see the shared module.

    Kept as a name here because the function objects below were written against it, but the
    arithmetic lives in panel/p1_potential_jump_torch.py so the production solver and this
    autograd path cannot drift apart.
    """
    return affine_state(B, rhs, J, nt, ng)


def _potential_jump_newton(pressure, mu_base, mapping, gamma, nt, max_nfev=12, tol=1e-12):
    """Newton on p(gamma) = 0; the shared implementation also returns the relative residual."""
    gamma, state, p, evaluations, scale, _relative, condition = potential_jump_newton(
        pressure, mu_base, mapping, gamma, nt, max_nfev=max_nfev, tol=tol)
    return gamma, state, p, evaluations, scale, condition


class _PotentialJumpImplicit(torch.autograd.Function):
    """Morino potential-jump Kutta closure with an implicit gradient in the ng unknowns.

    The wall equations are linear, so the wall solution is affine in the trailing-edge
    circulation; one least-squares pass with ng + 1 right-hand sides builds that map and a
    Newton iteration on the ng trailing-edge pressures p(gamma) = 0 replaces the full
    state-space pressure-Kutta least squares.  The adjoint system is ng x ng (81 x 81 at
    0.05 m) instead of ndof x ndof (3403 x 3403).
    """

    @staticmethod
    def forward(ctx, xyz, wake, reference_area, meta):
        wing = meta["wing"]
        alpha_deg = float(meta["alpha_deg"])
        include_ideal_drag = bool(meta["include_ideal_drag"])
        max_nfev = int(meta["max_nfev"])
        recompute = meta["recompute"]
        wake_xyz = wake if wake.numel() else None
        with torch.no_grad():
            model, base, mesh, A, W, b, mass, grad = galerkin_operators(
                wing, xyz, wake_xyz=wake_xyz, recompute=recompute)
            angle = torch.tensor(alpha_deg * np.pi / 180., dtype=xyz.dtype)
            U = torch.stack((torch.cos(angle), torch.zeros_like(angle), torch.sin(angle)))
            L = torch.linalg.cholesky(mass)
            B = torch.linalg.solve_triangular(L, torch.cat((A, W), dim=1), upper=False)
            rhs = torch.linalg.solve_triangular(L, (b @ U)[:, None], upper=False)[:, 0]
            J = torch.as_tensor(base["J"])
            nt, ng = A.shape[0], W.shape[1]
            mu_base, mapping = _affine_state(B, rhs, J, nt, ng)
            pairs = [p for group in base["te_face_pairs"] for p in group]
            pressure = TorchPressureModel(grad, pairs, B.shape[1])
            K = torch.stack([grad[top, 0].mean(dim=0) - grad[bottom, 0].mean(dim=0)
                             for top, bottom in pairs])
            K = torch.cat((K, torch.zeros((len(pairs), ng), dtype=xyz.dtype)), dim=1)
            basis = null_space(torch.cat((J, K), dim=0), rcond=1e-11)
            linear = basis @ torch.linalg.lstsq(B @ basis, rhs).solution
            gamma, state, p, evaluations, scale, condition = _potential_jump_newton(
                pressure, mu_base, mapping, linear[nt:], nt, max_nfev=max_nfev)
            p_inf = float(p.abs().max())
            # The Newton system is square, so a converged p is ~1e-15 by construction; that
            # makes |p| useless as an absolute gate.  Compare it with the pressure scale of
            # the initial linear-closure state instead, and never hand back a non-stationary
            # point whose implicit gradient would be silently wrong.
            relative_p = p_inf / max(scale, 1.0)
            if not bool(torch.isfinite(p).all()) or relative_p > 1e-10:
                raise RuntimeError(
                    "potential-jump Kutta did not converge; implicit gradient is not valid "
                    "(evaluations=%d, stationarity_inf=%.3g, relative=%.3g, max_nfev=%d)"
                    % (evaluations, p_inf, relative_p, max_nfev))
            # A converged residual does not make gamma unique; a near-null Jacobian direction
            # (trailing edge swept into the stream direction at a tip) would make the implicit
            # gradient just as arbitrary, so refuse it here too.
            if condition is not None and condition > 1e6:
                raise RuntimeError(
                    "potential-jump Kutta is ill-conditioned: cond(dp/dgamma)=%.3g exceeds "
                    "1e+06, so the trailing-edge circulation is not unique and the implicit "
                    "gradient is not valid (evaluations=%d, stationarity_inf=%.3g)"
                    % (condition, evaluations, p_inf))
            coefficients = _coefficients(state, grad, mesh["areas"], mesh["normals"],
                                         reference_area, angle, wing, xyz, include_ideal_drag)
        ctx.save_for_backward(xyz, wake, reference_area, gamma)
        ctx.meta = meta
        meta["sink"]["info"] = dict(
            status=0, success=True, kutta="potential_jump",
            stationarity=p_inf, stationarity_scale=scale, relative_step=relative_p,
            evaluations=evaluations, stationary_verified=True, condition=condition,
            physical_accuracy_accepted=False)
        return coefficients

    @staticmethod
    def backward(ctx, grad_coef):
        xyz, wake, reference_area, gamma = ctx.saved_tensors
        meta = ctx.meta
        wing = meta["wing"]
        alpha_deg = float(meta["alpha_deg"])
        include_ideal_drag = bool(meta["include_ideal_drag"])
        recompute = meta["recompute"]
        with torch.enable_grad():
            xg = xyz.detach().clone().requires_grad_(True)
            wg = wake.detach().clone().requires_grad_(True) if wake.numel() else wake
            ag = reference_area.detach().clone().requires_grad_(True)
            model, base, mesh, A, W, b, mass, grad = galerkin_operators(
                wing, xg, wake_xyz=(wg if wg.numel() else None), recompute=recompute)
            angle = torch.tensor(alpha_deg * np.pi / 180., dtype=xg.dtype)
            U = torch.stack((torch.cos(angle), torch.zeros_like(angle), torch.sin(angle)))
            L = torch.linalg.cholesky(mass)
            B = torch.linalg.solve_triangular(L, torch.cat((A, W), dim=1), upper=False)
            rhs = torch.linalg.solve_triangular(L, (b @ U)[:, None], upper=False)[:, 0]
            J = torch.as_tensor(base["J"])
            nt, ng = A.shape[0], W.shape[1]
            mu_base, mapping = _affine_state(B, rhs, J, nt, ng)
            state = torch.cat((mu_base + mapping @ gamma, gamma), dim=0)
            pairs = [p for group in base["te_face_pairs"] for p in group]
            pressure = TorchPressureModel(grad, pairs, B.shape[1])
            p = pressure(state)
            coef = _coefficients(state, grad, mesh["areas"], mesh["normals"], ag, angle,
                                 wing, xg, include_ideal_drag)
            jp = pressure.jacobian(state)
            dpdg = jp[:, :nt] @ mapping
            # The adjoint is one vector, not one per coefficient: dJ/dx = dJ/dx|_gamma
            # - lambda^T dp/dx with dp/dgamma^T lambda = dJ/dgamma.  Summing the per-coefficient
            # adjoints *unweighted* is only correct when every upstream weight is one (the
            # trap that made the first FD check of this closure pass).
            lam = torch.zeros(dpdg.shape[0], dtype=dpdg.dtype)
            for k in range(coef.numel()):
                dz = torch.autograd.grad(coef[k], state, retain_graph=True)[0]
                target = mapping.t() @ dz[:nt] + dz[nt:]
                lam = lam + grad_coef[k] * torch.linalg.solve(dpdg.t(), target).detach()
            total = (coef * grad_coef).sum() - (lam * p).sum()
            targets = [xg, ag] + ([wg] if wg.numel() else [])
            grads = torch.autograd.grad(total, targets, allow_unused=True)
        grad_wake = grads[2] if wg.numel() else None
        return grads[0], grad_wake, grads[1], None

def fixed_wake_geometry(wing, xyz, wake, reference_area, alpha_deg, max_nfev,
                        include_ideal_drag=False, recompute=None, kutta="pressure"):
    """Kutta-closed coefficients and the implicit gradient, at the cost of one assembly.

    kutta="pressure" (default) is the weighted pressure-Kutta least squares used by every
    existing study.  kutta="potential_jump" switches to the Morino closure solved by Newton
    in the ng trailing-edge unknowns (see docs/KUTTA_CONDITION.md); it agrees with the
    pressure solution to 1e-5 on fine meshes but finds a different root on coarse ones.
    """
    if wake is None:
        wake = torch.zeros(0, dtype=xyz.dtype)
    elif not torch.is_tensor(wake):
        wake = torch.as_tensor(wake, dtype=xyz.dtype)
    if not torch.is_tensor(reference_area):
        reference_area = torch.as_tensor(float(reference_area), dtype=xyz.dtype)
    sink = {}
    meta = dict(wing=wing, alpha_deg=alpha_deg, include_ideal_drag=include_ideal_drag,
                max_nfev=max_nfev, recompute=recompute, sink=sink)
    if kutta == "potential_jump":
        coefficients = _PotentialJumpImplicit.apply(xyz, wake, reference_area, meta)
    elif kutta == "pressure":
        coefficients = _FixedWakeImplicit.apply(xyz, wake, reference_area, meta)
    else:
        raise ValueError("unknown kutta closure %r" % (kutta,))
    return coefficients, sink["info"]