"""Torch implementation of the coupled G1/G2 P1 solve (Route B, slice S5).

Mirrors panel.p1_lifting_g2.solve_multi with torch.linalg operations in place of
NumPy/SciPy LAPACK: null_space, lstsq, the structured pressure-Kutta solve, the
nonlinear acceptance classification and the load integration. The module imports
neither numpy nor scipy. Results are CPU torch tensors and Python floats; the host
caller may convert the tensors with numpy.asarray without touching LAPACK.

Operator assembly is performed on the host; the default weak wall uses a
Torch-generated target quadrature and no NumPy/SciPy LAPACK.
"""

import math
import time

import torch

from .p1_pressure_kutta_torch_driver import TorchPressureModel, solve_pressure_kutta_torch
from .p1_rank_aware_torch import lstsq, null_space

DTYPE = torch.float64


def _t(value, device):
    return torch.as_tensor(value, dtype=DTYPE, device=device).contiguous()


def _i(value, device):
    return torch.as_tensor(value, dtype=torch.long, device=device)


def _acceptance_torch(*, result, fun, jac, z, wall_operator, wall_rhs,
                      pressure_residual, gradient_tolerance=1e-6,
                      pressure_tolerance=1e-4, wall_floor_ratio=1.05):
    """pressure_kutta_acceptance with torch linear algebra. Same predicates."""
    rows = int(wall_operator.shape[0])
    residual = fun(z)
    jf = jac(z)
    jf_norm = float(torch.linalg.vector_norm(jf))
    r_norm = float(torch.linalg.vector_norm(residual))
    gradient_absolute = float(torch.linalg.vector_norm(jf.mT @ residual, ord=float("inf")))
    gradient_relative = gradient_absolute/(jf_norm*r_norm+1e-300)
    floor = lstsq(wall_operator, wall_rhs)
    wall_floor = float(torch.linalg.vector_norm(wall_operator @ floor.x - wall_rhs)
                       / math.sqrt(rows))
    wall_rms = float(torch.linalg.vector_norm(residual[:rows])/math.sqrt(rows))
    pressure_rms = float(torch.linalg.vector_norm(pressure_residual)
                         / math.sqrt(int(pressure_residual.shape[0])))
    stationary = bool(gradient_relative <= gradient_tolerance)
    at_floor = bool(wall_floor > 0 and wall_rms <= wall_floor_ratio*wall_floor)
    kutta_ok = bool(pressure_rms <= pressure_tolerance)
    accepted = bool(result.success) or bool(int(result.status) == 0 and stationary
                                            and at_floor and kutta_ok)
    return dict(success=accepted, message=str(result.message), nfev=int(result.nfev),
                status=int(result.status),
                cost=(None if getattr(result, "cost", None) is None else float(result.cost)),
                optimality=(None if getattr(result, "optimality", None) is None
                            else float(result.optimality)),
                floor_method="torch lstsq of min ||B null(J) z - bw||",
                accepted_at_residual_floor=bool(accepted and not bool(result.success)),
                gradient_absolute=gradient_absolute,
                gradient_relative=gradient_relative,
                residual_norm=r_norm, wall_rms=wall_rms, wall_floor_rms=wall_floor,
                wall_over_floor=(None if wall_floor <= 0 else float(wall_rms/wall_floor)),
                pressure_rms=pressure_rms,
                criteria=dict(stationary=stationary, wall_at_floor=at_floor,
                              kutta=kutta_ok),
                tolerances=dict(gradient_relative=gradient_tolerance,
                                pressure_rms=pressure_tolerance,
                                wall_floor_ratio=wall_floor_ratio))


def solve_multi_torch(mesh, alpha_deg=5., *, kutta="pressure", kut_weight=5.,
                      max_nfev=80, operators=None, device="cpu"):
    """Solve wall P1 mu and all independent wake Gamma, on torch."""
    if kutta not in ("pressure", "signed_speed"):
        raise ValueError("kutta=pressure or signed_speed")
    if not isinstance(max_nfev, int) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    start = time.perf_counter()
    if operators is None:
        from .p1_galerkin_wall import assemble_galerkin_multi
        operators = assemble_galerkin_multi(mesh)
    A = _t(operators["A"], device)
    W = _t(operators["W"], device)
    J = _t(operators["J"], device)
    K = _t(operators["K"], device)
    grad = _t(operators["grad"], device)
    slices = operators.get("slices")
    if slices is None:
        # A hand-built operator set may omit the per-group slices; the mapping is a
        # property of the mesh, so derive it rather than require the caller to repeat it.
        slices = {}
        offset = 0
        for group in mesh.groups:
            width = group.gamma_hat.shape[1]
            slices[group.name] = slice(offset, offset + width)
            offset += width
    ms = operators["mesh"]
    nm, ng = int(A.shape[1]), int(W.shape[1])
    alpha = math.radians(float(alpha_deg))
    U = torch.tensor([math.cos(alpha), 0., math.sin(alpha)], dtype=DTYPE, device=device)
    centers, areas = _t(ms.centers, device), _t(ms.areas, device)
    normals = _t(ms.normals, device)
    weak = operators.get("wall_formulation") == "galerkin"
    if weak:
        rhs = _t(operators["b_xyz"], device) @ U
        L = torch.linalg.cholesky(_t(operators["M"], device))
        B = torch.linalg.solve_triangular(L, torch.hstack((A, W)), upper=False)
        bw = torch.linalg.solve_triangular(L, rhs[:, None], upper=False)[:, 0]
    else:
        rhs = -(centers @ U)
        weights = torch.sqrt(areas/areas.mean())
        B = weights.unsqueeze(-1)*torch.hstack((A, W))
        bw = weights*rhs
    constraint = torch.vstack((J, K))
    basis = null_space(constraint, rcond=1e-11, device=device)
    if int(basis.shape[1]) == 0:
        raise ValueError("All independent mu/Gamma DOFs consumed by constraints")
    solved = lstsq(B @ basis, bw, device=device)
    x = basis @ solved.x
    rank = int(solved.rank)
    singular = solved.singular
    pairs = [p for group in operators["te_face_pairs"] for p in group]
    model = TorchPressureModel(grad, pairs, nm+ng, device=device)
    nonlinear, pressure_seconds = None, None
    rr = None
    if kutta == "pressure":
        reduction = null_space(J, rcond=1e-11, device=device)

        def fun(z):
            state = reduction @ z
            return torch.cat((B @ state - bw, kut_weight*model(state)))

        def jac(z):
            state = reduction @ z
            return torch.vstack((B, kut_weight*model.jacobian(state))) @ reduction

        t_pressure = time.perf_counter()
        pressure = solve_pressure_kutta_torch(
            wall_operator=B, wall_rhs=bw, kutta_operator=J, grad=grad,
            te_face_pairs=pairs, x0=x, weight=kut_weight,
            max_nfev=max_nfev, device=device)
        rr = pressure.result
        pressure_seconds = time.perf_counter()-t_pressure
        x = pressure.state
        nonlinear = _acceptance_torch(result=rr, fun=fun, jac=jac, z=rr.x,
                                      wall_operator=B @ reduction, wall_rhs=bw,
                                      pressure_residual=model(x))
    mu, gamma = x[:nm], x[nm:]
    velocity = torch.einsum("fij,j->fi", grad, mu)
    cp = 1.-(velocity*velocity).sum(dim=1)
    faceforces = -(cp*areas).unsqueeze(-1)*normals
    net = faceforces.sum(dim=0)
    moment = torch.cross(centers - _t(mesh.reference_point, device), faceforces,
                         dim=1).sum(dim=0)
    lift = torch.tensor([-math.sin(alpha), 0., math.cos(alpha)], dtype=DTYPE,
                        device=device)
    reference_area = float(mesh.reference_area)
    total_cl = float(net @ lift/reference_area)
    per = []
    for group in mesh.groups:
        force = faceforces[_i(group.face_ids, device)].sum(dim=0)
        G = gamma[slices[group.name]]
        full = torch.cat((torch.zeros(1, dtype=DTYPE, device=device), G,
                          torch.zeros(1, dtype=DTYPE, device=device)))
        stations = _t(group.gamma_y, device)
        gamma_integral = float(torch.trapezoid(full, stations))
        per.append(dict(name=group.name,
                        CL_pressure=float(force @ lift/reference_area),
                        CL_gamma=2.*gamma_integral/reference_area,
                        gamma=G.detach().cpu().tolist(),
                        gamma_y=list(group.gamma_y),
                        Gamma_integral=gamma_integral,
                        wall_faces=len(group.face_ids),
                        wake_triangles=len(group.wake_faces)))
    mu_cpu = mu.detach().cpu()
    gamma_cpu = gamma.detach().cpu()
    return dict(
        alpha_deg=float(alpha_deg), kutta=kutta, CL=total_cl,
        CL_gamma=float(sum(item["CL_gamma"] for item in per)),
        CD_pressure=float(net @ U/reference_area),
        CM_y=float(moment[1]/(reference_area*float(mesh.reference_chord))),
        force=net.detach().cpu().tolist(), mu=mu_cpu, cp=cp.detach().cpu(),
        gamma=gamma_cpu, group_results=per,
        jump_rms=float(torch.linalg.vector_norm(J @ x)/math.sqrt(int(J.shape[0]))),
        signed_kutta_rms=float(torch.linalg.vector_norm(K @ x)/math.sqrt(int(K.shape[0]))),
        pressure_kutta_rms=float(torch.linalg.vector_norm(model(x))
                                 /math.sqrt(int(K.shape[0]))),
        wall_rms=float(torch.linalg.vector_norm(A @ mu + W @ gamma - rhs)
                       /math.sqrt(int(rhs.shape[0]))),
        wall_dual_rms=(float(torch.linalg.vector_norm(B @ x - bw)/math.sqrt(int(rhs.shape[0])))
                       if weak else None),
        wall_formulation="galerkin" if weak else "collocation",
        linear_constrained_rank=rank,
        linear_singular_min=float(singular[-1]) if len(singular) else float("nan"),
        nonlinear=nonlinear, physical_accuracy_accepted=False,
        pressure_solver="structured" if kutta == "pressure" else None,
        pressure_status=(rr.status if rr is not None else None),
        pressure_success=(bool(rr.success) if rr is not None else None),
        pressure_message=(rr.message if rr is not None else None),
        pressure_nfev=(getattr(rr, "nfev", None) if rr is not None else None),
        pressure_seconds=pressure_seconds,
        pressure_history=(getattr(rr, "history", None) if rr is not None else None),
        backend="torch", device=str(device), solve_seconds=time.perf_counter()-start)
