"""Issue #69: experimental rank-separated pressure Kutta selection for P1.

DPanel's lifting P1 wall equation is a DIRICHLET potential-trace equation,
not a Neumann collocation equation. An overdetermined P1 face-centre
collocation system generally has nonzero raw residual at its LS optimum.
This module tests whether well-separated, wall-weak modes can satisfy Kutta
without degrading that optimum. It does not replace solve_multi().
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import null_space, svd
from scipy.optimize import least_squares


@dataclass(frozen=True)
class RankControls:
    jump_rcond: float = 1e-11
    wall_rcond: float = 1e-11
    min_singular_gap: float = 100.0
    max_wall_relative_increase: float = .01
    max_nullspace_operator_ratio: float = 1e-10
    pressure_rms_tol: float = 1e-6
    max_nfev: int = 80


def pressure_squares(x, grad, pairs):
    """Signed TE upper-minus-lower |grad_s(mu)|**2, one row per station."""
    x = np.asarray(x, float)
    nm = grad.shape[2]
    v = np.einsum("fij,j->fi", grad, x[:nm], optimize=True)
    out = []
    for top, bottom in pairs:
        t, b = np.asarray(top, int), np.asarray(bottom, int)
        if not len(t) or not len(b) or min(t.min(), b.min()) < 0 or max(t.max(), b.max()) >= len(grad):
            raise ValueError("invalid pressure Kutta face-bank pairs")
        out.append(np.mean(np.sum(v[t]**2, axis=1)) -
                   np.mean(np.sum(v[b]**2, axis=1)))
    return np.asarray(out, float)


def pressure_jacobian(x, grad, pairs):
    """Exact derivative of pressure_squares with respect to [mu, Gamma]."""
    nm = grad.shape[2]
    v = np.einsum("fij,j->fi", grad, np.asarray(x)[:nm], optimize=True)
    out = []
    for top, bottom in pairs:
        t, b = np.asarray(top, int), np.asarray(bottom, int)
        dt = 2*np.einsum("fi,fij->j", v[t], grad[t])/len(t)
        db = 2*np.einsum("fi,fij->j", v[b], grad[b])/len(b)
        out.append(np.r_[dt-db, np.zeros(len(x)-nm)])
    return np.asarray(out)


def solve_wall_nullspace_kutta(B, rhs, J, grad, pairs, *,
                               seed=None, controls=None,
                               expected_kutta_count=None):
    """Minimise the actual wall LS first, then Kutta on a certified weak space.

    Every numerical-rank threshold and wall-residual increase is exposed.
    Success is only an experimental Kutta constraint result, not physical
    accuracy or a zero residual of the raw face-centre Dirichlet equation.
    """
    c = RankControls() if controls is None else controls
    settings = vars(c).copy()
    if any((not np.isfinite(v) or v <= 0) for v in settings.values()):
        raise ValueError("all rank controls must be positive/finite")
    if not 0 < c.jump_rcond < 1 or not 0 < c.wall_rcond < 1 or not isinstance(c.max_nfev, int):
        raise ValueError("invalid rank control")
    B = np.asarray(B, float)
    rhs = np.asarray(rhs, float).reshape(-1)
    J = np.asarray(J, float)
    grad = np.asarray(grad, float)
    if (B.ndim != 2 or rhs.shape != (B.shape[0],) or J.ndim != 2 or J.shape[1] != B.shape[1]
            or grad.ndim != 3 or grad.shape[1] != 3 or grad.shape[2] > B.shape[1]
            or len(pairs) == 0 or not all(np.isfinite(a).all() for a in (B, rhs, J, grad))):
        raise ValueError("incompatible wall, jump, or TE operators")
    N = null_space(J, rcond=c.jump_rcond)
    if not N.shape[1]:
        raise ValueError("TE jump consumes every state degree of freedom")
    E = B@N
    U, s, Vt = svd(E, full_matrices=True, lapack_driver="gesvd")
    s0 = float(s[0]) if len(s) else 0.
    rank = int(np.sum(s > s0*c.wall_rcond)) if s0 else 0
    d = E.shape[1]
    if 0 < rank < len(s):
        gap = float(s[rank-1]/s[rank]) if s[rank] else float("inf")
    elif rank > 0:
        gap = float("inf")
    else:
        gap = 0.
    diagnostics = dict(
        jump_rows=len(J), wall_rows=len(B), free_after_jump=d,
        numerical_wall_rank=rank, weak_mode_count=d-rank,
        wall_rcond=c.wall_rcond,
        sigma_max=s0, sigma_min_retained=float(s[rank-1]) if rank else None,
        sigma_max_discarded=float(s[rank]) if rank<len(s) else None,
        rank_boundary_gap=gap if np.isfinite(gap) else None,
        rank_gap_unbounded=not np.isfinite(gap),
        singular_values=s.tolist(),
        numerical_rank_not_physical_nullity=True)
    result = dict(status="not_accepted", rank=diagnostics, controls=settings,
                  wall_equation="Dirichlet P1 face-centre least-squares collocation",
                  physical_accuracy_accepted=False)
    if rank == 0:
        result["status"] = "blocked_zero_wall_rank"
        return result
    z_min = Vt[:rank].T@((U[:,:rank].T@rhs)/s[:rank])
    x_min = N@z_min
    r_min = B@x_min-rhs
    wall_min_norm = float(np.linalg.norm(r_min))
    r_kmin = pressure_squares(x_min, grad, pairs)
    jr = J@x_min
    result.update(
        wall_minimizer=x_min,
        wall_rms_minimizer=float(np.sqrt(np.mean(r_min*r_min))),
        jump_rms_minimizer=float(np.sqrt(np.mean(jr*jr))) if len(jr) else 0.,
        pressure_rms_wall_minimizer=float(np.sqrt(np.mean(r_kmin*r_kmin))),
        wall_stationarity_norm=float(np.linalg.norm(E.T@(E@z_min-rhs))))
    if d-rank == 0:
        result["status"] = "blocked_no_wall_nullspace"
        return result
    if gap < c.min_singular_gap:
        result["status"] = "blocked_no_spectral_gap"
        return result
    if expected_kutta_count is not None and d-rank != expected_kutta_count:
        result["status"] = "blocked_nullity_kutta_count_mismatch"
        return result
    T = N@Vt[rank:].T
    influence = float(np.linalg.norm(B@T, ord=2) /
                      max(float(np.linalg.norm(B, ord=2)), 1e-300))
    result["weak_mode_wall_operator_ratio"] = influence
    if influence > c.max_nullspace_operator_ratio:
        result["status"] = "blocked_weak_modes_alter_wall"
        return result
    if seed is None:
        t0 = np.zeros(T.shape[1])
    else:
        seed = np.asarray(seed, float)
        if seed.shape != (B.shape[1],) or not np.isfinite(seed).all():
            raise ValueError("invalid signed-speed seed")
        t0 = T.T@(seed-x_min)
    fun = lambda t: pressure_squares(x_min+T@t, grad, pairs)
    jac = lambda t: pressure_jacobian(x_min+T@t, grad, pairs)@T
    r_seed = fun(t0)
    fit = least_squares(fun, t0, jac=jac, max_nfev=c.max_nfev,
                        xtol=1e-11, ftol=1e-11, gtol=1e-11)
    x = x_min+T@fit.x
    wall = B@x-rhs
    kutta = fun(fit.x)
    # A consistent manufactured wall system can have a ~1e-15 minimum
    # residual. Relative growth against that number alone is meaningless.
    wall_delta = float(np.linalg.norm(wall)-wall_min_norm)
    wall_scale = max(wall_min_norm, 1e-10*float(np.linalg.norm(rhs)), 1e-12)
    w_growth = wall_delta/wall_scale
    k_rms = float(np.sqrt(np.mean(kutta*kutta)))
    sv_j = np.linalg.svd(jac(fit.x), compute_uv=False)
    jrank = int(np.sum(sv_j > sv_j[0]*1e-11)) if len(sv_j) and sv_j[0] else 0
    result.update(
        candidate=x, wall_rms=float(np.sqrt(np.mean(wall*wall))),
        wall_residual_max=float(np.max(np.abs(wall))),
        wall_relative_increase=w_growth, wall_abs_increase=wall_delta,
        wall_growth_reference_scale=wall_scale,
        pressure_kutta_rms=k_rms, pressure_kutta_max=float(np.max(np.abs(kutta))),
        pressure_seed_rms=float(np.sqrt(np.mean(r_seed*r_seed))),
        pressure_jacobian_rank=jrank, pressure_jacobian_singular_values=sv_j.tolist(),
        jump_rms=float(np.sqrt(np.mean((J@x)**2))) if len(J) else 0.,
        nonlinear=dict(success=bool(fit.success), message=str(fit.message),
                       nfev=int(fit.nfev), cost=float(fit.cost),
                       optimality=float(fit.optimality)))
    if w_growth > c.max_wall_relative_increase:
        result["status"] = "blocked_wall_residual_deterioration"
    elif k_rms > c.pressure_rms_tol:
        result["status"] = "blocked_pressure_not_satisfied"
    elif not fit.success:
        result["status"] = "blocked_nonlinear_termination"
    elif jrank < len(pairs):
        result["status"] = "blocked_degenerate_pressure_jacobian"
    else:
        result["status"] = "pressure_kutta_in_weak_wall_modes_not_physical_acceptance"
    return result


def solve_g2_rank_aware(mesh, alpha_deg=5., *, operators=None, controls=None):
    """Independent adapter using EXISTING G2 assembly and signed-Kutta seed."""
    from .p1_lifting_g2 import assemble_multi, solve_multi
    ops = assemble_multi(mesh) if operators is None else operators
    ms = ops["mesh"]
    theta = np.deg2rad(float(alpha_deg))
    u = np.array([np.cos(theta),0.,np.sin(theta)])
    weights = np.sqrt(ms.areas/ms.areas.mean())
    A,W = ops["A"],ops["W"]
    B = weights[:,None]*np.column_stack((A,W))
    b = -weights*(ms.centers@u)
    signed = solve_multi(mesh, alpha_deg, kutta="signed_speed", operators=ops)
    seed = np.r_[signed["mu"],signed["gamma"]]
    pairs = [p for group in ops["te_face_pairs"] for p in group]
    out = solve_wall_nullspace_kutta(
        B,b,ops["J"],ops["grad"],pairs,seed=seed,controls=controls,
        expected_kutta_count=len(ops["K"]))
    out["signed_speed"] = {k:signed[k] for k in
        ("CL","CL_gamma","CD_pressure","CM_y","wall_rms",
         "jump_rms","signed_kutta_rms","pressure_kutta_rms")}
    if "candidate" not in out:
        return out
    x = out["candidate"]
    mu, gamma = x[:A.shape[1]], x[A.shape[1]:]
    v = np.einsum("fij,j->fi",ops["grad"],mu)
    cp = 1-np.sum(v*v,axis=1)
    ff = -(cp*ms.areas)[:,None]*ms.normals
    force = ff.sum(axis=0)
    lift = np.array([-np.sin(theta),0.,np.cos(theta)])
    moment = np.cross(ms.centers-mesh.reference_point,ff).sum(axis=0)
    CL_gamma = sum(2*np.trapezoid(np.r_[0,gamma[ops["slices"][g.name]],0],g.gamma_y) /
                   mesh.reference_area for g in mesh.groups)
    out["candidate_state"] = dict(
        mu=mu,gamma=gamma,cp=cp,
        CL=float(force@lift/mesh.reference_area),
        CL_gamma=float(CL_gamma),
        CD_pressure=float(force@u/mesh.reference_area),
        CM_y=float(moment[1]/(mesh.reference_area*mesh.reference_chord)),
        raw_wall_rms=float(np.linalg.norm(A@mu+W@gamma+ms.centers@u) /
                           np.sqrt(len(ms.centers))),
        signed_kutta_rms=float(np.linalg.norm(ops["K"]@x) /
                               np.sqrt(len(ops["K"]))),
        physical_accuracy_accepted=False)
    return out
