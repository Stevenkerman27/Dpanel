"""Morino potential-jump Kutta closure, shared by the production and differentiable paths.

The wall equations are linear, so for a fixed circulation vector gamma the wall solution mu is
affine in gamma: mu = mu_base + mapping @ gamma.  The trailing-edge jump operator J pins
mu_upper - mu_lower = gamma, and the closure asks for the pressure difference across the
trailing edge to vanish, p(gamma) = 0.  One Newton pass on the ng unknowns gamma solves it;
because p depends on gamma only through mu, its Jacobian is dp/dgamma = (dp/dmu) @ mapping.

Why this lives in its own module: panel/p1_solve_torch.py documents that it imports neither
numpy nor scipy, and panel/p1_cst_gradient.py pulls both in plus the GRASS-plane path.  Keeping
the closure here lets the production solver and the autograd Function share one implementation
without either importing the other's dependencies.
"""

import torch

from .p1_rank_aware_torch import null_space


def _particular(Jc, Jg):
    """Min-norm particular solution of Jc @ mu = -Jg (zero if Jc is empty)."""
    if int(Jc.shape[0]) == 0:
        return torch.zeros((int(Jc.shape[1]), int(Jg.shape[1])), dtype=Jc.dtype,
                           device=Jc.device)
    gram = Jc @ Jc.t()
    try:
        return Jc.t() @ torch.linalg.solve(gram, -Jg)
    except Exception:  # rank-deficient constraint rows: fall back to least squares
        return torch.linalg.lstsq(Jc, -Jg).solution


def affine_state(B, rhs, J, nm, ng):
    """Build state(gamma) = [mu_base + mapping @ gamma; gamma] with J satisfied exactly."""
    Jc, Jg = J[:, :nm], J[:, nm:]
    particular = _particular(Jc, Jg)
    basis = null_space(Jc)
    eye = torch.eye(ng, dtype=B.dtype, device=B.device)
    targets = torch.cat((rhs[:, None],
                         rhs[:, None] - B[:, nm:] @ eye - B[:, :nm] @ particular), dim=1)
    free = torch.linalg.lstsq(B[:, :nm] @ basis, targets).solution
    mu = torch.cat((torch.zeros(nm, 1, dtype=B.dtype, device=B.device), particular), dim=1) \
        + basis @ free
    return mu[:, 0], mu[:, 1:] - mu[:, 0][:, None]


def potential_jump_newton(pressure, mu_base, mapping, gamma, nm, max_nfev=12, tol=1e-12):
    """Newton on p(gamma) = 0 with the exact Jacobian dp/dgamma = (dp/dmu) @ mapping.

    Returns (gamma, state, p, evaluations, scale, relative_residual).  Converged means the
    residual is small relative to its own initial scale, which is the scale-free form of the
    gate: the closure is a square system, so |p| -> 1e-15 is manufactured and an absolute
    threshold would only measure units.
    """
    state = torch.cat((mu_base + mapping @ gamma, gamma), dim=0)
    p = pressure(state)
    scale = max(float(p.abs().max()), 1e-30)
    evaluations = 0
    for _ in range(max_nfev):
        if float(p.abs().max()) <= tol:
            break
        jp = pressure.jacobian(state)
        dpdg = jp[:, :nm] @ mapping + jp[:, nm:]
        step = torch.linalg.solve(dpdg, p)
        gamma = gamma - step
        state = torch.cat((mu_base + mapping @ gamma, gamma), dim=0)
        p = pressure(state)
        evaluations += 1
        if not bool(torch.isfinite(p).all()):
            break
    relative = float(p.abs().max()) / max(scale, 1.0)
    return gamma, state, p, evaluations, scale, relative


def solve_potential_jump(B, rhs, J, pressure, gamma0, nm, max_nfev=12, tol=1e-12,
                         relative_tolerance=1e-10):
    """Solve the closure and return (state, diagnostics); raise if it did not converge."""
    ng = int(gamma0.numel())
    mu_base, mapping = affine_state(B, rhs, J, nm, ng)
    gamma, state, p, evaluations, scale, relative = potential_jump_newton(
        pressure, mu_base, mapping, gamma0, nm, max_nfev=max_nfev, tol=tol)
    diagnostics = dict(method="morino_potential_jump", evaluations=int(evaluations),
                       stationarity_inf=float(p.abs().max()), scale=float(scale),
                       relative_step=float(relative), tolerance=float(relative_tolerance),
                       max_nfev=int(max_nfev), success=True)
    if not bool(torch.isfinite(p).all()) or relative > relative_tolerance:
        diagnostics["success"] = False
        raise RuntimeError(
            "potential-jump Kutta did not converge; the result is not a converged closure "
            "(evaluations=%d, stationarity_inf=%.3g, relative=%.3g, tolerance=%.3g, "
            "max_nfev=%d)" % (evaluations, float(p.abs().max()), relative,
                              relative_tolerance, max_nfev))
    return state, diagnostics
