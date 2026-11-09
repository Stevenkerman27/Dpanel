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


def _condition(matrix):
    """Condition number of the closure Jacobian, or None when the SVD is skipped for cost."""
    if int(matrix.shape[0]) > 2048:
        return None
    singular = torch.linalg.svdvals(matrix)
    smallest = float(singular[-1])
    if smallest <= 0.0:
        return float("inf")
    return float(singular[0]) / smallest


def potential_jump_newton(pressure, mu_base, mapping, gamma, nm, max_nfev=12, tol=1e-12,
                          diagnostics=None, max_backtracks=12, step_tolerance=1e-10):
    """Newton on p(gamma) = 0 with the exact Jacobian dp/dgamma = (dp/dmu) @ mapping.

    Returns (gamma, state, p, evaluations, scale, relative_residual, condition).  Converged
    requires a small residual relative to the initial paired speed squared and a small
    Newton correction. An absolute residual threshold alone would only measure units.

    A small residual does NOT make the answer unique.  When the trailing edge is nearly swept
    into the stream direction at a tip, neighbouring jump rows become nearly collinear, the
    Jacobian condition number runs to 1e6, and Newton dumps an arbitrary circulation into the
    near-null direction while the residual still converges.  The largest condition number seen
    is therefore reported, and solve_potential_jump rejects the solution above a limit.
    """
    state = torch.cat((mu_base + mapping @ gamma, gamma), dim=0)
    p = pressure(state)
    tiny = torch.finfo(gamma.dtype).tiny
    scale = max(pressure.reference_scale(state) if hasattr(pressure, "reference_scale")
                else float(p.abs().max()), tiny)
    evaluations, backtracks, condition = 0, 0, None
    history = []
    gamma_scale = max(float(gamma.abs().max()), tiny)
    relative_step, dpdg = float("inf"), None
    for iteration in range(max_nfev + 1):
        if not bool(torch.isfinite(p).all()):
            break
        jp = pressure.jacobian(state)
        dpdg = jp[:, :nm] @ mapping + jp[:, nm:]
        current = _condition(dpdg)
        if current is not None:
            condition = current if condition is None else max(condition, current)
        # Check the Jacobian even when the initial residual is exactly zero.
        # Otherwise a singular root bypasses both the solve and the uniqueness gate.
        if current == float("inf"):
            break
        step = torch.linalg.solve(dpdg, p)
        sensitivity = float(torch.linalg.vector_norm(dpdg, ord=float("inf")))
        correction_scale = max(float(gamma.abs().max()), gamma_scale, scale / max(sensitivity, tiny))
        relative_step = float(step.abs().max()) / correction_scale
        history.append(dict(
            iteration=iteration, residual_relative=float(p.abs().max()) / scale,
            step_relative=relative_step, condition=current))
        if float(p.abs().max()) / scale <= tol and relative_step <= step_tolerance:
            break
        if iteration == max_nfev:
            break
        residual_norm = float(torch.linalg.vector_norm(p))
        damping, accepted = 1., False
        for attempt in range(max_backtracks + 1):
            trial_gamma = gamma - damping * step
            trial_state = torch.cat((mu_base + mapping @ trial_gamma, trial_gamma), dim=0)
            trial_p = pressure(trial_state)
            evaluations += 1
            if bool(torch.isfinite(trial_p).all()) and (
                    float(torch.linalg.vector_norm(trial_p)) <= (1. - 1e-4 * damping) * residual_norm):
                gamma, state, p = trial_gamma, trial_state, trial_p
                accepted = True
                break
            damping *= .5
            backtracks += 1
        history[-1]["damping"] = damping
        if not accepted:
            break
    relative = float(p.abs().max()) / scale
    if diagnostics is not None:
        scaled_condition = None
        if dpdg is not None:
            row_norm = torch.linalg.vector_norm(dpdg, dim=1).clamp_min(tiny)
            scaled = dpdg / row_norm[:, None]
            column_norm = torch.linalg.vector_norm(scaled, dim=0).clamp_min(tiny)
            scaled_condition = _condition(scaled / column_norm[None, :])
        diagnostics.update(
            relative_residual=relative, correction_relative=relative_step,
            condition_scaled=scaled_condition,
            backtracks=backtracks, history=history,
            gamma_norm=float(torch.linalg.vector_norm(gamma)))
    return gamma, state, p, evaluations, scale, relative, condition


def solve_potential_jump(B, rhs, J, pressure, gamma0, nm, max_nfev=12, tol=1e-12,
                         relative_tolerance=1e-10, condition_limit=1e6):
    """Solve the closure and return (state, diagnostics); raise if it did not converge.

    Two things can go wrong and both are refused rather than returned:

    * the Newton residual does not converge, so p(gamma) = 0 was never reached;
    * the residual converges but the Jacobian is too ill-conditioned for the answer to be
      unique (measured: 1.1e2 on a rectangular wing, 4.5e4 with a linear tip taper, 7.1e5 with
      a 62 degree trailing-edge sweep, 7.3e6 with the pure elliptic tip).  A converged point on
      a near-null direction carries a spurious tip circulation, so it is not a solution.
    """
    ng = int(gamma0.numel())
    mu_base, mapping = affine_state(B, rhs, J, nm, ng)
    trace = {}
    gamma, state, p, evaluations, scale, relative, condition = potential_jump_newton(
        pressure, mu_base, mapping, gamma0, nm, max_nfev=max_nfev, tol=tol, diagnostics=trace)
    diagnostics = dict(method="morino_potential_jump", evaluations=int(evaluations),
                       stationarity_inf=float(p.abs().max()), scale=float(scale),
                       relative_step=trace["correction_relative"], tolerance=float(relative_tolerance),
                       condition=condition, condition_limit=float(condition_limit),
                       max_nfev=int(max_nfev), success=True)
    diagnostics.update(trace)
    if condition is not None and condition > condition_limit:
        raise RuntimeError(
            "potential-jump Kutta is ill-conditioned: cond(dp/dgamma)=%.3g exceeds %.3g; "
            "the trailing-edge circulation is numerically unresolved "
            "(evaluations=%d, stationarity_inf=%.3g)"
            % (condition, condition_limit, evaluations, float(p.abs().max())))
    if not bool(torch.isfinite(p).all()) or relative > relative_tolerance \
            or trace["correction_relative"] > 1e-10:
        diagnostics["success"] = False
        raise RuntimeError(
            "potential-jump Kutta did not converge; the result is not a converged closure "
            "(evaluations=%d, stationarity_inf=%.3g, relative=%.3g, tolerance=%.3g, "
            "max_nfev=%d)" % (evaluations, float(p.abs().max()), relative,
                              relative_tolerance, max_nfev))
    return state, diagnostics
