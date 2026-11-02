"""Torch backend for the structured pressure-Kutta Gauss-Newton solve.

Route B (issue #92 comment 6569612502) requires the steady P1 solve to run in a
single PyTorch process: once torch has executed an op, the next NumPy/SciPy
LAPACK call in that interpreter dies with 0xC0000005 (measured, see
docs/ISSUE92_PHASE1_WAKE_EVALUATOR.md section 6f). The NumPy/SciPy backend
therefore cannot be used from inside a torch solve; this module is the torch
replacement for the reduced pressure-Kutta path, slice S1 of
docs/ISSUE92_TORCH_PORT_MAP.md.

Same algebra as panel/p1_pressure_kutta_reduced.py, call for call: the wall
block is factorised once by QR, the Gauss-Newton step is a Woodbury solve in
the pressure space, the stopping tests are scipy's gtol/ftol/xtol followed by
the relative-optimality and stagnation rules, and the line search is Armijo
backtracking. Only the linear algebra changes:

    np.linalg.qr                     -> torch.linalg.qr
    scipy.linalg.solve_triangular    -> torch.linalg.solve_triangular
    np.linalg.solve                  -> torch.linalg.solve
    np.linalg.norm                   -> torch.linalg.vector_norm
    scipy.optimize.OptimizeResult    -> TorchPressureResult (dataclass below)

The pressure residual and its Jacobian are supplied as callables that take and
return torch tensors. That is deliberate: the torch path must not hand NumPy
arrays back and forth, so the caller owns its torch implementation of the
pressure model. Nothing in this module imports scipy or calls NumPy linear
algebra.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

__all__ = ["TorchPressureResult", "TorchStructuredPressureKutta",
           "solve_structured_pressure_torch"]


@dataclass
class TorchPressureResult:
    """Mirror of the SciPy OptimizeResult fields the callers consume."""

    x: "torch.Tensor"
    fun: "torch.Tensor"
    cost: float
    optimality: float
    status: int
    success: bool
    message: str
    nfev: int
    nit: int
    relative_optimality: float = float("nan")
    backtracks: int = 0
    schur_size: int = 0
    free_dofs: int = 0
    method: str = "structured_gauss_newton_woodbury_torch"
    precompute_seconds: float | None = None
    history: list = field(default_factory=list)


def _as_tensor(value, name, dtype=torch.float64):
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, dtype=dtype)
    if value.dtype != dtype:
        value = value.to(dtype)
    if value.ndim != 2:
        raise ValueError("%s must be two-dimensional" % name)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("%s contains non-finite entries" % name)
    return value


class TorchStructuredPressureKutta:
    """Pre-factorised pressure-Kutta Gauss-Newton step, torch implementation.

    Parameters mirror StructuredPressureKutta except that the pressure callables
    are torch-native:

    pressure : callable
        pressure(v) -> tensor, with v the full state (length n_unknown) and the
        result a one-dimensional tensor of length n_pressure.
    pressure_jacobian : callable
        pressure_jacobian(v) -> (n_pressure, n_unknown) tensor; only the mu
        columns are used, as in the reference Jacobian.
    """

    def __init__(self, *, wall_operator, reduction, wall_rhs, pressure,
                 pressure_jacobian, weight=5., device=None, dtype=torch.float64):
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        B = _as_tensor(wall_operator, "wall_operator", dtype)
        N = _as_tensor(reduction, "reduction", dtype)
        if B.shape[1] != N.shape[0]:
            raise ValueError("wall_operator and reduction disagree on the unknown count")
        bw = torch.as_tensor(wall_rhs, dtype=dtype)
        if bw.ndim == 2 and bw.shape[1] == 1:
            bw = bw.reshape(-1)
        if bw.ndim != 1:
            raise ValueError("wall_rhs must have one entry per wall row")
        bw = bw.to(self.device)
        if bw.shape[0] != B.shape[0]:
            raise ValueError("wall_rhs must have one entry per wall row")
        if not bool(torch.isfinite(bw).all()):
            raise ValueError("wall_rhs contains non-finite entries")
        weight = float(weight)
        if not torch.isfinite(torch.tensor(weight)) or weight <= 0:
            raise ValueError("weight must be positive and finite")
        self.B = B.to(self.device)
        self.N = N.to(self.device)
        self.bw = bw
        self.pressure = pressure
        self.pressure_jacobian = pressure_jacobian
        self.weight = weight
        self.n_full = int(B.shape[1])
        self.n_free = int(N.shape[1])
        probe = torch.zeros(self.n_full, dtype=dtype, device=self.device)
        self.n_pressure = int(torch.as_tensor(self.pressure(probe)).reshape(-1).numel())
        M = self.B @ self.N
        Q, R = torch.linalg.qr(M, mode="reduced")
        diag = R.diagonal().abs()
        if diag.numel():
            if float(diag.min()) <= 1e-14 * max(float(diag.max()), 1e-300):
                raise ValueError(
                    "wall block is rank deficient after reduction (cond %.3e)"
                    % (float(diag.max()) / max(float(diag.min()), 1e-300)))
        self._M = M
        self._Q = Q
        self._R = R
        self._M_frobenius = float(torch.linalg.vector_norm(M))
        self.precompute_seconds = None
        self.condition_number_M = (float(diag.max() / diag.min())
                                   if diag.numel() and float(diag.min()) > 0 else None)

    # -- residual pieces -------------------------------------------------
    def _split(self, z):
        v = self.N @ z
        wall = self._M @ z - self.bw
        p = torch.as_tensor(self.pressure(v), dtype=self.dtype, device=self.device).reshape(-1)
        if p.numel() != self.n_pressure:
            raise ValueError("pressure residual changed size between iterations")
        return v, wall, p

    def residual(self, z):
        _, wall, p = self._split(z)
        return torch.cat((wall, self.weight * p))

    def cost(self, z):
        r = self.residual(z)
        return 0.5 * float(r @ r)

    def gradient(self, z):
        v, wall, p = self._split(z)
        P = torch.as_tensor(self.pressure_jacobian(v), dtype=self.dtype, device=self.device)
        if P.shape[0] != self.n_pressure or P.shape[1] != self.n_full:
            raise ValueError("pressure_jacobian must be (n_pressure, n_unknown)")
        F = self.weight * (P @ self.N)
        return self._M.T @ wall + F.T @ (self.weight * p), F

    # -- structured step --------------------------------------------------
    def _apply_inverse_gram(self, value):
        """Solve (M^T M) x = value through the QR factor, column-wise if needed."""
        arr = (value if isinstance(value, torch.Tensor)
               else torch.as_tensor(value, dtype=self.dtype, device=self.device))
        # torch.linalg.solve_triangular needs a matrix right-hand side, unlike
        # scipy, so a vector is promoted to one column and squeezed back.
        vector = arr.ndim == 1
        if vector:
            arr = arr.unsqueeze(-1)
        # Same elimination order as the reference: R^T first, then R. The other
        # order solves (R R^T)^-1 and is not a descent direction.
        forward = torch.linalg.solve_triangular(self._R.transpose(0, 1), arr, upper=False)
        out = torch.linalg.solve_triangular(self._R, forward, upper=True)
        return out.squeeze(-1) if vector else out

    def step(self, z):
        g, F = self.gradient(z)
        Ginv_b = self._apply_inverse_gram(-g)
        Ginv_Ft = self._apply_inverse_gram(F.transpose(0, 1))
        S = torch.eye(self.n_pressure, dtype=self.dtype, device=self.device) + F @ Ginv_Ft
        y = torch.linalg.solve(S, F @ Ginv_b)
        w = Ginv_b - Ginv_Ft @ y
        return w, g, S

    def _relative_optimality(self, g, F, r):
        jac_frobenius = torch.sqrt(
            torch.tensor(self._M_frobenius ** 2, dtype=self.dtype, device=self.device)
            + torch.sum(F * F))
        return float(torch.linalg.vector_norm(g, ord=float("inf"))
                     / max(float(jac_frobenius) * float(torch.linalg.vector_norm(r)), 1e-300))

    # -- driver -----------------------------------------------------------
    def solve(self, x0, *, max_nfev=100, ftol=1e-11, xtol=1e-11, gtol=1e-11,
              relative_gtol=1e-9, stagnation_gate=1e-6, stagnation_factor=1.05,
              armijo=1e-4, max_backtracks=12, record_history=True):
        z = torch.as_tensor(x0, dtype=self.dtype, device=self.device).reshape(-1).clone()
        if z.numel() != self.n_free:
            raise ValueError("x0 must have length %d" % self.n_free)
        v, wall, p = self._split(z)
        r = torch.cat((wall, self.weight * p))
        cost = 0.5 * float(r @ r)
        g, F = self.gradient(z)
        nfev = 1
        history = []
        status = 0
        message = "maximum number of function evaluations exceeded"
        optimality = float(torch.linalg.vector_norm(g, ord=float("inf")))
        relative = self._relative_optimality(g, F, r)
        best_relative = float("inf")
        stalled = 0
        accepted_steps = 0
        backtracks = 0
        while True:
            if record_history:
                history.append(dict(
                    iteration=accepted_steps, nfev=nfev, cost=cost,
                    gradient_inf=float(torch.linalg.vector_norm(g, ord=float("inf"))),
                    gradient_relative=float(relative),
                    wall_rms=float(torch.linalg.vector_norm(wall) / wall.numel() ** 0.5),
                    pressure_rms=float(torch.linalg.vector_norm(p) / p.numel() ** 0.5)))
            if optimality <= gtol:
                status = 1
                message = "gradient tolerance reached"
                break
            if relative <= relative_gtol:
                status = 4
                message = "relative gradient tolerance reached"
                break
            if relative <= stagnation_gate:
                if relative > stagnation_factor * best_relative:
                    stalled += 1
                    if stalled >= 2:
                        status = 5
                        message = "relative gradient stagnated"
                        break
                else:
                    stalled = 0
                best_relative = min(best_relative, relative)
            if nfev >= max_nfev:
                status = 0
                message = "maximum number of function evaluations exceeded"
                break
            w, g, S = self.step(z)
            slope = float(g @ w)
            if not bool(torch.isfinite(w).all()) or not bool(torch.isfinite(torch.tensor(slope))):
                status = 2
                message = "structured step is not finite"
                break
            if slope >= 0:
                status = 3
                message = "structured step is not a descent direction"
                break
            alpha = 1.0
            scale = max(1.0, float(torch.linalg.vector_norm(z)))
            accepted = False
            for _attempt in range(max_backtracks + 1):
                trial = z + alpha * w
                v_t, wall_t, p_t = self._split(trial)
                r_t = torch.cat((wall_t, self.weight * p_t))
                cost_t = 0.5 * float(r_t @ r_t)
                nfev += 1
                if cost_t <= cost + armijo * alpha * slope:
                    accepted = True
                    break
                alpha *= 0.5
                backtracks += 1
                if nfev >= max_nfev:
                    break
            if not accepted:
                status = 0
                message = "line search failed to reduce the residual"
                break
            reduction = cost - cost_t
            step_norm = float(torch.linalg.vector_norm(alpha * w))
            z, v, wall, p, r, cost = trial, v_t, wall_t, p_t, r_t, cost_t
            g, F = self.gradient(z)
            nfev += 1
            optimality = float(torch.linalg.vector_norm(g, ord=float("inf")))
            relative = self._relative_optimality(g, F, r)
            accepted_steps += 1
            if ftol > 0.0 and reduction <= ftol * cost:
                status = 2
                message = "ftol termination condition is satisfied"
                break
            if xtol > 0.0 and step_norm <= xtol * (xtol + scale):
                status = 3
                message = "xtol termination condition is satisfied"
                break
        return TorchPressureResult(
            x=z, fun=r, cost=cost, optimality=optimality, status=status,
            success=bool(status > 0), message=message, nfev=nfev, nit=accepted_steps,
            relative_optimality=relative, backtracks=backtracks,
            schur_size=self.n_pressure, free_dofs=self.n_free, history=history)


def solve_structured_pressure_torch(*, wall_operator, reduction, wall_rhs, pressure,
                                    pressure_jacobian, x0, weight=5., max_nfev=100,
                                    ftol=1e-11, xtol=1e-11, gtol=1e-11,
                                    max_backtracks=12, device=None,
                                    dtype=torch.float64):
    """Convenience wrapper: build the factorisation and run the torch driver."""
    t0 = time.perf_counter()
    problem = TorchStructuredPressureKutta(
        wall_operator=wall_operator, reduction=reduction, wall_rhs=wall_rhs,
        pressure=pressure, pressure_jacobian=pressure_jacobian, weight=weight,
        device=device, dtype=dtype)
    problem.precompute_seconds = time.perf_counter() - t0
    result = problem.solve(x0, max_nfev=max_nfev, ftol=ftol, xtol=xtol, gtol=gtol,
                           max_backtracks=max_backtracks)
    result.precompute_seconds = problem.precompute_seconds
    return result
