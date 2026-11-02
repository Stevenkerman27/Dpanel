"""Implicit adjoints for the constrained least-squares solves (Route B, slice S8).

The forward solves in this repository end in a constrained least-squares problem:

    x = argmin ||B x - b||  subject to  J x = 0

with B the wall operator, J the Kutta (and jump) rows and b the freestream right-hand side.
The pressure-Kutta branch solves the same problem with one more row block, [B; w Jp(x)],
rebuilt at every Gauss-Newton step.

Differentiating those solves by recording the factorisation is both expensive and, for the
pressure branch, impossible without a differentiable trust-region solver. This module
differentiates them implicitly instead: it takes the state as the solution of a linear system
and pushes the adjoint through ONE transposed triangular solve at the converged state.

Two levels are provided, because they are exact to different orders:

* model=None - the linear (signed-speed) solve. The adjoint is exact: the state is the unique
  minimiser of a quadratic, so the implicit-function theorem holds exactly.
* model=TorchPressureModel(...) - the Gauss-Newton solve the production pressure branch
  actually performs. The adjoint is taken of the stacked system at the converged state, so it
  inherits the Gauss-Newton approximation with an error of order the pressure residual, which
  the caller can measure (pressure_rms) and which the tools report.

Nothing here differentiates the SVD that builds the null-space basis, and nothing
differentiates through a LAPACK driver: autograd only ever sees the load functional.
"""
from __future__ import annotations

import torch

from .p1_rank_aware_torch import null_space

DTYPE = torch.float64


def _device(device):
    if device is None:
        return torch.device("cpu")
    return torch.device(device)


def _transposed_adjoint(q, r, basis, grad_state):
    """dL/db for x = basis * (R^-1 Q^T b) given dL/dx.

    With M = B*basis = QR the state is x = basis R^-1 Q^T b, so
    dL/db = Q R^-T basis^T dL/dx. The two triangular solves are the only linear work the
    adjoint needs; no factorisation is recomputed unless the operator moved (the
    Gauss-Newton case, where Jp depends on the state).
    """
    inner = (basis.mT @ grad_state).unsqueeze(-1)
    coef = torch.linalg.solve_triangular(r.mT, inner, upper=False).squeeze(-1)
    return q @ coef


class AdjointProblem:
    """A constrained least-squares problem that can report dL/dalpha implicitly.

    wall_operator (m, n) is B, kutta_operator (k, n) is J, centers (m, 3) and weights (m,)
    build the right-hand side from the freestream direction. Passing a model switches to the
    Gauss-Newton (pressure-Kutta) system; model must expose __call__(state) and
    jacobian(state) like panel.p1_pressure_kutta_torch_driver.TorchPressureModel.
    """

    def __init__(self, *, wall_operator, kutta_operator, centers, weights, model=None,
                 te_face_pairs=None, weight=5.0, rcond=1e-11, device=None,
                 hessian="exact", max_nfev=None):
        self.device = _device(device)
        self.wall_operator = torch.as_tensor(wall_operator, dtype=DTYPE, device=self.device)
        self.kutta_operator = torch.as_tensor(kutta_operator, dtype=DTYPE, device=self.device)
        self.centers = torch.as_tensor(centers, dtype=DTYPE, device=self.device)
        self.weights = torch.as_tensor(weights, dtype=DTYPE, device=self.device)
        self.model = model
        self.te_face_pairs = None if te_face_pairs is None else list(te_face_pairs)
        self.weight = float(weight)
        self.rcond = float(rcond)
        if hessian not in ("exact", "gauss_newton"):
            raise ValueError("hessian must be 'exact' or 'gauss_newton'")
        self.hessian = hessian
        self.max_nfev = None if max_nfev is None else int(max_nfev)
        self.last_pressure_residual = None
        self.last_pressure_solve = None
        rows, columns = int(self.wall_operator.shape[0]), int(self.wall_operator.shape[1])
        if rows < columns:
            raise ValueError("the wall operator must have at least as many rows as columns")
        self.basis = null_space(self.kutta_operator, rcond=self.rcond, device=self.device)
        if int(self.basis.shape[1]) == 0:
            raise ValueError("the constraints consume every degree of freedom")
        self._factor = self._factorise(self.wall_operator)

    def _factorise(self, operator):
        reduced = operator @ self.basis
        q, r = torch.linalg.qr(reduced, mode="reduced")
        if bool((torch.diagonal(r).abs() <= 0.0).any()):
            raise ValueError("the reduced operator is rank deficient; the adjoint is undefined")
        return q, r

    def freestream(self, alpha_deg):
        alpha = torch.deg2rad(torch.as_tensor(alpha_deg, dtype=DTYPE, device=self.device))
        zero = torch.zeros_like(alpha)
        return torch.stack((torch.cos(alpha), zero, torch.sin(alpha)))

    def direction_derivative(self, alpha_deg):
        """d(freestream)/d(alpha in degrees)."""
        alpha = torch.deg2rad(torch.as_tensor(alpha_deg, dtype=DTYPE, device=self.device))
        scale = torch.pi/180.0
        return scale*torch.stack((-torch.sin(alpha), torch.zeros_like(alpha),
                                  torch.cos(alpha)))

    def rhs(self, alpha_deg):
        """b = weights * (-centers . U), the production right-hand side."""
        return self.weights*(-(self.centers @ self.freestream(alpha_deg)))

    def rhs_jacobian(self, alpha_deg):
        """db/dalpha, one vector, for the scalar angle the tools differentiate."""
        return self.weights*(-(self.centers @ self.direction_derivative(alpha_deg)))

    def linear_state(self, alpha_deg):
        q, r = self._factor
        rhs = self.rhs(alpha_deg)
        coef = torch.linalg.solve_triangular(r, (q.mT @ rhs).unsqueeze(-1), upper=True)
        return self.basis @ coef.squeeze(-1)

    def state(self, alpha_deg):
        if self.model is None:
            return self.linear_state(alpha_deg)
        if self.te_face_pairs is None:
            raise ValueError("a pressure model needs te_face_pairs for the forward solve")
        from .p1_pressure_kutta_torch_driver import solve_pressure_kutta_torch
        start = self.linear_state(alpha_deg)
        options = {} if self.max_nfev is None else {"max_nfev": self.max_nfev}
        solve = solve_pressure_kutta_torch(
            wall_operator=self.wall_operator, wall_rhs=self.rhs(alpha_deg),
            kutta_operator=self.kutta_operator, grad=self.model.grad,
            te_face_pairs=self.te_face_pairs, x0=start, weight=self.weight,
            device=self.device, **options)
        self.last_pressure_residual = self.model(solve.state)
        self.last_pressure_solve = solve
        return solve.state

    def rhs_gradient(self, state, grad_state):
        """dL/db given dL/dstate, at the supplied converged state."""
        if self.model is None:
            q, r = self._factor
            return _transposed_adjoint(q, r, self.basis, grad_state)
        if self.hessian == "exact":
            return self._exact_rhs_gradient(state, grad_state)
        jacobian = self.model.jacobian(state)
        stacked = torch.vstack((self.wall_operator, self.weight*jacobian))
        q, r = self._factorise(stacked)
        full = _transposed_adjoint(q, r, self.basis, grad_state)
        return full[: int(self.wall_operator.shape[0])]

    def _face_weights(self, pressure):
        """Per-face multiplier of the second-order term, sum_p p_p d2p_p/dx2."""
        model = self.model
        weights = torch.zeros(int(model.grad.shape[0]), dtype=DTYPE, device=self.device)
        top = (pressure.unsqueeze(-1)*model.top_mask/model.top_count).reshape(-1)
        bottom = (pressure.unsqueeze(-1)*model.bot_mask/model.bot_count).reshape(-1)
        weights.index_add_(0, model.top_rows.reshape(-1), top)
        weights.index_add_(0, model.bot_rows.reshape(-1), -bottom)
        return weights

    def _exact_rhs_gradient(self, state, grad_state):
        """dL/db from the exact stationarity condition, not its Gauss-Newton model.

        The pressure residual is quadratic in the state, so the objective
        0.5||B x - b||^2 + 0.5 w^2 ||p(x)||^2 is quartic and its Gauss-Newton operator is not
        its Hessian: the difference is w^2 * sum_p p_p d2p_p/dx2, a state-independent
        quadratic form that the per-face weights below assemble exactly. Dropping it is only
        legitimate when the pressure residual is negligible, which is precisely what the S8
        readings show is false on the flat-valley CAD fixture.
        """
        model = self.model
        weight = self.weight
        jacobian = model.jacobian(state)
        stacked = torch.vstack((self.wall_operator, weight*jacobian))
        reduced = stacked @ self.basis
        hessian = reduced.mT @ reduced
        pressure = model(state)
        face_weights = self._face_weights(pressure)
        grad = model.grad
        extra = torch.einsum("fic,fid->cd", grad, grad*face_weights[:, None, None])
        wall_basis = self.basis[: int(model.n_wall)]
        hessian = hessian+2.0*weight*weight*(wall_basis.mT @ extra @ wall_basis)
        right = self.basis.mT @ grad_state
        multiplier = torch.linalg.solve(hessian, right.unsqueeze(-1)).squeeze(-1)
        return self.wall_operator @ (self.basis @ multiplier)

    def angle_gradient(self, alpha_deg, state, grad_state):
        """dL/dalpha in degrees: what gradcheck and the finite differences test."""
        return self.rhs_gradient(state, grad_state) @ self.rhs_jacobian(alpha_deg)

    def stationarity(self, state, alpha_deg):
        """Residual of the implicit assumption: F = 0 at the reported state.

        A non-zero value means the forward solve did not reach the point the adjoint
        differentiates, and the finite difference of the solve will disagree with the adjoint
        for reasons that have nothing to do with the adjoint.
        """
        residual = self.wall_operator.mT @ (self.wall_operator @ state-self.rhs(alpha_deg))
        if self.model is not None:
            pressure = self.model(state)
            residual = residual+self.weight*self.weight*(self.model.jacobian(state).mT
                                                         @ pressure)
        reference = torch.linalg.vector_norm(self.wall_operator.mT @ self.rhs(alpha_deg))
        return torch.linalg.vector_norm(residual)/torch.clamp(reference, min=1e-30)

    def reduced_hessian(self, state):
        """The reduced Hessian the exact adjoint inverts (diagnostic only)."""
        if self.model is None:
            reduced = self.wall_operator @ self.basis
            return reduced.mT @ reduced
        jacobian = self.model.jacobian(state)
        stacked = torch.vstack((self.wall_operator, self.weight*jacobian))
        reduced = stacked @ self.basis
        hessian = reduced.mT @ reduced
        if self.hessian == "gauss_newton":
            return hessian
        face_weights = self._face_weights(self.model(state))
        grad = self.model.grad
        extra = torch.einsum("fic,fid->cd", grad, grad*face_weights[:, None, None])
        wall_basis = self.basis[: int(self.model.n_wall)]
        return hessian+2.0*self.weight*self.weight*(wall_basis.mT @ extra @ wall_basis)

    def conditioning(self, state, alpha_deg):
        report = dict(stationarity=float(self.stationarity(state, alpha_deg)),
                      condition=float(torch.linalg.cond(self.reduced_hessian(state))))
        solve = self.last_pressure_solve
        if solve is not None:
            report.update(forward_nfev=int(solve.result.nfev),
                          forward_status=int(solve.result.status))
        return report


class _ImplicitState(torch.autograd.Function):
    """State as a function of the angle, with the implicit adjoint in backward."""

    @staticmethod
    def forward(ctx, alpha_deg, problem):
        state = problem.state(alpha_deg)
        ctx.save_for_backward(alpha_deg, state)
        ctx.problem = problem
        return state

    @staticmethod
    def backward(ctx, grad_state):
        alpha_deg, state = ctx.saved_tensors
        problem = ctx.problem
        gradient = problem.angle_gradient(alpha_deg, state, grad_state.detach())
        return gradient.reshape(()), None


def implicit_state(alpha_deg, problem):
    """Differentiable state; the backward pass is the implicit adjoint of the solve."""
    return _ImplicitState.apply(alpha_deg, problem)


def finite_difference_angle(function, alpha_deg, *, step=1e-5):
    """Central difference of a scalar function of the angle, in degrees."""
    left = function(alpha_deg - step)
    right = function(alpha_deg + step)
    return (right - left)/(2.0*step)
