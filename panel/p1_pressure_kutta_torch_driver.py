"""Torch driver for the nonlinear pressure-Kutta solve (issue #92, slice S3).

The production pressure-Kutta path (panel/p1_lifting_g1.py:492-526,
panel/p1_lifting_g2.py:294-316) solves

    min_t  || [ B ; w * J_pressure(N t) ] - [ bw ; 0 ] ||,   state = N t,

where N spans the null space of the Kutta rows J, B is the weighted wall block
and J_pressure is the top/bottom trailing-edge velocity difference. The reference
solver is scipy.optimize.least_squares (trust-region-reflective, ftol=xtol=gtol=1e-11,
max_nfev=80, analytic Jacobian); the numpy "structured" alternative is
panel/p1_pressure_kutta_reduced.py. This slice ports the *structured* algorithm to
torch, because a torch process cannot call into NumPy/SciPy LAPACK at all (0xC0000005,
docs/archive/issue/ISSUE92_PHASE1_WAKE_EVALUATOR.md section 6f) and because the structured step is
the one with a documented convergence path.

Two pieces live here:

* TorchPressureModel - the trailing-edge pressure residual and its derivatives
  from matched upper and lower velocity maps.
* solve_pressure_kutta_torch - reduce by the torch null space (S2), hand the
  pre-factorised problem to TorchStructuredPressureKutta (S1), and return the full
  state. No NumPy or SciPy call is made anywhere in this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from panel.p1_pressure_kutta_torch import TorchStructuredPressureKutta
from panel.p1_rank_aware_torch import null_space

__all__ = ["TorchPressureModel", "TorchPressureSolve", "solve_pressure_kutta_torch"]


class TorchPressureModel:
    """Squared-speed difference between the two banks at each TE station."""

    def __init__(self, velocity_maps, n_unknown, *, device=None, dtype=torch.float64):
        top, bottom = (torch.as_tensor(value, dtype=dtype, device=device)
                       for value in velocity_maps)
        if top.ndim != 3 or top.shape[1] != 3 or not top.numel() or bottom.shape != top.shape:
            raise ValueError("Recovered TE maps must have matching nonempty (stations,3,wall_dofs) shapes")
        self.n_pressure, _, self.n_wall = top.shape
        self.n_unknown = int(n_unknown)
        if self.n_unknown < self.n_wall:
            raise ValueError("n_unknown cannot be smaller than the wall block")
        self.velocity_maps = (top, bottom)

    def paired_velocity(self, state):
        """Upper/lower reconstructed vectors used for branch and scale checks."""
        return tuple(operator @ state[:self.n_wall] for operator in self.velocity_maps)

    def reference_scale(self, state):
        top, bottom = self.paired_velocity(state)
        return float(torch.maximum((top * top).sum(dim=1).max(),
                                    (bottom * bottom).sum(dim=1).max()))

    def __call__(self, state):
        top, bottom = self.paired_velocity(state)
        return (top ** 2).sum(dim=1) - (bottom ** 2).sum(dim=1)

    def jacobian(self, state):
        top, bottom = self.velocity_maps
        vt, vb = self.paired_velocity(state)
        wall = 2. * (torch.einsum("pi,pij->pj", vt, top) - torch.einsum("pi,pij->pj", vb, bottom))
        zeros = state.new_zeros((self.n_pressure, self.n_unknown - self.n_wall))
        return torch.cat((wall, zeros), dim=1)

    def reduced_curvature(self, residual, basis):
        """Sum p_i * Hessian(p_i) in reduced coordinates."""
        top, bottom = (operator @ basis[:self.n_wall] for operator in self.velocity_maps)
        weights = residual[:, None, None]
        columns = basis.shape[1]
        return 2. * (top.reshape(-1, columns).T @ (weights * top).reshape(-1, columns)
                     - bottom.reshape(-1, columns).T @ (weights * bottom).reshape(-1, columns))


@dataclass
class TorchPressureSolve:
    """Full state plus the reduced solve that produced it."""

    state: "torch.Tensor"
    reduced: "torch.Tensor"
    reduction: "torch.Tensor"
    result: object


def solve_pressure_kutta_torch(*, wall_operator, wall_rhs, kutta_operator, pressure_model,
                               x0, weight=5., max_nfev=80, rcond=1e-11, device=None,
                               dtype=torch.float64):
    """Reduced pressure-Kutta solve on tensors; returns the full state.

    Parameters are the production objects: the weighted wall block and its right
    hand side, the Kutta row block whose null space is the feasible subspace,
    the trailing-edge pressure model, and the linear initial state.
    """
    device = torch.device(device) if device is not None else torch.device("cpu")
    wall = torch.as_tensor(wall_operator, dtype=dtype, device=device)
    rhs = torch.as_tensor(wall_rhs, dtype=dtype, device=device).reshape(-1)
    kutta = torch.as_tensor(kutta_operator, dtype=dtype, device=device)
    start = torch.as_tensor(x0, dtype=dtype, device=device).reshape(-1)
    if wall.ndim != 2 or kutta.ndim != 2:
        raise ValueError("wall_operator and kutta_operator must be two-dimensional")
    if wall.shape[1] != kutta.shape[1] or start.shape[0] != wall.shape[1]:
        raise ValueError("wall_operator, kutta_operator and x0 disagree on the unknown count")
    if wall.shape[0] != rhs.shape[0]:
        raise ValueError("wall_rhs must have one entry per wall row")
    reduction = null_space(kutta, rcond=rcond, device=device, dtype=dtype)
    if reduction.shape[1] == 0:
        raise ValueError("All independent mu/Gamma DOFs consumed by constraints")
    solver = TorchStructuredPressureKutta(wall_operator=wall, reduction=reduction,
                                          wall_rhs=rhs, pressure=pressure_model,
                                          pressure_jacobian=pressure_model.jacobian,
                                          weight=weight, device=device, dtype=dtype)
    reduced0 = reduction.mT @ start
    result = solver.solve(reduced0, max_nfev=max_nfev)
    state = reduction @ result.x
    return TorchPressureSolve(state=state, reduced=result.x, reduction=reduction, result=result)
