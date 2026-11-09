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

* TorchPressureModel - the trailing-edge pressure residual and its Jacobian, on
  tensors, with the same padded-gather algebra as the NumPy loops (the pair lists
  are ragged, so they are packed into padded index tensors once).
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


def _pack_pairs(pairs, device):
    """Pack ragged (top, bottom) face-index lists into padded tensors plus masks."""
    if len(pairs) == 0:
        raise ValueError("pressure model needs at least one trailing-edge pair")
    tops = [torch.as_tensor(list(map(int, top)), dtype=torch.long, device=device) for top, _ in pairs]
    bots = [torch.as_tensor(list(map(int, bot)), dtype=torch.long, device=device) for _, bot in pairs]
    for indices, label in ((tops, "top"), (bots, "bottom")):
        if any(index.numel() == 0 for index in indices):
            raise ValueError("empty %s face list in a trailing-edge pair" % label)
    width = max(int(index.numel()) for index in tops + bots)

    def pad(indices):
        rows = torch.zeros((len(indices), width), dtype=torch.long, device=device)
        mask = torch.zeros((len(indices), width), dtype=torch.float64, device=device)
        for row, index in enumerate(indices):
            rows[row, : index.numel()] = index
            mask[row, : index.numel()] = 1.0
        return rows, mask

    top_rows, top_mask = pad(tops)
    bot_rows, bot_mask = pad(bots)
    return top_rows, top_mask, bot_rows, bot_mask


class TorchPressureModel:
    """Trailing-edge pressure residual and Jacobian on tensors.

    pressure(state)  = mean_top |v|^2 - mean_bottom |v|^2 per pair
    jacobian(state)  = d pressure / d state, with zeros in the Gamma columns

    where v = grad . mu and mu is the wall part of the state. The per-pair mean is
    taken over the declared face list, exactly as the NumPy implementation does.
    """

    def __init__(self, grad, te_face_pairs, n_unknown, *, device=None, dtype=torch.float64,
                 velocity_maps=None):
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        grad = torch.as_tensor(grad, dtype=dtype, device=self.device)
        if grad.ndim != 3:
            raise ValueError("grad must have shape (faces, 3, wall_dofs)")
        if grad.shape[0] == 0 or grad.shape[2] == 0:
            raise ValueError("grad must not be empty")
        self.grad = grad
        self.n_wall = int(grad.shape[2])
        self.n_unknown = int(n_unknown)
        if self.n_unknown < self.n_wall:
            raise ValueError("n_unknown cannot be smaller than the wall block")
        (self.top_rows, self.top_mask,
         self.bot_rows, self.bot_mask) = _pack_pairs(te_face_pairs, self.device)
        self.n_pressure = int(self.top_rows.shape[0])
        if int(self.bot_rows.shape[0]) != self.n_pressure:
            raise ValueError("top and bottom pair lists must have the same length")
        self.top_count = self.top_mask.sum(dim=1, keepdim=True)
        self.bot_count = self.bot_mask.sum(dim=1, keepdim=True)
        self.velocity_maps = None
        if velocity_maps is not None:
            top, bottom = (torch.as_tensor(value, dtype=dtype, device=self.device)
                           for value in velocity_maps)
            expected = (self.n_pressure, 3, self.n_wall)
            if top.shape != expected or bottom.shape != expected:
                raise ValueError("Recovered TE maps must be (pairs,3,wall_dofs)")
            self.velocity_maps = (top, bottom)

    def velocity(self, state):
        if state.ndim != 1:
            raise ValueError("state must be one-dimensional")
        return torch.einsum("fij,j->fi", self.grad, state[: self.n_wall])

    def _pair_mean(self, values, rows, mask, count):
        gathered = values[rows].squeeze(-1) * mask
        return gathered.sum(dim=1, keepdim=True) / count

    def paired_velocity(self, state):
        """Upper/lower reconstructed vectors used for branch and scale checks."""
        if self.velocity_maps is not None:
            return tuple(operator @ state[:self.n_wall] for operator in self.velocity_maps)
        velocity = self.velocity(state)
        top = (velocity[self.top_rows] * self.top_mask[:, :, None]).sum(dim=1) / self.top_count
        bottom = (velocity[self.bot_rows] * self.bot_mask[:, :, None]).sum(dim=1) / self.bot_count
        return top, bottom

    def reference_scale(self, state):
        top, bottom = self.paired_velocity(state)
        return float(torch.maximum((top * top).sum(dim=1).max(),
                                    (bottom * bottom).sum(dim=1).max()))

    def __call__(self, state):
        if self.velocity_maps is not None:
            top, bottom = self.velocity_maps
            return ((top @ state[:self.n_wall]) ** 2).sum(dim=1) \
                - ((bottom @ state[:self.n_wall]) ** 2).sum(dim=1)
        speed = (self.velocity(state) ** 2).sum(dim=1, keepdim=True)
        top = self._pair_mean(speed, self.top_rows, self.top_mask, self.top_count)
        bottom = self._pair_mean(speed, self.bot_rows, self.bot_mask, self.bot_count)
        return (top - bottom).reshape(-1)

    def jacobian(self, state):
        if self.velocity_maps is not None:
            top, bottom = self.velocity_maps
            wall = 2. * (torch.einsum("pi,pij->pj", top @ state[:self.n_wall], top)
                          - torch.einsum("pi,pij->pj", bottom @ state[:self.n_wall], bottom))
            zeros = state.new_zeros((self.n_pressure, self.n_unknown - self.n_wall))
            return torch.cat((wall, zeros), dim=1)
        velocity = self.velocity(state)
        pieces = []
        for rows in (self.top_rows, self.bot_rows):
            faces = velocity[rows]                      # (pairs, width, 3)
            jac = torch.einsum("pfi,pfij->pfj", faces, self.grad[rows])
            mask = self.top_mask if rows is self.top_rows else self.bot_mask
            count = self.top_count if rows is self.top_rows else self.bot_count
            pieces.append((jac * mask[:, :, None]).sum(dim=1, keepdim=True) / count[:, :, None])
        wall = 2.0 * (pieces[0] - pieces[1]).reshape(self.n_pressure, self.n_wall)
        zeros = torch.zeros((self.n_pressure, self.n_unknown - self.n_wall),
                            dtype=self.dtype, device=self.device)
        return torch.cat((wall, zeros), dim=1)


@dataclass
class TorchPressureSolve:
    """Full state plus the reduced solve that produced it."""

    state: "torch.Tensor"
    reduced: "torch.Tensor"
    reduction: "torch.Tensor"
    result: object


def solve_pressure_kutta_torch(*, wall_operator, wall_rhs, kutta_operator, grad,
                               te_face_pairs, x0, weight=5., max_nfev=80,
                               rcond=1e-11, device=None, dtype=torch.float64,
                               pressure_model=None):
    """Reduced pressure-Kutta solve on tensors; returns the full state.

    Parameters are the production objects: the weighted wall block and its right
    hand side, the Kutta row block whose null space is the feasible subspace, the
    face gradient, the ragged trailing-edge pairs, and the linear initial state.
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
    model = pressure_model if pressure_model is not None else TorchPressureModel(
        grad, te_face_pairs, wall.shape[1], device=device, dtype=dtype)
    solver = TorchStructuredPressureKutta(wall_operator=wall, reduction=reduction,
                                          wall_rhs=rhs, pressure=model,
                                          pressure_jacobian=model.jacobian,
                                          weight=weight, device=device, dtype=dtype)
    reduced0 = reduction.mT @ start
    result = solver.solve(reduced0, max_nfev=max_nfev)
    state = reduction @ result.x
    return TorchPressureSolve(state=state, reduced=result.x, reduction=reduction, result=result)
