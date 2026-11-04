"""Torch port of the rank-aware decision points of the P1 steady solve (issue #92, S2).

The NumPy/SciPy path decides twice per solve:

* panel.p1_lifting_g1.solve builds the feasible subspace with
  null_space(constraints, rcond=1e-11) and then solves the wall equations with
  lstsq(B @ N, bw, lapack_driver="gelsd");
* the coupled path and the alpha sensitivity repeat the same two calls.

This module reproduces both with torch.linalg.svd and reports what the decision
depended on: the rank, the null-space dimension, the singular value that was
truncated, the smallest singular value that was kept, and their ratio. The ratio
is the quantity that decides whether a backend difference of ~1e-14 in a singular
vector can flip the rank - the reason the port map asks for a gap table per
fixture instead of a pass word.

Two facts shape the parity claim:

* the null-space basis is not unique (any orthogonal basis of the same subspace is
  correct), so a basis-vector comparison between backends is meaningless; the
  unique things to compare are the subspace itself, the rank, the singular values
  and the projected solution N @ t;
* scipy.linalg.null_space truncates at s.max() * rcond and keeps the trailing
  right-singular vectors; gelsd truncates at rcond * s.max() with rcond defaulting
  to max(M, N) * eps. Both thresholds are reproduced exactly, defaults included.

No NumPy or SciPy import here: a NumPy LAPACK call after torch has executed in the
same interpreter dies with 0xC0000005
(docs/archive/issue/ISSUE92_PHASE1_WAKE_EVALUATOR.md section 6f).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


def _tensor(value, *, dtype=torch.float64, device=None):
    if isinstance(value, torch.Tensor):
        out = value.to(dtype=dtype)
        return out if device is None else out.to(device=device)
    return torch.as_tensor(value, dtype=dtype, device=device)


@dataclass
class RankReport:
    """What a rank decision was made on."""

    rows: int
    columns: int
    rank: int
    null_dim: int
    largest: float
    smallest_kept: float
    largest_discarded: float
    rcond: float
    threshold: float
    basis_dimension: int = 0
    notes: dict = field(default_factory=dict)

    def as_dict(self):
        return dict(rows=self.rows, columns=self.columns, rank=self.rank,
                    null_dim=self.null_dim, largest=self.largest,
                    smallest_kept=self.smallest_kept,
                    largest_discarded=self.largest_discarded,
                    rcond=self.rcond, threshold=self.threshold,
                    basis_dimension=self.basis_dimension,
                    gap_ratio=self.gap_ratio, margin=self.margin,
                    notes=dict(self.notes))

    @property
    def gap_ratio(self):
        """Smallest kept singular value divided by the largest discarded one."""
        if self.largest_discarded <= 0.0:
            return float("inf")
        if self.smallest_kept <= 0.0:
            return 0.0
        return self.smallest_kept / self.largest_discarded

    @property
    def margin(self):
        """How far the decisive singular value sits from the threshold (1 = on it)."""
        if self.threshold <= 0.0:
            return float("inf")
        decisive = self.smallest_kept if self.smallest_kept > 0.0 else self.largest_discarded
        return decisive / self.threshold


def _check_rcond(rcond, name="rcond"):
    if rcond is None:
        return None
    value = float(rcond)
    if not (value > 0.0) or value != value or value == float("inf"):
        raise ValueError(name + " must be a positive finite number")
    return value


def _matrix(matrix, *, dtype, device):
    values = _tensor(matrix, dtype=dtype, device=device)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("matrix must not be empty")
    if not torch.isfinite(values).all():
        raise ValueError("matrix contains non-finite entries")
    return values


def _describe(singular, rcond, rows, columns, rank):
    largest = float(singular[0]) if singular.numel() else 0.0
    threshold = largest * rcond
    kept = singular[singular > threshold]
    discarded = singular[singular <= threshold]
    return RankReport(rows=rows, columns=columns, rank=rank,
                      null_dim=columns - rank, largest=largest,
                      smallest_kept=float(kept.min()) if kept.numel() else 0.0,
                      largest_discarded=float(discarded.max()) if discarded.numel() else 0.0,
                      rcond=rcond, threshold=threshold)


def null_space(matrix, *, rcond=1e-11, dtype=torch.float64, device=None, report=False):
    """Trailing right-singular vectors, with scipy.linalg.null_space semantics.

    Truncation is s > s.max() * rcond; the basis is (columns, null_dim) and
    orthonormal by construction, like the SciPy result.
    """
    rcond = _check_rcond(rcond)
    values = _matrix(matrix, dtype=dtype, device=device)
    _, singular, vh = torch.linalg.svd(values, full_matrices=True)
    threshold = singular[0] * rcond
    rank = int((singular > threshold).sum())
    basis = vh[rank:].mT.contiguous()
    if not report:
        return basis
    info = _describe(singular, rcond, int(values.shape[0]), int(values.shape[1]), rank)
    info.basis_dimension = int(basis.shape[1])
    return basis, info


@dataclass
class LeastSquaresResult:
    """Minimum-norm truncated least-squares solution with its decision record."""

    x: torch.Tensor
    rank: int
    residual: float
    singular: torch.Tensor
    report: RankReport

    def as_dict(self):
        return dict(rank=self.rank, residual=self.residual,
                    singular=[float(value) for value in self.singular],
                    report=self.report.as_dict())


def lstsq(matrix, rhs, *, rcond=None, dtype=torch.float64, device=None):
    """Minimum-norm least squares with gelsd truncation semantics.

    rcond defaults to max(rows, columns) * eps like LAPACK gelsd; singular values at
    or below rcond * s.max() are treated as zero, so the result is the minimum-norm
    solution of the truncated problem. A one-dimensional right-hand side is accepted.
    """
    rcond = _check_rcond(rcond)
    values = _matrix(matrix, dtype=dtype, device=device)
    target = _tensor(rhs, dtype=dtype, device=device)
    rows, columns = int(values.shape[0]), int(values.shape[1])
    vector = target.ndim == 1
    if vector:
        target = target.unsqueeze(-1)
    if target.ndim != 2 or int(target.shape[0]) != rows:
        raise ValueError("right-hand side must have the number of rows of the matrix")
    if rcond is None:
        rcond = max(rows, columns) * float(torch.finfo(values.dtype).eps)
    u, singular, vh = torch.linalg.svd(values, full_matrices=False)
    threshold = singular[0] * rcond
    keep = singular > threshold
    inverse = torch.where(keep, 1.0 / singular, torch.zeros_like(singular))
    coefficients = u.mT @ target
    solution = vh.mT @ (inverse.unsqueeze(-1) * coefficients)
    residual = torch.linalg.vector_norm(values @ solution - target)
    rank = int(keep.sum())
    report = _describe(singular, rcond, rows, columns, rank)
    report.basis_dimension = columns
    if vector:
        solution = solution.squeeze(-1)
    return LeastSquaresResult(x=solution, rank=rank, residual=float(residual),
                              singular=singular.detach(), report=report)


def constrained_solve(wall_operator, wall_rhs, constraints, *, rcond_null=1e-11,
                      rcond_lstsq=None, dtype=torch.float64, device=None):
    """The G1/G2 initial solve: project the wall equations onto the feasible subspace.

    Mirrors exact_constrained_solve in panel.p1_lifting_g1.solve:

        N = null_space(constraints, rcond=1e-11)
        t = lstsq(B @ N, bw)
        z = N @ t

    Returns the solution in the original unknown space plus both decision records.
    """
    basis, null_info = null_space(constraints, rcond=rcond_null, dtype=dtype,
                                  device=device, report=True)
    if basis.shape[1] == 0:
        raise ValueError("No degrees of freedom after Kutta constraints")
    operator = _tensor(wall_operator, dtype=dtype, device=device)
    rhs = _tensor(wall_rhs, dtype=dtype, device=device)
    reduced = operator @ basis
    solved = lstsq(reduced, rhs, rcond=rcond_lstsq, dtype=dtype, device=device)
    return basis @ solved.x, dict(null_space=null_info.as_dict(),
                                  least_squares=solved.as_dict())
