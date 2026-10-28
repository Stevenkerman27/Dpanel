"""CPU half-only streamed GMRES and near-field block preconditioner.

This optional host reference uses JAX-streamed HalfPanelOperator.matvec and
rmatvec. It retains no dense wall AIC but does NOT supply a JAX custom VJP;
solve_half_fields is the proven dense implicit geometry-gradient reference.
All iterative residuals refer to the true, unpreconditioned half equation.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator, gmres

from .kernel import triangle_potential
from .half_native import HalfPanelOperator


def _spatial_blocks(centers, ids, max_size):
    if len(ids) <= max_size:
        return [np.asarray(ids, dtype=np.int32)]
    axis = np.ptp(centers[ids], axis=0).argmax()
    order = ids[np.argsort(centers[ids, axis], kind="stable")]
    mid = len(order)//2
    return (_spatial_blocks(centers, order[:mid], max_size)
            + _spatial_blocks(centers, order[mid:], max_size))


class HalfNearFieldPreconditioner:
    """P=[[local D+image D,0],[-TE incidence,half circulation mass]]."""

    def __init__(self, operator: HalfPanelOperator, *, max_block_size=32):
        if max_block_size <= 0:
            raise ValueError("Positive near-field block size required")
        self.op = operator
        centers = np.asarray(operator.center)
        self.blocks = _spatial_blocks(
            centers, np.arange(operator.nf), max_block_size)
        self.factors = []
        for block in self.blocks:
            ids = jnp.asarray(block)
            targets = operator.center[ids]
            _, direct = triangle_potential(
                targets, operator.tri[ids], interior_diagonal=True,
                block_size=0, target_panel_ids=ids, source_panel_ids=ids)
            _, reflected = triangle_potential(
                targets, operator.image[ids], block_size=0)
            local = np.asarray(direct+reflected)
            fac = la.lu_factor(local)
            if np.any(np.abs(np.diag(fac[0])) < 1e-13):
                raise ValueError("Singular local block; no diagonal regularization")
            self.factors.append(fac)
        self.T = np.asarray(operator.T)
        self.mass_factor = la.lu_factor(np.asarray(operator.M))

    def apply(self, rhs, *, transpose=False):
        rhs = np.asarray(rhs, dtype=np.float64)
        if rhs.shape != (self.op.state_size,):
            raise ValueError("Preconditioner RHS length mismatch")
        n = self.op.nf
        out = np.empty_like(rhs)
        if not transpose:
            for ids, fac in zip(self.blocks, self.factors):
                out[ids] = la.lu_solve(fac, rhs[ids])
            out[n:] = la.lu_solve(
                self.mass_factor, rhs[n:]+self.T@out[:n])
        else:
            out[n:] = la.lu_solve(self.mass_factor, rhs[n:], trans=1)
            shifted = rhs[:n]+self.T.T@out[n:]
            for ids, fac in zip(self.blocks, self.factors):
                out[ids] = la.lu_solve(fac, shifted[ids], trans=1)
        return out


def solve_half_gmres(operator: HalfPanelOperator, *, tolerance=1e-10,
                     restart=40, max_cycles=20, preconditioner=None,
                     transpose=False, rhs=None):
    """Solve true half equation in CPU float64; forward and A.T supported.

    A wrong/mismatched preconditioner, non-convergence and non-finite
    residual are hard errors. This wrapper is not differentiable by itself.
    """
    if jax.default_backend() != "cpu":
        raise NotImplementedError("Host half GMRES is CPU-only")
    if tolerance <= 0 or restart < 1 or max_cycles < 1:
        raise ValueError("Invalid GMRES tolerances or limits")
    b = np.asarray(operator.rhs() if rhs is None else rhs, dtype=np.float64)
    n = operator.state_size
    if b.shape != (n,) or not np.isfinite(b).all():
        raise ValueError("Invalid/nonfinite half GMRES RHS")
    action = jax.jit(operator.rmatvec if transpose else operator.matvec)
    action(jnp.asarray(b)).block_until_ready()
    calls = [0]

    def apply(v):
        calls[0] += 1
        return np.array(action(jnp.asarray(v)), dtype=np.float64, copy=True)

    A = LinearOperator((n, n), matvec=apply, dtype=np.float64)
    M = None
    if preconditioner is not None:
        if preconditioner.op is not operator:
            raise ValueError("Half preconditioner belongs to a different geometry")
        M = LinearOperator(
            (n, n),
            matvec=lambda v: preconditioner.apply(v, transpose=transpose),
            dtype=np.float64)
    history = []
    solution, info = gmres(
        A, b, M=M, rtol=tolerance, atol=0.,
        restart=restart, maxiter=max_cycles,
        callback=history.append, callback_type="pr_norm")
    residual = float(np.linalg.norm(apply(solution)-b) /
                     max(np.linalg.norm(b), 1e-300))
    if info != 0 or residual > tolerance or not np.isfinite(residual):
        raise RuntimeError(
            f"Half GMRES did not converge: info={info}, residual={residual}")
    return solution, dict(
        transpose=bool(transpose),
        preconditioned=bool(M is not None),
        iterations=len(history),
        matvec_calls=calls[0],
        true_relative_residual=residual)
