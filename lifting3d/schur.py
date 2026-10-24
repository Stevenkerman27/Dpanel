"""Exact block elimination of potential and circulation increments."""
import time

import numpy as np
from scipy.sparse.linalg import splu


def solve_step(a, b, c, d, residual):
    """Solve [A B; C D] delta=-r without factoring the augmented matrix."""
    started = time.perf_counter()
    n = a.shape[0]
    factor = splu(a.tocsc())
    potential_rhs = factor.solve(-residual[:n])
    response = factor.solve(b.toarray())
    schur = d.toarray()-c@response
    circulation_rhs = -residual[n:]-c@potential_rhs
    circulation = np.linalg.solve(schur, circulation_rhs)
    potential = potential_rhs-response@circulation
    update = np.r_[potential, circulation]
    block_residual = np.r_[a@potential+b@circulation, c@potential+d@circulation]+residual
    relative = float(np.linalg.norm(block_residual)/np.linalg.norm(residual))
    assert np.isfinite(update).all() and relative < 1e-9, "Schur solve did not satisfy the original block equations"
    singular_values = np.linalg.svd(schur, compute_uv=False)
    pivots = abs(factor.U.diagonal())
    diagnostics = dict(
        linear_solver="exact_schur", linear_relative_residual=relative,
        potential_block_unknowns=n, circulation_unknowns=d.shape[0],
        potential_lu_nonzeros=int(factor.L.nnz+factor.U.nnz),
        potential_lu_min_abs_pivot=float(pivots.min()),
        potential_lu_pivot_ratio=float(pivots.min()/pivots.max()),
        schur_min_singular_value=float(singular_values[-1]),
        schur_max_singular_value=float(singular_values[0]),
        schur_singular_value_ratio=float(singular_values[-1]/singular_values[0]),
        factorization_permutation="SuperLU default COLAMD, same as augmented reference",
        regularization="none; dense Schur solve uses all circulation unknowns",
        linear_solve_seconds=time.perf_counter()-started)
    return update, diagnostics
