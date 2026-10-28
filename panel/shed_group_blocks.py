"""Research-only algebra for independent shed-group weak Kutta blocks.

The caller must evaluate each group's W from its own actual wake geometry;
this module does not infer shedding edges, change the CAD or implement a
physically accepted multi-wing solver. It verifies *coupled* block algebra.
"""
from __future__ import annotations

import numpy as np


def te_weak_blocks(te_basis, te_weights, upper_faces, lower_faces, wall_count):
    """Return T, M and unscaled SPD mass from the SAME P1 TE quadrature as v1.

    Basis[e,q,i], weights[e,q]; T=(integrals.T @ incidence)/measure[:,None]
    M=mass/measure[:,None], mass_ij=sum_eq weights_eq * B_i * B_j.
    In general, mass != integrals.T @ integrals.
    """
    B = np.asarray(te_basis, dtype=float)
    w = np.asarray(te_weights, dtype=float)
    up = np.asarray(upper_faces)
    lo = np.asarray(lower_faces)
    if (B.ndim != 3 or not B.shape[0] or not B.shape[1] or not B.shape[2]
            or w.shape != B.shape[:2] or up.shape != (len(B),)
            or lo.shape != (len(B),) or not np.issubdtype(up.dtype, np.integer)
            or not np.issubdtype(lo.dtype, np.integer) or not isinstance(wall_count, int)
            or wall_count < 1 or not np.isfinite(B).all()
            or not np.isfinite(w).all() or np.any(w <= 0)
            or np.any(up < 0) or np.any(lo < 0)
            or np.any(up >= wall_count) or np.any(lo >= wall_count)
            or np.any(up == lo)):
        raise ValueError("Invalid grouped TE Gauss quadrature or bank faces")
    integ = np.einsum("eq,eqi->ei", w, B)
    mass = np.einsum("eq,eqi,eqj->ij", w, B, B)
    measure = integ.sum(axis=0)
    if (np.any(measure <= 0) or not np.isfinite(measure).all()):
        raise ValueError("Group TE basis has unsupported circulation DOF")
    if not np.allclose(mass, mass.T, atol=1e-13, rtol=1e-13):
        raise ValueError("Grouped TE mass must be symmetric")
    try:
        np.linalg.cholesky(mass)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Grouped TE mass must be positive definite") from exc
    incidence = np.zeros((len(B), wall_count))
    incidence[np.arange(len(B)), up] += 1.
    incidence[np.arange(len(B)), lo] -= 1.
    T = (integ.T @ incidence) / measure[:, None]
    M = mass / measure[:, None]
    return T, M, mass


def assemble_shed_groups(D, b, groups):
    """Assemble [D,W1,...;-T1,M1,...] with independent Gamma DOFs.

    groups contains (group_id, W[N,ng], T[ng,N], M[ng,ng]); zero groups
    means nonlifting potential body D mu=b. Cross-group coupling is through mu.
    """
    D, b = np.asarray(D, dtype=float), np.asarray(b, dtype=float)
    if (D.ndim != 2 or D.shape[0] != D.shape[1] or not len(D)
            or b.shape != (len(D),) or not np.isfinite(D).all()
            or not np.isfinite(b).all()):
        raise ValueError("Require finite square wetted-wall D and matching RHS")
    seen, converted = set(), []
    for entry in groups:
        if not isinstance(entry, (tuple, list)) or len(entry) != 4:
            raise ValueError("Group must provide (id,W,T,M)")
        name, W, T, M = entry
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("Group names must be nonempty and unique")
        seen.add(name)
        W, T, M = (np.asarray(a, dtype=float) for a in (W,T,M))
        ng = W.shape[1] if W.ndim == 2 else 0
        if (not ng or W.shape != (len(D),ng) or
                T.shape != (ng,len(D)) or M.shape != (ng,ng) or
                not np.isfinite(W).all() or not np.isfinite(T).all()
                or not np.isfinite(M).all()):
            raise ValueError(f"Group {name}: invalid W/T/M shapes or values")
        converted.append((name,W,T,M))
    sizes = [len(D)] + [x[1].shape[1] for x in converted]
    A = np.zeros((sum(sizes),sum(sizes)),dtype=float)
    A[:len(D),:len(D)] = D
    rhs = np.r_[b, np.zeros(sum(sizes[1:]))]
    slices = {"mu": slice(0,len(D))}
    pos = len(D)
    for name,W,T,M in converted:
        ng = W.shape[1]
        sl = slice(pos,pos+ng)
        slices[name] = sl
        A[:len(D),sl] = W
        A[sl,:len(D)] = -T
        A[sl,sl] = M
        pos += ng
    return A,rhs,slices


def solve_reference(A, rhs):
    """Dense CPU algebra oracle; not a scalable or physical production solver."""
    A, rhs = np.asarray(A,dtype=float), np.asarray(rhs,dtype=float)
    if A.ndim!=2 or A.shape[0]!=A.shape[1] or rhs.shape!=(len(A),):
        raise ValueError("Invalid grouped linear system")
    x = np.linalg.solve(A,rhs)
    relative = np.linalg.norm(A@x-rhs)/max(np.linalg.norm(rhs),1e-300)
    if not np.isfinite(relative) or relative>1e-10:
        raise ValueError("Grouped dense oracle failed true residual")
    return x,relative
