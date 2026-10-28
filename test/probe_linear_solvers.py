"""Compare dense LU with dense-matvec GMRES on EXACTLY the same saved AIC.

Usage:
  python test/probe_linear_solvers.py runs/backend_probes/vortexad_aic.npz
  python test/probe_linear_solvers.py runs/backend_probes/vortexad_aic.npz --rtol 1e-8

All GMRES operators below retain the dense A in memory. This is NOT FMM,
H-matrix, or a matrix-free O(N) panel method. Record actual matrix source
and validate both primal and transposed-adjoint residuals.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import LinearOperator, gmres


def residual(A, x, b):
    return float(np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300))


def timed_gmres(A, b, *, transpose, rtol, restart, maxiter, jacobi):
    mat = A.T if transpose else A
    n = mat.shape[0]
    calls = [0]
    history = []

    def mv(v):
        calls[0] += 1
        return mat @ v

    operator = LinearOperator((n, n), matvec=mv, dtype=A.dtype)
    M = None
    if jacobi:
        diag = np.diag(mat)
        if np.min(np.abs(diag)) <= max(np.max(np.abs(diag)) * 1e-13, 1e-14):
            raise ValueError("Jacobi not safe: near-zero AIC diagonal")
        M = LinearOperator((n, n), matvec=lambda v: v / diag, dtype=A.dtype)

    start = time.perf_counter()
    x, info = gmres(
        operator, b, M=M, rtol=rtol, atol=0.0, restart=restart,
        maxiter=maxiter, callback=history.append, callback_type="pr_norm",
    )
    duration = time.perf_counter() - start
    return {
        "method": ("GMRES+Jacobi" if jacobi else "GMRES"),
        "problem": "transpose" if transpose else "primal",
        "seconds": duration, "info": int(info),
        "iterations": len(history), "matvec_calls": calls[0],
        "reported_last_preconditioned_residual": float(history[-1]) if history else None,
        "true_relative_residual": residual(mat, x, b),
        "_x": x,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", type=Path)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--restart", type=int, default=40)
    ap.add_argument("--maxiter", type=int, default=20, help="Restart cycles")
    ap.add_argument("--max-n", type=int, default=3000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if not 0 < args.rtol < 1 or min(args.restart, args.maxiter, args.max_n) <= 0:
        raise ValueError("Invalid benchmark limits")

    with np.load(args.npz, allow_pickle=False) as saved:
        A = np.ascontiguousarray(saved["A"], dtype=np.float64)
        b = np.asarray(saved["b"], dtype=np.float64).reshape(-1)
        origin = str(saved["source_kind"]) if "source_kind" in saved else "unknown"
    n = len(b)
    if A.shape != (n, n) or n > args.max_n or not np.isfinite(A).all() or not np.isfinite(b).all():
        raise ValueError("Invalid/nonfinite/oversize saved dense AIC")

    # Use the same real matrix for both methods; report LU factor/solve separately.
    t0 = time.perf_counter()
    lu, piv = lu_factor(A)
    t_factor = time.perf_counter() - t0
    t0 = time.perf_counter()
    x_lu = lu_solve((lu, piv), b)
    t_primal = time.perf_counter() - t0
    # Deterministic adjoint RHS; not a substitute for real structural loss weights.
    g = np.random.default_rng(20261024).standard_normal(n)
    t0 = time.perf_counter()
    y_lu = lu_solve((lu, piv), g, trans=1)
    t_adjoint = time.perf_counter() - t0

    reports = []
    for transpose, rhs, truth in [(False, b, x_lu), (True, g, y_lu)]:
        for jacobi in (False, True):
            try:
                result = timed_gmres(
                    A, rhs, transpose=transpose, rtol=args.rtol,
                    restart=args.restart, maxiter=args.maxiter,
                    jacobi=jacobi,
                )
                result["relative_solution_difference_vs_lu"] = float(
                    np.linalg.norm(result["_x"] - truth) /
                    max(np.linalg.norm(truth), 1e-300)
                )
                del result["_x"]
                reports.append(result)
            except Exception as exc:
                reports.append({
                    "method": "GMRES+Jacobi" if jacobi else "GMRES",
                    "problem": "transpose" if transpose else "primal",
                    "error": f"{type(exc).__name__}: {exc}",
                })
    report = {
        "matrix_source": origin, "n": n,
        "A_bytes": int(A.nbytes), "LU_factor_bytes": int(lu.nbytes),
        "precision": "float64", "rtol": args.rtol,
        "restart": args.restart, "maxiter_restart_cycles": args.maxiter,
        "LU": {
            "factor_seconds": t_factor, "primal_solve_seconds": t_primal,
            "adjoint_transpose_solve_seconds": t_adjoint,
            "primal_true_relative_residual": residual(A, x_lu, b),
            "adjoint_true_relative_residual": residual(A.T, y_lu, g),
        },
        "GMRES": reports,
        "caveat": "LinearOperator wraps stored dense A; no matrix-free acceleration. Timings exclude AIC assembly and autodiff.",
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
