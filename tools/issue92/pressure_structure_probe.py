"""Measure the structure and per-iteration cost of the pressure-Kutta Gauss-Newton step.

The reference solve builds the full Jacobian [B; c P] N (about 1811 x 1600 on the
elliptic fixture) and hands it to scipy.optimize.least_squares, which takes a dense
SVD every iteration. This probe measures what that costs and what a structured step
(constant factorisation of the wall block plus a Schur complement on the pressure
rows) would cost instead, including how closely the two steps agree.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve, lstsq, null_space

ROOT = Path(__file__).resolve().parents[2]


def _jacobian(ops, v, nm, ng, weight):
    vel = np.einsum("fij,j->fi", ops["grad"], v[:nm])
    nk = len(ops["te_pairs"])
    P = np.zeros((nk, nm + ng))
    for i, (top, bot) in enumerate(ops["te_pairs"]):
        P[i, :nm] = (2 * np.mean(np.einsum("fi,fij->fj", vel[top], ops["grad"][top]), axis=0)
                     - 2 * np.mean(np.einsum("fi,fij->fj", vel[bot], ops["grad"][bot]), axis=0))
    return vel, P


def _pressure(ops, vel):
    return np.array([np.mean(np.sum(vel[t] ** 2, axis=1)) - np.mean(np.sum(vel[b] ** 2, axis=1))
                     for t, b in ops["te_pairs"]])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="g1_elliptic")
    ap.add_argument("--weight", type=float, default=5.)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/issue92/pressure_structure_probe.json")
    args = ap.parse_args(argv)

    from panel.p1_lifting_g1 import assemble
    from tools.issue92 import fixtures
    from tools.issue92.profile_production import _wing_object

    arrays, entry = fixtures.load(args.fixture)
    wing = _wing_object(args.fixture, entry)
    t0 = time.perf_counter(); ops = assemble(wing); t_assemble = time.perf_counter() - t0
    A, W, C, K, GH = ops["A"], ops["W"], ops["C"], ops["K"], ops["GH"]
    nm, ng = A.shape[1], W.shape[1]
    alpha_deg = float(entry.get("meta", {}).get("alpha_deg", 5.))
    alpha = np.deg2rad(alpha_deg)
    U = np.array([np.cos(alpha), 0., np.sin(alpha)])
    rhs = -ops["mesh"].centers @ U
    scale = np.sqrt(ops["mesh"].areas / np.mean(ops["mesh"].areas))
    B = scale[:, None] * np.column_stack([A, W]); bw = scale * rhs
    Jc = np.block([C, -GH])
    N = null_space(Jc, rcond=1e-11)
    nfree = N.shape[1]
    t0 = time.perf_counter(); M = B @ N; t_M = time.perf_counter() - t0
    t0 = time.perf_counter(); G = M.T @ M; t_G = time.perf_counter() - t0
    t0 = time.perf_counter(); cG = cho_factor(G, lower=True); t_chol = time.perf_counter() - t0
    svM = np.linalg.svd(M, compute_uv=False)
    cond_M = float(svM[0] / svM[-1])
    t0 = time.perf_counter()
    z, _, rank, svB = lstsq(B @ N, bw, lapack_driver="gelsd")
    t_lin = time.perf_counter() - t0
    v = N @ z
    t0 = time.perf_counter(); vel, P = _jacobian(ops, v, nm, ng, args.weight); t_pjac = time.perf_counter() - t0
    p = _pressure(ops, vel)
    r1 = B @ v - bw
    r = np.r_[r1, args.weight * p]
    nk = len(p)
    t0 = time.perf_counter(); Jfull = np.vstack([B, args.weight * P]) @ N; t_jac = time.perf_counter() - t0
    svJ = np.linalg.svd(Jfull, compute_uv=False)
    t0 = time.perf_counter(); w_svd = np.linalg.lstsq(Jfull, -r, rcond=None)[0]; t_svd = time.perf_counter() - t0
    t0 = time.perf_counter(); H = Jfull.T @ Jfull; g = Jfull.T @ r; t_ne = time.perf_counter() - t0
    t0 = time.perf_counter(); w_ne = cho_solve(cho_factor(H, lower=True), -g); t_chol_solve = time.perf_counter() - t0
    F = args.weight * (P @ N)
    t0 = time.perf_counter()
    b = -(M.T @ r1) - (F.T @ (args.weight * p))
    Gi_b = cho_solve(cG, b)
    Gi_Ft = cho_solve(cG, F.T)
    S = np.eye(nk) + F @ Gi_Ft
    y = np.linalg.solve(S, F @ Gi_b)
    w_wood = Gi_b - Gi_Ft @ y
    t_wood = time.perf_counter() - t0
    ref = float(np.linalg.norm(w_svd))
    payload = dict(
        fixture=args.fixture, n_wall_faces=int(len(A)), n_mu=int(nm), n_gamma=int(ng),
        n_free=int(nfree), n_pressure=int(nk), jacobian_shape=list(Jfull.shape),
        rank_wall_operator=int(rank),
        seconds=dict(assemble=float(t_assemble), build_M=float(t_M), build_G=float(t_G),
                     factor_G=float(t_chol), linear_start=float(t_lin), pressure_jacobian=float(t_pjac),
                     jacobian_formation=float(t_jac), gauss_newton_svd=float(t_svd),
                     normal_equations=float(t_ne), cholesky_of_normal_equations=float(t_chol_solve),
                     structured_step=float(t_wood)),
        conditioning=dict(cond_M=cond_M, cond_J=float(svJ[0] / svJ[-1]),
                          sigma_max_J=float(svJ[0]), sigma_min_J=float(svJ[-1]),
                          sigma_max_M=float(svM[0]), sigma_min_M=float(svM[-1])),
        step_agreement=dict(
            normal_equations=float(np.linalg.norm(w_ne - w_svd) / max(ref, 1e-300)),
            structured=float(np.linalg.norm(w_wood - w_svd) / max(ref, 1e-300)),
            step_norm=ref, residual_norm=float(np.linalg.norm(r))),
        speedup_vs_reference=dict(
            jacobian_plus_svd=float((t_jac + t_svd) / max(t_wood, 1e-9)),
            normal_equations_total=float((t_jac + t_ne + t_chol_solve) / max(t_wood, 1e-9))))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
