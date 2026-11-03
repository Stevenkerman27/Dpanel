"""Is the g1_elliptic stopping branch a property of the backend or of roundoff?

Runs the same structured pressure-Kutta algorithm from initial states perturbed by a
relative epsilon, on the frozen g1_elliptic system, in each backend, and reports which
stopping rule fires. If the branch assignment flips with 1e-13 perturbations inside one
backend, the branch is numerical noise, not a semantic difference between libraries.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
EPSILONS = (0.0, 1e-13, -1e-13, 1e-11, -1e-11, 1e-9, -1e-9)


def load(work, name="g1_elliptic"):
    path = pathlib.Path(work) / (name + "_system.npz")
    with np.load(path) as data:
        return dict(B=data["B"], bw=data["bw"], kutta=data["kutta"], grad=data["grad"],
                    x0=data["x0"], max_nfev=int(data["max_nfev"]),
                    weight=float(data["weight"]),
                    pairs=[(list(map(int, t)), list(map(int, b))) for t, b in zip(
                        np.split(data["top_flat"], data["top_ptr"][1:-1]),
                        np.split(data["bot_flat"], data["bot_ptr"][1:-1]))])


def numpy_stage(work, out_path):
    from scipy.linalg import null_space
    try:
    from panel.p1_pressure_kutta_reduced import StructuredPressureKutta
except ImportError as error:  # pragma: no cover - historical evidence tool
    raise SystemExit(
        "this evidence tool re-runs the retired NumPy chain; check out tag "
        "numpy-path-frozen-s8 to use it (its recorded answers are docs/issue92/references/)"
    ) from error

    system = load(work)
    grad, pairs = system["grad"], system["pairs"]
    n_wall = grad.shape[2]

    def model(state):
        velocity = np.einsum("fij,j->fi", grad, state[:n_wall])
        pressure = np.array([np.mean(np.sum(velocity[top] ** 2, axis=1))
                             - np.mean(np.sum(velocity[bot] ** 2, axis=1)) for top, bot in pairs])
        rows = np.array([2 * np.mean(np.einsum("fi,fij->fj", velocity[top], grad[top]), axis=0)
                         - 2 * np.mean(np.einsum("fi,fij->fj", velocity[bot], grad[bot]), axis=0)
                         for top, bot in pairs])
        jacobian = np.column_stack((rows, np.zeros((rows.shape[0], state.size - n_wall))))
        return pressure, jacobian

    reduction = null_space(system["kutta"], rcond=1e-11)
    solver = StructuredPressureKutta(wall_operator=system["B"], reduction=reduction,
                                     wall_rhs=system["bw"],
                                     pressure=lambda s: model(s)[0],
                                     pressure_jacobian=lambda s: model(s)[1],
                                     weight=system["weight"])
    rows = []
    for eps in EPSILONS:
        x0 = system["x0"] * (1.0 + eps) if eps else system["x0"].copy()
        out = solver.solve(reduction.T @ x0, max_nfev=system["max_nfev"])
        rows.append(dict(epsilon=eps, status=int(out.status), nfev=int(out.nfev),
                         nit=int(out.nit), message=str(out.message),
                         cost=float(out.cost), optimality=float(out.optimality),
                         relative_optimality=float(out.relative_optimality)))
        print("%-8s numpy status %d nfev %3d cost %.16e rel_opt %.3e %s"
              % (eps, out.status, out.nfev, out.cost, out.relative_optimality, out.message), flush=True)
    pathlib.Path(out_path).write_text(json.dumps({"backend": "numpy", "rows": rows}, indent=2),
                                      encoding="utf-8")


def torch_stage(work, out_path):
    import torch
    from panel.p1_pressure_kutta_torch_driver import solve_pressure_kutta_torch

    system = load(work)
    rows = []
    for eps in EPSILONS:
        x0 = system["x0"] * (1.0 + eps) if eps else system["x0"].copy()
        out = solve_pressure_kutta_torch(wall_operator=system["B"], wall_rhs=system["bw"],
                                         kutta_operator=system["kutta"], grad=system["grad"],
                                         te_face_pairs=system["pairs"], x0=x0,
                                         weight=system["weight"], max_nfev=system["max_nfev"])
        result = out.result
        rows.append(dict(epsilon=eps, status=int(result.status), nfev=int(result.nfev),
                         nit=int(result.nit), message=str(result.message),
                         cost=float(result.cost), optimality=float(result.optimality),
                         relative_optimality=float(result.relative_optimality)))
        print("%-8s torch status %d nfev %3d cost %.16e rel_opt %.3e %s"
              % (eps, result.status, result.nfev, result.cost, result.relative_optimality,
                 result.message), flush=True)
    pathlib.Path(out_path).write_text(json.dumps({"backend": "torch", "rows": rows}, indent=2),
                                      encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("numpy", "torch"))
    parser.add_argument("--work", default=str(ROOT / "runs/scratch/issue92_loose/s3"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.stage == "numpy":
        numpy_stage(args.work, args.out)
    else:
        torch_stage(args.work, args.out)


if __name__ == "__main__":
    main()
