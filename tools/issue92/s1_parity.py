"""S1 parity: torque the reduced pressure-Kutta solve onto torch, compare.

The two backends cannot share an interpreter: once torch runs an op, the next
NumPy LAPACK call dies with 0xC0000005 (docs/ISSUE92_PHASE1_WAKE_EVALUATOR.md
section 6f). So parity is measured across processes:

    python -m tools.issue92.s1_parity numpy --out a.npz   (.conda_env, no torch)
    python -m tools.issue92.s1_parity torch --out b.npz   (myml, torch)
    python -m tools.issue92.s1_parity compare a.npz b.npz

The shared problem is tools/issue92/s1_problem.py (fixed seed, structural
shapes of the reduced path, nonzero wall floor).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from tools.issue92.s1_problem import N_FREE, N_PRESSURE, build, pressure_numpy


def run_numpy(out, scenario="base", quadratic=None):
    from panel.p1_pressure_kutta_reduced import solve_structured_pressure

    problem = build(scenario=scenario, quadratic=quadratic)
    pressure, jacobian = pressure_numpy(problem)
    t0 = time.perf_counter()
    result = solve_structured_pressure(
        wall_operator=problem["B"], reduction=problem["N"], wall_rhs=problem["bw"],
        pressure=pressure, pressure_jacobian=jacobian, x0=problem["x0"], weight=5.)
    total = time.perf_counter() - t0
    history = result.history
    np.savez(out, backend=np.array("numpy"), x=np.asarray(result.x, float),
             fun=np.asarray(result.fun, float), cost=float(result.cost),
             optimality=float(result.optimality), status=int(result.status),
             nfev=int(result.nfev), nit=int(result.nit),
             relative_optimality=float(result.relative_optimality),
             backtracks=int(result.backtracks), message=np.array(result.message),
             scenario=np.array(problem["scenario"]),
             precompute_seconds=float(result.precompute_seconds), total_seconds=total,
             history_cost=np.array([h["cost"] for h in history]),
             history_nfev=np.array([h["nfev"] for h in history]))
    return dict(backend="numpy", status=int(result.status), nfev=int(result.nfev),
                nit=int(result.nit), cost=float(result.cost),
                message=result.message, total_seconds=total,
                precompute_seconds=float(result.precompute_seconds))


def run_torch(out, device="cpu", dtype="float64", scenario="base", quadratic=None):
    import torch

    from panel.p1_pressure_kutta_torch import solve_structured_pressure_torch

    torch_dtype = torch.float64 if dtype == "float64" else torch.float32
    problem = build(scenario=scenario, quadratic=quadratic)
    tensors = {k: torch.as_tensor(problem[k]) for k in ("B", "N", "bw", "K", "target")}
    tensors = {k: v.to(device=device, dtype=torch_dtype) for k, v in tensors.items()}
    n_mu = problem["n_mu"]
    K, c2, target = tensors["K"], problem["c2"], tensors["target"]

    def pressure(v):
        mu = v[:n_mu]
        return K @ mu + c2 * mu[:N_PRESSURE] ** 2 - target

    def jacobian(v):
        mu = v[:n_mu]
        block = K.clone()
        idx = torch.arange(N_PRESSURE, device=mu.device)
        block[idx, idx] = block[idx, idx] + 2.0 * c2 * mu[:N_PRESSURE]
        P = torch.zeros((N_PRESSURE, v.numel()), dtype=mu.dtype, device=mu.device)
        P[:, :n_mu] = block
        return P

    t0 = time.perf_counter()
    result = solve_structured_pressure_torch(
        wall_operator=tensors["B"], reduction=tensors["N"], wall_rhs=tensors["bw"],
        pressure=pressure, pressure_jacobian=jacobian,
        x0=torch.as_tensor(problem["x0"], dtype=torch_dtype, device=device), weight=5.,
        device=device, dtype=torch_dtype)
    total = time.perf_counter() - t0
    history = result.history
    np.savez(out, backend=np.array("torch"), x=result.x.detach().cpu().numpy(),
             fun=result.fun.detach().cpu().numpy(), cost=float(result.cost),
             optimality=float(result.optimality), status=int(result.status),
             nfev=int(result.nfev), nit=int(result.nit),
             relative_optimality=float(result.relative_optimality),
             backtracks=int(result.backtracks), message=np.array(result.message),
             precompute_seconds=float(result.precompute_seconds), total_seconds=total,
             history_cost=np.array([h["cost"] for h in history]),
             history_nfev=np.array([h["nfev"] for h in history]),
             scenario=np.array(problem["scenario"]))
    return dict(backend="torch", device=device, dtype=dtype, status=int(result.status),
                nfev=int(result.nfev), nit=int(result.nit), cost=float(result.cost),
                message=result.message, total_seconds=total,
                precompute_seconds=float(result.precompute_seconds))


TOLERANCES = dict(x_relative=1e-11, cost_relative=1e-11, optimality_absolute=1e-9)


def compare(a_path, b_path):
    a = np.load(a_path, allow_pickle=False)
    b = np.load(b_path, allow_pickle=False)
    xa, xb = a["x"], b["x"]
    scale = max(float(np.linalg.norm(xa)), 1e-300)
    x_rel = float(np.linalg.norm(xb - xa)) / scale
    cost_rel = abs(float(b["cost"]) - float(a["cost"])) / max(abs(float(a["cost"])), 1e-300)
    opt_abs = abs(float(b["optimality"]) - float(a["optimality"]))
    checks = {
        "same_status": int(a["status"]) == int(b["status"]),
        "same_nfev": int(a["nfev"]) == int(b["nfev"]),
        "x_relative": x_rel <= TOLERANCES["x_relative"],
        "cost_relative": cost_rel <= TOLERANCES["cost_relative"],
        "optimality": opt_abs <= TOLERANCES["optimality_absolute"],
    }
    report = dict(
        tolerances=TOLERANCES,
        numpy=dict(status=int(a["status"]), nfev=int(a["nfev"]), nit=int(a["nit"]),
                   cost=float(a["cost"]), optimality=float(a["optimality"]),
                   message=str(a["message"]), total_seconds=float(a["total_seconds"])),
        torch=dict(status=int(b["status"]), nfev=int(b["nfev"]), nit=int(b["nit"]),
                   cost=float(b["cost"]), optimality=float(b["optimality"]),
                   message=str(b["message"]), total_seconds=float(b["total_seconds"])),
        scenarios=dict(numpy=str(a["scenario"]), torch=str(b["scenario"])),
        backtracks=dict(numpy=int(a["backtracks"]), torch=int(b["backtracks"])),
        differences=dict(x_relative=x_rel, cost_relative=cost_rel, optimality_absolute=opt_abs),
        checks=checks, passed=bool(all(checks.values())))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("numpy", "torch"):
        stage = sub.add_parser(name)
        stage.add_argument("--out", required=True)
        stage.add_argument("--scenario", default="base",
                            choices=("base", "hard", "backtrack", "stiff"))
        stage.add_argument("--quadratic", type=float, default=None)
        if name == "torch":
            stage.add_argument("--device", default="cpu")
            stage.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    cmp_parser = sub.add_parser("compare")
    cmp_parser.add_argument("a")
    cmp_parser.add_argument("b")
    cmp_parser.add_argument("--json")
    args = parser.parse_args(argv)
    if args.stage == "numpy":
        summary = run_numpy(args.out, scenario=args.scenario, quadratic=args.quadratic)
    elif args.stage == "torch":
        summary = run_torch(args.out, device=args.device, dtype=args.dtype,
                            scenario=args.scenario, quadratic=args.quadratic)
    else:
        report = compare(args.a, args.b)
        print(json.dumps(report, indent=2))
        if args.json:
            with open(args.json, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
        return 0 if report["passed"] else 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())