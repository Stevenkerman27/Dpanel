"""A/B the reference least_squares pressure-Kutta solve against the structured one.

Per fixture and per solver: the assembly is shared, so the comparison reports the
nonlinear stage alone (reported by the solver itself as pressure_seconds), the nfev
budget actually used, the residual history, and the final loads and diagnostics so
that agreement can be checked before any default changes.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

LOADS = ("CL", "CD_pressure", "CL_gamma", "CM_y", "wall_rms", "pressure_kutta_rms",
         "jump_rms", "signed_kutta_rms")


def _compare(reference, candidate):
    out = {}
    for key in LOADS:
        a = float(reference[key]); b = float(candidate[key])
        out[key] = dict(reference=a, candidate=b, delta=b - a,
                        relative=(None if a == 0 else (b - a) / abs(a)))
    return out


def _entry(name):
    from tools.issue92 import fixtures

    arrays, entry = fixtures.load(name)
    return entry


def run_single(name, solver, repeat=1, max_nfev=100):
    from panel.p1_lifting_g1 import solve
    from tools.issue92.profile_production import _wing_object

    entry = _entry(name)
    wing = _wing_object(name, entry)
    alpha = float(entry.get("meta", {}).get("alpha_deg", 5.))
    best_nonlinear = float("inf"); best_total = float("inf"); result = None
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        result = solve(wing, alpha, kutta="pressure", max_nfev=max_nfev,
                       pressure_solver=solver)
        total = time.perf_counter() - t0
        best_total = min(best_total, total)
        best_nonlinear = min(best_nonlinear, float(result.get("pressure_seconds") or total))
    return dict(solver=solver, seconds_total=best_total, seconds_nonlinear=best_nonlinear,
                nfev=result.get("pressure_nfev"), status=result.get("pressure_status"),
                message=result.get("pressure_message"),
                precompute_seconds=result.get("pressure_precompute_seconds"),
                loads={k: float(result[k]) for k in LOADS},
                history=result.get("pressure_history"))


def run_multi(name, solver, repeat=1, max_nfev=80):
    from panel.p1_lifting_g2 import solve_multi
    from tools.issue92.profile_production import _multi_object

    entry = _entry(name)
    mesh = _multi_object(entry)
    best_nonlinear = float("inf"); best_total = float("inf"); result = None
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        result = solve_multi(mesh, 5., kutta="pressure", max_nfev=max_nfev,
                             pressure_solver=solver)
        total = time.perf_counter() - t0
        best_total = min(best_total, total)
        best_nonlinear = min(best_nonlinear, float(result.get("pressure_seconds") or total))
    nonlinear = result.get("nonlinear") or {}
    return dict(solver=solver, seconds_total=best_total, seconds_nonlinear=best_nonlinear,
                nfev=result.get("pressure_nfev"), status=nonlinear.get("status"),
                message=nonlinear.get("message"),
                accepted=nonlinear.get("success"),
                loads={k: float(result[k]) for k in LOADS},
                nonlinear=nonlinear, history=result.get("pressure_history"))


def _record(payload, name, runs):
    entry = {"solvers": runs}
    if "reference" in runs and "structured" in runs:
        entry["comparison"] = _compare(runs["reference"]["loads"], runs["structured"]["loads"])
        entry["speedup_nonlinear"] = (runs["reference"]["seconds_nonlinear"]
                                     / max(runs["structured"]["seconds_nonlinear"], 1e-9))
        entry["speedup_total"] = (runs["reference"]["seconds_total"]
                                  / max(runs["structured"]["seconds_total"], 1e-9))
    payload["fixtures"][name] = entry


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default="g1_wing_rect,g1_wing_camber,g2_conventional,g2_tail_low")
    ap.add_argument("--solvers", default="reference,structured")
    ap.add_argument("--max-nfev", type=int, default=100)
    ap.add_argument("--multi-max-nfev", type=int, default=80)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/issue92/pressure_solver_bench.json")
    args = ap.parse_args(argv)

    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    payload = {"max_nfev": args.max_nfev, "multi_max_nfev": args.multi_max_nfev,
               "repeat": args.repeat, "fixtures": {}}
    for name in [f.strip() for f in args.fixtures.split(",") if f.strip()]:
        multi = name.startswith("g2_")
        budget = args.multi_max_nfev if multi else args.max_nfev
        runs = {}
        for solver in solvers:
            runner = run_multi if multi else run_single
            runs[solver] = runner(name, solver, args.repeat, budget)
            print("%-18s %-10s nfev=%-4s nonlin=%7.3f s total=%7.2f s  CL=%.10f  pk_rms=%.3e"
                  % (name, solver, runs[solver]["nfev"], runs[solver]["seconds_nonlinear"],
                     runs[solver]["seconds_total"], runs[solver]["loads"]["CL"],
                     runs[solver]["loads"]["pressure_kutta_rms"]), flush=True)
        _record(payload, name, runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
