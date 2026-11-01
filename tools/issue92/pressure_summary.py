"""Summarise the pressure solver A/B JSON into a compact table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=Path("docs/issue92/pressure_solver_bench.json"))
    args = ap.parse_args(argv)
    payload = json.loads(args.path.read_text(encoding="utf-8"))
    print("%-20s %-10s %5s %10s %10s %12s %12s %11s"
          % ("fixture", "solver", "nfev", "nonlin_s", "cost", "CL", "pk_rms", "stop"))
    for name, entry in payload["fixtures"].items():
        for solver in ("reference", "structured"):
            run = entry["solvers"].get(solver)
            if not run:
                continue
            hist = run.get("history") or []
            cost = hist[-1]["cost"] if hist else None
            print("%-20s %-10s %5s %10.3f %s %s %s %11s"
                  % (name, solver, run.get("nfev"), run.get("seconds_nonlinear"),
                     "%.15e" % cost if cost is not None else "n/a",
                     "%.10f" % run["loads"]["CL"],
                     "%.3e" % run["loads"]["pressure_kutta_rms"],
                     run.get("status")))
        comp = entry.get("comparison") or {}
        if comp:
            print("%-20s %-10s dCL=%+.3e rel=%.2e dpk=%+.2e speedup_nonlin=%.2fx speedup_total=%.2fx"
                  % (name, "delta", comp["CL"]["delta"], comp["CL"]["relative"],
                     comp["pressure_kutta_rms"]["delta"], entry["speedup_nonlinear"],
                     entry["speedup_total"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
