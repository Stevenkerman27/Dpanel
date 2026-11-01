"""Diagnose the structured pressure solve iteration by iteration on one fixture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="g1_elliptic")
    ap.add_argument("--max-nfev", type=int, default=100)
    ap.add_argument("--solver", default="structured")
    args = ap.parse_args(argv)

    from panel.p1_lifting_g1 import solve
    from tools.issue92.profile_production import _wing_object
    from tools.issue92 import fixtures

    arrays, entry = fixtures.load(args.fixture)
    wing = _wing_object(args.fixture, entry)
    alpha = float(entry.get("meta", {}).get("alpha_deg", 5.))
    result = solve(wing, alpha, kutta="pressure", max_nfev=args.max_nfev,
                   pressure_solver=args.solver)
    history = result["pressure_history"] or []
    print("solver", args.solver, "nfev", result["pressure_nfev"], "status", result["pressure_status"],
          "message", result["pressure_message"], "cost", result["pressure_cost"],
          "optimality", result["pressure_optimality"], "CL", repr(result["CL"]),
          "pk_rms", result["pressure_kutta_rms"])
    prev = None
    for row in history:
        rel = None if not prev else (prev - row["cost"]) / max(prev, 1e-300)
        print("%3d nfev=%-4d cost=%.15e grad=%.3e rel=%.3e wall=%.3e pk=%.3e reduction_rel=%s"
              % (row["iteration"], row["nfev"], row["cost"], row["gradient_inf"],
                 row.get("gradient_relative", float("nan")),
                 row["wall_rms"], row["pressure_rms"],
                 "n/a" if rel is None else "%.3e" % rel))
        prev = row["cost"]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
