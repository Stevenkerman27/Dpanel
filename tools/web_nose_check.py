"""Rebuild and solve the default Web wing on one physical mesh size."""
import argparse
import json
from pathlib import Path
import sys
import time

from tools.web.pipeline import Job, build, mesh_surface, solve


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--size", type=float, required=True)
parser.add_argument("--alpha", type=float, default=5.)
parser.add_argument("--ncpu", type=int, default=2)
args = parser.parse_args()
args.out = args.out.resolve()
args.out.mkdir(parents=True, exist_ok=False)
started = time.perf_counter()
job = Job(args.out)
wing = build(job, {})
surface, mesh = mesh_surface(job, wing["params"], args.size, args.alpha, ncpu=args.ncpu)
result = solve(job, surface, "fixed", args.alpha, 0, wing["reference"], ncpu=args.ncpu)
report = dict(command=[sys.executable, *sys.argv], size_m=args.size, alpha_deg=args.alpha,
              ncpu=args.ncpu, mesh=mesh, CL=result["CL"], CD_pressure=result["CD"],
              CM=result["CM"], linear_residual_relative=result["residual"],
              unknowns=result["unknowns"], solver_seconds=result["seconds"],
              elapsed_seconds=time.perf_counter()-started, physical_accuracy_accepted=False)
(args.out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2), flush=True)
