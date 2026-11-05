"""Run one panel solve in a child process.

The solvers link jax/LAPACK. A faulty native runtime can take the interpreter
down with an access violation (see README for the OpenBLAS tuning defect).
Running the solve out of process keeps the HTTP server alive and turns a native
crash into a job error that names the exit code.

  python -m tools.web.solve_runner --surface mesh/surface.json --out solve \
      --wake fixed --alpha 4 --iterations 4 --reference '{"area": 4.5, ...}'
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _plain(value):
    """numpy scalars/arrays and nested containers to JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return number if number == number and abs(number) != float("inf") else None


def solve_fixed(surface):
    """Server-side dense reference solve: one half domain, fixed flat wake."""
    from panel.run_half_groups import run as run_half

    print("稠密参考解（半域, 固定尾迹）", flush=True)
    report = run_half(surface, gradient=False, v1_check=False, max_panels=20000)
    solve = report["solve"]
    rows = []
    for row in report["group_rows"]:
        gamma = [float(v) for v in row["gamma"]]
        n = len(gamma)
        rows.append(dict(group=row["group_id"],
                         gamma=[[i / max(n - 1, 1), g] for i, g in enumerate(gamma)],
                         gamma_sum=row["gamma_sum"],
                         kutta_residual_rms=row["kutta_residual_rms"]))
    field = None
    cp_faces = solve.get("Cp_faces") or []
    try:
        data = json.loads(Path(surface).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    faces = data.get("faces") or []
    if cp_faces and len(cp_faces) == len(faces):
        field = dict(vertices=data["coordinates"], faces=faces, cp=cp_faces,
                     wake_faces=data.get("wake_faces") or [],
                     chord=data.get("reference_chord"))
    return dict(mode="fixed", CL=solve["CL"], CD=solve["CD_pressure"], CM=solve["CM"],
                cp=[solve["Cp_min"], solve["Cp_max"]],
                residual=solve["linear_residual_relative"],
                seconds=solve["hot_seconds"], unknowns=solve["unknowns"],
                panels=solve["wall_panels"], group_rows=rows, history=None,
                field=field)


def solve_free(surface, alpha, iterations, reference):
    """Steady free wake: the wake shape and mu/Gamma are solved together."""
    from panel.p1_lifting_g1 import from_dpanel_half_surface
    from panel.p1_lifting_g2 import combine_wings
    from panel.p1_steady_free_wake import solve_steady_free_wake

    print("装载半域面板", flush=True)
    wing = from_dpanel_half_surface(surface)
    model = combine_wings([wing], reference_area=reference["area"],
                          reference_chord=reference["chord"],
                          reference_point=tuple(reference["point"]))
    print("自由尾迹: alpha=%g, iterations=%d, 每轮同时重解 mu/Gamma" % (alpha, iterations),
          flush=True)

    def save(iteration, mesh, state, record):
        cl = record.get("CL")
        jump = record.get("jump_rms")
        print("iter %d: CL=%s jump_rms=%s" % (
            iteration,
            "-" if cl is None else "%+.6f" % float(cl),
            "-" if jump is None else "%.2e" % float(jump)), flush=True)

    value = solve_steady_free_wake(model, alpha, kutta="pressure",
                                   iterations=int(iterations), relaxation=0.25,
                                   save_callback=save)
    history = []
    for row in value["history"]:
        history.append(dict(iteration=row.get("iteration"), CL=row.get("CL"),
                            CD=row.get("CD_pressure"), jump_rms=row.get("jump_rms")))
    final = value["state"]
    rows = []
    for result in final.get("group_results", []):
        gamma = [float(v) for v in result["gamma"]]
        n = len(gamma)
        rows.append(dict(group=result["name"],
                         gamma=[[i / max(n - 1, 1), g] for i, g in enumerate(gamma)],
                         gamma_sum=result.get("Gamma_integral"),
                         CL_gamma=result.get("CL_gamma")))
    last = history[-1] if history else {}
    return dict(mode="free", CL=last.get("CL"), CD=last.get("CD"),
                CM=final.get("CM_y"), cp=None, residual=None,
                seconds=value.get("seconds"),
                converged=bool(value.get("converged")),
                geometry_stable=bool(value.get("geometry_stable")),
                iterations=value.get("iterations"), history=history,
                group_rows=rows, group_results=_plain(final.get("group_results")))


def main(argv=None):
    parser = argparse.ArgumentParser(description="one in-process panel solve")
    parser.add_argument("--surface", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--wake", default="fixed", choices=["fixed", "free"])
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--reference", default="{}")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.wake == "free":
        result = solve_free(Path(args.surface), args.alpha, args.iterations,
                            json.loads(args.reference))
    else:
        result = solve_fixed(Path(args.surface))
    result["runner_seconds"] = time.time() - started
    (out / "result.json").write_text(json.dumps(_plain(result)), encoding="utf-8")
    print("求解完成: CL=%s" % ("-" if result.get("CL") is None else "%+.6f" % result["CL"]),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
