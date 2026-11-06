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
import os
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


def solve_fixed_p0(surface):
    """Server-side dense reference solve: one half domain, fixed flat wake."""
    from panel.run_half_groups import run as run_half

    print("稠密参考解（半域, 固定尾迹）", flush=True)
    report = run_half(surface, gradient=False, v1_check=False, max_panels=20000,
                      progress=print)
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
                timings=report["timings"],
                panels=solve["wall_panels"], group_rows=rows, history=None,
                field=field)


def solve_fixed(surface, alpha, reference, wall_formulation="galerkin"):
    """Fixed wake with the same full-span P1 model as the free-wake solver."""
    from panel.p1_lifting_g1 import from_dpanel_half_surface
    from panel.p1_lifting_g2 import combine_wings, solve_multi

    data = json.loads(Path(surface).read_text(encoding="utf-8"))
    structured = data.get("schema") == "dpanel.grass_fixed_topology.v1"
    if structured:
        from panel.grass_mesh import load_grass_wing
        wing = load_grass_wing(data)
    else:
        wing = from_dpanel_half_surface(surface)
    model = combine_wings([wing], reference_area=reference.get("area", wing.reference_area),
                          reference_chord=reference.get("chord", wing.chord),
                          reference_point=tuple(reference.get("point", wing.reference_point)))
    print("P1 %s 固定尾迹（半翼镜像为全翼）" % wall_formulation, flush=True)
    state = solve_multi(model, alpha, kutta="pressure", wall_formulation=wall_formulation)
    faces = data.get("faces") or []
    cp = state["cp"][:len(faces)]
    vertices = data["coordinates"]
    wake_faces = data.get("wake_faces") or []
    if structured:
        offset = len(vertices)
        vertices = vertices + wing.wake_xyz.tolist()
        wake_faces = (wing.wake_faces + offset).tolist()
    field = dict(vertices=vertices, faces=faces, cp=cp.tolist(),
                 wake_faces=wake_faces, chord=wing.chord, mirror=not structured)
    rows = []
    for group in state["group_results"]:
        gamma = group["gamma"]
        n = len(gamma)
        rows.append(dict(group=group["name"],
                         gamma=[[i / max(n - 1, 1), float(g)] for i, g in enumerate(gamma)],
                         gamma_sum=group["Gamma_integral"], CL_gamma=group["CL_gamma"]))
    nonlinear = state.get("nonlinear") or {}
    warning = None if nonlinear.get("success") else "压力 Kutta 求解未通过数值终止条件，结果仅供诊断。"
    return dict(mode="fixed", wall_formulation=state["wall_formulation"],
                CL=state["CL"], CD=state["CD_pressure"], CM=state["CM_y"],
                cp=[float(cp.min()), float(cp.max())],
                residual=(state["wall_dual_rms"] if state["wall_formulation"]=="galerkin"
                          else state["wall_rms"]),
                nonlinear=nonlinear, warning=warning, seconds=state["solve_seconds"],
                unknowns=len(state["mu"])+len(state["gamma"]), panels=len(faces),
                group_rows=rows, history=None, field=field,
                physical_accuracy_accepted=False)


def solve_free(surface, alpha, iterations, reference, wall_formulation="galerkin",
               wake_method="sheet"):
    """Steady free wake: the wake shape and mu/Gamma are solved together."""
    from panel.p1_lifting_g1 import from_dpanel_half_surface
    from panel.p1_lifting_g2 import combine_wings
    from panel.p1_steady_free_wake import solve_steady_free_wake

    print("装载半域面板", flush=True)
    data = json.loads(Path(surface).read_text(encoding="utf-8"))
    if data.get("schema") == "dpanel.grass_fixed_topology.v1":
        from panel.grass_mesh import load_grass_wing
        wing = load_grass_wing(data)
    else:
        wing = from_dpanel_half_surface(surface)
    model = combine_wings([wing], reference_area=reference.get("area", wing.reference_area),
                          reference_chord=reference.get("chord", wing.chord),
                          reference_point=tuple(reference.get("point", wing.reference_point)))
    print("自由尾迹: method=%s, alpha=%g, iterations=%d, 每轮同时重解 mu/Gamma" % (wake_method, alpha, iterations),
          flush=True)

    def save(iteration, mesh, state, record):
        cl = record.get("CL")
        jump = record.get("jump_rms")
        print("iter %d: CL=%s jump_rms=%s" % (
            iteration,
            "-" if cl is None else "%+.6f" % float(cl),
            "-" if jump is None else "%.2e" % float(jump)), flush=True)

    if wake_method == "filament":
        if wall_formulation != "galerkin":
            raise ValueError("涡线自由尾迹目前只支持 P1 弱形式壁面")
        from panel.p1_filament_free_wake import (resample_structured_wake, structure_cad_wake,
                                                 solve_filament_free_wake)
        if data.get("schema") == "dpanel.grass_fixed_topology.v1":
            model = resample_structured_wake(model, wake_panels=8)
        else:
            model = structure_cad_wake(model, wake_panels=8)
        print("涡线尾迹: 重建为 8 段结构化尾迹，涡核=0.05c，展向细分=2", flush=True)
        value = solve_filament_free_wake(model, alpha, iterations=int(iterations),
                    relaxation=0.25, core_radius=0.05*model.reference_chord,
                    subdivisions=2, intersection_guard=True, save_callback=save)
    elif wake_method == "sheet":
        value = solve_steady_free_wake(model, alpha, kutta="pressure",
                                       iterations=int(iterations), relaxation=0.25,
                                       wall_formulation=wall_formulation,
                                       save_callback=save)
    else:
        raise ValueError("未知自由尾迹方法")
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
    return dict(mode="free", wake_method=wake_method, wall_formulation=final["wall_formulation"],
                CL=last.get("CL"), CD=last.get("CD"),
                CM=final.get("CM_y"), cp=None, residual=None,
                seconds=value.get("seconds"),
                converged=bool(value.get("converged")),
                geometry_stable=(bool(value["geometry_stable"])
                                 if "geometry_stable" in value else None),
                status=value.get("status"), iterations=len(history)-1, history=history,
                group_rows=rows, group_results=_plain(final.get("group_results")),
                physical_accuracy_accepted=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description="one in-process panel solve")
    parser.add_argument("--surface", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--wake", default="fixed", choices=["fixed", "free", "free_filament"])
    parser.add_argument("--wall-formulation", default="galerkin",
                        choices=["galerkin", "collocation", "p0"])
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--reference", default="{}")
    parser.add_argument("--ncpu", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args(argv)
    if not 1 <= args.ncpu <= (os.cpu_count() or 1):
        parser.error("--ncpu must be between 1 and the logical CPU count")
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = str(args.ncpu)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.wake in ("free", "free_filament"):
        if args.wall_formulation == "p0":
            parser.error("p0 is only available for a fixed wake")
        result = solve_free(Path(args.surface), args.alpha, args.iterations,
                            json.loads(args.reference), args.wall_formulation,
                            wake_method="filament" if args.wake=="free_filament" else "sheet")
    elif args.wall_formulation == "p0":
        result = solve_fixed_p0(Path(args.surface))
    else:
        result = solve_fixed(Path(args.surface), args.alpha, json.loads(args.reference),
                             args.wall_formulation)
    result["runner_seconds"] = time.time() - started
    result["ncpu"] = args.ncpu
    (out / "result.json").write_text(json.dumps(_plain(result)), encoding="utf-8")
    print("求解完成: CL=%s" % ("-" if result.get("CL") is None else "%+.6f" % result["CL"]),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
