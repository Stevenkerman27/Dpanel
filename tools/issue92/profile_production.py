"""Issue #92 Phase 0: stage profile of the real NumPy/SciPy production paths.

Measures, on the frozen fixtures written by tools.issue92.fixtures:

  * geometry/mesh construction, P1 influence assembly, the constrained linear
    solve, Kutta iteration, pressure/load recovery;
  * panel/unknown counts, matrix shape and density;
  * wall clock per stage, cProfile attribution inside solve, peak RSS;
  * a numerical fingerprint (CL, CL_gamma, CD_pressure, CM_y, residuals) that the
    PyTorch parity work in Phase 1 must reproduce.

The profiler calls the same public entry points as the shipped CLIs
(panel.p1_lifting_g1.solve / panel.p1_lifting_g2.solve_multi), so the numbers are
production numbers, not a reduced toy path.

Usage:
  python -m tools.issue92.profile_production --out docs/issue92/phase0_profile.json
  python -m tools.issue92.profile_production --only g1_wing_rect g2_conventional --repeat 2
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import platform
import pstats
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("PANEL_ROOT", Path.cwd()))

import numpy as np

from panel.p1_lifting_g1 import Wing, assemble, from_dpanel_half_surface, make_wing, solve
from panel.p1_lifting_g2 import assemble_multi, combine_wings, solve_multi
from panel.velocity_profile import peak_rss_mb
from tools.issue92 import fixtures


def _versions():
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
            "numpy": np.__version__}
    for name in ("scipy", "jax", "torch"):
        try:
            module = __import__(name)
            info[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            info[name] = "unavailable: " + type(exc).__name__
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    except Exception:
        info["cuda_available"] = False
    return info


def _threads():
    return {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "XLA_FLAGS": os.environ.get("XLA_FLAGS")}


def _profile(func):
    profiler = cProfile.Profile()
    profiler.enable()
    value = func()
    profiler.disable()
    stats = pstats.Stats(profiler, stream=io.StringIO()).sort_stats("tottime")
    rows = []
    for (filename, line, name), (cc, nc, tt, ct, _callers) in stats.stats.items():
        if "panel" not in filename.replace("\\", "/"):
            continue
        rows.append({"function": name, "file": Path(filename).name + ":" + str(line),
                     "calls": nc, "tottime": round(tt, 4), "cumtime": round(ct, 4)})
    rows.sort(key=lambda row: row["tottime"], reverse=True)
    return value, rows[:15]


def _verify(name, entry):
    """Rebuild the fixture in memory and prove the profiled geometry is the frozen one."""
    arrays, _meta = fixtures.builders()[name]()
    _per_array, digest = fixtures._digest(arrays)
    return digest == entry["digest"], digest


def _wing_counters(wing, ops):
    A = np.asarray(ops["A"])
    ng = int(np.asarray(ops["W"]).shape[1])
    return {"n_wall_faces": int(len(wing.faces)),
            "n_wall_nodes": int(len(wing.xyz)),
            "n_mu_dofs": int(A.shape[1]),
            "n_gamma_dofs": ng,
            "n_kutta_rows": int(len(ops["K"])),
            "n_wake_triangles": int(len(wing.wake_faces)),
            "matrix_shape": [int(A.shape[0]), int(A.shape[1] + ng)],
            "A_nnz_fraction": round(float(np.count_nonzero(A)) / A.size, 6),
            "A_bytes_mb": round(A.nbytes / 1048576.0, 3),
            "scaled_B_bytes_mb": round((A.shape[0] * (A.shape[1] + ng) * 8) / 1048576.0, 3)}


def _fingerprint(result):
    keys = ("CL", "CL_gamma", "CD_pressure", "CM_y", "wall_rms",
            "pressure_kutta_rms", "signed_kutta_rms", "jump_rms", "constrained_rank")
    out = {}
    for key in keys:
        value = result.get(key)
        out[key] = None if value is None else float(value)
    return out


def _resolve_source(value):
    """Resolve a fixture source that may carry a stale absolute checkout prefix.

    Phase 0 was measured inside the runs/fix_master worktree, so several manifest
    entries still name D:/3D/Projects/panel/runs/fix_master/... . The same files
    live in this checkout, so fall back to the longest suffix below cases/ or
    runs/ before failing.
    """
    path = Path(value)
    if path.exists():
        return path
    text = str(value).replace("\\", "/")
    for marker in ("cases/", "runs/"):
        index = text.find(marker)
        if index < 0:
            continue
        try:
            return fixtures._first_existing(text[index:])
        except FileNotFoundError:
            continue
    raise FileNotFoundError("fixture source does not exist in this checkout: %s" % value)


def _wing_object(name, entry):
    meta = entry["meta"]
    if "source" in meta:
        return from_dpanel_half_surface(_resolve_source(meta["source"]))
    return make_wing(chord_panels=meta["chord_panels"], span_panels=meta["span_panels"],
                     thickness=meta["thickness"], camber=meta["camber"],
                     span=meta["span"], chord=meta["chord"],
                     wake_panels=meta["wake_panels"], wake_length=5.)


def _multi_object(entry):
    tail_origin = entry["meta"]["tail_origin"]
    return combine_wings([
        make_wing(chord_panels=4, span_panels=6, span=4., chord=1., camber=.02, wake_panels=2, wake_length=5.),
        make_wing(chord_panels=4, span_panels=4, span=1.8, chord=.55, camber=0., wake_panels=2, wake_length=5.)],
        origins=[(0., 0., 0.), tuple(tail_origin)], names=["main_wing", "raised_tail"])


def profile_wing(name, entry, repeat=1):
    meta = entry["meta"]
    wing = _wing_object(name, entry)
    t0 = time.perf_counter()
    ops = assemble(wing)
    t_assemble = time.perf_counter() - t0
    t0 = time.perf_counter()
    result = solve(wing, meta["alpha_deg"], kutta=meta["kutta"])
    t_solve = time.perf_counter() - t0
    profiled, rows = _profile(lambda: solve(wing, meta["alpha_deg"], kutta=meta["kutta"]))
    for _ in range(max(0, repeat - 1)):
        solve(wing, meta["alpha_deg"], kutta=meta["kutta"])
    drift = float(np.max(np.abs(np.asarray(profiled["mu"]) - np.asarray(result["mu"]))))
    return {"kind": "wing", "meta": meta, "counters": _wing_counters(wing, ops),
            "stages_seconds": {"assemble": round(t_assemble, 4),
                               "solve_total": round(t_solve, 4)},
            "profile_top": rows, "profile_rerun_mu_max_abs_drift": drift,
            "fingerprint": _fingerprint(result),
            "peak_rss_mb": round(peak_rss_mb(), 1)}


def profile_multi(name, entry):
    meta = entry["meta"]
    mesh = _multi_object(entry)
    t0 = time.perf_counter()
    ops = assemble_multi(mesh)
    t_assemble = time.perf_counter() - t0
    t0 = time.perf_counter()
    result = solve_multi(mesh, meta["alpha_deg"], kutta=meta["kutta"], operators=ops)
    t_solve = time.perf_counter() - t0
    profiled, rows = _profile(
        lambda: solve_multi(mesh, meta["alpha_deg"], kutta=meta["kutta"], operators=ops))
    A = np.asarray(ops["A"])
    ng = int(sum(group.gamma_hat.shape[1] for group in mesh.groups))
    counters = {"n_wall_faces": int(len(mesh.faces)),
                "n_wall_nodes": int(mesh.ndof_mu),
                "n_mu_dofs": int(A.shape[1]),
                "n_gamma_dofs": ng,
                "n_groups": len(mesh.groups),
                "matrix_shape": [int(A.shape[0]), int(A.shape[1] + ng)],
                "A_nnz_fraction": round(float(np.count_nonzero(A)) / A.size, 6),
                "A_bytes_mb": round(A.nbytes / 1048576.0, 3),
                "n_wake_triangles": int(sum(len(g.wake_faces) for g in mesh.groups))}
    drift = float(np.max(np.abs(np.asarray(profiled["mu"]) - np.asarray(result["mu"]))))
    return {"kind": "multi", "meta": meta, "counters": counters,
            "stages_seconds": {"assemble_multi": round(t_assemble, 4),
                               "solve_multi_total": round(t_solve, 4)},
            "profile_top": rows, "profile_rerun_mu_max_abs_drift": drift,
            "fingerprint": _fingerprint(result), "peak_rss_mb": round(peak_rss_mb(), 1)}


def profile_wake(iterations=1):
    """One coupled free-wake (B2) iteration: the induced-velocity stage of production.

    Uses the same CAD case the issue #84 batteries use, with the audit gate so the
    reported wall clock is the sampler plus the update, not a gate abort.
    """
    from panel.p1_coupled_material_wake import solve_coupled_material_wake
    from tools.issue84.wake_node_rejection import build_case
    from panel import velocity_profile

    source = fixtures._first_existing("runs/cases/wingbody/half_cad_wingbody_v1/surface.json")
    mesh, alpha = build_case([str(source)])
    t0 = time.perf_counter()
    ops = assemble_multi(mesh)
    t_assemble = time.perf_counter() - t0
    t0 = time.perf_counter()
    initial = solve_multi(mesh, alpha, kutta="pressure", operators=ops)
    t_initial = time.perf_counter() - t0
    velocity_profile.reset()
    velocity_profile.enable(True)
    t0 = time.perf_counter()
    result = solve_coupled_material_wake(mesh, alpha, iterations=iterations,
                                         gate_mode="off",
                                         save_callback=lambda *a, **k: None)
    t_wake = time.perf_counter() - t0
    velocity_profile.enable(False)
    counters = velocity_profile.snapshot()["counters"]
    history = result.get("history") or []
    records = [{"iteration": index,
                "map_residual_chord": row.get("map_residual_chord"),
                "relaxation": (row.get("incoming_update") or {}).get("relaxation"),
                "status": (row.get("incoming_update") or {}).get("status")}
               for index, row in enumerate(history)]
    return {"kind": "wake", "meta": {"source": str(source).replace("\\", "/"),
                                     "alpha_deg": float(alpha), "iterations": iterations,
                                     "gate_mode": "off", "velocity_method": "analytic_edges"},
            "counters": {"n_wall_faces": int(len(mesh.faces)),
                         "n_wake_nodes": int(sum(len(g.wake_xyz) for g in mesh.groups)),
                         "n_wake_triangles": int(sum(len(g.wake_faces) for g in mesh.groups)),
                         "n_unknowns": int(mesh.ndof_mu + sum(g.gamma_hat.shape[1]
                                                             for g in mesh.groups)),
                         "evaluate_many_calls": counters.get("total.evaluate_many.calls", 0),
                         "evaluate_many_targets": counters.get("total.evaluate_many.targets", 0),
                         "limits_targets": counters.get("limits.targets", 0),
                         "refined_calls": counters.get("limits.refined_calls", 0)},
            "stages_seconds": {"assemble_multi": round(t_assemble, 4),
                               "initial_solve_multi": round(t_initial, 4),
                               "coupled_wake_iteration": round(t_wake, 4)},
            "wake_records": records,
            "status": result.get("status"),
            "solver_seconds": round(float(result.get("seconds", 0.0)), 3),
            "fingerprint": _fingerprint(initial),
            "peak_rss_mb": round(peak_rss_mb(), 1)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Issue #92 Phase 0 production profile")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/issue92/phase0_profile.json")
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--repeat", type=int, default=1,
                        help="extra warm solves per fixture (timing stability)")
    parser.add_argument("--wake", action="store_true",
                        help="also profile one coupled free-wake iteration on the CAD case")
    parser.add_argument("--wake-iterations", type=int, default=1)
    args = parser.parse_args(argv)
    manifest = json.loads((ROOT / "docs/issue92/fixtures_manifest.json").read_text(encoding="utf-8"))
    available = [name for name, entry in manifest["fixtures"].items() if entry["status"] == "ok"]
    names = args.only or available
    payload = {"schema": "dpanel.issue92.phase0_profile.v1",
               "physical_accuracy_accepted": False,
               "command": "python -m tools.issue92.profile_production",
               "environment": _versions(), "threads": _threads(),
               "profile_overhead_note": ("cProfile rows are for attribution only; "
                                         "stages_seconds are wall clock without the profiler"),
               "fixtures": {}}
    for name in names:
        entry = manifest["fixtures"][name]
        if entry["status"] != "ok":
            payload["fixtures"][name] = {"status": "unavailable", "reason": entry["reason"]}
            continue
        began = time.perf_counter()
        verified, digest = _verify(name, entry)
        if entry["meta"]["kind"] == "wing":
            record = profile_wing(name, entry, args.repeat)
        else:
            record = profile_multi(name, entry)
        record["fixture_digest"] = entry["digest"]
        record["fixture_verified"] = bool(verified)
        record["rebuild_digest"] = digest
        if not verified:
            record["status"] = "digest_mismatch"
        record["fixture_npz_sha256"] = entry["npz_sha256"]
        record["elapsed_seconds"] = round(time.perf_counter() - began, 3)
        payload["fixtures"][name] = record
        stages = record["stages_seconds"]
        assemble_s = stages.get("assemble", stages.get("assemble_multi"))
        solve_s = stages.get("solve_total", stages.get("solve_multi_total"))
        print("%-20s %-5s faces=%-5d assemble=%8.3fs solve=%9.3fs CL=%+.9f rss=%.0fMB" % (
            name, entry["meta"]["kind"], record["counters"]["n_wall_faces"],
            assemble_s, solve_s, record["fingerprint"]["CL"], record["peak_rss_mb"]))
    if args.wake:
        began = time.perf_counter()
        record = profile_wake(args.wake_iterations)
        record["elapsed_seconds"] = round(time.perf_counter() - began, 3)
        payload["wake"] = record
        print("wake               faces=%-5d nodes=%-5d iteration=%8.3fs" % (
            record["counters"]["n_wall_faces"], record["counters"]["n_wake_nodes"],
            record["stages_seconds"]["coupled_wake_iteration"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote " + str(args.out))
    return payload


if __name__ == "__main__":
    main()
