"""Compare the Kutta closure and the wake model as numbers, with no thresholds.

Each planform is solved twice per wake model: fixed (the prescribed wake the mesh
carries) and free (vertices convected by the wake velocity, then Gamma resolved
again).  Every configuration reports CL, the near-field pressure drag, the Trefftz
plane drag and, for the elliptic planform, the span efficiency whose exact value
is 1.  Nothing here is an acceptance threshold: the report shows what changed, a
reader decides whether the change is physically defensible.

    python -m tools.wake_closure_study --iterations 10 --out runs/verification/wake_closure

The manifest records the commit, whether the tree was dirty, library versions and
the geometry hash, so a later reader can tell the numbers belong to this code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SPAN = 8.0
ASPECT_RATIO = 6.0
ALPHA_DEG = 4.0


def _git(*args, root="."):
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                             timeout=20)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def environment():
    import scipy
    import torch

    return dict(commit=_git("rev-parse", "HEAD"), dirty=bool(_git("status", "--porcelain")),
                platform=platform.platform(), python=sys.version.split()[0],
                numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__,
                torch_threads=int(torch.get_num_threads()),
                generated=time.strftime("%Y-%m-%dT%H:%M:%S"))


def planform(name, aspect_ratio=6.0, sections=9):
    """Return (y, chord, leading_edge_x) for rect, swept30 or elliptic.

    GRASS sections must start at the root origin with strictly increasing y, so these are
    half wings (0 .. b/2); the solver mirrors them.
    """
    from tools.calibration import _elliptic_planform

    if name in ("rect", "swept30"):
        y = 0.5 * SPAN * np.linspace(0.0, 1.0, sections)
        chord = np.full(sections, SPAN / aspect_ratio)
        le_x = y * np.tan(np.deg2rad(30.0)) if name == "swept30" else np.zeros(sections)
        return y, chord, le_x
    if name == "elliptic":
        y, chord, _ = _elliptic_planform(aspect_ratio, sections=sections)
        return y, chord, np.zeros(sections)
    raise ValueError("unknown planform %r" % (name,))


def build(name, chord_panels=6, span_panels_per_segment=4):
    from panel.grass_mesh import build_grass_mesh, load_grass_wing

    y, chord, le_x = planform(name, aspect_ratio=ASPECT_RATIO)
    sections = len(y)
    code = np.r_[np.full(8, 0.17), np.full(8, -0.17), 0.0, 0.0]
    geometry = dict(leading_edge_xyz=np.c_[le_x, y, np.zeros(sections)],
                    chord=chord, twist_radians=np.zeros(sections),
                    cst=np.tile(code, (sections, 1)))
    wing = load_grass_wing(build_grass_mesh(geometry, 1.0, chord_panels=chord_panels,
                                            span_panels_per_segment=span_panels_per_segment))
    digest = hashlib.sha256(np.ascontiguousarray(wing.xyz, float).tobytes()).hexdigest()
    return wing, digest


def _trefftz(wing, state):
    from tools.calibration import _trefftz as calibration_trefftz

    return calibration_trefftz(wing, state)


def _row(wing, state, name, wake, kutta, seconds, aspect_ratio=6.0, extra=None):
    drag = _trefftz(wing, state)
    # Half wings start at the root origin, so the mirrored span is twice max(y) and the
    # mirrored area twice the stored reference area.  Using ptp(y) alone would halve AR.
    half_span = float(np.max(np.asarray(wing.xyz, float)[:, 1]))
    # reference_area already carries the mirrored (full) wing, so this is the geometric AR.
    geometric = 4.0 * half_span ** 2 / float(wing.reference_area)
    aspect_ratio = float(aspect_ratio)
    optimum = float(state["CL_gamma"]) ** 2 / (np.pi * aspect_ratio)
    row = dict(planform=name, wake=wake, kutta=kutta,
               kutta_effective=state.get("kutta_effective", kutta),
               kutta_fell_back=state.get("kutta_fallback") is not None,
               faces=int(len(wing.xyz)), aspect_ratio=aspect_ratio,
               geometric_aspect_ratio=geometric,
               CL=float(state["CL"]), CL_gamma=float(state["CL_gamma"]),
               CD_pressure=float(state["CD_pressure"]),
               CDi_trefftz=float(drag["CDi_ideal"]),
               e_span=float(optimum / drag["CDi_ideal"]), seconds=seconds)
    if extra:
        row.update(extra)
    return row


def run(iterations=10, relaxation=0.25, planforms=("rect", "swept30", "elliptic"),
        closures=("pressure", "potential_jump")):
    from panel import p1_lifting_g1 as g1
    from panel.p1_lifting_g2 import combine_wings
    from panel.p1_steady_free_wake import solve_steady_free_wake

    rows = []
    for name in planforms:
        wing, digest = build(name)
        area = float(wing.reference_area)
        for kutta in closures:
            started = time.perf_counter()
            state = g1.solve(wing, ALPHA_DEG, kutta=kutta, wall_formulation="collocation")
            rows.append(_row(wing, state, name, "fixed", kutta,
                             time.perf_counter() - started, aspect_ratio=ASPECT_RATIO,
                             extra=dict(geometry_sha256=digest)))
            print(json.dumps({k: rows[-1][k] for k in
                              ("planform", "wake", "kutta", "CL", "e_span", "seconds")}),
                  flush=True)

            history = []

            def save(iteration, current, current_state, record):
                history.append(dict(iteration=int(iteration),
                                    CL=float(current_state["CL"]),
                                    max_wake_z=float(np.max(np.abs(
                                        np.asarray(current.groups[0].wake_xyz, float)[:, 2])))))

            mesh = combine_wings([wing], reference_area=area, reference_chord=1.0,
                                reference_point=np.array([0.25, 0.0, 0.0]))
            started = time.perf_counter()
            try:
                result = solve_steady_free_wake(mesh, ALPHA_DEG, kutta=kutta,
                                                iterations=iterations,
                                                relaxation=relaxation,
                                                wall_formulation="collocation",
                                                save_callback=save)
            except RuntimeError as error:
                rows.append(dict(planform=name, wake="free", kutta=kutta, status="raised",
                                 error=str(error)[:400], history=history,
                                 seconds=time.perf_counter() - started))
                print(json.dumps({k: rows[-1][k] for k in ("planform", "wake", "kutta", "error")}),
                      flush=True)
                continue
            rows.append(_row(wing, result["state"], name, "free", kutta,
                             time.perf_counter() - started, aspect_ratio=ASPECT_RATIO,
                             extra=dict(geometry_sha256=digest, status=result["status"],
                                  converged=bool(result["converged"]),
                                  geometry_stable=bool(result["geometry_stable"]),
                                  iterations=int(iterations), history=history)))
            print(json.dumps({k: rows[-1][k] for k in
                              ("planform", "wake", "kutta", "CL", "e_span", "status",
                               "seconds")}), flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--relaxation", type=float, default=0.25)
    parser.add_argument("--planform", action="append", default=None)
    parser.add_argument("--closure", action="append", default=None)
    parser.add_argument("--out", type=Path,
                        default=Path("runs/verification/wake_closure"))
    args = parser.parse_args(argv)
    rows = run(args.iterations, args.relaxation,
               tuple(args.planform or ("rect", "swept30", "elliptic")),
               tuple(args.closure or ("pressure", "potential_jump")))
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    payload = dict(protocol=dict(alpha_deg=ALPHA_DEG, span=SPAN,
                                 iterations=args.iterations, relaxation=args.relaxation,
                                 wall_formulation="collocation",
                                 drag="Trefftz plane, exact piecewise linear"),
                   environment=environment(), rows=rows)
    path = args.out / ("wake_closure_%s.json" % stamp)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
