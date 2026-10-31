"""Fan order / strips convergence of overall coefficients on ten GRASS wings.

For each geometry the coupled P1 material-wake solve is run once with the
analytic (unregularized) edge kernel as the fan-independent reference and once
per projection-fan configuration.  One geometry update is applied per run, so the
recorded coefficients include the wake-velocity quadrature effect; the wall
solve itself is quadrature independent.

Usage (repository root of runs/fix_master):
    python -m tools.grass_fan_convergence.run_sweep --out docs/issue84/grass_fan_convergence.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from panel.p1_coupled_material_wake import solve_coupled_material_wake
from panel.p1_lifting_g2 import combine_wings
from panel.velocity_profile import peak_rss_mb
from tools.grass_fan_convergence.cases import cases
from tools.grass_fan_convergence.wing_builder import build_wing, naca_code

DEFAULT_CONFIGS = ("analytic", "analytic_open", "8:2/16:2", "16:4/32:4",
                   "24:8/32:8")


def resolve_airfoils(sections):
    cache = {}
    for s in sections:
        code = s["code"]
        if isinstance(code, tuple) and code[0] == "naca4":
            key = code[1]
            if key not in cache:
                cache[key] = naca_code(key)
            s["code"] = cache[key]
        elif isinstance(code, str):
            cache.setdefault(code, None)
    return sections


def build_mesh(case):
    sections = resolve_airfoils([dict(s) for s in case["sections"]])
    wing = build_wing(sections, span_panels=case["span_panels"],
                      chord_panels=case["chord_panels"],
                      wake_panels=case["wake_panels"],
                      wake_length=case.get("wake_length", 5.0), name=case["name"])
    mesh = combine_wings([wing], names=[case["name"]], reference_chord=wing.chord,
                         reference_point=np.zeros(3))
    return wing, mesh


def _jsonable(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def config_kwargs(label, *, core_radius):
    """label is 'analytic' or 'order:strips/refine_order:refine_strips'."""
    if label == "analytic":
        return dict(velocity_method="analytic_edges")
    if label == "analytic_open":
        # Same kernel, but every wake node is accepted: isolates the effect of
        # the per-node gate (which nodes get advected) from the kernel itself.
        return dict(velocity_method="analytic_edges", epsilon_tol=1e3,
                    quadrature_tol=1e3)
    open_gate = label.endswith("@open")
    if open_gate:
        # Same fan configuration, but every wake node is accepted: the kernel
        # effect only, with the per-node gate removed.
        label = label[: -len("@open")]
    base, _, refined = label.partition("/")
    fan_order, fan_strips = (int(v) for v in base.split(":"))
    if not refined:
        refined = "%d:8" % max(48, 2 * fan_order)
    ref_order, ref_strips = (int(v) for v in refined.split(":"))
    kwargs = dict(velocity_method="rm_regularized_p1", core_radius=core_radius,
                  area_quadrature="projection_fan", fan_order=fan_order,
                  fan_strips=fan_strips, fan_refinement_order=ref_order,
                  fan_refinement_strips=ref_strips)
    if open_gate:
        kwargs.update(epsilon_tol=1e3, quadrature_tol=1e3)
    return kwargs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="all")
    ap.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--core-radius", type=float, default=0.01)
    ap.add_argument("--order", type=int, default=48)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    wanted = [c.strip() for c in args.configs.split(",") if c.strip()]
    selected = cases()
    if args.cases != "all":
        keep = {c.strip() for c in args.cases.split(",")}
        selected = [c for c in selected if c["name"] in keep]

    records = {}
    payload = dict(schema="dpanel.grass_fan_convergence.v1", alpha_deg=args.alpha,
                   iterations=args.iterations, core_radius=args.core_radius,
                   order=args.order, refinement_rule="paired per config (base, 2x order)",
                   configs=wanted, records=[], physical_accuracy_accepted=False)
    if args.resume and out.exists():
        old = json.loads(out.read_text(encoding="utf-8"))
        if old.get("schema") != payload["schema"]:
            raise SystemExit("--resume: incompatible schema in " + str(out))
        payload = old
        payload["configs"] = wanted
        payload["refinement_rule"] = "paired per config"
        for r in old["records"]:
            records[(r["case"], r["config"])] = r
        print("resume: %d records" % len(records), flush=True)

    for case in selected:
        t_case = time.perf_counter()
        wing, mesh = build_mesh(case)
        geometry = dict(faces=int(len(wing.faces)), wake_faces=int(len(wing.wake_faces)),
                        stations=int(wing.gamma_hat.shape[1]),
                        planform_area=float(wing.reference_area),
                        span=float(wing.span), chord=float(wing.chord))
        print("[%s] faces=%d wake=%d stations=%d area=%.4f (%.1fs)" % (
            case["name"], geometry["faces"], geometry["wake_faces"],
            geometry["stations"], geometry["planform_area"],
            time.perf_counter() - t_case), flush=True)
        for label in wanted:
            if (case["name"], label) in records:
                continue
            t0 = time.perf_counter()
            kwargs = config_kwargs(label, core_radius=args.core_radius)
            result = solve_coupled_material_wake(
                mesh, args.alpha, kutta="pressure", iterations=args.iterations,
                order=args.order, **kwargs)
            first = result["history"][0]
            update = first.get("outgoing_update") or {}
            wakes = first.get("wakes", {})
            main_wake = next(iter(wakes.values())) if wakes else {}
            state = result["state"]
            record = dict(
                case=case["name"], config=label, note=case["note"], geometry=geometry,
                status=result["status"], seconds=time.perf_counter() - t0,
                velocity_method=kwargs.get("velocity_method", "analytic_edges"),
                fan_order=kwargs.get("fan_order"),
                fan_strips=kwargs.get("fan_strips"),
                fan_refinement_order=kwargs.get("fan_refinement_order"),
                fan_refinement_strips=kwargs.get("fan_refinement_strips"),
                CL=float(state["CL"]), CL_gamma=float(state["CL_gamma"]),
                CD_pressure=float(state["CD_pressure"]), CM_y=float(state["CM_y"]),
                pressure_kutta_rms=float(state.get("pressure_kutta_rms", np.nan)),
                wall_rms=float(state.get("wall_rms", np.nan)),
                update_accepted=update.get("accepted"),
                update_relaxation=update.get("relaxation"),
                update_moved_nodes=update.get("moved_nodes"),
                update_displacement_rms=update.get("displacement_rms"),
                eligibility=(main_wake.get("eligibility") or {}),
                peak_rss_mb=peak_rss_mb())
            records[(case["name"], label)] = record
            payload["records"] = sorted(records.values(),
                                        key=lambda r: (r["case"], wanted.index(r["config"])
                                                       if r["config"] in wanted else -1))
            payload["configs"] = wanted
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=1, sort_keys=True,
                                      default=_jsonable) + "\n",
                           encoding="utf-8")
            print("  %-8s %6.1fs status=%-26s CL=%+.8f CLg=%+.8f CD_p=%+.8f CMy=%+.8f "
                  "acc=%s" % (label, record["seconds"], record["status"], record["CL"],
                              record["CL_gamma"], record["CD_pressure"], record["CM_y"],
                              record["update_accepted"]), flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
