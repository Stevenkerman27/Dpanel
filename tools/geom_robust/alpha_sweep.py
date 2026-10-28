"""Fixed-mesh angle sweep for one saved half surface (issue #38).

The mesh, trailing edges, wake and the Gamma station degrees of freedom are
untouched; only the freestream direction rotates about y at the same speed,
with zero sideslip. Central differences give dCL/dalpha without assuming
CL(0) = 0.
"""
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from panel.half_native import solve_half_fields
from panel.run_half_native import load_half_surface


def sha256_file(path):
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def freestream(alpha_deg, speed):
    angle = math.radians(alpha_deg)
    return np.asarray([speed * math.cos(angle), 0.0, speed * math.sin(angle)])


def prandtl(ar):
    return 2.0 * math.pi * ar / (ar + 2.0)


def helmbold(ar):
    return 2.0 * math.pi * ar / (2.0 + math.sqrt(ar * ar + 4.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alphas", default="-5,-2.5,0,2.5,5")
    parser.add_argument("--hot-repeat", type=int, default=2)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a new output path; preserve prior sweeps")
    jax.config.update("jax_enable_x64", True)

    surface = json.loads(args.surface.read_text(encoding="utf-8"))
    fields = load_half_surface(args.surface)
    solver = jax.jit(solve_half_fields)
    speed = float(np.linalg.norm(np.asarray(fields["freestream"])))
    alphas = [float(a) for a in args.alphas.split(",")]

    rows = []
    first_seconds = None
    for index, alpha in enumerate(alphas):
        replaced = dict(fields)
        replaced["freestream"] = jnp.asarray(freestream(alpha, speed))
        started = time.perf_counter()
        out = solver(**replaced)
        elapsed = time.perf_counter() - started
        if index == 0:
            first_seconds = elapsed
        hot = []
        for _ in range(max(args.hot_repeat, 1)):
            begin = time.perf_counter()
            solver(**replaced)
            hot.append(time.perf_counter() - begin)
        rows.append(dict(
            alpha_deg=alpha,
            CL=float(out["CL"]), CD_pressure=float(out["CD_pressure"]),
            CM_y=float(np.asarray(out["CM"])[1]),
            true_residual=float(np.linalg.norm(np.asarray(out["linear_residual_half"]))
                                / max(np.linalg.norm(np.asarray(out["rhs_half"])), 1e-300)),
            first_seconds=elapsed, hot_seconds=min(hot),
        ))
        print("alpha %+5.1f  CL %+.9f  CD %+.9e  CM_y %+.9e  resid %.2e"
              % (alpha, rows[-1]["CL"], rows[-1]["CD_pressure"],
                 rows[-1]["CM_y"], rows[-1]["true_residual"]))

    by_alpha = {row["alpha_deg"]: row["CL"] for row in rows}

    def central(h):
        plus, minus = h, -h
        if plus not in by_alpha or minus not in by_alpha:
            return None
        return (by_alpha[plus] - by_alpha[minus]) / (2.0 * math.radians(h))

    half_span = float(np.max(np.abs(np.asarray(surface["coordinates"])[:, 1])))
    span = 2.0 * half_span
    area = float(surface["reference_area"])
    ar = span * span / area
    five = by_alpha.get(5.0)
    legacy = five / math.radians(5.0) if five is not None else None
    slope_h25 = central(2.5)
    slope_h5 = central(5.0)
    zero = by_alpha.get(0.0)
    report = dict(
        schema="dpanel.half_alpha_sweep.v1",
        status="completed",
        physical_accuracy_accepted=False,
        note="Fixed mesh, trailing edges, wake and Gamma degrees of freedom. Only "
             "the freestream direction rotates about +y at constant speed with "
             "zero sideslip, so the z=0 wake stays where the mesh put it and its "
             "kinematic misalignment at non-zero alpha is a model limitation, not "
             "a discretisation error.",
        source_surface=str(args.surface),
        surface_sha256=sha256_file(args.surface),
        environment=dict(python=platform.python_version(), jax=jax.__version__,
                         platform=platform.platform()),
        span=span, reference_area=area, aspect_ratio=ar,
        speed=speed,
        alphas=alphas,
        rows=rows,
        cl_zero=zero,
        single_point_legacy_slope=legacy,
        single_point_legacy_definition="CL(5 deg) / alpha_rad; this is not dCL/dalpha",
        central_difference=dict(
            h_2p5=slope_h25, h_5=slope_h5,
            agreement=None if (slope_h25 is None or slope_h5 is None)
                else abs(slope_h25 - slope_h5) / abs(slope_h5),
        ),
        odd_even=dict(
            cl_even_part=0.5 * (by_alpha.get(2.5, float("nan")) + by_alpha.get(-2.5, float("nan"))),
            cl_odd_part=0.5 * (by_alpha.get(2.5, float("nan")) - by_alpha.get(-2.5, float("nan"))),
            note="for a symmetric section the even part is the camber/finite-mesh "
                 "offset and the odd part carries the slope",
        ),
        references=dict(
            prandtl=prandtl(ar), helmbold=helmbold(ar),
            applicability="both are low-speed lifting-line style estimates for a "
                          "single isolated unswept wing; neither is a physical "
                          "truth value and neither is an acceptance threshold, and "
                          "they do not apply to the T-tail configuration because "
                          "its tail carries no Kutta condition",
            prandtl_formula="2*pi*AR/(AR+2)",
            helmbold_formula="2*pi*AR/(2+sqrt(AR^2+4))",
        ),
        first_solve_seconds=first_seconds,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    dev25 = "" if slope_h25 is None else "%.6f" % slope_h25
    dev5 = "" if slope_h5 is None else "%.6f" % slope_h5
    print("AR %.4f  CL(0) %s  legacy %s  dCL/dalpha(h=2.5) %s  (h=5) %s"
          % (ar, "%.9f" % zero if zero is not None else "n/a",
             "%.6f" % legacy if legacy is not None else "n/a", dev25, dev5))
    print("Prandtl %.6f  Helmbold %.6f" % (prandtl(ar), helmbold(ar)))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
