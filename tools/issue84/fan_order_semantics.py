"""Probe which flag actually controls the projection-fan area integral (issue #84 P0).

limits(field, ..., order=...) documents a Gauss-Legendre order audit. In the
regularized path the fan branch of RegularizedP1SheetField.evaluate_many calls
_area_velocity(point, order), and that method forwards self.fan_order / self.fan_strips
to area_velocity, ignoring its own order argument. This tool measures the consequence
on a few fixed targets instead of asserting anything about the code:

  * same fan order, different limits order  -> equal mean velocity, zero quadrature error;
  * different fan order, same limits order  -> different mean velocity.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from panel.p1_coupled_material_wake import TotalVelocity, limits, triangle_data
from panel.p1_lifting_g2 import assemble_multi, solve_multi
from tools.issue84.wake_node_rejection import build_case


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-half-surface", nargs="+", required=True)
    parser.add_argument("--case", default="g2", choices=("g1", "g2"))
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--core-radius", type=float, default=0.01)
    parser.add_argument("--fan-strips", type=int, default=8)
    parser.add_argument("--points", type=int, default=4)
    parser.add_argument("--ladder", type=float, nargs="+", default=[1., .5])
    parser.add_argument("--eps-fraction", type=float, default=.05)
    parser.add_argument("--out", type=Path, default=Path("docs/issue84/fan_order_semantics.json"))
    args = parser.parse_args(argv)

    started = time.perf_counter()
    mesh, alpha_deg = build_case(args.main_half_surface, case=args.case, alpha_deg=args.alpha)
    ops = assemble_multi(mesh)
    state = solve_multi(mesh, alpha_deg, kutta="pressure", operators=ops)
    group = mesh.groups[0]
    tri, normal, area, _shape, edges = triangle_data(group.wake_xyz, group.wake_faces)
    points = tri.mean(axis=1)[:args.points]
    normals = normal[:args.points]
    scales = edges.min(axis=1)[:args.points]
    solve_seconds = time.perf_counter() - started

    def measure(fan_order, limits_order):
        field = TotalVelocity(mesh, state, method="rm_regularized_p1", core_radius=args.core_radius,
                              area_quadrature="projection_fan", fan_order=fan_order,
                              fan_strips=args.fan_strips)
        t0 = time.perf_counter()
        samples = limits(field, points, normals, scales, eps_fraction=args.eps_fraction,
                         ladder=args.ladder, order=limits_order, tag="probe")
        return dict(fan_order=fan_order, fan_strips=args.fan_strips, limits_order=limits_order,
                    mean=samples["mean"], epsilon_error=samples["epsilon_error"],
                    quadrature_error=samples["quadrature_error"],
                    seconds=time.perf_counter()-t0)

    runs = [measure(28, 48), measure(28, 28), measure(16, 48)]
    uref = float(np.linalg.norm(runs[0]["mean"], axis=1).max())

    def compare(a, b):
        delta = np.linalg.norm(runs[a]["mean"]-runs[b]["mean"], axis=1)
        return dict(pair="%s vs %s" % (
            "fan%d/limits%d" % (runs[a]["fan_order"], runs[a]["limits_order"]),
            "fan%d/limits%d" % (runs[b]["fan_order"], runs[b]["limits_order"])),
            max_abs_difference=float(delta.max()),
            max_relative_difference=float(delta.max()/uref))

    payload = dict(
        schema="dpanel.issue84.fan_order_semantics.v1",
        python=platform.python_version(), numpy=np.__version__,
        case=args.case, alpha_deg=alpha_deg, core_radius=args.core_radius,
        fan_strips=args.fan_strips, ladder=list(args.ladder), points=args.points,
        solve_seconds=solve_seconds, total_seconds=time.perf_counter()-started,
        runs=[{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in row.items()}
              for row in runs],
        comparisons=[compare(0, 1), compare(0, 2)],
        reading=dict(
            limits_order_is_inert="fan order fixed, limits order 48 vs 28 give the same mean",
            fan_order_is_binding="fan 28 vs fan 16 give different means",
            quadrature_error_in_fan_path="refined pass re-evaluates the same fan order, so the "
                                         "quadrature channel cannot detect under-resolution"),
        status="completed")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for row in payload["comparisons"]:
        print(row["pair"], "max|d| =", "%.6e" % row["max_abs_difference"],
              "rel =", "%.6e" % row["max_relative_difference"], flush=True)
    for run in runs:
        print("fan%d/limits%d qerror_max %.3e eps_max %.3e %.2f s" % (
            run["fan_order"], run["limits_order"], float(np.max(run["quadrature_error"])),
            float(np.max(run["epsilon_error"])), run["seconds"]), flush=True)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
