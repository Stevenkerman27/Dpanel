"""Report trailing-edge strip warnings for a GRASS mesh, or for a built-in demo planform.

The potential-jump Kutta closure loses uniqueness where the trailing edge sweeps into the
stream; panel/te_strip_audit.py explains the limits.  This tool is the command-line front
end: it reads a saved grass mesh, prints the report and exits 1 when the audit warns, so a
case generator can refuse to hand a bad discretization to an optimizer.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from panel.grass_mesh import build_grass_mesh, load_grass_wing, read_grass_mesh
from panel.te_strip_audit import bounded_elliptic_sections, te_strip_diagnostics


def _demo_wing(name):
    if name == "bounded":
        geometry = bounded_elliptic_sections(sections=9)
    else:
        count = 9
        theta = np.linspace(0.0, 0.5 * np.pi * 0.97, count)
        cst = np.zeros((count, 18))
        cst[:, :8] = 0.06
        cst[:, 8:16] = -0.06
        geometry = dict(
            leading_edge_xyz=np.c_[np.zeros(count), 4.0 * np.sin(theta), np.zeros(count)],
            chord=2.0 * np.cos(theta), twist_radians=np.zeros(count), cst=cst)
    return load_grass_wing(build_grass_mesh(geometry, 1.0, chord_panels=6,
                                            span_panels_per_segment=8))


def _report_text(report):
    lines = ["TE stations: %d, wake length %.3f" % (len(report["y"]), report["wake_length"]),
             "max TE sweep  %.1f deg   (limit %.0f)"
             % (report["max_sweep_deg"], report["sweep_limit_deg"]),
             "min strip     %.5f      max aspect %.0f (limit %.0f)"
             % (report["min_strip_width"], report["max_strip_aspect"],
                report["strip_aspect_limit"]),
             "tip/root chord %.4f    (limit %.3f)"
             % (report["tip_chord_ratio"], report["tip_chord_ratio_limit"]),
             "ok: %s" % report["ok"]]
    for warning in report["warnings"]:
        lines.append("  %-12s station %3d  %s"
                     % (warning["kind"], warning["station"], warning["message"]))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mesh", type=Path, help="grass mesh JSON written by build_grass_mesh")
    source.add_argument("--demo", choices=("bounded", "cusp"), help="built-in elliptic planform")
    parser.add_argument("--chord-panels", type=int, default=6)
    parser.add_argument("--span-panels-per-segment", type=int, default=8)
    parser.add_argument("--sweep-limit", type=float)
    parser.add_argument("--aspect-limit", type=float)
    parser.add_argument("--tip-ratio-limit", type=float)
    parser.add_argument("--json", type=Path, help="write the full report as JSON")
    args = parser.parse_args(argv)

    if args.mesh:
        # read_grass_mesh checks the saved display mesh against the saved solver mesh and
        # skips the GRASS rebuild, so a report costs no torch import.
        _, wing = read_grass_mesh(args.mesh)
    else:
        wing = _demo_wing(args.demo)
    limits = dict(
        sweep_limit=args.sweep_limit if args.sweep_limit is not None else None,
        aspect_limit=args.aspect_limit if args.aspect_limit is not None else None,
        tip_ratio_limit=args.tip_ratio_limit if args.tip_ratio_limit is not None else None)
    limits = {key: value for key, value in limits.items() if value is not None}
    report = te_strip_diagnostics(wing, **limits)
    print(_report_text(report))
    if args.json:
        serialisable = {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                        for key, value in report.items()}
        args.json.write_text(json.dumps(serialisable, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
