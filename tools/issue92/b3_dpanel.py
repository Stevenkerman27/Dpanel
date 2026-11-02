"""Issue #92 B3: DPanel side of the four-geometry VSPAERO protocol.

Builds each case exactly as the repository solves it (analytic make_wing for the
rectangular wing, the frozen CAD surface otherwise), runs the steady fixed-wake
signed-speed Kutta, and records CL, circulation lift, pressure drag, the
Trefftz-plane induced drag and the spanwise circulation at the declared
stations. Nothing here is new physics: it calls panel.p1_lifting_g1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("PANEL_ROOT", Path.cwd()))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from panel.p1_lifting_g1 import make_wing, solve, from_dpanel_half_surface  # noqa: E402
from tools.issue92.trefftz_full import trefftz_cd_full  # noqa: E402


def _wing(spec):
    if spec["kind"] == "rect":
        return make_wing(**spec["make_wing"])
    path = ROOT / spec["surface"]
    return from_dpanel_half_surface(path)


def _trefftz(wing, gamma, spec):
    """Induced drag from the full-span circulation that also defines CL_gamma."""
    stations = np.asarray(wing.gamma_y, float)
    if len(stations) != len(gamma) + 2:
        return None
    cd, _ = trefftz_cd_full(stations, np.r_[0.0, gamma, 0.0], area=float(spec["sref"]))
    return float(cd)


def run_case(name, spec, *, alphas, out_dir):
    wing = _wing(spec)
    rows = []
    for alpha in alphas:
        t0 = time.perf_counter()
        res = solve(wing, alpha_deg=float(alpha), kutta="signed_speed")
        seconds = time.perf_counter() - t0
        gamma = np.asarray(res["gamma"], float)
        row = dict(
            alpha_deg=float(alpha),
            CL=float(res["CL"]),
            CL_gamma=float(res["CL_gamma"]),
            CD_pressure=float(res["CD_pressure"]),
            CM_y=float(res["CM_y"]),
            CD_trefftz=_trefftz(wing, gamma, spec),
            n_wall_faces=int(res["diagnostics"]["n_wall_faces"]),
            constrained_rank=int(res["constrained_rank"]),
            wall_rms=float(res["wall_rms"]),
            kutta_rms=float(res["signed_kutta_rms"]),
            seconds=float(seconds),
            stations=[float(v) for v in np.asarray(wing.gamma_y, float)],
            gamma_stations=[float(v) for v in gamma],
        )
        rows.append(row)
        print("%-8s a=%4.1f CL %+.5f CLg %+.5f CDp %+.6f CDt %s CM %+.5f" % (
            name, alpha, row["CL"], row["CL_gamma"], row["CD_pressure"],
            "%.6f" % row["CD_trefftz"] if row["CD_trefftz"] else "n/a", row["CM_y"]), flush=True)
    return dict(case=name, kind=spec["kind"], geometry_sha256=spec["geometry_sha256"],
                levels=spec.get("dpanel_levels", ["cad"]), rows=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="B3 DPanel side")
    parser.add_argument("--cases", type=Path, default=ROOT / "docs/issue92/b3_cases.json")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/issue92/b3/dpanel.json")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args(argv)
    doc = json.loads(args.cases.read_text(encoding="utf-8"))
    alphas = doc.get("alphas_deg", [0.0, 2.0, 4.0, 6.0])
    results = []
    for name, spec in doc["cases"].items():
        if args.only and name not in args.only:
            continue
        results.append(run_case(name, spec, alphas=alphas, out_dir=args.out.parent))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(cases=results), indent=2), encoding="utf-8")
    print("wrote", args.out)
    return results


if __name__ == "__main__":
    main()
