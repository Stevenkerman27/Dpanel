"""One rung of the VSPAERO thick-panel ladder, built through vspapi.

Geometry and meshing come from the SAME OpenVSP build for every rung, which
matters: a 3.55-built mesh and a 3.52-built mesh with the same panel count give
CL(5 deg) = 0.712 vs 0.655.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vspapi.api import Case, WingSpec          # noqa: E402
from vspapi.config import SweepConfig          # noqa: E402

OUT = ROOT / "runs" / "vsp_conv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tess-w", type=int, required=True)
    ap.add_argument("--sect-tess-u", type=int, required=True)
    ap.add_argument("--alphas", default="0,5")
    ap.add_argument("--ncpu", type=int, default=8)
    ap.add_argument("--mode", default="panel")
    args = ap.parse_args()

    alphas = [float(v) for v in args.alphas.split(",")]
    workdir = OUT / ("api_%s_%s" % (args.mode, args.tag))
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    spec = WingSpec(spanlist=[1.0], chordlist=[0.2, 0.1],
                    tess_w=args.tess_w, sect_tess_u=[args.sect_tess_u])
    case = Case(workdir, name="trap")
    case.new_model()
    case.add_wing(spec)
    case.write_model()
    geom = case.compute_geometry(args.mode)
    info = case.assert_mode(geom, args.mode, spec.planform_area)

    cfg = SweepConfig(sref=spec.planform_area, cref=spec.chordlist[0], bref=2.0,
                      alpha=alphas, vinf=100.0, rho=0.002377, rec_ref=1.0e7,
                      symmetry=1, wake_iters=5, wake_nodes=8, wake_relax=1.0,
                      far_dist=-1.0)
    result = case.run(cfg, ncpu=args.ncpu)
    lod = result["lod"] or []
    cl_lod = None if not lod else float(lod[-1]["sum_cl_dspan"])
    summary = {
        "tag": args.tag,
        "mode": args.mode,
        "tess_w": args.tess_w,
        "sect_tess_u": args.sect_tess_u,
        "mesh": info,
        "alphas": alphas,
        "ncpu": args.ncpu,
        "exit_code": result["exit_code"],
        "seconds": result["seconds"],
        "polar": result["polar"],
        "health": result["health"],
        "lod_cl_integral": cl_lod,
    }
    (workdir / "ladder.json").write_text(json.dumps(summary, indent=2, default=float),
                                         encoding="utf-8")
    rows = " | ".join("a=%.1f CLwtot=%.5f CLi=%.4f" % (r["AoA"], r["CLwtot"], r["CLi"])
                      for r in result["polar"])
    print("%-6s panels=%5d %s %.0fs %s" % (args.tag, info["panels"], rows,
                                           result["seconds"], cl_lod))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())