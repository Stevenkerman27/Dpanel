"""Inspect one VSPAERO case directory: mesh header, polar columns, wake-iteration history, span loads.

Replaces the one-off `runs/_an_user*.py` probes that were deleted in the 2026-10-27 cleanup.
Everything here goes through `vspapi.results`, so the column handling documented in
docs/VSPAERO_PANEL_POLAR_DIAGNOSIS.md is the same code path.

Usage:
    .conda_env\\python.exe runs\\vsp_conv\\inspect_case.py runs\\vsp_conv\\evidence\\l3
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vspapi.mesh import measure, read_vspgeom  # noqa: E402
from vspapi.results import read_history, read_lod, read_polar  # noqa: E402


def _fmt(value: float, width: int = 10, digits: int = 5) -> str:
    return f"{value:>{width}.{digits}f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("case", help="directory holding trap.polar / trap.history / trap.lod / trap.vspgeom")
    args = ap.parse_args(argv)
    case = pathlib.Path(args.case)

    geom_path = case / "trap.vspgeom"
    if geom_path.exists():
        geom = read_vspgeom(geom_path)
        info = measure(geom)
        lo, hi = info["bbox_min"], info["bbox_max"]
        types = info["panel_types"]
        print(f"mesh      : {info['points']} nodes / {info['panels']} panels / area {info['area']:.5f}")
        print(f"            x {lo[0]:+.4f}..{hi[0]:+.4f}  y {lo[1]:+.4f}..{hi[1]:+.4f}  z {lo[2]:+.5f}..{hi[2]:+.5f}")
        print(f"            panel types {dict(sorted(types.items()))}  symmetry={info['symmetry']}")

    polar_path = case / "trap.polar"
    if polar_path.exists():
        rows = read_polar(polar_path)
        cols = list(rows[0].keys())
        print(f"polar     : {len(rows)} rows, {len(cols)} columns")
        show = [c for c in ("AoA", "CLo", "CLi", "CLtot", "CDo", "CDi", "CDtot", "CLwtot", "CDwtot", "CMytot") if c in cols]
        print("            " + " ".join(f"{c:>9}" for c in show))
        for row in rows:
            print("            " + " ".join(_fmt(row[c], 9, 5) for c in show))
        if "CLwtot" in cols and "CLi" in cols:
            for row in rows:
                hi, hw = row['CLi'], row['CLwtot']
                if abs(hw) > 1e-9 and abs(hi / hw) > 1.5:
                    print(f"            ! CLi/CLwtot = {hi / hw:+.3f} on AoA {row['AoA']:.2f}: "
                          "CLi is the unreliable column, read CLwtot")

    hist_path = case / "trap.history"
    if hist_path.exists():
        rows = read_history(hist_path)
        keys = ("AoA", "CLi", "CDi", "CLwtot")
        print(f"history   : {len(rows)} wake iterations")
        print("            " + " ".join(f"{k:>12}" for k in keys))
        for row in rows:
            print("            " + " ".join(f"{row[k]:>12.6g}" for k in keys))

    lod_path = case / "trap.lod"
    if lod_path.exists():
        blocks = read_lod(lod_path)
        print(f"span load : {len(blocks)} alpha block(s)")
        for i, blk in enumerate(blocks):
            print(f"            block {i}: {blk['strips']:4d} strips, "
                  f"sum(Cl*dSpan) {blk['sum_cl_dspan']:+.5f}, "
                  f"sum(Cd*dSpan) {blk['sum_cd_dspan']:+.5f}, cl_max {blk['cl_max']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())