"""CAD fixture for the cases/vsp trapezoidal wing (NACA 2412, taper 0.5).

Geometry is taken from the actual vspgeom vertices of cases/vsp/trap.vspgeom:
straight leading edge at x=0, trailing edge x = 0.2 - 0.1|y|, half span 1.0,
root chord 0.2, tip chord 0.1, NACA 2412 camber line, zero twist, z=0.

The solid is built full span (-1..1) and symmetric about y=0, because
panel.cad_half cuts the half domain itself.  The declared trailing edge is the
full-span TE polyline of both halves.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "audit" / "runtime" / "python"))
import gmsh  # noqa: E402

NCPU = 8
SECTION_X = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, 81)))
HALF_SPAN, ROOT_CHORD, TIP_CHORD = 1.0, 0.2, 0.1
TC, CAMBER, CAMBER_LOC = 0.12, 0.02, 0.4
MAC = 0.15555556


def profile(tc, camber, position):
    x = SECTION_X
    t = 5.0 * tc * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                    + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    mean = np.zeros_like(x)
    slope = np.zeros_like(x)
    if camber > 0.0:
        fwd = x < position
        mean[fwd] = camber / position ** 2 * (2.0 * position * x[fwd] - x[fwd] ** 2)
        slope[fwd] = 2.0 * camber / position ** 2 * (position - x[fwd])
        aft = ~fwd
        mean[aft] = camber / (1.0 - position) ** 2 * (
            (1.0 - 2.0 * position) + 2.0 * position * x[aft] - x[aft] ** 2)
        slope[aft] = 2.0 * camber / (1.0 - position) ** 2 * (position - x[aft])
    ang = np.arctan(slope)
    up = np.column_stack([x - t * np.sin(ang), mean + t * np.cos(ang)])
    lo = np.column_stack([x + t * np.sin(ang), mean - t * np.cos(ang)])
    up[0] = lo[0] = [0.0, 0.0]
    up[-1] = lo[-1] = [1.0, 0.0]
    return up, lo


def chord_at(y):
    return ROOT_CHORD + (TIP_CHORD - ROOT_CHORD) * abs(y) / HALF_SPAN


def wing_wire(occ, y, chord, le_x, z0):
    up, lo = profile(TC, CAMBER, CAMBER_LOC)
    tu = [occ.addPoint(float(le_x + chord * a), float(y), float(z0 + chord * b))
          for a, b in up]
    tl = [occ.addPoint(float(le_x + chord * a), float(y), float(z0 + chord * b))
          for a, b in lo[::-1]]
    return occ.addWire([occ.addSpline(tu), occ.addSpline(tl)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sections", type=int, default=21)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("trap")
    gmsh.option.setNumber("General.NumThreads", NCPU)
    occ = gmsh.model.occ
    y = np.linspace(-HALF_SPAN, HALF_SPAN, args.sections)
    wires = [wing_wire(occ, float(y[i]), chord_at(float(y[i])), 0.0, 0.0)
             for i in range(len(y))]
    solids = [e for e in occ.addThruSections(wires, makeSolid=True, makeRuled=True)
              if e[0] == 3]
    if len(solids) != 1:
        raise ValueError("loft returned %d solids" % len(solids))
    occ.synchronize()
    volume = float(occ.getMass(*solids[0]))
    lo = np.array(occ.getBoundingBox(*solids[0])[0:3])
    hi = np.array(occ.getBoundingBox(*solids[0])[3:6])
    gmsh.write(str(out / "wing.step"))
    gmsh.finalize()

    te = [(float(0.0 + chord_at(float(y[i]))), float(y[i]), 0.0)
          for i in range(len(y))]
    half_area = 0.5 * float(
        np.sum(0.5 * (chord_at(y[1:]) + chord_at(y[:-1])) * np.diff(y)))
    # _half_loads mirrors the one-sided force, so the reference area must be
    # the full wing area (both halves), matching the other geometric cases.
    declaration = dict(points=te, chord=MAC, reference_area=2.0 * half_area)
    (out / "trailing_edge.json").write_text(
        json.dumps(declaration, indent=1), encoding="utf-8")

    # re-import to prove the written STEP is one positive solid
    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("validate")
    imported = gmsh.model.occ.importShapes(str((out / "wing.step")),
                                          highestDimOnly=True)
    gmsh.model.occ.synchronize()
    imp = [e for e in imported if e[0] == 3]
    vol2 = [float(gmsh.model.occ.getMass(*e)) for e in imp]
    gmsh.finalize()

    info = dict(half_span=HALF_SPAN, root_chord=ROOT_CHORD, tip_chord=TIP_CHORD,
                tc=TC, camber=CAMBER, camber_location=CAMBER_LOC,
                sections=int(len(y)), volume=volume,
                bbox_low=lo.tolist(), bbox_high=hi.tolist(),
                reference_area=declaration["reference_area"],
                reference_chord=MAC,
                half_wing_area=half_area,
                imported_solid_count=len(imp), imported_volumes=vol2,
                step_bytes=(out / "wing.step").stat().st_size)
    (out / "geometry.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
