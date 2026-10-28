"""Build reusable STEP fixtures for geometric-robustness cases.

Every fixture is a closed solid (or solid set) mirror symmetric about y=0,
plus the declared full-span trailing-edge polyline of the lifting surface
whose wake the current single-sheet half contract can represent.

All cases are built with a unit root chord, so 'panel.cad_half --size' is in
root chords and every case is directly comparable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "audit" / "runtime" / "python"))
import gmsh

NCPU = 8
SECTION_X = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, 81)))


def profile(tc, camber, position):
    """Closed NACA-4 style section with an exactly sharp LE and TE."""
    x = SECTION_X
    t = 5.0 * tc * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                    + 0.2843 * x**3 - 0.1036 * x**4)
    mean = np.zeros_like(x)
    slope = np.zeros_like(x)
    if camber > 0.0:
        fwd = x < position
        mean[fwd] = camber / position**2 * (2.0 * position * x[fwd] - x[fwd]**2)
        slope[fwd] = 2.0 * camber / position**2 * (position - x[fwd])
        aft = ~fwd
        mean[aft] = camber / (1.0 - position)**2 * (
            (1.0 - 2.0 * position) + 2.0 * position * x[aft] - x[aft]**2)
        slope[aft] = 2.0 * camber / (1.0 - position)**2 * (position - x[aft])
    ang = np.arctan(slope)
    up = np.column_stack([x - t * np.sin(ang), mean + t * np.cos(ang)])
    lo = np.column_stack([x + t * np.sin(ang), mean - t * np.cos(ang)])
    up[0] = lo[0] = [0.0, 0.0]
    up[-1] = lo[-1] = [1.0, 0.0]
    return up, lo


def wing_wire(occ, y, chord, le_x, z0, tc, camber, position):
    up, lo = profile(tc, camber, position)
    tu = [occ.addPoint(float(le_x + chord * a), float(y), float(z0 + chord * b))
          for a, b in up]
    tl = [occ.addPoint(float(le_x + chord * a), float(y), float(z0 + chord * b))
          for a, b in lo[::-1]]
    return occ.addWire([occ.addSpline(tu), occ.addSpline(tl)])


def fin_wire(occ, z, chord, le_x, tc):
    up, lo = profile(tc, 0.0, 0.4)
    tu = [occ.addPoint(float(le_x + chord * a), float(chord * b), float(z))
          for a, b in up]
    tl = [occ.addPoint(float(le_x + chord * a), float(chord * b), float(z))
          for a, b in lo[::-1]]
    return occ.addWire([occ.addSpline(tu), occ.addSpline(tl)])


def loft(occ, wires, name):
    out = occ.addThruSections(wires, makeSolid=True, makeRuled=True)
    solids = [entity for entity in out if entity[0] == 3]
    if len(solids) != 1:
        raise ValueError(name + ": loft returned %d solids" % len(solids))
    return solids[0]


def begin(name):
    gmsh.clear()
    gmsh.model.add(name)
    gmsh.option.setNumber("General.NumThreads", NCPU)


def validate_step(path, expected):
    begin("validate_" + path.stem)
    imported = gmsh.model.occ.importShapes(str(path.resolve()), highestDimOnly=True)
    gmsh.model.occ.synchronize()
    solids = [e for e in imported if e[0] == 3]
    volumes = [float(gmsh.model.occ.getMass(*e)) for e in solids]
    return dict(imported_solid_count=len(solids),
                imported_solid_volumes=volumes,
                matches_expected=len(solids) == expected,
                positive_volume=bool(np.isfinite(volumes).all() and min(volumes) > 0.0))


def trapezoid(y, c):
    return float(np.sum(0.5 * (c[1:] + c[:-1]) * np.diff(y)))


def swept_sections(half_span, root_chord, tip_chord, sweep_deg, count, x_shift=0.0):
    y = np.linspace(-half_span, half_span, count)
    frac = np.abs(y) / half_span
    chord = root_chord + (tip_chord - root_chord) * frac
    le = (np.abs(y) * math.tan(math.radians(sweep_deg))
          + 0.25 * (root_chord - chord) + x_shift)
    return y, chord, le


def elliptic_sections(half_span, root_chord, tip_chord, count):
    grid = np.linspace(0.0, math.pi, count)
    y = np.sort(half_span * np.cos(grid))
    frac = np.abs(y) / half_span
    chord = tip_chord + (root_chord - tip_chord) * np.sqrt(
        np.maximum(0.0, 1.0 - frac**2))
    le = 0.25 * (root_chord - chord)
    return y, chord, le


def bwb_sections(half_span, root_chord, tip_chord, count):
    frac = np.linspace(0.0, 1.0, count)
    chord = tip_chord + (root_chord - tip_chord) * (1.0 - frac)**0.90
    chord[-1] = tip_chord
    tc = 0.10 + 0.12 * (1.0 - frac)**0.80
    le = 0.35 * frac * half_span
    y = np.concatenate((-np.flip(frac[1:]), frac)) * half_span
    chord = np.concatenate((np.flip(chord[1:]), chord))
    tc = np.concatenate((np.flip(tc[1:]), tc))
    le = np.concatenate((np.flip(le[1:]), le))
    return y, chord, le, tc


def case_swept30():
    y, chord, le = swept_sections(2.0, 1.0, 0.50, 30.0, 9)
    te = [(float(le[i] + chord[i]), float(y[i]), 0.0) for i in range(len(y))]

    def builder(occ):
        wires = [wing_wire(occ, float(y[i]), float(chord[i]), float(le[i]), 0.0,
                           0.12, 0.02, 0.4) for i in range(len(y))]
        return [loft(occ, wires, "swept30")]

    return dict(name="swept30", builder=builder, te=te, sref=trapezoid(y, chord),
                solids=1, half_span=2.0,
                notes="LE sweep 30 deg, taper 0.5, AR 5.33, NACA 2412")


def case_elliptic():
    y, chord, le = elliptic_sections(2.0, 1.0, 0.05, 21)
    te = [(float(le[i] + chord[i]), float(y[i]), 0.0) for i in range(len(y))]

    def builder(occ):
        wires = [wing_wire(occ, float(y[i]), float(chord[i]), float(le[i]), 0.0,
                           0.12, 0.0, 0.4) for i in range(len(y))]
        return [loft(occ, wires, "elliptic")]

    return dict(name="elliptic", builder=builder, te=te,
                sref=trapezoid(y, chord), solids=1, half_span=2.0,
                notes="elliptical planform, truncated tip chord 0.05, NACA 0012")


def case_bwb():
    y, chord, le, tc = bwb_sections(1.0, 1.0, 0.20, 13)
    te = [(float(le[i] + chord[i]), float(y[i]), 0.0) for i in range(len(y))]

    def builder(occ):
        wires = [wing_wire(occ, float(y[i]), float(chord[i]), float(le[i]), 0.0,
                           float(tc[i]), 0.015, 0.45) for i in range(len(y))]
        return [loft(occ, wires, "bwb")]

    return dict(name="bwb", builder=builder, te=te, sref=trapezoid(y, chord),
                solids=1, half_span=1.0,
                notes="blended wing body, root t/c 0.22 -> tip 0.10, AR 1.07")


def conv_parts():
    wy, wc, wl = swept_sections(1.6, 1.0, 0.50, 5.0, 7, x_shift=0.35)
    # Root chord 0.38, LE at 3.10 so the whole root section sits inside the
    # fin box; a T-tail is the only way to bury a raised tail root at y=0.
    hy, hc, hl = swept_sections(0.7, 0.38, 0.24, 10.0, 5, x_shift=3.10)
    fin_z = np.asarray([0.10, 0.32, 0.55, 0.78])
    fin_c = np.asarray([0.45, 0.42, 0.36, 0.28])
    fin_l = np.asarray([3.00, 3.10, 3.22, 3.38])
    fin_tc = 0.06 / fin_c
    return (wy, wc, wl, hy, hc, hl, fin_z, fin_c, fin_l, fin_tc)


def case_conv():
    wy, wc, wl, hy, hc, hl, fz, fc, fl, ftc = conv_parts()
    te_wing = [(float(wl[i] + wc[i]), float(wy[i]), 0.0) for i in range(len(wy))]

    def builder(occ):
        # The half contract hardcodes the wake sheet at z=0, so the only
        # surface that can carry a Kutta condition is the wing, whose TE lies
        # on that plane.  The T-tail sits above the wake sheet and therefore
        # contributes no trailing vorticity: it is a wall with a Dirichlet
        # condition only.  This is a contract limit, not a modelling choice.
        wing = loft(occ, [wing_wire(occ, float(wy[i]), float(wc[i]), float(wl[i]),
                                    0.0, 0.12, 0.02, 0.4)
                          for i in range(len(wy))], "conv_wing")
        htail = loft(occ, [wing_wire(occ, float(hy[i]), float(hc[i]), float(hl[i]),
                                     0.24, 0.10, 0.0, 0.4)
                           for i in range(len(hy))], "conv_htail")
        fin = loft(occ, [fin_wire(occ, float(fz[i]), float(fc[i]), float(fl[i]),
                                  float(ftc[i])) for i in range(len(fz))],
                   "conv_fin")
        return [wing, htail, fin]

    return dict(name="conv", builder=builder, te=te_wing,
                sref=trapezoid(wy, wc), solids=3, half_span=1.6,
                body=dict(center=[1.3, 0.0, 0.0], axes=[2.5, 0.25, 0.25]),
                notes="fuselage + low wing (Kutta, TE at z=0) + T-tail above "
                      "the wake plane (no Kutta) + centreline fin")


def build_body(case, path, meta):
    begin("body_" + case["name"])
    occ = gmsh.model.occ
    center = case["body"]["center"]
    axes = case["body"]["axes"]
    body = occ.addSphere(center[0], center[1], center[2], 1.0)
    occ.dilate([(3, body)], center[0], center[1], center[2], *axes)
    copy = occ.copy([(3, body)])[0][1]
    left = occ.addBox(center[0] - 60.0, -60.0, -60.0, 60.0, 120.0, 120.0)
    right = occ.addBox(center[0], -60.0, -60.0, 60.0, 120.0, 120.0)
    left_halves, _ = occ.intersect([(3, body)], [(3, left)],
                                   removeObject=True, removeTool=True)
    right_halves, _ = occ.intersect([(3, copy)], [(3, right)],
                                    removeObject=True, removeTool=True)
    solids = [e for e in left_halves + right_halves if e[0] == 3]
    if len(solids) != 2:
        raise ValueError("ellipsoid split produced %d solids" % len(solids))
    occ.synchronize()
    gmsh.write(str(path))
    meta["body_step_validation"] = validate_step(path, 2)
    expected = 4.0 / 3.0 * math.pi * float(np.prod(axes))
    measured = sum(meta["body_step_validation"]["imported_solid_volumes"])
    meta["body_volume_relative_error"] = abs(measured - expected) / expected
    return measured


def build(case, target):
    target.mkdir(parents=True)
    meta = dict(name=case["name"], notes=case["notes"], solids=case["solids"],
                half_span=case["half_span"], reference_area=case["sref"],
                reference_chord=1.0, status="built")
    begin("wing_" + case["name"])
    occ = gmsh.model.occ
    solids = case["builder"](occ)
    occ.synchronize()
    wing_path = target / "wing.step"
    gmsh.write(str(wing_path))
    meta["wing_step_validation"] = validate_step(wing_path, case["solids"])
    if case.get("body"):
        build_body(case, target / "body.step", meta)
        meta["body"] = case["body"]
    te = np.asarray(case["te"], dtype=float)
    declaration = dict(points=te.tolist(), chord=1.0,
                       reference_area=float(case["sref"]),
                       origin=[0.0, 0.0, 0.0])
    (target / "trailing_edge.json").write_text(
        json.dumps(declaration, indent=2) + "\n", encoding="utf-8")
    meta["te_points"] = int(len(te))
    meta["te_first"] = te[0].tolist()
    meta["te_last"] = te[-1].tolist()
    meta["wing_step_sha256"] = hashlib.sha256(wing_path.read_bytes()).hexdigest()
    if (target / "body.step").is_file():
        meta["body_step_sha256"] = hashlib.sha256(
            (target / "body.step").read_bytes()).hexdigest()
    meta["te_sha256"] = hashlib.sha256(
        (target / "trailing_edge.json").read_bytes()).hexdigest()
    (target / "build.json").write_text(json.dumps(meta, indent=2) + "\n",
                                       encoding="utf-8")
    return meta


CASES = dict(swept30=case_swept30, elliptic=case_elliptic, bwb=case_bwb)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    out = args.out.resolve()
    requested = args.only or ["swept30", "elliptic", "bwb", "conv"]
    records = []
    gmsh.initialize()
    gmsh.option.setNumber("General.NumThreads", NCPU)
    try:
        for name in requested:
            if name == "conv":
                records.append(build(case_conv(), out / "conv"))
            else:
                records.append(build(CASES[name](), out / name))
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()
    (out / "cases.json").write_text(json.dumps(records, indent=2) + "\n",
                                    encoding="utf-8")
    summary = [dict(name=r["name"], solids=r["solids"],
                    sref=round(r["reference_area"], 6),
                    te_points=r["te_points"],
                    wing_ok=r["wing_step_validation"]["matches_expected"])
               for r in records]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
