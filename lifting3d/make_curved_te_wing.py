"""Generate the curved-trailing-edge CAD regression case (issue #100).

The wing is a symmetric NACA 0012 planform whose trailing-edge point is lifted
parabolically along the span: z_te(y) = amp*(y/half_span)**2. Sections are
lofted with makeRuled=True, so the trailing edge is the piecewise-linear 3-D
polyline declared in trailing_edge.json (one point per section). This is the
positive counterpart to run_cad_input_checks.py: the declared TE leaves z=0
and must be accepted.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def half_thickness(x, tc):
    return 5.0*tc*(0.2969*math.sqrt(max(x, 0.0)) - 0.126*x - 0.3516*x*x
                   + 0.2843*x**3 - 0.1036*x**4)


def section_curves(y, count, tc, amp, half_span):
    """Upper (LE->TE) and lower (TE->LE) polylines of one section."""
    n = int(count)
    xs = [0.5*(1.0-math.cos(math.pi*i/(n-1.0))) for i in range(n)]
    z_te = amp*(y/half_span)**2
    upper = [[x, y, z_te+half_thickness(x, tc)] for x in xs]
    lower = [[x, y, z_te-half_thickness(x, tc)] for x in xs]
    return upper, lower


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sections", type=int, default=9)
    parser.add_argument("--count", type=int, default=41)
    parser.add_argument("--amplitude", type=float, default=0.25)
    parser.add_argument("--half-span", type=float, default=2.0)
    parser.add_argument("--thickness", type=float, default=0.12)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import gmsh
    ys = np.linspace(-args.half_span, args.half_span, args.sections)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("curved_te_wing")
        occ = gmsh.model.occ
        wires = []
        for y in ys:
            upper, lower = section_curves(y, args.count, args.thickness,
                                          args.amplitude, args.half_span)
            upper_tags = [occ.addPoint(x, yy, z, tag=-1, meshSize=0.0) for x, yy, z in upper]
            lower_tags = [occ.addPoint(x, yy, z, tag=-1, meshSize=0.0) for x, yy, z in lower]
            wires.append(occ.addCurveLoop([occ.addBSpline(upper_tags),
                                           occ.addBSpline(lower_tags)]))
        occ.synchronize()
        occ.addThruSections(wires, makeSolid=True, makeRuled=True)
        occ.synchronize()
        solids = [tag for dim, tag in gmsh.model.getEntities(3)]
        if not solids:
            raise RuntimeError("loft produced no solid")
        volume = float(occ.getMass(3, solids[-1]))
        if not volume > 0.0:
            raise RuntimeError("loft produced a non-positive volume")
        gmsh.write(str(args.out/"wing.step"))
    finally:
        gmsh.finalize()
    points = [[1.0, float(y), float(args.amplitude*(y/args.half_span)**2)] for y in ys]
    declaration = dict(chord=1.0, reference_area=float(2.0*args.half_span),
                       origin=[0.0, 0.0, 0.0], points=points)
    (args.out/"trailing_edge.json").write_text(json.dumps(declaration, indent=2), encoding="utf-8")
    (args.out/"case.json").write_text(json.dumps(dict(
        family="curved_te_wing", volume=volume, sections=int(args.sections),
        points_per_section=int(args.count), amplitude=args.amplitude,
        half_span=args.half_span, thickness=args.thickness,
        te_points=len(points), source="lifting3d/make_curved_te_wing.py"), indent=2),
        encoding="utf-8")
    print(json.dumps(dict(volume=volume, te_z_max=float(max(p[2] for p in points)),
                          te_points=len(points))))


if __name__ == "__main__":
    main()
