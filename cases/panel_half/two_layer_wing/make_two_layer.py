"""Build a two-layer (biplane-like) wing STEP with two independent horizontal TE wakes."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "audit" / "runtime" / "python"))
sys.path.insert(0, str(ROOT / "runs" / "pushrepo" / "tools" / "geom_robust"))
import gmsh
from make_cases import loft, profile, swept_sections, trapezoid, wing_wire

HALF_SPAN = 1.6
ROOT_CHORD = 1.0
TIP_CHORD = 0.5
SWEEP_DEG = 5.0
TC = 0.12
CAMBER = 0.02
POSITION = 0.4
SECTIONS = 9
Z_LAYERS = (("lower_wing", 0.0), ("upper_wing", 0.24))


def main():
    y, chord, le = swept_sections(HALF_SPAN, ROOT_CHORD, TIP_CHORD, SWEEP_DEG, SECTIONS)
    sref_single = trapezoid(y, chord)
    gmsh.initialize()
    gmsh.option.setNumber("General.NumThreads", 8)
    gmsh.model.add("two_layer_wing")
    occ = gmsh.model.occ
    groups = []
    for name, z0 in Z_LAYERS:
        wires = [wing_wire(occ, float(y[i]), float(chord[i]), float(le[i]), z0,
                           TC, CAMBER, POSITION) for i in range(len(y))]
        solid = loft(occ, wires, name)
        te = [[float(le[i] + chord[i]), float(y[i]), float(z0)] for i in range(len(y))]
        groups.append(dict(group_id=name, z=float(z0), points=te,
                           role="lifting", kind="attached_lifting_TE",
                           root_condition="symmetry_plane"))
    occ.synchronize()
    solids = [e for e in gmsh.model.getEntities(3)]
    volumes = [float(occ.getMass(*e)) for e in solids]
    if len(solids) != len(Z_LAYERS) or min(volumes) <= 0:
        raise ValueError("Expected one positive-volume solid per layer")
    step = HERE / "two_layer.step"
    gmsh.write(str(step))
    declaration = dict(
        schema="dpanel.half_wake_groups_declaration.v1",
        chord=ROOT_CHORD, reference_area=2.0 * sref_single,
        origin=[0.0, 0.0, 0.0],
        notes=("Two identical tapered swept wings stacked at z=0 and z=0.24, "
               "each with its own full-span trailing edge and its own horizontal "
               "wake sheet; reference area is the sum of both planforms."),
        groups=groups)
    (HERE / "groups.json").write_text(json.dumps(declaration, indent=2) + "\n",
                                      encoding="utf-8")
    build = dict(step=str(step), solids=len(solids), volumes=volumes,
                 sref_single=sref_single, sref_total=2.0 * sref_single,
                 half_span=HALF_SPAN, root_chord=ROOT_CHORD, tip_chord=TIP_CHORD,
                 sweep_deg=SWEEP_DEG, tc=TC, camber=CAMBER, sections=SECTIONS,
                 z_layers=[float(z) for _, z in Z_LAYERS],
                 te_tip=[float(le[-1] + chord[-1]), float(y[-1]), float(Z_LAYERS[0][1])])
    (HERE / "build.json").write_text(json.dumps(build, indent=2) + "\n",
                                     encoding="utf-8")
    gmsh.finalize()
    print(json.dumps(build, indent=2))


if __name__ == "__main__":
    main()