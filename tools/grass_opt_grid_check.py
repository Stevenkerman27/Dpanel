"""Re-evaluate the four-section baseline and optimized geometry on finer grids."""
import argparse
import json
from pathlib import Path

import numpy as np

from panel.drag_consistency_audit import ideal_trefftz_piecewise_linear
from panel.p1_cst_gradient import grass_section_geometry
from panel.p1_lifting_g2 import combine_wings, solve_multi

parser = argparse.ArgumentParser()
parser.add_argument("--result", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    raise ValueError("output must be new")
data = json.loads(args.result.read_text(encoding="utf-8"))
rows = []
for chord_panels, span_panels in ((3, 2), (4, 3), (6, 4)):
    for name in ("initial", "final"):
        wing, _, _, _ = grass_section_geometry(
            data["sections_" + name], chord_panels=chord_panels,
            span_panels_per_segment=span_panels)
        state = solve_multi(combine_wings([wing]), data["alpha"])
        gamma = np.r_[0., state["group_results"][0]["gamma"], 0.]
        ideal = ideal_trefftz_piecewise_linear(
            wing.gamma_y, gamma, U=1., density=1., reference_area=wing.reference_area)
        row = dict(case=name, chord_panels=chord_panels, span_panels_per_segment=span_panels,
                   panels=len(wing.faces), CL=state["CL"], CD=state["CD_pressure"],
                   CDi=ideal["CDi_ideal"], nonlinear=state["nonlinear"])
        rows.append(row)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print("%s panels=%d CL=%.8f CDp=%.8f CDi=%.8f" %
              (name, row["panels"], row["CL"], row["CD"], row["CDi"]), flush=True)
