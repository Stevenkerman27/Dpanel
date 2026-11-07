"""Compare GRASS gradients with production P1 centered differences on one topology."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from panel.grass_mesh import build_grass_mesh, read_grass_mesh, reconstruct_grass_mesh
from panel.drag_consistency_audit import ideal_trefftz_piecewise_linear
from panel.p1_cst_gradient import fixed_wake_geometry, grass_section_geometry
from panel.p1_lifting_g2 import combine_wings, solve_multi

parser = argparse.ArgumentParser()
source = parser.add_mutually_exclusive_group(required=True)
source.add_argument("--surface", type=Path)
source.add_argument("--default-size", type=float)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--step", type=float, default=1e-5)
parser.add_argument("--alpha", type=float, default=4.)
parser.add_argument("--ideal-drag", action="store_true")
parser.add_argument("--four-sections", action="store_true")
args = parser.parse_args()
if args.out.exists() or args.step <= 0:
    raise ValueError("output must be new and step positive")
if args.surface is None:
    from panel.grass_mesh import load_grass_wing
    from tools.web.params import GRASS_DEFAULTS
    data = build_grass_mesh(GRASS_DEFAULTS, args.default_size)
    saved = load_grass_wing(data)
else:
    data, saved = read_grass_mesh(args.surface)
geometry, wing, xyz, wake, area = reconstruct_grass_mesh(data)
values, info = fixed_wake_geometry(wing, xyz, wake, area, args.alpha, 160,
                                   include_ideal_drag=args.ideal_drag)
formal = solve_multi(combine_wings([saved]), args.alpha)
variables = [("leading_edge_xyz", (1, 0)), ("leading_edge_xyz", (1, 1)),
             ("leading_edge_xyz", (1, 2)), ("chord", (0,)), ("chord", (1,)),
             ("twist_radians", (0,)), ("twist_radians", (1,)),
             ("cst", (0, 0)), ("cst", (1, 8)), ("cst", (1, 16))]
if args.four_sections:
    if len(data["sections"]["chord"]) != 4:
        raise ValueError("four-section check requires exactly four sections")
    variables = [("chord", (i,)) for i in range(4)] + [("twist_radians", (i,)) for i in range(4)] + [("cst", (i, 0)) for i in range(4)]
rows = []
for name, index in variables:
    derivative = [torch.autograd.grad(value, geometry[name], retain_graph=True)[0][index].item()
                  for value in values]
    results = []
    for direction in (-1, 1):
        changed = {key: value.detach().clone() for key, value in geometry.items()}
        changed[name][index] += direction * args.step
        moved, _, _, _ = grass_section_geometry(
            changed, chord_panels=data["chord_panels"],
            span_panels_per_segment=data["span_panels_per_segment"],
            wake_panels=data["wake_panels"], wake_length=data["wake_length"])
        for field in ("faces", "dof_face", "wake_faces", "gamma_hat", "te_pairs"):
            if not np.array_equal(getattr(moved, field), getattr(saved, field)):
                raise ValueError("perturbation changed frozen topology: " + field)
        state = solve_multi(combine_wings([moved]), args.alpha)
        result = [state["CL"], state["CD_pressure"]]
        if args.ideal_drag:
            gamma = np.r_[0., state["group_results"][0]["gamma"], 0.]
            ideal = ideal_trefftz_piecewise_linear(moved.gamma_y, gamma, U=1., density=1.,
                                                    reference_area=moved.reference_area)
            result.append(ideal["CDi_ideal"])
        results.append(result)
    finite = (np.asarray(results[1]) - results[0]) / (2 * args.step)
    rows.append(dict(name=name, index=index, derivative=derivative,
                     finite_difference=finite.tolist(),
                     abs_error=np.abs(np.asarray(derivative) - finite).tolist()))
formal_forward = [formal["CL"], formal["CD_pressure"]]
if args.ideal_drag:
    gamma = np.r_[0., formal["group_results"][0]["gamma"], 0.]
    ideal = ideal_trefftz_piecewise_linear(saved.gamma_y, gamma, U=1., density=1.,
                                            reference_area=saved.reference_area)
    formal_forward.append(ideal["CDi_ideal"])
record = dict(surface=str(args.surface) if args.surface else None,
              sections=data["sections"], chord_panels=data["chord_panels"],
              span_panels_per_segment=data["span_panels_per_segment"],
              wake_panels=data["wake_panels"], wake_length=data["wake_length"],
              step=args.step, alpha=args.alpha,
              panels=len(saved.faces), stationarity=info["stationarity"],
              gradient_forward=values.detach().tolist(),
              formal_forward=formal_forward, variables=rows,
              max_absolute_error=np.max([row["abs_error"] for row in rows], axis=0).tolist())
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps(dict(panels=record["panels"], max_absolute_error=record["max_absolute_error"])))
