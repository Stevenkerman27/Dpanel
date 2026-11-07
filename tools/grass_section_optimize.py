"""Optimize four GRASS sections with a fixed-topology P1 fixed-wake model."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

from panel.grass_mesh import build_grass_mesh
from panel.drag_consistency_audit import ideal_trefftz_piecewise_linear
from panel.p1_cst_gradient import fixed_wake_geometry, grass_section_geometry
from panel.p1_lifting_g2 import combine_wings, solve_multi
from tools.web.params import GRASS_DEFAULTS, normalise

parser = argparse.ArgumentParser()
parser.add_argument("--case", type=Path)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--size", type=float, default=.5)
parser.add_argument("--alpha", type=float, default=4.)
parser.add_argument("--iterations", type=int, default=20)
parser.add_argument("--objective", choices=("induced", "pressure"), default="induced")
parser.add_argument("--chord-panels", type=int, default=3)
parser.add_argument("--span-panels-per-segment", type=int, default=2)
args = parser.parse_args()
if args.out.exists():
    raise ValueError("output directory must be new")
if args.case:
    sections = json.loads(args.case.read_text(encoding="utf-8"))
else:
    defaults = GRASS_DEFAULTS
    sections = {
        name: [(np.asarray(values[0]) * (1 - t) + np.asarray(values[1]) * t).tolist()
               for t in np.linspace(0., 1., 4)]
        for name, values in defaults.items()
    }
sections = normalise({"grass_sections": sections})[0]["grass_sections"]
if len(sections["chord"]) != 4:
    raise ValueError("this study requires exactly four sections")
mesh = build_grass_mesh(sections, args.size, chord_panels=args.chord_panels,
                        span_panels_per_segment=args.span_panels_per_segment)
args.out.mkdir(parents=True)
(args.out / "initial_surface.json").write_text(json.dumps(mesh, allow_nan=False), encoding="utf-8")
design0 = np.column_stack((sections["chord"], sections["twist_radians"],
                           np.asarray(sections["cst"])[:, :17]))
x0 = design0.ravel()
delta = np.tile(np.r_[.05, .035, np.full(16, .04), .02], 4)
bounds = list(zip(x0 - delta, x0 + delta))
leading = torch.tensor(sections["leading_edge_xyz"], dtype=torch.float64)
area0 = float(2 * np.trapezoid(np.asarray(sections["chord"]),
                               np.asarray(sections["leading_edge_xyz"])[:, 1]))
history = []
cache = {}


def evaluate(x):
    if "x" in cache and np.array_equal(x, cache["x"]):
        return cache
    variable = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    matrix = variable.reshape(4, 19)
    geometry = dict(leading_edge_xyz=leading, chord=matrix[:, 0], twist_radians=matrix[:, 1],
                    cst=torch.cat((matrix[:, 2:], torch.zeros((4, 1), dtype=torch.float64)), dim=1))
    try:
        wing, xyz, wake, area = grass_section_geometry(
            geometry, chord_panels=mesh["chord_panels"],
            span_panels_per_segment=mesh["span_panels_per_segment"],
            wake_panels=mesh["wake_panels"], wake_length=mesh["wake_length"])
        values, info = fixed_wake_geometry(wing, xyz, wake, area, args.alpha, 160,
                                           include_ideal_drag=True)
        jac = np.stack([torch.autograd.grad(value, variable, retain_graph=True)[0].detach().numpy()
                        for value in values])
        area_jac = torch.autograd.grad(area, variable)[0].detach().numpy()
        record = dict(evaluation=len(history) + 1, CL=float(values[0].detach()),
                      CD=float(values[1].detach()), CDi=float(values[2].detach()), area=float(area.detach()),
                      stationarity=info["stationarity"], x=x.tolist())
        history.append(record)
        with (args.out / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print("eval %d CL %.9f CDp %.9f CDi %.9f area %.9f" %
              (record["evaluation"], record["CL"], record["CD"], record["CDi"], record["area"]), flush=True)
        cache.clear()
        cache.update(x=x.copy(), CL=record["CL"], CD=record["CD"], CDi=record["CDi"], area=record["area"],
                     jac=jac, area_jac=area_jac)
        return cache
    except Exception as exc:
        with (args.out / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(error=str(exc), x=x.tolist())) + "\n")
        raise


baseline = evaluate(x0)
cl0 = baseline["CL"]
constraints = [
    dict(type="ineq", fun=lambda x: evaluate(x)["CL"] - cl0,
         jac=lambda x: evaluate(x)["jac"][0]),
    dict(type="eq", fun=lambda x: evaluate(x)["area"] - area0,
         jac=lambda x: evaluate(x)["area_jac"]),
]
objective_key = "CDi" if args.objective == "induced" else "CD"
objective_index = 2 if args.objective == "induced" else 1
result = minimize(lambda x: evaluate(x)[objective_key], x0,
                  jac=lambda x: evaluate(x)["jac"][objective_index], bounds=bounds,
                  constraints=constraints, method="SLSQP",
                  options=dict(maxiter=args.iterations, ftol=1e-8, disp=True))
final = result.x.reshape(4, 19)
final_sections = dict(leading_edge_xyz=sections["leading_edge_xyz"],
                      chord=final[:, 0].tolist(), twist_radians=final[:, 1].tolist(),
                      cst=np.column_stack((final[:, 2:], np.zeros(4))).tolist())
final_mesh = build_grass_mesh(final_sections, args.size, chord_panels=mesh["chord_panels"],
                              span_panels_per_segment=mesh["span_panels_per_segment"])
if (final_mesh["chord_panels"] != mesh["chord_panels"] or
        final_mesh["span_panels_per_segment"] != mesh["span_panels_per_segment"]):
    raise ValueError("final mesh size selected a different topology; fixed-topology result retained in history")
(args.out / "final_surface.json").write_text(json.dumps(final_mesh, allow_nan=False), encoding="utf-8")
formal = []
for candidate in (sections, final_sections):
    wing, _, _, _ = grass_section_geometry(
        candidate, chord_panels=mesh["chord_panels"],
        span_panels_per_segment=mesh["span_panels_per_segment"])
    state = solve_multi(combine_wings([wing]), args.alpha)
    gamma = np.r_[0., state["group_results"][0]["gamma"], 0.]
    ideal = ideal_trefftz_piecewise_linear(wing.gamma_y, gamma, U=1., density=1.,
                                            reference_area=wing.reference_area)
    formal.append(dict(CL=state["CL"], CD=state["CD_pressure"], CDi=ideal["CDi_ideal"]))
final_result = evaluate(result.x)
report = dict(success=bool(result.success), status=int(result.status), message=str(result.message),
              iterations=int(result.nit), evaluations=len(history), sections_initial=sections,
              sections_final=final_sections, panel_count=len(mesh["faces"]), alpha=args.alpha,
              target="minimize %s CD, CL >= initial CL, reference area fixed" % args.objective,
              bounds="chord +/-0.05 m, twist +/-0.035 rad, CST weights +/-0.04, LE weight +/-0.02",
              initial=dict(CL=cl0, CD=history[0]["CD"], CDi=history[0]["CDi"], area=area0),
              final={key: final_result[key] for key in ("CL", "CD", "CDi", "area")},
              formal_initial=formal[0], formal_final=formal[1],
              final_constraint=dict(lift=final_result["CL"]-cl0, area=final_result["area"]-area0),
              physical_accuracy_accepted=False)
(args.out / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps(dict(success=report["success"], message=report["message"],
                      initial=report["initial"], final=formal[1],
                      constraints=report["final_constraint"])), flush=True)
