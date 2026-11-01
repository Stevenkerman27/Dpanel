"""Does the small fan sensitivity survive once the wake is actually advanced?

Same wing, eight geometry updates instead of one, four velocity settings.
"""
import json
import numpy as np
from tools.grass_fan_convergence.cases import cases
from tools.grass_fan_convergence.run_sweep import resolve_airfoils
from tools.grass_fan_convergence.wing_builder import build_wing
from panel.p1_lifting_g2 import combine_wings
from panel.p1_coupled_material_wake import solve_coupled_material_wake

RUNS = [
    ("analytic", dict(velocity_method="analytic_edges")),
    ("analytic_open", dict(velocity_method="analytic_edges", epsilon_tol=1e3, quadrature_tol=1e3)),
    ("8:2/16:2", dict(velocity_method="rm_regularized_p1", area_quadrature="projection_fan",
                      core_radius=0.01, fan_order=8, fan_strips=2, fan_refinement_order=16,
                      fan_refinement_strips=2)),
    ("16:4/32:4", dict(velocity_method="rm_regularized_p1", area_quadrature="projection_fan",
                       core_radius=0.01, fan_order=16, fan_strips=4, fan_refinement_order=32,
                       fan_refinement_strips=4)),
]

case = [c for c in cases() if c["name"] == "g01_rect_ar6"][0]
sections = resolve_airfoils([dict(s) for s in case["sections"]])
wing = build_wing(sections, span_panels=case["span_panels"], chord_panels=case["chord_panels"],
                  wake_panels=case["wake_panels"], wake_length=5.0, name=case["name"])
mesh = combine_wings([wing], names=[case["name"]], reference_chord=wing.chord,
                     reference_point=np.zeros(3))
print("faces=%d ngamma=%d stations=%d" % (len(wing.faces), wing.gamma_hat.shape[1],
      len(wing.gamma_y)), flush=True)

out = []
for label, kwargs in RUNS:
    res = solve_coupled_material_wake(mesh, 5.0, kutta="pressure", iterations=8,
                                      order=48, **kwargs)
    hist = []
    for h in res["history"]:
        upd = h.get("outgoing_update") or {}
        hist.append(dict(iteration=int(h.get("iteration", -1)),
                         CL=float(h.get("CL", np.nan)),
                         CD_p=float(h.get("CD_pressure", np.nan)),
                         CMy=float(h.get("CM_y", np.nan)),
                         moved=upd.get("moved_nodes"),
                         relaxation=upd.get("relaxation"),
                         accepted=upd.get("accepted")))
    final = res["state"]
    row = dict(label=label, status=res["status"], iterations=int(res["iterations"]),
               seconds=float(res["seconds"]), CL=float(final["CL"]),
               CL_gamma=float(final["CL_gamma"]), CD_p=float(final["CD_pressure"]),
               CM_y=float(final["CM_y"]), history=hist)
    out.append(row)
    print("%-12s status=%-22s CL=%+.6f CD_p=%+.6f CMy=%+.6f  %.1fs" %
          (label, row["status"], row["CL"], row["CD_p"], row["CM_y"], row["seconds"]), flush=True)
    print("   CL by update: " + " ".join("%+.5f" % h["CL"] for h in hist), flush=True)

with open("docs/issue84/fan_wake_convergence.json", "w", encoding="utf-8") as fh:
    json.dump(dict(bridge="bank fixed from all face vertices", runs=out), fh, indent=1)
