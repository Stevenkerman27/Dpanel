"""Same wing, same fan configuration, three gate modes: cost and load effect."""
import json
import numpy as np
from tools.grass_fan_convergence.cases import cases
from tools.grass_fan_convergence.run_sweep import resolve_airfoils
from tools.grass_fan_convergence.wing_builder import build_wing
from panel.p1_lifting_g2 import combine_wings
from panel.p1_coupled_material_wake import solve_coupled_material_wake

case = [c for c in cases() if c["name"] == "g01_rect_ar6"][0]
sections = resolve_airfoils([dict(s) for s in case["sections"]])
wing = build_wing(sections, span_panels=case["span_panels"], chord_panels=case["chord_panels"],
                  wake_panels=case["wake_panels"], wake_length=5.0, name=case["name"])
mesh = combine_wings([wing], names=[case["name"]], reference_chord=wing.chord,
                     reference_point=np.zeros(3))
rows = []
for mode in ("enforce", "audit", "off"):
    kwargs = dict(velocity_method="rm_regularized_p1", area_quadrature="projection_fan",
                  core_radius=0.01, fan_order=8, fan_strips=2,
                  fan_refinement_order=16, fan_refinement_strips=2)
    if mode == "off":
        kwargs = {k: v for k, v in kwargs.items() if k not in ("fan_refinement_order",
                                                               "fan_refinement_strips")}
    res = solve_coupled_material_wake(mesh, 5.0, kutta="pressure", iterations=3, order=48,
                                      gate_mode=mode, **kwargs)
    row = dict(gate_mode=mode, seconds=float(res["seconds"]), status=res["status"],
               CL=float(res["state"]["CL"]), CD_p=float(res["state"]["CD_pressure"]),
               CM_y=float(res["state"]["CM_y"]),
               frozen=[r["wakes"][g]["eligibility"]["rejected"] for r in res["history"]
                       for g in r["wakes"]],
               audited=[r["wakes"][g]["eligibility"]["audit_rejected"] for r in res["history"]
                        for g in r["wakes"]])
    rows.append(row)
    print("%-8s %6.1fs CL=%+.6f CD_p=%+.6f CMy=%+.6f frozen=%s audited=%s" %
          (mode, row["seconds"], row["CL"], row["CD_p"], row["CM_y"],
           row["frozen"], row["audited"]), flush=True)
with open("docs/issue84/gate_mode_ab.json", "w", encoding="utf-8") as fh:
    json.dump(dict(note="rectangular wing, fan 8:2/16:2, 3 updates", runs=rows), fh, indent=1)
