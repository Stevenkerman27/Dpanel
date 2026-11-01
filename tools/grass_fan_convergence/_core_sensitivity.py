"""Core-radius sensitivity: the RM kernel is a model, not an error budget.

One geometry update, fan 8:2, the regularisation length scaled from 0.04 down
to 0.0025 chord; CL should approach the analytic-edge value as core -> 0.
"""
import json
import time
from pathlib import Path

import numpy as np

from tools.grass_fan_convergence.cases import cases
from tools.grass_fan_convergence.run_sweep import resolve_airfoils
from tools.grass_fan_convergence.wing_builder import build_wing
from panel.p1_lifting_g2 import combine_wings
from panel.p1_coupled_material_wake import solve_coupled_material_wake

case = [c for c in cases() if c["name"] == "g08_dihedral_washout"][0]
sections = resolve_airfoils([dict(s) for s in case["sections"]])
wing = build_wing(sections, span_panels=case["span_panels"], chord_panels=case["chord_panels"],
                  wake_panels=case["wake_panels"], wake_length=5.0, name=case["name"])
mesh = combine_wings([wing], names=[case["name"]], reference_chord=wing.chord,
                     reference_point=np.zeros(3))
rows = []
analytic = solve_coupled_material_wake(mesh, 5.0, kutta="pressure", iterations=1,
                                       gate_mode="audit")
rows.append(dict(method="analytic_edges", core=None, CL=float(analytic["state"]["CL"]),
                 CD_p=float(analytic["state"]["CD_pressure"]), seconds=float(analytic["seconds"])))
print("analytic_edges        CL=%+.6f CD_p=%+.6f %.1fs" %
      (rows[-1]["CL"], rows[-1]["CD_p"], rows[-1]["seconds"]), flush=True)
for core in (0.04, 0.02, 0.01, 0.005, 0.0025):
    started = time.perf_counter()
    res = solve_coupled_material_wake(mesh, 5.0, kutta="pressure", iterations=1,
                                      gate_mode="audit", velocity_method="rm_regularized_p1",
                                      area_quadrature="projection_fan", core_radius=core,
                                      fan_order=8, fan_strips=2,
                                      fan_refinement_order=16, fan_refinement_strips=2)
    row = dict(method="rm_regularized_p1", core=core, CL=float(res["state"]["CL"]),
               CD_p=float(res["state"]["CD_pressure"]), seconds=time.perf_counter()-started)
    rows.append(row)
    print("core=%.4f            CL=%+.6f CD_p=%+.6f %.1fs" % (core, row["CL"], row["CD_p"],
                                                            row["seconds"]), flush=True)
Path("docs/issue84/core_radius_sensitivity.json").write_text(
    json.dumps(dict(case=case["name"], runs=rows), indent=1), encoding="utf-8")
print("wrote docs/issue84/core_radius_sensitivity.json")
