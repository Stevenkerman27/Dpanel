"""Pilot: does the projection-fan configuration move the overall coefficients?"""
from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np

from tools.issue84.wake_node_rejection import build_case
from panel.p1_coupled_material_wake import solve_coupled_material_wake

SURFACE = ["D:/3D/Projects/panel/runs/half_cad_wingbody_v1/surface.json"]
mesh, alpha = build_case(SURFACE, case="g2", alpha_deg=5.0)
print("faces", len(mesh.faces), "wake faces",
      sum(len(g.wake_faces) for g in mesh.groups), flush=True)

configs = [
    ("analytic_edges", dict(velocity_method="analytic_edges")),
    ("fan 8:2", dict(velocity_method="rm_regularized_p1", core_radius=.01,
                     area_quadrature="projection_fan", fan_order=8, fan_strips=2,
                     fan_refinement_order=48, fan_refinement_strips=8)),
    ("fan 28:8", dict(velocity_method="rm_regularized_p1", core_radius=.01,
                      area_quadrature="projection_fan", fan_order=28, fan_strips=8,
                      fan_refinement_order=48, fan_refinement_strips=8)),
]
rows = []
for label, kwargs in configs:
    t0 = time.perf_counter()
    result = solve_coupled_material_wake(mesh, alpha, kutta="pressure", iterations=1,
                                         **kwargs)
    state = result["state"]
    row = dict(config=label, seconds=time.perf_counter()-t0,
               status=result["status"], CL=float(state["CL"]),
               CL_gamma=float(state["CL_gamma"]),
               CD_pressure=float(state["CD_pressure"]),
               CM_y=float(state["CM_y"]),
               update=(result["history"][0].get("outgoing_update") or {}).get("accepted"),
               iterations=int(result["iterations"]))
    rows.append(row)
    print("%-16s %7.1fs status=%-18s CL=%+.9f CLg=%+.9f CD_p=%+.9f CMy=%+.9f acc=%s" % (
        label, row["seconds"], row["status"], row["CL"], row["CL_gamma"],
        row["CD_pressure"], row["CM_y"], row["update"]), flush=True)

base = rows[0]
for row in rows[1:]:
    print("%-16s dCL=%+.3e dCD_p=%+.3e dCM_y=%+.3e" % (
        row["config"], row["CL"]-base["CL"], row["CD_pressure"]-base["CD_pressure"],
        row["CM_y"]-base["CM_y"]), flush=True)
out = Path("docs/issue84/fan_coefficient_pilot.json")
out.write_text(json.dumps(dict(rows=rows), indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("wrote", out, flush=True)
