"""Why do analytic and projection-fan coefficients differ for g02?"""
import numpy as np
from panel.p1_coupled_material_wake import solve_coupled_material_wake
from tools.grass_fan_convergence.cases import cases
from tools.grass_fan_convergence.run_sweep import build_mesh

case = [c for c in cases() if c["name"] == "g02_taper05_ar7"][0]
wing, mesh = build_mesh(case)
print("faces", len(mesh.faces), "wake", len(wing.wake_faces), flush=True)

runs = [
    ("analytic", dict(velocity_method="analytic_edges"), {}),
    ("fan_gated", dict(velocity_method="rm_regularized_p1", core_radius=0.01,
                       area_quadrature="projection_fan", fan_order=8, fan_strips=2,
                       fan_refinement_order=16, fan_refinement_strips=2), {}),
    ("fan_open", dict(velocity_method="rm_regularized_p1", core_radius=0.01,
                      area_quadrature="projection_fan", fan_order=8, fan_strips=2,
                      fan_refinement_order=16, fan_refinement_strips=2),
     dict(epsilon_tol=1e3, quadrature_tol=1e3)),
]
for label, kw, extra in runs:
    res = solve_coupled_material_wake(mesh, 5.0, kutta="pressure", iterations=1,
                                      order=48, **kw, **extra)
    st = res["state"]
    wake = next(iter(res["history"][0]["wakes"].values()))
    elig = wake.get("eligibility") or {}
    upd = (res["history"][0].get("outgoing_update") or {})
    print("%-10s CL=%+.8f CD_p=%+.8f CMy=%+.8f status=%s" % (
        label, st["CL"], st["CD_pressure"], st["CM_y"], res["status"]), flush=True)
    print("   eligibility", {k: (int(v) if isinstance(v, (int, np.integer)) else v)
                             for k, v in elig.items()}, flush=True)
    print("   update accepted=%s relaxation=%s moved=%s" % (
        upd.get("accepted"), upd.get("relaxation"), upd.get("moved_nodes")), flush=True)
