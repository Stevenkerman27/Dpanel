"""g02: is the analytic-vs-fan CL gap caused by the advected node set or the kernel?"""
import numpy as np
from panel.p1_coupled_material_wake import solve_coupled_material_wake
from tools.grass_fan_convergence.cases import cases
from tools.grass_fan_convergence.run_sweep import build_mesh

case = [c for c in cases() if c["name"] == "g02_taper05_ar7"][0]
wing, mesh = build_mesh(case)
print("faces", len(mesh.faces), "wake", len(wing.wake_faces), flush=True)

fan = dict(velocity_method="rm_regularized_p1", core_radius=0.01,
           area_quadrature="projection_fan", fan_order=8, fan_strips=2,
           fan_refinement_order=16, fan_refinement_strips=2)
runs = [
    ("analytic it0", dict(velocity_method="analytic_edges"), {}, 0),
    ("analytic it1", dict(velocity_method="analytic_edges"), {}, 1),
    ("analytic open", dict(velocity_method="analytic_edges"),
     dict(epsilon_tol=1e3, quadrature_tol=1e3), 1),
    ("fan8 it0", fan, {}, 0),
    ("fan8 it1", fan, {}, 1),
]
for label, kw, extra, iters in runs:
    res = solve_coupled_material_wake(mesh, 5.0, kutta="pressure", iterations=iters,
                                      order=48, **kw, **extra)
    st = res["state"]
    hist = res["history"][0] if res["history"] else {}
    wake = next(iter(hist.get("wakes", {}).values()), {})
    upd = hist.get("outgoing_update") or {}
    print("%-14s CL=%+.8f CLg=%+.8f CD_p=%+.8f CMy=%+.8f elig=%s moved=%s relax=%s" % (
        label, st["CL"], st["CL_gamma"], st["CD_pressure"], st["CM_y"],
        {k: int(v) for k, v in (wake.get("eligibility") or {}).items()
         if isinstance(v, (int, float, np.integer))},
        upd.get("moved_nodes"), upd.get("relaxation")), flush=True)
