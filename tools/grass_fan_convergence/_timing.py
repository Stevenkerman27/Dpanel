"""Timing probe: cost of one coupled solve per fan configuration."""
import time
from panel.p1_coupled_material_wake import solve_coupled_material_wake
from tools.grass_fan_convergence.cases import cases
from tools.grass_fan_convergence.run_sweep import build_mesh

case = [c for c in cases() if c["name"] == "g01_rect_ar6"][0]
wing, mesh = build_mesh(case)
print("faces", len(mesh.faces), "wake faces", len(wing.wake_faces), flush=True)

runs = [
    ("analytic", dict(velocity_method="analytic_edges")),
    ("8:2/16:2", dict(velocity_method="rm_regularized_p1", core_radius=0.01,
                     area_quadrature="projection_fan", fan_order=8, fan_strips=2,
                     fan_refinement_order=16, fan_refinement_strips=2)),
    ("16:4/32:4", dict(velocity_method="rm_regularized_p1", core_radius=0.01,
                      area_quadrature="projection_fan", fan_order=16, fan_strips=4,
                      fan_refinement_order=32, fan_refinement_strips=4)),
    ("28:8/48:8", dict(velocity_method="rm_regularized_p1", core_radius=0.01,
                      area_quadrature="projection_fan", fan_order=28, fan_strips=8,
                      fan_refinement_order=48, fan_refinement_strips=8)),
]
for label, kw in runs:
    t0 = time.perf_counter()
    res = solve_coupled_material_wake(mesh, 5.0, kutta="pressure", iterations=1,
                                      order=48, **kw)
    st = res["state"]
    print("%-10s %7.1fs CL=%+.8f CD_p=%+.8f CMy=%+.8f status=%s" % (
        label, time.perf_counter() - t0, st["CL"], st["CD_pressure"], st["CM_y"],
        res["status"]), flush=True)
