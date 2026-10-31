"""Smoke the elliptic and .dat-section geometries after the builder fixes."""
from tools.grass_fan_convergence.cases import cases
from tools.grass_fan_convergence.run_sweep import build_mesh
from panel.p1_coupled_material_wake import solve_coupled_material_wake

want = {"g05_elliptic_ar10", "g06_glider_ar12", "g10_camber4412_ar7"}
for case in cases():
    if case["name"] not in want:
        continue
    wing, mesh = build_mesh(case)
    res = solve_coupled_material_wake(mesh, 5.0, kutta="pressure", iterations=0,
                                      order=48, velocity_method="analytic_edges")
    st = res["state"]
    print("%s faces=%d wake=%d area=%.4f CL=%+.6f CD_p=%+.6f CMy=%+.6f status=%s"
          % (case["name"], len(wing.faces), len(wing.wake_faces), mesh.reference_area,
             st["CL"], st["CD_pressure"], st["CM_y"], res["status"]), flush=True)
