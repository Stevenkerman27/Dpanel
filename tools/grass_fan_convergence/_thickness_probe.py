"""Is the low lift of the make_wing/GRASS family a thickness (cusp TE) effect?"""
from panel.p1_lifting_g1 import make_wing
from panel.p1_lifting_g2 import combine_wings
from panel.p1_coupled_material_wake import solve_coupled_material_wake

for thickness in (0.12, 0.06, 0.02, 0.005):
    wing = make_wing(chord_panels=8, span_panels=6, thickness=thickness,
                     camber=0.0, span=6.0, chord=1.0, wake_panels=3)
    mesh = combine_wings([wing], names=["wing"])
    res = solve_coupled_material_wake(mesh, 5.0, kutta="pressure", iterations=0,
                                      order=48, velocity_method="analytic_edges")
    st = res["state"]
    print("thickness=%.3f faces=%d CL=%+.8f CLg=%+.8f CD_p=%+.8f CMy=%+.8f status=%s"
          % (thickness, len(wing.faces), st["CL"], st["CL_gamma"], st["CD_pressure"],
             st["CM_y"], res["status"]), flush=True)
