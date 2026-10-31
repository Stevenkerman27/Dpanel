"""Spanwise refinement + quadrature order sensitivity, Galerkin vs collocation."""
import numpy as np
from tools.grass_fan_convergence.cases import cases
from tools.grass_fan_convergence.run_sweep import resolve_airfoils
from tools.grass_fan_convergence.wing_builder import build_wing
from panel.p1_lifting_g2 import combine_wings, solve_multi
from panel.p1_galerkin_solver import solve_galerkin_multi

case = [c for c in cases() if c["name"] == "g01_rect_ar6"][0]
sections = resolve_airfoils([dict(s) for s in case["sections"]])
for span in (3, 6, 9):
    wing = build_wing(sections, span_panels=span, chord_panels=8,
                      wake_panels=2, wake_length=5.0, name="g01")
    mesh = combine_wings([wing], names=["g01"], reference_chord=wing.chord,
                         reference_point=np.zeros(3))
    col = solve_multi(mesh, 5.0, kutta="pressure")
    gal = solve_galerkin_multi(mesh, 5.0, kutta="pressure")
    print("span_panels=%d faces=%4d colloc CL=%+.6f CD_p=%+.6f | galerkin CL=%+.6f CD_p=%+.6f"
          % (span, len(wing.faces), col["CL"], col["CD_pressure"],
             gal["CL"], gal["CD_pressure"]), flush=True)
    if span == 3:
        for order in (5, 8):
            g8 = solve_galerkin_multi(mesh, 5.0, kutta="pressure", order=order)
            print("   order=%d galerkin CL=%+.6f CD_p=%+.6f"
                  % (order, g8["CL"], g8["CD_pressure"]), flush=True)
