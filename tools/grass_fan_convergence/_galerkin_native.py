"""Same panel counts: make_wing (DPanel native) vs the GRASS bridge, under Galerkin."""
import numpy as np
from panel.p1_lifting_g1 import make_wing
from panel.p1_lifting_g2 import combine_wings, solve_multi
from panel.p1_galerkin_solver import solve_galerkin_multi

for span_panels in (3, 6, 9):
    wing = make_wing(chord_panels=8, span_panels=span_panels, thickness=0.12,
                     camber=0.0, span=6.0, chord=1.0, wake_panels=2, wake_length=5.0)
    mesh = combine_wings([wing], names=["native"], reference_chord=wing.chord,
                         reference_point=np.zeros(3))
    col = solve_multi(mesh, 5.0, kutta="pressure")
    gal = solve_galerkin_multi(mesh, 5.0, kutta="pressure")
    print("make_wing span_panels=%d faces=%4d colloc CL=%+.6f | galerkin CL=%+.6f CD_p=%+.6f"
          % (span_panels, len(wing.faces), col["CL"], gal["CL"], gal["CD_pressure"]), flush=True)
