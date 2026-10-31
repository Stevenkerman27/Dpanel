"""Diagnose the GRASS bridge: circulation basis and loads vs the native wing."""
import numpy as np
from panel.p1_lifting_g1 import make_wing
from panel.p1_lifting_g2 import combine_wings
from panel.p1_galerkin_solver import solve_galerkin_multi
from tools.grass_fan_convergence.cases import cases
from tools.grass_fan_convergence.run_sweep import resolve_airfoils
from tools.grass_fan_convergence.wing_builder import build_wing


def report(tag, wing):
    mesh = combine_wings([wing], names=[tag], reference_chord=wing.chord,
                         reference_point=np.zeros(3))
    gal = solve_galerkin_multi(mesh, 5.0, kutta="pressure")
    g = np.asarray(gal["gamma"], float)
    print("%-10s faces=%4d ndof_mu=%3d ngamma=%2d gamma_y=[%.2f..%.2f] Sref=%.3f"
          % (tag, len(wing.faces), wing.ndof_mu, g.size, wing.gamma_y[0], wing.gamma_y[-1],
             wing.reference_area), flush=True)
    print("   CL=%+.6f CLg=%+.6f CD_p=%+.6f  sum(gamma*dy)=%+.6f  gamma=%s"
          % (gal["CL"], gal["CL_gamma"], gal["CD_pressure"],
             float(np.trapezoid(np.r_[0., g, 0.], wing.gamma_y)),
             np.array2string(g, precision=3, max_line_width=200)), flush=True)
    print("   te_pairs=%d te_up=%s te_down=%s wake_faces=%d"
          % (len(wing.te_pairs), np.shape(wing.te_up), np.shape(wing.te_down),
             len(wing.wake_faces)), flush=True)


report("native", make_wing(chord_panels=8, span_panels=9, thickness=0.12, camber=0.0,
                           span=6.0, chord=1.0, wake_panels=2, wake_length=5.0))
case = [c for c in cases() if c["name"] == "g01_rect_ar6"][0]
sections = resolve_airfoils([dict(s) for s in case["sections"]])
report("grass", build_wing(sections, span_panels=9, chord_panels=8, wake_panels=2,
                           wake_length=5.0, name="g01"))
