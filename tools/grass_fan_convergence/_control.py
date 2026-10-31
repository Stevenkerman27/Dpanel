"""Control: GRASS rectangular NACA0012 vs panel.p1_lifting_g1.make_wing."""
import numpy as np
from panel.p1_lifting_g1 import make_wing
from panel.p1_lifting_g2 import combine_wings, solve_multi
from tools.grass_fan_convergence.wing_builder import build_wing, naca_code

def report(tag, wing):
    tri = wing.xyz[wing.faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    up = wing.face_bank == 1
    lo = wing.face_bank == -1
    print(tag, "faces", len(wing.faces), "dof", wing.ndof_mu, "gamma", wing.gamma_hat.shape[1],
          "upper nz+ %.3f" % np.mean(n[up, 2] > 0), "lower nz- %.3f" % np.mean(n[lo, 2] < 0),
          "area", wing.reference_area, flush=True)
    mesh = combine_wings([wing], names=[tag], reference_chord=wing.chord,
                         reference_point=np.zeros(3))
    st = solve_multi(mesh, 5.0, kutta="pressure")
    print("   CL=%+.6f CLg=%+.6f CD_p=%+.6f CMy=%+.6f pk=%.2e" % (
        st["CL"], st["CL_gamma"], st["CD_pressure"], st["CM_y"],
        st.get("pressure_kutta_rms", np.nan)), flush=True)

report("make_wing", make_wing(chord_panels=8, span_panels=6, thickness=.12, span=6., chord=1.,
                              camber=0., wake_panels=3, wake_length=5.))
code12 = naca_code("0012")
sec = [dict(y=k * 1.5, chord=1.0, code=code12) for k in range(5)]
report("grass_rect", build_wing(sec, span_panels=6, chord_panels=8, wake_panels=3,
                                wake_length=5.0, name="grass_rect"))
