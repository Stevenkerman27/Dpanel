"""A/B: collocation Dirichlet least squares vs P1 Galerkin on the GRASS wings."""
import copy
import numpy as np
from tools.grass_fan_convergence.cases import cases
from tools.grass_fan_convergence.run_sweep import resolve_airfoils
from tools.grass_fan_convergence.wing_builder import build_wing
from panel.p1_lifting_g2 import combine_wings
from panel.p1_lifting_g2 import solve_multi
from panel.p1_galerkin_solver import solve_galerkin_multi


def mesh_for(case, *, chord_panels=None):
    sections = resolve_airfoils([dict(s) for s in case["sections"]])
    wing = build_wing(sections, span_panels=case["span_panels"],
                      chord_panels=chord_panels or case["chord_panels"],
                      wake_panels=case["wake_panels"],
                      wake_length=case.get("wake_length", 5.0), name=case["name"])
    mesh = combine_wings([wing], names=[case["name"]], reference_chord=wing.chord,
                         reference_point=np.zeros(3))
    return wing, mesh


def pick(res, *names):
    for key in names:
        if key in res:
            return float(res[key])
    return float("nan")


def row(tag, faces, res):
    print("%-28s faces=%4d CL=%+.6f CLg=%+.6f CD_p=%+.6f CMy=%+.6f  %s"
          % (tag, faces, pick(res, "CL_pressure", "CL"), pick(res, "CL_gamma"),
             pick(res, "CD_pressure", "CD_p"), pick(res, "CM_y"),
             res.get("note", "")), flush=True)


for name in ("g01_rect_ar6", "g02_taper05_ar7", "g10_camber4412_ar7"):
    case = [c for c in cases() if c["name"] == name][0]
    for chord in (case["chord_panels"], 12):
        wing, mesh = mesh_for(case, chord_panels=chord)
        nfaces = len(wing.faces)
        col = solve_multi(mesh, 5.0, kutta="pressure")
        row("collocation " + name, nfaces, col)
        gal = solve_galerkin_multi(mesh, 5.0, kutta="pressure")
        gal["note"] = "order3"
        row("galerkin     " + name, nfaces, gal)
