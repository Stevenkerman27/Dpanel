"""Ten GRASS-plane CST wing geometries for the fan order/strips convergence study.

Aspect ratios quoted in the notes are computed as b^2/S. The AR suffixes in the
case names are legacy placeholders and are NOT the geometric aspect ratio (for
example g02 is AR 9.33, not 7).

Each case is a full-span wing built by tools.grass_fan_convergence.wing_builder
(GRASS-plane thick_wing_topology / thick_wing_half_mesh + Kulfan CST codec).
The set deliberately spans the planform/section space that stresses the panel
method: rectangular, tapered, swept, elliptic, glider-high-AR, slender delta,
dihedral+washout, thick symmetric, strongly cambered.
"""
from __future__ import annotations

import numpy as np


def synthetic(number):
    return ("naca4", number)


def linear(y_tip, n, *, root_chord, tip_chord, sweep_deg=0.0, dihedral_deg=0.0,
           twist_deg=0.0, airfoil=None):
    ys = np.linspace(0.0, y_tip, n)
    frac = ys / y_tip
    return [dict(y=float(y), chord=float(root_chord + (tip_chord - root_chord) * f),
                 le_x=float(np.tan(np.radians(sweep_deg)) * y),
                 z=float(np.tan(np.radians(dihedral_deg)) * y),
                 twist_deg=float(twist_deg * f), code=airfoil)
            for y, f in zip(ys, frac)]


def elliptic(y_tip, n, *, root_chord, airfoil, sweep_deg=0.0):
    # stop short of the singular zero-chord tip (chord = cos(theta) would vanish)
    theta = np.linspace(0.0, 0.46 * np.pi, n)
    ys = y_tip * np.sin(theta)
    return [dict(y=float(y), chord=float(root_chord * np.cos(t)),
                 le_x=float(np.tan(np.radians(sweep_deg)) * y),
                 code=airfoil)
            for y, t in zip(ys, theta)]


def cases():
    return [
        dict(name="g01_rect_ar6", airfoil=synthetic("0012"),
             sections=linear(3.0, 4, root_chord=1.0, tip_chord=1.0,
                             airfoil=synthetic("0012")),
             span_panels=3, chord_panels=8, wake_panels=2,
             note="rectangular planform, AR 6.00, symmetric NACA 0012"),
        dict(name="g02_taper05_ar7", airfoil=synthetic("2412"),
             sections=linear(3.5, 4, root_chord=1.0, tip_chord=0.5,
                             airfoil=synthetic("2412")),
             span_panels=3, chord_panels=8, wake_panels=2,
             note="taper 0.5, AR 9.33, cambered NACA 2412"),
        dict(name="g03_swept30_ar6", airfoil=synthetic("2412"),
             sections=linear(3.0, 4, root_chord=1.0, tip_chord=0.6,
                             sweep_deg=30.0, airfoil=synthetic("2412")),
             span_panels=3, chord_panels=8, wake_panels=2,
             note="30 deg leading-edge sweep, taper 0.6, AR 7.50"),
        dict(name="g04_swept45_taper02", airfoil=synthetic("0012"),
             sections=linear(2.0, 4, root_chord=1.0, tip_chord=0.2,
                             sweep_deg=45.0, airfoil=synthetic("0012")),
             span_panels=3, chord_panels=8, wake_panels=2,
             note="45 deg sweep, taper 0.2, thin symmetric section, AR 6.67"),
        dict(name="g05_elliptic_ar10", airfoil=synthetic("4412"),
             sections=elliptic(5.0, 6, root_chord=1.0, airfoil=synthetic("4412")),
             span_panels=3, chord_panels=8, wake_panels=2,
             note="elliptic planform, AR 12.70, NACA 4412"),
        dict(name="g06_glider_ar12", airfoil="naca63a210",
             sections=linear(4.0, 4, root_chord=0.7, tip_chord=0.3,
                             sweep_deg=5.0, airfoil="naca63a210"),
             span_panels=3, chord_panels=8, wake_panels=2,
             note="high-AR glider wing, AR 16.00, NACA 63A210 laminar section"),
        dict(name="g07_delta_ar2p5", airfoil=synthetic("0012"),
             sections=linear(1.5, 4, root_chord=1.4, tip_chord=0.2,
                             sweep_deg=50.0, airfoil=synthetic("0012")),
             span_panels=3, chord_panels=8, wake_panels=2,
             note="slender delta-like planform, AR 3.75, 50 deg sweep"),
        dict(name="g08_dihedral_washout", airfoil=synthetic("2412"),
             sections=linear(3.0, 4, root_chord=1.0, tip_chord=0.7,
                             dihedral_deg=5.0, twist_deg=-4.0,
                             airfoil=synthetic("2412")),
             span_panels=3, chord_panels=8, wake_panels=2,
             note="5 deg dihedral with 4 deg linear washout, AR 7.06"),
        dict(name="g09_thick0018_ar5", airfoil=synthetic("0018"),
             sections=linear(2.5, 4, root_chord=1.0, tip_chord=0.8,
                             airfoil=synthetic("0018")),
             span_panels=3, chord_panels=8, wake_panels=2,
             note="18 percent thick symmetric section, AR 5.56"),
        dict(name="g10_camber4412_ar7", airfoil="naca4412",
             sections=linear(3.5, 4, root_chord=1.0, tip_chord=0.7,
                             sweep_deg=8.0, airfoil="naca4412"),
             span_panels=3, chord_panels=8, wake_panels=2,
             note="strongly cambered NACA 4412 with 8 deg sweep, AR 8.24"),
    ]
