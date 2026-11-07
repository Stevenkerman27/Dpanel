"""Physics anchors: analytic limits, not replays of our own past answers.

The regression references in docs/issue92/references/ only prove that behaviour did not
change. These tests ask whether the answer is right, using limits that depend on no
implementation of ours:

* a wing of large aspect ratio must approach the two-dimensional lift slope. Fitting
  1/CL_alpha against 1/AR and extrapolating to 1/AR -> 0 gives a_inf; a flat plate gives
  2*pi = 6.283, and a 12%-thick section sits a few percent above it.
* an unswept, untwisted elliptic planform must have span efficiency
  e = CDi / (CL^2 / (pi*AR)) close to one, and better than a rectangular wing of the same
  span and area. CDi comes from the Trefftz plane, so it is induced drag only; the
  near-field CD_pressure is not usable for this (it went negative on the elliptic case).
* the potential-jump closure must refuse a mesh on which its own Jacobian is too
  ill-conditioned for the trailing-edge circulation to be unique, instead of returning a
  converged-looking answer (see the tip section below).

Measured 2026-11-07/08 (myml: torch 2.13.0+cu132, CPU), alpha = 2 deg, collocation wall,
chord_panels = 8, span_panels = 4*AR, thickness 0.12, wake_panels = 2, wake_length = 5:

* potential_jump, AR 8/16/24 -> CL_alpha 4.9353/5.6294/5.8854, fit a_inf = 6.62189
  (+5.39% over 2*pi, consistent with 12% thickness), slope 0.426935 (lifting line
  predicts 1/pi = 0.318310);
* pressure, same sweep -> fit a_inf = 6.95530 (+10.70%), slope 1.125701, i.e. 3.54x the
  lifting-line slope: the pressure closure's aspect-ratio dependence is wrong.

Chordwise refinement at AR = 16, thickness 0.12 (CL):

| closure | c4 | c8 | c16 |
| --- | --- | --- | --- |
| pressure | 0.188409 | 0.150047 | 0.025672 |
| potential_jump | 0.208363 | 0.196360 | 0.189290 |

and for thickness 0.02 the pressure closure even goes negative (c4 0.139365, c8 0.053398,
c16 -0.095238) while potential_jump stays put (0.185201/0.180503/0.171499). The two
strict xfail markers below keep that defect visible: an XPASS means the defect was fixed
and the marker should become a plain assertion.

Elliptic planform (span 8, root chord 2, GRASS sections, CST 0.17): the planform is cut at
0.85 span and closed with a LINEAR tip taper to chord 0.24, because cutting a pure ellipse
at 0.97 span sweeps the trailing edge into the stream direction (77 deg) and makes the
potential-jump closure singular at the tip (measured below). Coarse mesh: 9 sections,
chord_panels 6, 4 span panels per segment, 1560 faces, AR 5.123, e = 0.9265, CL_alpha =
4.4341 against the lifting-line 4.518 (-1.9%); a rectangular wing of the same span and area
gives e = 0.8678. Refined to 13 sections and 8 span panels per segment (6176 faces) the
same bounded-tip planform gives e = 0.9884, i.e. it converges towards the elliptic optimum.

Tip conditioning, same 6176-face mesh, only the tip shape changed (cond = condition number
of dp/dgamma at the closure solution, measured by instrumenting torch.linalg.solve):

| tip | TE sweep | cond | gamma sign changes | CL |
| --- | --- | --- | --- | --- |
| rectangular wing, 96 span panels | 0 deg | 1.1e2 | 0 | 0.276199 |
| ellipse to 0.85 + linear taper 0.24 | 54 deg | 4.5e4 | 0 | 0.302000 |
| ellipse to 0.90 + linear taper 0.12 | 62 deg | 7.1e5 | 0 | 0.303038 |
| pure ellipse cut at 0.97 span | 77 deg | 7.3e6 | 12, gamma down to -2.13 | 0.253478 |

The last row is a converged state (stationarity 3e-15, jump residual 3e-15) that is not a
solution: Newton puts an arbitrary 2.13 circulation into the near-null direction, and the
wake sheet there is 7 strips of width 0.007 and length 20 chords (aspect ratio ~2900).
solve_potential_jump now raises above condition 1e6, which is what the last test checks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panel import p1_lifting_g1 as g1  # noqa: E402
from panel.grass_mesh import build_grass_mesh, load_grass_wing  # noqa: E402
from tools.issue92.trefftz_full import trefftz_cd_full  # noqa: E402


ALPHA_DEG = 2.0
TWO_PI = 2.0 * np.pi
SPAN = 8.0
ROOT_CHORD = 2.0


def _wing(aspect_ratio, chord_panels=8, thickness=0.12):
    return g1.make_wing(chord_panels=chord_panels,
                        span_panels=int(round(4 * aspect_ratio)),
                        thickness=thickness, camber=0.0, span=float(aspect_ratio),
                        chord=1.0, wake_panels=2, wake_length=5.)


def _lift_slope(kutta, chord_panels=8):
    aspect_ratios = np.array([8.0, 16.0, 24.0])
    slopes = []
    for aspect_ratio in aspect_ratios:
        state = g1.solve(_wing(aspect_ratio, chord_panels), ALPHA_DEG, kutta=kutta,
                         wall_formulation="collocation")
        slopes.append(float(state["CL"]) / np.deg2rad(ALPHA_DEG))
    slope, intercept = np.polyfit(1.0 / aspect_ratios, 1.0 / np.asarray(slopes), 1)
    return float(1.0 / intercept), float(slope)


def test_potential_jump_recovers_the_two_dimensional_lift_slope():
    a_inf, slope = _lift_slope("potential_jump")
    assert 0.95 * TWO_PI < a_inf < 1.15 * TWO_PI, a_inf
    assert abs(slope - 1.0 / np.pi) < 0.15, slope


@pytest.mark.xfail(strict=True,
                   reason="pressure closure: AR slope 1.126 vs 1/pi=0.318, a_inf +10.7%")
def test_pressure_kutta_has_the_lifting_line_aspect_ratio_slope():
    _, slope = _lift_slope("pressure")
    assert abs(slope - 1.0 / np.pi) < 0.15, slope


def _cl_at(aspect_ratio, chord_panels, kutta):
    state = g1.solve(_wing(aspect_ratio, chord_panels), ALPHA_DEG, kutta=kutta,
                     wall_formulation="collocation")
    return float(state["CL"])


def test_potential_jump_survives_chordwise_refinement():
    coarse, fine = _cl_at(16, 8, "potential_jump"), _cl_at(16, 16, "potential_jump")
    assert abs(fine - coarse) / coarse < 0.05, (coarse, fine)


@pytest.mark.xfail(strict=True,
                   reason="pressure closure: CL 0.150047 -> 0.025672 from c8 to c16")
def test_pressure_kutta_survives_chordwise_refinement():
    coarse, fine = _cl_at(16, 8, "pressure"), _cl_at(16, 16, "pressure")
    assert abs(fine - coarse) / coarse < 0.30, (coarse, fine)


def _sections(sections=9, blend=0.85, tip_chord=0.24):
    """Unswept elliptic planform, cut at 'blend' span and closed with a linear tip taper.

    A pure ellipse cut near the tip sweeps the trailing edge into the stream direction and
    makes the potential-jump Jacobian singular there; see the module docstring.
    """
    theta = np.linspace(0.0, 0.5 * np.pi * 0.97, sections)
    y = 0.5 * SPAN * np.sin(theta)
    chord = ROOT_CHORD * np.cos(theta)
    y0 = 0.5 * SPAN * blend
    c0 = ROOT_CHORD * np.sqrt(1.0 - blend ** 2)
    ramp = y > y0
    chord[ramp] = c0 + (tip_chord - c0) * (y[ramp] - y0) / (y[-1] - y0)
    return y, chord


def _grass_wing(y, chord, chord_panels=6, span_panels_per_segment=4):
    sections = len(y)
    code = np.r_[np.full(8, 0.17), np.full(8, -0.17), 0.0, 0.0]
    geometry = dict(
        leading_edge_xyz=np.c_[np.zeros(sections), y, np.zeros(sections)],
        chord=chord,
        twist_radians=np.zeros(sections),
        cst=np.tile(code, (sections, 1)))
    mesh = build_grass_mesh(geometry, 1.0, chord_panels=chord_panels,
                            span_panels_per_segment=span_panels_per_segment)
    return load_grass_wing(mesh)


def _span_efficiency(wing, alpha_deg=4.0):
    state = g1.solve(wing, alpha_deg, kutta="potential_jump",
                     wall_formulation="collocation")
    gamma = np.asarray(state["gamma"], float)
    stations = np.asarray(wing.gamma_y, float)
    cd, _ = trefftz_cd_full(stations, np.r_[0.0, gamma, 0.0],
                            area=float(wing.reference_area))
    aspect_ratio = float(np.ptp(wing.xyz[:, 1])) ** 2 / float(wing.reference_area)
    optimum = float(state["CL"]) ** 2 / (np.pi * aspect_ratio)
    return cd / optimum, float(state["CL"]) / np.deg2rad(alpha_deg), aspect_ratio


def test_elliptic_planform_reaches_unit_span_efficiency():
    elliptic = _grass_wing(*_sections())
    efficiency, lift_slope, aspect_ratio = _span_efficiency(elliptic)
    assert 0.90 < efficiency < 1.02, efficiency
    lifting_line = TWO_PI / (1.0 + TWO_PI / (np.pi * aspect_ratio))
    assert abs(lift_slope - lifting_line) / lifting_line < 0.05, (lift_slope, lifting_line)
    span = float(np.ptp(elliptic.xyz[:, 1]))
    rectangular = g1.make_wing(chord_panels=6, span_panels=32, thickness=0.12,
                               camber=0.0, span=span,
                               chord=float(elliptic.reference_area) / span,
                               wake_panels=2, wake_length=5.)
    rectangular_efficiency, _, _ = _span_efficiency(rectangular)
    assert rectangular_efficiency < efficiency - 0.03, (rectangular_efficiency, efficiency)


def test_potential_jump_refuses_an_ill_conditioned_tip():
    """A pure ellipse cut at 0.97 span converges to a residual of 1e-15 and is still wrong.

    Measured: cond(dp/dgamma) = 1.6e6 on this 3096-face mesh (7.3e6 on the 6176-face one),
    with the trailing-edge circulation carrying an arbitrary tip component. The closure must
    raise instead of returning it, otherwise every consumer sees a plausible CL.
    """
    y, chord = _sections(sections=9, blend=0.97, tip_chord=0.0942)
    wing = _grass_wing(y, chord, chord_panels=6, span_panels_per_segment=8)
    with pytest.raises(RuntimeError, match="ill-conditioned"):
        g1.solve(wing, 4.0, kutta="potential_jump", wall_formulation="collocation")
