"""Trailing-edge strip audit: is the spanwise discretization fit for a Kutta closure?

The Morino potential-jump closure solves p(gamma) = 0 on a square ng x ng system, so its
dp/dgamma matrix has to be invertible.  When the trailing edge runs nearly parallel to the
stream, neighbouring TE rows become nearly collinear and the closure loses that property.
Measured cond(dp/dgamma) against geometry (GRASS elliptic wing, span 8, root chord 2, P1
wall, collocation closure):

    configuration                        TE sweep   min strip   strip aspect   cond
    rectangular, 96 span panels            0.0 deg     0.0833         120      1.1e2
    elliptic cut 0.85, taper to chord 0.24 53.8 deg    0.0270         807      4.5e4
    elliptic cut 0.90, taper to chord 0.12 62.2 deg    0.0135        1622      7.1e5
    elliptic cut 0.97 (cusp tip)           74.0 deg    0.0135        1624      1.6e6

Strip aspect alone does not decide: a bounded tip refined to 8 span panels per segment has
aspect 1613 and stays well conditioned, so the aspect limit is only a degenerate-mesh
catch-all.  The gate is the sweep angle and the tip chord ratio.

The same sweep under a Galerkin wall is roughly two decades better conditioned, so the
limits below are guidance for the collocation closure, not a statement about the geometry
alone.  A cusp tip (chord -> 0 with d(chord)/dy -> infinity) is a meshing artifact: real
wings have a finite tip chord, and the taper angle it produces is what the limits describe.

This module only reports.  It never silently changes a mesh: the tip has to be cut and
capped in the section definition, which is a geometry decision the user owns.
"""

from __future__ import annotations

import math

import numpy as np

# Limits measured on the table above; see the module docstring for the evidence.
# Warnings are capped per kind so a cusp tip does not print one line per strip.
WARNINGS_PER_KIND = 3
SWEEP_LIMIT_DEG = 60.0
STRIP_ASPECT_LIMIT = 4000.0
TIP_CHORD_RATIO_LIMIT = 0.03


def _te_stations(wing):
    """Trailing-edge points, one per span station, and their spanwise locations."""
    y = np.asarray(wing.te_pair_y, dtype=float)
    geo = np.asarray(wing.te_geo, dtype=int)
    if len(geo) == len(y) + 2:
        geo = geo[1:-1]
    elif len(geo) != len(y):
        raise ValueError("te_geo has %d entries for %d TE stations" % (len(geo), len(y)))
    return np.asarray(wing.xyz, dtype=float)[geo], y


def te_strip_diagnostics(wing, *, sweep_limit=SWEEP_LIMIT_DEG,
                         aspect_limit=STRIP_ASPECT_LIMIT,
                         tip_ratio_limit=TIP_CHORD_RATIO_LIMIT):
    """Geometry of every trailing-edge strip, plus the warnings a Kutta closure cares about.

    Returns a dict with per-strip arrays (width, sweep_deg, chord, aspect), the maxima and
    minima that drive the limits, a list of warnings and an "ok" flag.  A warning carries
    the station index, the measured value and a message naming the limit, so a caller can
    either print it or refuse to solve.
    """
    points, y = _te_stations(wing)
    if len(y) < 2:
        raise ValueError("a wing needs at least two TE stations")
    width = np.abs(np.diff(y))
    if np.any(width <= 0.0):
        raise ValueError("TE stations must strictly increase in y")
    delta = np.diff(points, axis=0)
    sweep = np.degrees(np.arctan2(np.abs(delta[:, 0]), np.abs(delta[:, 1])))
    # The LE of the GRASS and make_wing families sits at the origin plane, so |(x, z)| is
    # the chord of that station.
    chord = np.hypot(points[:, 0], points[:, 2])
    wake = np.asarray(wing.wake_xyz, dtype=float)
    wake_length = float(np.ptp(wake[:, 0])) if wake.size else 0.0
    aspect = wake_length / width

    warnings = []

    def warn(kind, station, value, limit, message):
        warnings.append(dict(kind=kind, station=int(station), value=float(value),
                             limit=float(limit), message=message))

    for k in np.argsort(sweep)[::-1]:
        if sweep[k] <= sweep_limit:
            break
        warn("te_sweep", k, sweep[k], sweep_limit,
             "TE strip %d (y=%.4f) is swept %.1f deg to the stream (limit %.0f): the two "
             "trailing-edge rows there are nearly parallel, so the potential-jump closure "
             "cannot separate the circulations" % (k, y[k], sweep[k], sweep_limit))
    for k in np.argsort(aspect)[::-1]:
        if aspect[k] <= aspect_limit:
            break
        warn("strip_aspect", k, aspect[k], aspect_limit,
             "wake strip %d is %.4f wide but %.2f long (aspect %.0f, limit %.0f): refine "
             "the span, not the chord, or cut the tip earlier"
             % (k, width[k], wake_length, aspect[k], aspect_limit))
    # Both tips are interior stations of the full-span mesh (the tip caps are excluded), so
    # compare the narrower one with the root chord, which is the largest on a tapered wing.
    root_chord = float(chord.max())
    tip_ratio = float(chord.min() / root_chord) if root_chord > 0.0 else float("inf")
    if tip_ratio < tip_ratio_limit:
        warn("cusp_tip", len(chord) - 1, tip_ratio, tip_ratio_limit,
             "tip chord is %.3f = %.1f%% of the root chord (limit %.1f%%): a cusp tip is a "
             "meshing artifact, cut the tip and give it a finite chord"
             % (chord.min(), 100.0 * tip_ratio, 100.0 * tip_ratio_limit))

    counts = {}
    for warning in warnings:
        counts[warning["kind"]] = counts.get(warning["kind"], 0) + 1
    kept = []
    seen = {}
    for warning in warnings:
        kind = warning["kind"]
        seen[kind] = seen.get(kind, 0) + 1
        if seen[kind] <= WARNINGS_PER_KIND:
            warning = dict(warning, total=counts[kind])
            kept.append(warning)

    return dict(
        y=np.asarray(y, dtype=float),
        width=width,
        sweep_deg=sweep,
        chord=chord,
        aspect=aspect,
        wake_length=wake_length,
        max_sweep_deg=float(sweep.max()),
        min_strip_width=float(width.min()),
        max_strip_aspect=float(aspect.max()),
        tip_chord_ratio=tip_ratio,
        sweep_limit_deg=float(sweep_limit),
        strip_aspect_limit=float(aspect_limit),
        tip_chord_ratio_limit=float(tip_ratio_limit),
        warnings=kept,
        warning_counts=counts,
        ok=not warnings,
    )


def bounded_elliptic_sections(sections=9, *, span=8.0, root_chord=2.0,
                              tip_fraction=0.85, tip_chord=None, sweep_fraction=0.97,
                              thickness=0.12):
    """Elliptic planform with a finite tip chord, ready for build_grass_mesh.

    tip_fraction is where the pure ellipse stops and the linear taper to tip_chord starts.
    The default (0.85, tip chord 0.24 for a 2.0 root chord) gives a 53.8 deg TE sweep,
    which stays below the audit sweep limit; a pure ellipse cut at 0.97 gives 74 deg and
    warns.  thickness sets the CST half-thickness (0.12 is about 12% t/c).
    """
    if not 0.2 <= tip_fraction <= 0.98:
        raise ValueError("tip_fraction must be between 0.2 and 0.98")
    count = int(sections)
    if count < 3:
        raise ValueError("an elliptic planform needs at least three sections")
    half = 0.5 * float(span)
    theta = np.linspace(0.0, 0.5 * math.pi * float(sweep_fraction), count)
    y = half * np.sin(theta)
    chord = float(root_chord) * np.cos(theta)
    y0 = half * float(tip_fraction)
    c0 = float(root_chord) * math.sqrt(max(0.0, 1.0 - float(tip_fraction) ** 2))
    if tip_chord is None:
        tip_chord = 0.12 * float(root_chord)
    tip_chord = float(tip_chord)
    ramp = y > y0
    if np.any(ramp):
        chord[ramp] = c0 + (tip_chord - c0) * (y[ramp] - y0) / (y[-1] - y0)
    half_thickness = 0.5 * float(thickness)
    cst = np.zeros((count, 18))
    cst[:, :8] = half_thickness
    cst[:, 8:16] = -half_thickness
    return dict(leading_edge_xyz=np.c_[np.zeros(count), y, np.zeros(count)],
                chord=chord, twist_radians=np.zeros(count), cst=cst)
