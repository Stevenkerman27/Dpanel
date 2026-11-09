"""Prandtl lifting-line reference for a finite-span, uncambered wing.

Classical monoplane equation in Glauert's sine series, used as an independent
reference for the surface-panel solver:

    sum_n A_n sin(n theta) [ 4 b / (a0 c(theta)) + n / sin(theta) ] = alpha,

with y = -(b/2) cos(theta), b the full span, a0 the two-dimensional section
lift-curve slope and c(theta) the chord. Then

    CL = pi AR A1,  CDi = pi AR sum_n n A_n^2,  e = A1^2 / sum_n n A_n^2.

Only odd terms are kept: exact for a wing with no twist, no camber and a chord
distribution symmetric in y. Everything compared against this module is of that
kind.

Pass the chord either as samples over the input span (then it is interpolated
onto the cosine stations) or as a callable of y. Samples are fine — a sampled
elliptic wing with 801 points returns e = 1.0000 — but the area integral uses
the samples, so give a grid fine enough near the tips to represent the chord.

The section slope is 2*pi by default (thin airfoil). A 12% thick section has a
slightly higher slope, so a panel wing is expected to sit a few percent above
this reference; that bias is stated where the reference is used, not hidden by
tuning a0.

A rectangular planform with a sharp tip is a poor reference case: the loading
is discontinuous at the side edge and the sine series converges slowly, so e
keeps drifting with the number of terms. Prefer planforms with a finite tip
chord, which is also what the solver's bounded-tip meshes have.
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def _odd_terms(count: int) -> np.ndarray:
    return 2.0 * np.arange(1, count + 1) - 1.0


def lifting_line(y=None, chord=None, *, chord_of_y=None, span=None, alpha_deg=0.0,
                 section_slope=TWO_PI, terms=63):
    """Return the lifting-line state of a planform symmetric in y.

    Give either ``y`` (strictly increasing, whole span) plus ``chord`` samples of
    the same length, or just ``chord_of_y``. Returns ``cl_alpha`` (per radian),
    ``cl`` at ``alpha_deg``, ``cdi``, ``e``, the planform ``span``/``area``/
    ``aspect_ratio`` and the circulation ``gamma`` over ``y`` (when given).
    """
    if chord_of_y is None:
        if y is None or chord is None:
            raise ValueError("give y and chord, or chord_of_y")
        y = np.asarray(y, dtype=float)
        chord = np.asarray(chord, dtype=float)
        if y.ndim != 1 or y.shape != chord.shape or y.size < 2:
            raise ValueError("y and chord must be equal-length 1-D arrays")
        if np.any(np.diff(y) <= 0.0):
            raise ValueError("y must be strictly increasing")
        if np.any(chord < 0.0) or not np.any(chord > 0.0):
            raise ValueError("chord must be non-negative and not all zero")
        semispan = 0.5 * (y[-1] - y[0])
        span = 2.0 * semispan
        area = float(np.trapezoid(chord, y))
        if abs(y[0] + semispan) > 1e-9 * max(span, 1.0):
            raise ValueError("y must be symmetric about zero")
        def chord_at(stations):
            return np.interp(stations, y, chord)
    else:
        if y is not None or chord is not None:
            raise ValueError("give either samples or chord_of_y, not both")
        if span is None or not np.isfinite(span) or span <= 0.0:
            raise ValueError("chord_of_y needs a positive span")
        span = float(span)
        semispan = 0.5 * span
        # Midpoint rule over a fine grid; the caller passes a smooth chord function.
        grid = np.linspace(-semispan, semispan, 4001)
        area = float(np.trapezoid(np.asarray(chord_of_y(grid), dtype=float), grid))
        def chord_at(stations):
            return np.asarray(chord_of_y(stations), dtype=float)

    if terms < 1:
        raise ValueError("terms must be at least 1")
    # Collocate on twice as many stations as harmonics and solve in the least
    # squares sense: the square system is ill-conditioned (rows near the tip carry
    # 1/sin(theta)), and a squared solve then returns a spurious harmonic at some
    # term counts. Overdetermined collocation keeps e stable (measured: the square
    # solve gave e 0.999 / 0.999 / 0.829 / 0.999 at 31 / 63 / 127 / 255 terms).
    stations = 2 * terms + 1
    theta = (np.arange(1, stations + 1) - 0.5) * np.pi / stations
    sine_theta = np.sin(theta)

    station_y = -semispan * np.cos(theta)
    station_c = chord_at(station_y)
    if np.any(station_c <= 0.0):
        raise ValueError("chord vanishes inside the cosine stations")
    n = _odd_terms(terms)
    sine = np.sin(np.outer(theta, n))
    matrix = sine * (4.0 * span / (section_slope * station_c)[:, None]
                     + n[None, :] / sine_theta[:, None])
    unit, *_ = np.linalg.lstsq(matrix, np.ones(stations), rcond=None)
    weighted = float(np.sum(n * unit * unit))
    alpha = np.deg2rad(alpha_deg)
    result = {
        "span": span,
        "area": area,
        "aspect_ratio": span * span / area,
        "cl_alpha": float(np.pi * (span * span / area) * unit[0]),
        "cl": float(np.pi * (span * span / area) * unit[0] * alpha),
        "cdi": float(np.pi * (span * span / area) * weighted * alpha * alpha),
        "e": float(unit[0] ** 2 / weighted) if weighted > 0.0 else 0.0,
        "terms": terms,
    }
    result["gamma"] = 4.0 * span * (sine @ unit) * alpha
    return result