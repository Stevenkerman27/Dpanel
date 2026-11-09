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

The section slope is 2*pi by default (thin airfoil). Finite-thickness comparisons
need an independently measured section slope; a thickness correction must not be
fitted to the three-dimensional answer. Rectangular-wing circulation also tends
to zero at the tips, although its gradients need more terms to resolve.
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
    ``aspect_ratio`` and ``gamma = Gamma/U_inf`` over the returned ``y``.
    Sample inputs retain their original y; callable inputs return cosine stations
    plus both tips. ``modes`` and ``coefficients`` describe the sine expansion at
    alpha_deg, using the full-span convention Gamma = 2*b*U_inf*sum(A_n sin(n theta)).
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
        # Integrate the caller's chord on a fine grid, including the tips.
        grid = np.linspace(-semispan, semispan, 4001)
        area = float(np.trapezoid(np.asarray(chord_of_y(grid), dtype=float), grid))
        def chord_at(stations):
            return np.asarray(chord_of_y(stations), dtype=float)

    if not isinstance(terms, (int, np.integer)) or isinstance(terms, bool) or terms < 1:
        raise ValueError("terms must be a positive integer")
    if not np.isfinite(section_slope) or section_slope <= 0.0:
        raise ValueError("section_slope must be positive and finite")
    # Mirror rows are identical for odd modes. N whole-span symmetric stations
    # therefore supply only ceil(N/2) independent equations for N unknowns.
    stations = 2 * terms + 1
    theta = (np.arange(1, stations + 1) - 0.5) * np.pi / stations
    sine_theta = np.sin(theta)

    station_y = -semispan * np.cos(theta)
    station_c = chord_at(station_y)
    if not np.isfinite(station_c).all() or np.any(station_c <= 0.0):
        raise ValueError("chord vanishes inside the cosine stations")
    if not np.allclose(station_c, station_c[::-1], rtol=1e-8, atol=1e-12):
        raise ValueError("odd modes require chord symmetric about zero")
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
    output_y = y.copy() if y is not None else np.r_[-semispan, station_y, semispan]
    output_theta = np.arccos(np.clip(-output_y / semispan, -1.0, 1.0))
    result["y"] = output_y
    result["modes"] = n
    result["coefficients"] = unit * alpha
    result["gamma"] = 2.0 * span * (np.sin(np.outer(output_theta, n)) @ unit) * alpha
    result["gamma"][[0, -1]] = 0.0
    return result
