"""Read-only Kutta-Joukowski lift diagnostic.

The physical lift from a horizontal trailing edge uses spanwise dy, whereas
weak Kutta mass matrices legitimately integrate along actual TE arc length ds.
"""
from __future__ import annotations
import numpy as np


def circulation_lift_measures(station_y, station_hat, gamma, te_edges, te_basis,
                              *, density, speed, reference_area):
    """Return physical spanwise, exposed-edge and legacy arclength measures.

    station_hat[k,j] maps DOFs to stations; te_basis[e,2,j] is the
    Gauss basis at two ordered actual trailing-edge endpoints te_edges[e,2,3].
    For a buried/contact root, station and exposed-edge integrals may differ.
    """
    y = np.asarray(station_y, dtype=float)
    H = np.asarray(station_hat, dtype=float)
    g = np.asarray(gamma, dtype=float)
    e = np.asarray(te_edges, dtype=float)
    B = np.asarray(te_basis, dtype=float)
    flow = np.asarray((density, speed, reference_area), dtype=float)
    if (y.ndim != 1 or len(y) < 2 or H.ndim != 2
            or H.shape[0] != len(y) or H.shape[1] == 0
            or g.shape != (H.shape[1],) or e.ndim != 3
            or e.shape[1:] != (2,3) or len(e) == 0
            or B.shape != (len(e),2,len(g))
            or not all(np.isfinite(a).all() for a in (y,H,g,e,B,flow))
            or np.any(flow <= 0) or np.any(np.diff(y) <= 0)):
        raise ValueError("Invalid spanwise circulation integration inputs")
    dy = e[:,1,1]-e[:,0,1]
    if np.any(dy <= 0):
        raise ValueError("Trailing-edge edges must be ordered outboard in y")
    ds = np.linalg.norm(e[:,1]-e[:,0], axis=1)
    # Station basis integrates exactly for P1; tip DOF=-1 stays zero.
    span = np.einsum("k,kj->j",.5*np.diff(y),H[:-1]+H[1:])
    exposed = np.einsum("e,eqj->j",.5*dy,B)
    legacy = np.einsum("e,eqj->j",.5*ds,B)
    coeff = 4.0/(speed*reference_area)
    true_cl = float(coeff*(span@g))
    exposed_cl = float(coeff*(exposed@g))
    legacy_cl = float(coeff*(legacy@g))
    return dict(
        CL_gamma=true_cl, CL_gamma_exposed_TE_span=exposed_cl,
        CL_gamma_legacy_TE_arclength=legacy_cl,
        CL_legacy_minus_span=legacy_cl-true_cl,
        CL_exposed_minus_span=exposed_cl-true_cl,
        span_measure=span.tolist(),
        exposed_te_span_measure=exposed.tolist(),
        legacy_te_arclength_measure=legacy.tolist(),
        max_abs_span_coverage_gap=float(np.max(np.abs(span-exposed))),
        span_projection="true_dy_not_trailing_edge_ds")
