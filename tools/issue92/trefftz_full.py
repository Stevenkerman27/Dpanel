"""Trefftz-plane induced drag for a full-span station layout.

The repository helper runs/trefftz_drag.py assumes half-wing stations (0..b/2)
and mirrors them internally. DPanel's Wing.gamma_y actually spans the whole
wing (-b/2..b/2), so feeding it to that helper double-covers the planform and
returns a drag that is wrong by an O(1) factor. This function takes the full
span layout directly and keeps the principal-value log quadrature.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HELPER = Path(__file__).resolve().parents[2] / "runs" / "trefftz_drag.py"


def _pv_log(y0, a, b):
    if a <= y0 <= b:
        return float(np.log(abs((y0 - a) / (b - y0))))
    return float(np.log(abs((y0 - a) / (y0 - b))))


def trefftz_cd_full(stations, gamma, *, area, rho=1.0, U=1.0):
    y = np.asarray(stations, float)
    g = np.asarray(gamma, float)
    if len(y) != len(g):
        raise ValueError("stations and gamma must have the same length")
    if len(y) < 3:
        raise ValueError("need at least three stations")
    if not np.all(np.diff(y) > 0):
        order = np.argsort(y)
        y, g = y[order], g[order]
    a, b = y[:-1], y[1:]
    dy = np.diff(y)
    dgdy = np.diff(g) / dy
    y_mid = 0.5 * (a + b)
    g_mid = 0.5 * (g[:-1] + g[1:])
    w = np.empty_like(y_mid)
    for i, y0 in enumerate(y_mid):
        logt = np.array([_pv_log(y0, aa, bb) for aa, bb in zip(a, b)])
        w[i] = -float(np.sum(dgdy * logt)) / (4.0 * np.pi)
    drag = -rho * float(np.sum(g_mid * w * dy))
    return drag / (0.5 * rho * U * U * float(area)), w


def elliptic_self_test(n=201, b=4.0, c=1.0, cl=0.6):
    """Full-span elliptic loading: CD must approach cl^2 / (pi AR)."""
    y = np.linspace(-b / 2.0, b / 2.0, n)
    g0 = 2.0 * cl * c / np.pi
    g = g0 * np.sqrt(np.clip(1.0 - (2.0 * y / b) ** 2, 0.0, None))
    cd, _ = trefftz_cd_full(y, g, area=b * c)
    return cd, cl * cl / (np.pi * (b / c))


if __name__ == "__main__":
    for n in (51, 101, 201, 401):
        cd, theory = elliptic_self_test(n)
        print("n=%3d CD=%.8f theory=%.8f ratio=%.5f" % (n, cd, theory, cd / theory))
