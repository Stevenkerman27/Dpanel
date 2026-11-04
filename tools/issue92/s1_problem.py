"""Deterministic structural problem for the S1 pressure-Kutta torch parity.

The reduced pressure-Kutta solve (panel/p1_pressure_kutta_reduced.py) runs on
B (weighted wall operator), N (feasible-subspace basis), bw (weighted wall
target) and a few nonlinear pressure rows that depend on mu only. This module
builds a problem with exactly that structure so the NumPy and torch backends
can be compared without assembling a mesh:

* M = B N is well conditioned (QR is stable) but the wall block has a nonzero
  residual floor, because bw contains a component outside range(B N) - that is
  the regime the G2 acceptance logic is written for;
* the pressure residual is quadratic in mu, so the Gauss-Newton iteration needs
  several steps and a working line search.

Shapes are the elliptic fixture's, scaled down by 0.2: n_wall 600, n_unknown
300, n_free 200, n_pressure 24. Everything is drawn from a fixed seed, so both
processes build bit-identical inputs.

This is scaffolding, not a production path: it exists to check that the torch
linear algebra reproduces the reference decisions (status, nfev) and the
iterate, not to model a wing.
"""
from __future__ import annotations

import numpy as np

SEED = 20261102
N_WALL = 600
N_UNKNOWN = 300
N_FREE = 200
N_PRESSURE = 24
PRESSURE_QUADRATIC = 0.2
FLOOR_NOISE = 1e-2


SCENARIOS = ("base", "hard", "backtrack", "stiff")


def build(seed=SEED, scenario="base", quadratic=None):
    """Return the arrays of the parity problem as a dict of NumPy arrays.

    "base" starts at zero with a mild quadratic term. "hard" starts far from the
    solution with a strong quadratic term, so the Armijo line search has to
    backtrack and the stopping rule sees several rejected trials - that is the
    branch the reference solve spends most of its logic on.
    """
    if scenario not in SCENARIOS:
        raise ValueError("scenario must be one of %s" % (SCENARIOS,))
    rng = np.random.default_rng(seed)
    n_mu = N_UNKNOWN // 2
    B = rng.standard_normal((N_WALL, N_UNKNOWN)) / np.sqrt(N_WALL)
    raw = rng.standard_normal((N_UNKNOWN, N_FREE))
    N, _ = np.linalg.qr(raw)
    z_true = 0.5 * rng.standard_normal(N_FREE)
    bw = B @ N @ z_true + FLOOR_NOISE * rng.standard_normal(N_WALL)
    K = rng.standard_normal((N_PRESSURE, n_mu))
    target = rng.standard_normal(N_PRESSURE)
    if scenario == "hard":
        c2 = 5.0
        x0 = 2.0 * rng.standard_normal(N_FREE)
    elif scenario == "backtrack":
        # strong enough that the Armijo search has to halve the step many times
        # (27 backtracks measured) while the iteration still terminates cleanly
        c2 = 20.0
        x0 = 2.0 * rng.standard_normal(N_FREE)
    elif scenario == "stiff":
        # deliberately outside the production regime: 50x the quadratic term and
        # a line-search failure after ~31 iterations, so roundoff is amplified
        # into a visible iterate difference. Reported, not gated.
        c2 = 50.0
        x0 = 3.0 * rng.standard_normal(N_FREE)
    else:
        c2 = PRESSURE_QUADRATIC
        x0 = np.zeros(N_FREE)
    if quadratic is not None:
        c2 = float(quadratic)
    return dict(B=B, N=N, bw=bw, K=K, target=target, c2=c2,
                n_mu=n_mu, x0=x0, scenario=scenario)


def pressure_numpy(problem):
    """NumPy pressure residual for the problem returned by build()."""
    K, c2, target = problem["K"], problem["c2"], problem["target"]
    n_mu, n_pressure = problem["n_mu"], N_PRESSURE

    def pressure(v):
        mu = np.asarray(v, float)[:n_mu]
        return K @ mu + c2 * mu[:n_pressure] ** 2 - target

    def jacobian(v):
        mu = np.asarray(v, float)[:n_mu]
        block = np.array(K, copy=True)
        block[np.arange(n_pressure), np.arange(n_pressure)] += 2.0 * c2 * mu[:n_pressure]
        P = np.zeros((n_pressure, N_UNKNOWN))
        P[:, :n_mu] = block
        return P

    return pressure, jacobian
