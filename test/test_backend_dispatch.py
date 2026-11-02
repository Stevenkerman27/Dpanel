"""Backend dispatch on the public G1/G2 entry points (issue #92, slice S6, revised in S9).

Route B option A (user's decision): torch is the production default, the NumPy/SciPy
chain stays reachable as the explicit evidence generator. The two paths run the same
algorithm with different roundoff, so decisions must match and state/loads are compared
with a declared bound.
"""
import numpy as np
import pytest

import panel.p1_backend as backend_module
import panel.p1_lifting_g1 as g1
import panel.p1_lifting_g2 as g2

pytest.importorskip("torch")

THRESHOLDS = dict(loads_relative=1e-9, cp_absolute=1e-10)


def _wing():
    return g1.make_wing(chord_panels=3, span_panels=4, thickness=.12, camber=.0,
                        span=2., chord=1., wake_panels=2)


def _mesh():
    wing = _wing()
    return g2.combine_wings([wing], origins=[(0., 0., 0.)], names=["main_wing"])


def test_unknown_backend_is_rejected():
    wing = _wing()
    with pytest.raises(ValueError, match="backend must be"):
        g1.solve(wing, 3., kutta="signed_speed", backend="quantum")
    with pytest.raises(ValueError, match="backend must be"):
        g2.solve_multi(_mesh(), 3., kutta="signed_speed", backend="quantum")


def test_torch_backend_refuses_the_scipy_reference_solver():
    wing = _wing()
    with pytest.raises(ValueError, match="structured"):
        g1.solve(wing, 3., kutta="pressure", pressure_solver="reference",
                 backend="torch")
    with pytest.raises(ValueError, match="structured"):
        g2.solve_multi(_mesh(), 3., kutta="pressure", pressure_solver="reference",
                       backend="torch")


def test_resolve_backend_reports_where_the_choice_came_from(monkeypatch):
    assert backend_module.resolve_backend("numpy") == ("numpy", "explicit")
    assert backend_module.resolve_backend("torch") == ("torch", "explicit")
    monkeypatch.setenv("PANEL_P1_BACKEND", "numpy")
    assert backend_module.resolve_backend() == ("numpy", "environment")
    monkeypatch.delenv("PANEL_P1_BACKEND")
    assert backend_module.resolve_backend() == ("torch", "default")


def test_resolve_backend_falls_back_only_when_torch_is_missing(monkeypatch):
    monkeypatch.delenv("PANEL_P1_BACKEND", raising=False)
    monkeypatch.setattr(backend_module, "torch_available", lambda: False)
    assert backend_module.resolve_backend() == ("numpy", "default")
    with pytest.raises(ValueError, match="not importable"):
        backend_module.resolve_backend("torch")


def test_single_wing_backend_parity_signed_kutta():
    wing = _wing()
    base = g1.solve(wing, 4., kutta="signed_speed", backend="numpy")
    other = g1.solve(wing, 4., kutta="signed_speed", backend="torch")
    for key in ("CL", "CL_gamma", "CD_pressure", "CM_y"):
        assert abs(other[key] - base[key]) <= THRESHOLDS["loads_relative"] * max(
            abs(base[key]), 1e-12), key
    assert other["constrained_rank"] == base["constrained_rank"]
    assert np.max(np.abs(np.asarray(other["cp"]) - base["cp"])) <= THRESHOLDS["cp_absolute"]
    assert other["backend"] == "torch" and base["backend"] == "numpy"


def test_single_wing_backend_parity_pressure_kutta():
    wing = _wing()
    base = g1.solve(wing, 4., kutta="pressure", pressure_solver="structured",
                    backend="numpy")
    other = g1.solve(wing, 4., kutta="pressure", pressure_solver="structured",
                     backend="torch")
    for key in ("pressure_status", "pressure_nfev", "pressure_message"):
        assert other[key] == base[key], key
    assert other["pressure_success"] == base["pressure_success"]
    for key in ("CL", "CL_gamma", "CD_pressure", "CM_y"):
        assert abs(other[key] - base[key]) <= THRESHOLDS["loads_relative"] * max(
            abs(base[key]), 1e-12), key
    assert other["nonlinear"] is not None
    assert other["nonlinear"]["status"] == base["pressure_status"]


def test_multi_backend_parity_pressure_kutta():
    mesh = _mesh()
    base = g2.solve_multi(mesh, 4., kutta="pressure", pressure_solver="structured",
                          backend="numpy")
    other = g2.solve_multi(mesh, 4., kutta="pressure", pressure_solver="structured",
                           backend="torch")
    for key in ("pressure_status", "pressure_nfev", "pressure_message"):
        assert other[key] == base[key], key
    assert other["linear_constrained_rank"] == base["linear_constrained_rank"]
    for key in ("CL", "CL_gamma", "CD_pressure", "CM_y"):
        assert abs(other[key] - base[key]) <= THRESHOLDS["loads_relative"] * max(
            abs(base[key]), 1e-12), key
    assert np.max(np.abs(np.asarray(other["mu"]) - base["mu"])) <= 1e-9


def test_the_default_backend_is_torch_and_says_so(monkeypatch):
    monkeypatch.delenv("PANEL_P1_BACKEND", raising=False)
    out = g1.solve(_wing(), 4., kutta="signed_speed")
    assert out["backend"] == "torch" and out["backend_source"] == "default"
    explicit = g1.solve(_wing(), 4., kutta="signed_speed", backend="numpy")
    assert explicit["backend"] == "numpy" and explicit["backend_source"] == "explicit"
