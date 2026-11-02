"""Backend dispatch after the NumPy chain was retired (issue #92, slices S6, S9, S10).

S6 wired backend='numpy'|'torch' into the public G1/G2 entry points so the two
implementations could be compared. S9 made torch the default and kept the NumPy path as
the explicit evidence generator. S10 retires that path: its code is tag
numpy-path-frozen-s8 and its recorded answers are docs/issue92/references/, replayed by
test/test_frozen_bundle_parity.py.

What is left to test here is the contract of the surviving path: the retired names must
fail closed with a message that says where the answers live, the resolution must report
where the choice came from, and a default call must reach torch.
"""
import numpy as np
import pytest

import panel.p1_backend as backend_module
import panel.p1_lifting_g1 as g1
import panel.p1_lifting_g2 as g2

pytest.importorskip("torch")


def _wing():
    return g1.make_wing(chord_panels=3, span_panels=4, thickness=.12, camber=.0,
                        span=2., chord=1., wake_panels=2)


def _mesh():
    wing = _wing()
    return g2.combine_wings([wing], origins=[(0., 0., 0.)], names=["main_wing"])


def test_the_numpy_backend_is_retired():
    message = "was retired with the NumPy solve chain"
    with pytest.raises(ValueError, match=message):
        backend_module.resolve_backend("numpy")
    with pytest.raises(ValueError, match=message):
        g1.solve(_wing(), 3., kutta="signed_speed", backend="numpy")
    with pytest.raises(ValueError, match=message):
        g2.solve_multi(_mesh(), 3., kutta="signed_speed", backend="numpy")


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="backend must be"):
        g1.solve(_wing(), 3., kutta="signed_speed", backend="quantum")
    with pytest.raises(ValueError, match="backend must be"):
        g2.solve_multi(_mesh(), 3., kutta="signed_speed", backend="quantum")


def test_the_scipy_reference_solver_is_retired():
    message = "was retired with the NumPy chain"
    with pytest.raises(ValueError, match=message):
        g1.solve(_wing(), 3., kutta="pressure", pressure_solver="reference")
    with pytest.raises(ValueError, match=message):
        g2.solve_multi(_mesh(), 3., kutta="pressure", pressure_solver="reference")


def test_resolve_backend_reports_where_the_choice_came_from(monkeypatch):
    monkeypatch.delenv("PANEL_P1_BACKEND", raising=False)
    assert backend_module.resolve_backend("torch") == ("torch", "explicit")
    assert backend_module.resolve_backend() == ("torch", "default")
    monkeypatch.setenv("PANEL_P1_BACKEND", "torch")
    assert backend_module.resolve_backend() == ("torch", "environment")


def test_a_program_without_torch_is_told_where_the_answers_are(monkeypatch):
    monkeypatch.delenv("PANEL_P1_BACKEND", raising=False)
    monkeypatch.setattr(backend_module, "torch_available", lambda: False)
    with pytest.raises(ValueError, match="requires torch"):
        backend_module.resolve_backend()
    with pytest.raises(ValueError, match="not importable"):
        backend_module.resolve_backend("torch")


def test_the_default_call_reaches_torch_and_says_so(monkeypatch):
    monkeypatch.delenv("PANEL_P1_BACKEND", raising=False)
    out = g1.solve(_wing(), 4., kutta="signed_speed")
    assert out["backend"] == "torch" and out["backend_source"] == "default"
    for key in ("CL", "CL_gamma", "CD_pressure", "CM_y"):
        assert np.isfinite(out[key]), key
    assert out["constrained_rank"] > 0


def test_the_pressure_branch_also_reaches_torch():
    out = g1.solve(_wing(), 4., kutta="pressure", pressure_solver="structured")
    assert out["backend"] == "torch"
    assert out["pressure_status"] is not None
    assert out["nonlinear"] is not None
    multi = g2.solve_multi(_mesh(), 4., kutta="pressure", pressure_solver="structured")
    assert multi["backend"] == "torch"
    assert np.isfinite(multi["CL"]) and multi["linear_constrained_rank"] > 0
