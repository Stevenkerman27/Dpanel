"""The public wake-velocity surface must be three methods, not a matrix of backends."""
from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from panel import wake_velocity_methods as methods
from panel.p1_coupled_material_wake import TotalVelocity

ROOT = Path(__file__).resolve().parents[1]


def _tiny_mesh_state():
    """One wall triangle and one wake triangle, both unit-scale."""
    wall_xyz = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    wake_xyz = np.array([[1., 0., 0.], [2., 0., 0.], [1., 1., 0.]])
    xyz = np.vstack([wall_xyz, wake_xyz])
    mesh = SimpleNamespace(xyz=xyz, faces=np.array([[0, 1, 2]]),
                           dof_face=np.array([[0, 1, 2]]), groups=[
                               SimpleNamespace(name="wake", wake_xyz=wake_xyz,
                                               wake_faces=np.array([[0, 1, 2]]),
                                               gamma_hat=np.ones((3, 1)))])
    state = dict(mu=np.array([0.1, 0.2, 0.3]), gamma=np.array([0.2]), alpha_deg=5.)
    return mesh, state


def test_registry_has_exactly_three_models():
    assert methods.PUBLIC_METHODS == ("analytic_edges", "rm_regularized_p1")
    assert methods.DOCUMENTED_ONLY_METHODS == ("lagrangian_filaments",)
    assert methods.ACCEPTED_METHODS == methods.PUBLIC_METHODS + methods.DEPRECATED_METHODS
    indices = [methods.METHOD_SPECS[m]["index"] for m in
               methods.PUBLIC_METHODS + methods.DOCUMENTED_ONLY_METHODS]
    assert sorted(indices) == [1, 2, 3]
    assert methods.DEFAULT_METHOD == "analytic_edges"


def test_quadrature_backends_are_not_velocity_methods():
    for backend in methods.AREA_QUADRATURES:
        assert backend not in methods.ACCEPTED_METHODS
        assert backend not in methods.METHOD_SPECS
    for mode in ("projection_fan", "projection_fan_adaptive", "legacy_duffy"):
        with pytest.raises(ValueError) as err:
            methods.describe(mode)
        assert "quadrature settings belong to area_quadrature" in str(err.value)
        assert methods.ARCHITECTURE_DOC in str(err.value)


def test_documented_only_method_is_refused_with_its_reason():
    reason = methods.wiring_error("lagrangian_filaments")
    assert "not coupled to B2" in reason and methods.ARCHITECTURE_DOC in reason
    assert methods.wiring_error("legacy_gauss").startswith("legacy_gauss is a deprecated")
    assert methods.wiring_error("analytic_edges") is None


def test_total_velocity_accepts_the_public_surface_only():
    mesh, state = _tiny_mesh_state()
    field = TotalVelocity(mesh, state)
    assert field.method == "analytic_edges"
    assert field.method_spec["index"] == 1
    regularized = TotalVelocity(mesh, state, method="rm_regularized_p1", core_radius=0.05)
    assert regularized.method_spec["index"] == 2
    for bad in ("projection_fan", "projection_fan_adaptive", "lagrangian_filaments"):
        with pytest.raises(ValueError) as err:
            TotalVelocity(mesh, state, method=bad)
        assert "velocity_method must be one of" in str(err.value)
        assert "analytic_edges, rm_regularized_p1, legacy_gauss" in str(err.value)


def test_area_quadrature_fails_closed_under_method_one():
    mesh, state = _tiny_mesh_state()
    with pytest.raises(ValueError) as err:
        TotalVelocity(mesh, state, area_quadrature="projection_fan")
    assert "requires method='rm_regularized_p1'" in str(err.value)
    with pytest.raises(ValueError) as err:
        TotalVelocity(mesh, state, area_quadrature="projection_fan_adaptive")
    assert "requires method='rm_regularized_p1'" in str(err.value)
    with pytest.raises(ValueError) as err:
        TotalVelocity(mesh, state, method="rm_regularized_p1", core_radius=0.05,
                      area_quadrature="not_a_backend")
    assert "area_quadrature must be" in str(err.value)


def test_legacy_gauss_stays_replayable_but_is_not_public():
    mesh, state = _tiny_mesh_state()
    field = TotalVelocity(mesh, state, method="legacy_gauss")
    assert field.method == "legacy_gauss"
    assert field.method_spec["status"].startswith("deprecated")
    assert not methods.is_public("legacy_gauss")
    assert methods.is_selectable("legacy_gauss") is True


def test_cli_describes_the_three_models_without_running_a_case(tmp_path=None):
    out = subprocess.run([sys.executable, "-m", "panel.run_p1_coupled_material_wake",
                          "--describe-velocity-methods"], cwd=ROOT,
                         capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr
    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    assert lines[0].startswith("method")
    listed = [ln.split()[0] for ln in lines[1:] if not ln.startswith("Methods 1 and 2")]
    assert listed[:4] == ["legacy_gauss", "analytic_edges", "rm_regularized_p1",
                          "lagrangian_filaments"]
    assert "quadrature order/strips are numerical controls" in out.stdout.lower()


def test_cli_rejects_a_quadrature_backend_as_a_method(tmp_path=None):
    from panel import run_p1_coupled_material_wake as cli
    # argparse rejects a backend name or the unwired model before any case runs,
    # so no evidence directory is created.
    for bad in ("projection_fan", "projection_fan_adaptive", "lagrangian_filaments"):
        with pytest.raises(SystemExit) as err:
            cli.main(["--velocity-method", bad])
        assert err.value.code == 2
    with pytest.raises(SystemExit) as err:
        cli.main(["--out"])
    assert err.value.code == 2
    with pytest.raises(SystemExit) as err:
        cli.main([])
    assert err.value.code == 2


def test_doc_page_covers_the_required_distinctions():
    text = (ROOT / methods.ARCHITECTURE_DOC).read_text(encoding="utf-8")
    for needle in ("representation", "evaluator", "update",
                   "Method 1", "Method 2", "Method 3",
                   "Rosenhead-Moore", "Biot-Savart",
                   "fan_order", "fan_strips", "wake node count",
                   "wake iteration count", "physical_accuracy_accepted",
                   "not coupled"):
        assert needle.lower() in text.lower(), needle
    assert "projection_fan" in text and "area_quadrature" in text
