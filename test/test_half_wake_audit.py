"""Acceptance tests for panel/audit_half_wake.py on the mirrored half fixture.

The fixture is the same conforming thick half wing used by
test_half_native_panel.py (cases/panel_half/thick_wing_two_strip.json):
20 wall triangles, 4 wake triangles and 2 circulation unknowns per half.
The audit is only ever read-only; these tests never touch production modules.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import jax

jax.config.update("jax_enable_x64", True)

# core/triangle_kernel.py is the audited numba field kernel; panel/audit.py has
# the same requirement, so an environment without numba cannot run either audit.
pytest.importorskip("numba", reason="half-wake audit needs the numba kernel")

from panel.audit_half_wake import (  # noqa: E402
    audit, audit_wake, build_mirror_state, check_mirror_equivariance,
    check_potential_gradient, check_potential_kernel, check_wall_tangency,
    evaluate_field, main, probe_points, wake_samples,
)

SURFACE = (Path(__file__).resolve().parents[1] / "cases" / "panel_half"
           / "thick_wing_two_strip.json")


@pytest.fixture(scope="module")
def context():
    return build_mirror_state(SURFACE)


@pytest.fixture(scope="module")
def report(context):
    return audit(SURFACE, offsets=(1e-2, 1e-3))


def test_fixture_is_the_expected_small_half_wing(context):
    assert context["half_panels"] == 20
    assert context["half_wake"] == 4
    assert context["half_gamma"] == 2
    assert len(context["triangles"]) == 40
    assert len(context["wake"]) == 8
    assert len(context["gamma"]) == 3
    assert np.all(context["wake_centers"][:4, 1] > 0)
    assert np.all(context["wake_centers"][4:, 1] < 0)


def test_mirror_expansion_reproduces_the_full_oracle(context):
    check = context["mirror_check"]
    for name in ("expanded_state_vs_full", "half_mu_vs_full_positive",
                 "half_gamma_vs_full_positive",
                 "half_wake_doublet_vs_full_positive",
                 "half_wake_slope_vs_full_positive"):
        assert check[name]["relative_to_peak"] < 1e-12, (name, check[name])


def test_wall_dirichlet_is_exact_and_normal_residual_is_discrete(context):
    # The Dirichlet condition is satisfied to round-off; the normal velocity is
    # not imposed by the Dirichlet solve, so it is only O(h) on this 20-panel
    # fixture (panel/audit.py reports the same quantity as a gate, not an
    # identity, and its 1e-6 threshold belongs to converged meshes).
    check = check_wall_tangency(context)
    assert check["interior_potential_rms"] < 1e-12, check
    assert check["normal_rms"] < 0.05, check
    assert check["normal_max"] < 0.25, check


def test_velocity_is_the_gradient_of_the_total_potential(context):
    check = check_potential_gradient(context, steps=(1e-3, 1e-4))
    assert check["probes"] >= 12
    for entry in check["steps"]:
        assert entry["relative_to_peak_speed"] < 1e-6, entry
    # Halving the step must not increase the first-order error by orders.
    assert check["steps"][1]["max_abs"] <= 3 * check["steps"][0]["max_abs"]


def test_numba_potential_matches_the_independent_jax_kernel(context):
    check = check_potential_kernel(context)
    assert check["relative_to_peak"] < 1e-12, check


def test_two_sided_identity_and_mirror_equivariance(report):
    offset = report["wake_report"]["by_offset"]
    for block in offset:
        assert block["all"]["identity_abs_max"] < 1e-12, block["all"]
    check = report["checks"]["mirror_equivariance"]
    assert check["pairs"] == 4
    assert check["flux_max_abs"] < 1e-12, check
    assert check["dcp_max_abs"] < 1e-12, check
    # Vmean.grad(Gamma) is a scalar invariant, so it is symmetric under the
    # mirror while Vy and dGamma/dy are separately antisymmetric.
    assert check["transport_symmetry_max_abs"] < 1e-12, check
    assert check["vy_antisymmetry_max_abs"] < 1e-12, check
    assert check["slope_sum_max_abs"] < 1e-12, check
    # The doublet strength is the potential jump, also a scalar invariant.
    assert check["mu_symmetry_max_abs"] < 1e-12, check


def test_image_contribution_is_not_negligible(report):
    check = report["checks"]["image_contribution"]
    assert check["image_rms_over_speed"] > 1e-3, check
    assert check["image_plus_rms_over_speed"] > 1e-3, check


def test_offsets_are_finite_and_area_fractions_sum_to_one(report):
    for block in report["wake_report"]["by_offset"]:
        near, far = block["near"], block["far"]
        assert np.isfinite([near["flux_rms"], far["flux_rms"],
                            near["dcp_rms"], far["dcp_rms"]]).all()
        assert abs(near["area_fraction"] + far["area_fraction"] - 1) < 1e-12
        assert near["samples"] + far["samples"] == block["all"]["samples"]


def test_positive_half_only_evaluation_differs(report):
    half = report["positive_half_only"]["near_far"]["all"]
    full = report["wake_report"]["by_offset"][0]["all"]
    assert half["samples"] == full["samples"] // 2
    assert half["flux_rms"] != full["flux_rms"]


def test_report_records_provenance_and_never_accepts_accuracy(report):
    assert report["schema"] == "dpanel.half_wake_audit.v1"
    assert report["physical_accuracy_accepted"] is False
    assert len(report["surface_sha256"]) == 64
    assert len(report["state_sha256"]) == 64
    assert len(report["code_sha256"]) == 64
    assert report["geometry"]["full_panels"] == 40
    json.dumps(report, allow_nan=False)
    assert report["checks"]["units"]["constraint"]


def test_cli_writes_a_new_report_and_refuses_to_overwrite():
    # pytest's tmp_path is unscannable under this sandbox (WinError 5), so the
    # CLI test owns a short-lived directory next to the worktree instead.
    import shutil
    base = Path.cwd() / f".tmp_wake_audit_{os.getpid()}"
    out = base / "wake" / "audit.json"
    assert main(["--surface", str(SURFACE), "--out", str(out),
                 "--offsets", "1e-2", "--steps", "1e-3"]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["schema"] == "dpanel.half_wake_audit.v1"
    assert written["wake_report"]["by_offset"][0]["offset_over_sqrt_area"] == 1e-2
    with pytest.raises(FileExistsError):
        main(["--surface", str(SURFACE), "--out", str(out)])
    shutil.rmtree(base, ignore_errors=True)


def test_oversized_half_mesh_is_rejected_before_the_dense_oracle():
    with pytest.raises(MemoryError):
        build_mirror_state(SURFACE, max_half_panels=4)


def test_streaming_blocks_do_not_change_the_field(context):
    targets = wake_samples(context)[:, 0, :]
    coarse, _ = evaluate_field(targets, context, chunk=7)
    fine, _ = evaluate_field(targets, context, chunk=1024)
    np.testing.assert_allclose(coarse, fine, rtol=0., atol=0.)
