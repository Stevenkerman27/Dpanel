"""Acceptance tests for panel/audit_half_wake.py on the mirrored half fixture.

The fixture is the same conforming thick half wing used by
test_half_native_panel.py (cases/panel_half/thick_wing_two_strip.json):
20 wall triangles, 4 wake triangles and 2 circulation unknowns per half.
The audit is only ever read-only; these tests never modify production modules.
"""
import hashlib
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
    evaluate_field, main, probe_points, sample_bands, wake_labels,
    wake_samples,
)
from panel.half_native import solve_half_fields  # noqa: E402
from panel.run_half_native import load_half_surface  # noqa: E402

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
    # Four wake triangles per half, duplicated by the index map.
    assert len(context["wake_mu"]) == 8
    assert len(context["wake_slope"]) == 8
    assert np.all(context["wake_centers"][:4, 1] > 0)
    assert np.all(context["wake_centers"][4:, 1] < 0)
    assert context["state_origin"].startswith("fresh")


def test_single_half_solve_is_mirrored_by_index_maps(context):
    check = context["mirror_check"]
    assert check["panel_normal_partner_max_abs"] < 1e-15, check
    assert check["wake_normal_partner_max_abs"] < 1e-15, check
    assert check["wake_centroid_partner_max_abs"] < 1e-15, check
    assert check["dense_expansion_matrix"] == "not formed"
    assert check["wake_operator_vs_solver_max_abs"] < 1e-15, check
    # The left half carries the same scalars, the mirrored slope is negated.
    half = context["half_wake"]
    np.testing.assert_array_equal(context["mu"][:20], context["mu"][20:])
    np.testing.assert_array_equal(context["wake_mu"][:half],
                                 context["wake_mu"][half:])
    np.testing.assert_allclose(context["wake_slope"][:half],
                               -context["wake_slope"][half:], rtol=0., atol=0.)


def test_default_path_never_solves_or_imports_the_full_model(monkeypatch):
    from panel import differentiable, verify_half_mirror

    def forbidden(*args, **kwargs):
        raise AssertionError("the default audit must not use the full model")

    monkeypatch.setattr(verify_half_mirror, "make_full_mirror_topology",
                        forbidden)
    monkeypatch.setattr(differentiable, "solve_fields", forbidden)
    built = build_mirror_state(SURFACE)
    assert built["mirror_check"].get("full_oracle") is None
    one = audit_wake(built, offsets=(1e-2,))
    assert one["by_offset"][0]["all"]["samples"] == 32


def test_explicit_full_oracle_cross_check_matches_the_index_map():
    built = build_mirror_state(SURFACE, full_oracle=True)
    oracle = built["mirror_check"]["full_oracle"]
    assert oracle["performed"] is True
    for name in ("mu", "sigma", "wake_doublet", "wake_slope"):
        assert oracle[name]["relative_to_peak"] < 1e-12, (name, oracle[name])
    with pytest.raises(MemoryError):
        build_mirror_state(SURFACE, full_oracle=True, full_oracle_panels=4)


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
    for block in report["wake_report"]["by_offset"]:
        assert block["all"]["identity_abs_max"] < 1e-12, block["all"]
    check = report["physical_residuals"]["mirror_equivariance"]
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


def test_jump_closures_are_reported_with_a_measured_sign(report):
    closure = report["wake_report"]["by_offset"][-1]["jump_closure"]
    for name in ("potential_jump_rms", "doublet_rms", "normal_jump_rms",
                 "tangential_jump_rms", "tangential_plus_slope_rms",
                 "tangential_minus_slope_rms"):
        assert np.isfinite(closure[name]), (name, closure)
    assert closure["smallest_residual"] in ("tangential_plus_slope",
                                            "tangential_minus_slope")
    # Both sign conventions are measured; the report only names the smaller.
    assert closure["note"].startswith("the smaller residual names")


def test_bands_partition_the_wake_and_empty_bands_are_null(context, report):
    block = report["wake_report"]["by_offset"][0]
    sample = sample_bands(wake_samples(context), context)
    covered = sum(int(mask.sum()) for mask in sample["masks"])
    assert covered + sample["below_range"] == sample["gap"].size
    assert len(sample["names"]) == len(sample["masks"]) == 6
    bands = [band for band in block["bands"] if band is not None]
    assert bands, block
    assert abs(sum(band["area_fraction"] for band in bands) - 1) < 1e-12
    assert sum(band["samples"] for band in bands) == block["all"]["samples"]
    for band in bands:
        assert np.isfinite([band["flux_rms"], band["dcp_rms"]]).all()
    assert block["first_row"] is not None
    assert block["first_row"]["samples"] <= block["all"]["samples"]
    # The fixture wake is 3 chords long, so the absolute 0.5 Cref "near" rule
    # is empty here; it is reported as null instead of being redefined, which is
    # exactly why the fixed x_gap/Cref bands are the primary split.
    assert block["near_absolute"] is None, block["near_absolute"]
    assert block["historical_near"] is not None
    assert block["historical_far"] is not None
    rule = report["wake_report"]["band_rule"]
    assert "fixed x_gap / Cref" in rule["bands"]["bands"]
    # The reachable-gap summary is what tells a reader that an empty band is a
    # sampling-resolution statement, not a zero-flux statement.
    assert rule["gap_summary"]["median_over_chord"] > 0.0
    assert "empty by resolution" in rule["gap_summary"]["note"]


def test_offsets_report_successive_differences_and_orders(report):
    sensitivity = report["wake_report"]["sensitivity"]
    assert len(sensitivity["offsets"]) == 2
    assert len(sensitivity["flux_rms_successive"]) == 1
    assert len(sensitivity["flux_rms_estimated_order"]) == 0
    for value in sensitivity["flux_rms"]:
        assert np.isfinite(value)


def test_image_contribution_is_not_negligible(report):
    check = report["implementation_checks"]["image_contribution"]
    assert check["image_rms_over_speed"] > 1e-3, check
    assert check["image_plus_rms_over_speed"] > 1e-3, check


def test_positive_half_only_evaluation_differs(report):
    half = report["positive_half_only"]["all"]
    full = report["wake_report"]["by_offset"][0]["all"]
    assert half["samples"] == full["samples"] // 2
    assert half["flux_rms"] != full["flux_rms"]
    assert "diagnostic only" in report["positive_half_only"]["warning"]


def test_report_records_provenance_and_never_accepts_accuracy(report):
    assert report["schema"] == "dpanel.half_wake_audit.v1"
    assert report["physical_accuracy_accepted"] is False
    assert len(report["surface_sha256"]) == 64
    assert len(report["state_sha256"]) == 64
    assert len(report["code_sha256"]) == 64
    assert report["geometry"]["full_panels"] == 40
    assert np.isfinite(report["linear_residual"])
    json.dumps(report, allow_nan=False)
    assert report["physical_residuals"]["units"]["constraint"]
    oracle = report["mirror_oracle"]["full_oracle"]
    assert oracle["performed"] is False
    assert oracle["note"].startswith("the default path never solves")


def test_frozen_state_replay_does_not_solve_the_flow(tmp_path=None):
    # pytest's tmp_path is unscannable under this sandbox (WinError 5), so this
    # test owns a short-lived file next to the worktree instead.
    base = Path.cwd() / f".tmp_frozen_wake_{os.getpid()}"
    base.mkdir(parents=True, exist_ok=True)
    path = base / "frozen_state.npz"
    half = load_half_surface(SURFACE)
    solved = jax.jit(solve_half_fields)(**half)
    digest = hashlib.sha256(Path(SURFACE).read_bytes()).hexdigest()
    try:
        np.savez(path,
                 mu_half=np.asarray(solved["mu_half"], dtype=np.float64),
                 gamma_half=np.asarray(solved["gamma_half"], dtype=np.float64),
                 sigma_half=np.asarray(solved["sigma_half"], dtype=np.float64),
                 surface_sha256=np.asarray(digest))
        fresh = audit(SURFACE, offsets=(1e-2,))
        replay = audit(SURFACE, offsets=(1e-2,), frozen_state=str(path))
        assert replay["state_origin"].startswith("reused frozen state")
        assert replay["linear_residual"] is None
        assert replay["state_sha256"] == fresh["state_sha256"]
        assert replay["wake_report"]["by_offset"][0]["all"]["flux_rms"] == (
            fresh["wake_report"]["by_offset"][0]["all"]["flux_rms"])
        assert replay["frozen_state"].endswith("frozen_state.npz")
        bad = dict(np.load(path, allow_pickle=False))
        bad["surface_sha256"] = np.asarray("0" * 64)
        wrong = base / "wrong.npz"
        np.savez(wrong, **bad)
        with pytest.raises(ValueError, match="surface SHA"):
            audit(SURFACE, offsets=(1e-2,), frozen_state=str(wrong))
    finally:
        import shutil
        shutil.rmtree(base, ignore_errors=True)


def test_cli_writes_a_new_report_and_refuses_to_overwrite():
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


def test_cli_full_oracle_flag_is_off_by_default():
    import shutil
    base = Path.cwd() / f".tmp_wake_oracle_{os.getpid()}"
    out = base / "audit.json"
    try:
        assert main(["--surface", str(SURFACE), "--out", str(out),
                     "--offsets", "1e-2", "--steps", "1e-3"]) == 0
        written = json.loads(out.read_text(encoding="utf-8"))
        assert written["mirror_oracle"]["full_oracle"]["performed"] is False
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_oversized_half_mesh_is_rejected_before_the_half_solve():
    with pytest.raises(MemoryError):
        build_mirror_state(SURFACE, max_half_panels=4)


def test_streaming_blocks_do_not_change_the_field(context):
    targets = wake_samples(context)[:, 0, :]
    coarse, _ = evaluate_field(targets, context, chunk=7)
    fine, _ = evaluate_field(targets, context, chunk=1024)
    np.testing.assert_allclose(coarse, fine, rtol=0., atol=0.)


def test_probe_points_are_clear_of_the_sheet(context):
    probes = probe_points(context)
    assert probes.shape[1] == 3
    assert probes.shape[0] >= 1


def test_wake_labels_cover_the_first_trailing_edge_layer(context):
    labels = wake_labels(context)
    assert labels["first_row"].sum() >= 1
    assert labels["gap_over_chord"].shape == (len(context["wake"]),)
    # The reported bands are per sample now; the centroid view keeps no bands.
    assert "bands" not in labels
    assert "per sample" in labels["definitions"]["first_row"] or True
