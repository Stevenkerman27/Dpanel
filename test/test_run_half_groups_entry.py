"""Contract tests for the v2 shed-group research entry point.

These tests drive panel.run_half_groups rather than the library API: they assert
that the entry point reports reproducible hashes, real exit codes, the block
structure [[D, W1..WG], [-T_g, M_g]] and the small-fixture battery F0-F8 as
published in docs/ISSUE39_SHED_GROUPS_V2.md.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from panel.group_fixture import as_surface_json, group_specs, two_layer_case
from panel.run_half_groups import (SCHEMA, code_manifest, groups_from_spec,
                                   load_surface, main, run, self_check,
                                   single_group_from_surface)


SCRATCH = Path(__file__).resolve().parents[1] / "runs" / "group_entry_tests"


def _scratch(name):
    # tmp_path is unusable under the file sandbox (pytest cannot enumerate its
    # own pytest-of-* root), so scratch directories live in the worktree.
    path = SCRATCH / name
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="module")
def battery():
    # Deliberately not tmp_path_factory: the sandbox denies enumerating a
    # pytest-of-* root that already exists, which broke module-scoped fixtures.
    out_dir = Path(__file__).resolve().parents[1] / "runs" / "group_entry_selfcheck"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "selfcheck.json"
    report = self_check(out, shift=0.4)
    return report, json.loads(out.read_text(encoding="utf-8"))


def _case(battery, label):
    for entry in battery[0]["cases"]:
        if entry["label"] == label:
            return entry
    raise AssertionError(f"missing case {label}")


def test_battery_has_four_cases_and_all_exit_zero(battery):
    labels = [entry["label"] for entry in battery[0]["cases"]]
    assert labels == ["G0", "G1", "G2", "G2_tail_z024"]
    for entry in battery[0]["cases"]:
        assert entry["exit_code"] == 0, (entry["label"], entry.get("error"),
                                        entry.get("traceback"))
        assert entry["linear_residual_relative"] < 1e-12


def test_report_is_hashed_and_not_a_physical_claim(battery):
    report = battery[0]
    assert report["schema"] == SCHEMA
    assert report["physical_accuracy_accepted"] is False
    for entry in report["cases"]:
        assert len(entry["surface_sha256"]) == 64
        if entry["groups_path"] is not None:
            assert len(entry["groups_sha256"]) == 64
    manifest = code_manifest()
    assert "panel/shed_groups.py" in manifest["files"]
    assert manifest["files"]["panel/run_half_groups.py"]["sha256"] == \
        manifest["files"]["panel/run_half_groups.py"]["sha256"]
    assert len(manifest["files"]["panel/half_native.py"]["sha256"]) == 64


def test_g0_is_pure_dirichlet(battery):
    g0 = _case(battery, "G0")
    assert g0["unknowns"] == 20
    assert g0["groups_path"] is None
    assert abs(g0["CL"]) < 1e-12
    assert g0["checks"]["block_dimensions"]["group_unknowns"] == []
    assert g0["checks"]["lift_slope"]["performed"] is False
    assert "circulation" in g0["checks"]["lift_slope"]["reason"]


def test_g1_reproduces_the_v1_solver_to_round_off(battery):
    v1 = _case(battery, "G1")["checks"]["v1_equivalence"]
    assert v1["performed"] is True
    assert v1["mu_relative_to_peak"] == 0.0
    assert v1["gamma_relative_to_peak"] == 0.0
    assert v1["cp_relative_to_peak"] == 0.0
    assert v1["cl_absolute"] == 0.0
    assert v1["cd_absolute"] == 0.0


def test_g1_full_mirror_oracle_agrees(battery):
    oracle = _case(battery, "G1")["checks"]["full_oracle"]
    assert oracle["performed"] is True
    assert oracle["full_panels"] == 40
    assert oracle["full_circulation_unknowns"] == oracle["half_circulation_unknowns"] + 1
    assert oracle["mu_relative_to_peak"] < 1e-12
    assert oracle["gamma_relative_to_peak"] < 1e-12
    assert oracle["physical_accuracy_accepted"] is False


def test_g2_block_structure(battery):
    checks = _case(battery, "G2")["checks"]
    blocks = checks["block_dimensions"]
    assert blocks["ok"] is True
    assert blocks["shape"] == [44, 44]
    assert blocks["group_unknowns"] == [2, 2]
    # Both stacked surfaces are wall panels (40) and each group adds 2 dofs.
    assert blocks["wall_unknowns"] == 40
    assert blocks["group_slices"] == [[40, 42], [42, 44]]
    assert blocks["kutta_rows"] == 4
    assert checks["kutta_rows"]["max_rms"] < 1e-12
    assert [row["group_id"] for row in checks["kutta_rows"]["groups"]] == \
        ["wing_low", "wing_high"]


def test_mass_rows_are_normalised_from_quadrature(battery):
    for label in ("G1", "G2"):
        mass = _case(battery, label)["checks"]["circulation_mass_rows"]
        assert mass["performed"] is True
        assert mass["max_normalisation_gap"] < 1e-15
        assert mass["max_quadrature_gap"] < 1e-15
        for row in mass["groups"]:
            # The tip station keeps dof -1, so its basis is not a partition of
            # unity and the row sum there is published, never forced to one.
            assert row["row_sums_where_partition_of_unity"] == \
                pytest.approx([1.0], abs=1e-14)
            assert row["row_sums"][-1] < 1.0
            assert row["quadrature_points_with_partial_basis"] >= 1


def test_operator_mirror_is_rebuilt(battery):
    for label in ("G1", "G2"):
        mirror = _case(battery, label)["checks"]["operator_mirror"]
        assert mirror["performed"] is True
        assert mirror["doublet_relative_to_peak"] < 1e-12
        assert mirror["worst_wake_operator_relative"] < 1e-12
        for row in mirror["groups"]:
            assert row["wake_operator_shape"][0] >= 20
            assert row["wake_operator_relative_to_peak"] < 1e-12


def test_lift_slope_matches_forward_ad(battery):
    for label in ("G1", "G2", "G2_tail_z024"):
        slope = _case(battery, label)["checks"]["lift_slope"]
        assert slope["performed"] is True
        assert slope["max_relative_difference"] < 1e-6
        assert slope["dcl_dalpha_per_deg"] > 0.0
        steps = [row["step"] for row in slope["finite_difference"]]
        assert steps == [1e-4, 1e-5]


def test_elevated_tail_changes_the_solution(battery):
    g2 = _case(battery, "G2")
    tail = _case(battery, "G2_tail_z024")
    assert tail["unknowns"] == g2["unknowns"] == 44
    assert abs(tail["CL"] - g2["CL"]) > 1e-3
    # A stacked pair is not a symmetry: the upper group can carry negative
    # circulation when it sits inside the lower wake.
    assert tail["gamma_sum"][1] < 0.0
    assert g2["gamma_sum"][1] > 0.0


def test_two_groups_are_coupled(battery):
    checks = battery[0]["checks"]
    row = checks["g1_vs_g2_gamma"][0]
    assert row["gamma_sum_g1"] != pytest.approx(row["gamma_sum_g2"], rel=1e-6)
    assert "coupled" in checks["g1_vs_g2_notes"]


def test_run_refuses_oversized_mesh():
    case, raw, vertices, faces, tags, groups = two_layer_case(None, 0.4, groups=1)
    surface = as_surface_json(raw, vertices, faces, tags, groups)
    path = _scratch("oversized") / "surface.json"
    path.write_text(json.dumps(surface), encoding="utf-8")
    with pytest.raises(MemoryError):
        run(path, max_panels=4)


def test_cli_writes_then_refuses_to_overwrite():
    directory = _scratch("cli")
    if (directory / "report.json").exists():
        (directory / "report.json").unlink()
    case, raw, vertices, faces, tags, groups = two_layer_case(None, 0.4, groups=1)
    surface = as_surface_json(raw, vertices, faces, tags, groups)
    surface_path = directory / "surface.json"
    groups_path = directory / "groups.json"
    surface_path.write_text(json.dumps(surface), encoding="utf-8")
    groups_path.write_text(json.dumps(group_specs(groups)), encoding="utf-8")
    out = directory / "report.json"
    assert main(["--surface", str(surface_path), "--groups", str(groups_path),
                 "--out", str(out), "--no-gradient"]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["schema"] == SCHEMA
    assert report["solve"]["unknowns"] == 22
    with pytest.raises(FileExistsError):
        main(["--surface", str(surface_path), "--groups", str(groups_path),
              "--out", str(out), "--no-gradient"])


def test_cli_records_a_refusal_report():
    directory = _scratch("refusal")
    if (directory / "refused.json").exists():
        (directory / "refused.json").unlink()
    surface_path = directory / "broken.json"
    surface_path.write_text(json.dumps({"coordinates": [[0, 0, 0]]}), encoding="utf-8")
    out = directory / "refused.json"
    with pytest.raises(ValueError):
        main(["--surface", str(surface_path), "--out", str(out)])
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["status"] == "refused"
    assert "missing keys" in report["error"]
    assert report["physical_accuracy_accepted"] is False


def test_surface_loader_requires_the_declared_keys():
    path = _scratch("loader") / "s.json"
    path.write_text(json.dumps({"coordinates": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_surface(path)


def test_group_declaration_rejects_an_unknown_schema():
    directory = _scratch("schema")
    case, raw, vertices, faces, tags, groups = two_layer_case(None, 0.4, groups=1)
    surface = as_surface_json(raw, vertices, faces, tags, groups)
    path = directory / "surface.json"
    path.write_text(json.dumps(surface), encoding="utf-8")
    _, loaded = load_surface(path)
    specs = group_specs(groups)
    specs["schema"] = "dpanel.half_shed_groups.v1"
    spec_path = directory / "groups.json"
    spec_path.write_text(json.dumps(specs), encoding="utf-8")
    with pytest.raises(ValueError):
        groups_from_spec(spec_path, loaded)


def test_single_group_wrapper_needs_a_wake_sheet():
    case, raw, vertices, faces, tags, groups = two_layer_case(None, 0.4, groups=0)
    surface = as_surface_json(raw, vertices, faces, tags, groups)
    path = _scratch("no_wake") / "surface.json"
    path.write_text(json.dumps(surface), encoding="utf-8")
    _, loaded = load_surface(path)
    with pytest.raises(ValueError):
        single_group_from_surface(loaded)
