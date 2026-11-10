"""Registry and format checks for the issue #115 calibration suite."""
import json

import pytest

from tools import calibration


def test_every_case_of_issue_115_is_registered():
    assert sorted(calibration.CASES) == ["C01", "C02", "C03", "C04", "C05", "C06"]


def test_reference_tables_are_frozen_with_their_conditions():
    data = calibration._reference()
    tables = data["tables"]
    assert len(tables["C01_cylinder_cp"]["rows"]) == 9
    assert len(tables["C02_sphere_cp"]["rows"]) == 8
    assert len(tables["C03_joukowski"]["rows"]) == 3
    assert len(tables["C04_plate"]["rows"]) == 5
    assert len(tables["C05_elliptic"]["rows"]) == 5
    assert len(tables["C06_ladson_80grit"]["rows"]) == 17
    assert len(tables["C06_gregory_cp_alpha0"]["rows"]) == 25
    assert data["conditions"]["C06_ladson_80grit"]["reynolds_chord"] == 6.0e6
    assert data["conditions"]["C06_gregory_cp_alpha0"]["reynolds_chord"] == 2.88e6


def test_blocked_cases_report_the_reason_and_keep_the_reference():
    for name in ("C01", "C03", "C06"):
        record = calibration.CASES[name]()
        assert record["status"] == "blocked"
        assert record["notes"], name
        assert record["differences"], name
        assert record["physical_acceptance"] == "not_assessed"


def test_ladson_fit_uses_the_declared_band_only():
    record = calibration.case_c06()
    rows = calibration._reference()["tables"]["C06_ladson_80grit"]["rows"]
    band = [row for row in rows if abs(row[0]) <= 8.3]
    assert len(band) == 7
    assert record["differences"]["ladson_fit_slope_per_deg"] == pytest.approx(0.107994, abs=2e-4)


def test_summary_fields_follow_the_issue_template(tmp_path):
    record = calibration._summary("X", status="blocked", notes=["n"])
    required = {"case_id", "geometry_source", "geometry_sha256", "solver_commit",
                "solver_mode", "U_inf", "rho_inf", "mach", "reynolds_chord", "alpha_deg",
                "transition", "reference_area", "mesh", "algebraic_residual",
                "max_normal_velocity_ratio", "cp_l2_error", "cp_linf_error", "cl_pressure",
                "cl_circulation", "cd_pressure", "cdi_trefftz", "moment_quarter_chord",
                "physical_acceptance"}
    assert required <= set(record)
    assert record["physical_acceptance"] == "not_assessed"
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["case_id"] == "X"


def test_cli_runs_only_the_requested_case(tmp_path, capsys):
    assert calibration.main(["--case", "C03", "--out", str(tmp_path)]) == 0
    written = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert [record["case_id"] for record in written] == ["C03_joukowski_A0_4_8"]
    assert "C03_joukowski_A0_4_8" in capsys.readouterr().out
