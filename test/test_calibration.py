"""Registry and format checks for the inviscid calibration suite."""
import json

from tools import calibration


def test_default_cli_runs_only_active_p1_cases(tmp_path, monkeypatch):
    for name in calibration.CASES:
        monkeypatch.setitem(calibration.CASES, name, lambda name=name: calibration._summary(name, status="ran"))
    monkeypatch.setattr(calibration, "case_c05",
                        lambda *, refine: calibration._summary("C05", status="ran"))
    assert calibration.main(["--out", str(tmp_path)]) == 0
    written = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert [record["case_id"] for record in written] == ["C02", "C04", "C05"]


def test_inviscid_reference_tables_have_expected_rows():
    data = calibration._reference()
    tables = data["tables"]
    assert len(tables["C01_cylinder_cp"]["rows"]) == 9
    assert len(tables["C02_sphere_cp"]["rows"]) == 8
    assert len(tables["C03_joukowski"]["rows"]) == 3
    assert len(tables["C04_plate"]["rows"]) == 5
    assert len(tables["C05_elliptic"]["rows"]) == 5


def test_blocked_cases_report_the_reason_and_keep_the_reference():
    for name in ("C01", "C03"):
        record = calibration.CASES[name]()
        assert record["status"] == "blocked"
        assert record["notes"], name
        assert record["differences"], name
        assert record["physical_acceptance"] == "not_assessed"


def test_summary_fields_follow_the_issue_template(tmp_path):
    refinement = {"rows": [{"faces": 1560}, {"faces": 4640}]}
    record = calibration._summary("X", status="ran", refinement=refinement)
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
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["case_id"] == "X"
    assert written["refinement"] == refinement


def test_cli_runs_only_the_requested_case(tmp_path, capsys):
    assert calibration.main(["--case", "C03", "--out", str(tmp_path)]) == 0
    written = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert [record["case_id"] for record in written] == ["C03_joukowski_A0_4_8"]
    assert "C03_joukowski_A0_4_8" in capsys.readouterr().out
