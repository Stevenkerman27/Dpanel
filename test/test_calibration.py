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


def test_summary_json_contains_aerodynamic_report_fields(tmp_path):
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


def test_cli_runs_only_the_requested_case(tmp_path, monkeypatch):
    monkeypatch.setitem(calibration.CASES, "C02", lambda: calibration._summary("C02", status="ran"))
    assert calibration.main(["--case", "C02", "--out", str(tmp_path)]) == 0
    written = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert [record["case_id"] for record in written] == ["C02"]
