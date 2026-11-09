"""Numerical trend checks, without treating thin-wing theory as a thick-wing truth."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from panel.p1_lifting_g1 import make_wing, solve


def test_fixed_gamma_physical_probes_improve_under_galerkin_refinement(tmp_path):
    subprocess.run(
        [sys.executable, "-m", "tools.lifting_line_study", "--case", "fixed-gamma",
         "--out", str(tmp_path), "--chords", "4", "8", "16", "--walls", "galerkin"],
        cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True)
    rows = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["rows"]
    assert all(row["gamma_fixed_error"] == 0. and row["jump_max"] < 1e-12 for row in rows)
    np.testing.assert_allclose([row["cl_gamma"] for row in rows], rows[0]["cl_gamma"], atol=1e-13)
    for quantity in ("potential_change_max", "speed_change_max", "cp_change_max"):
        assert rows[2][quantity] < rows[1][quantity], quantity


def test_same_family_roots_from_multiple_seeds_and_both_alpha_sweeps(tmp_path):
    subprocess.run(
        [sys.executable, "-m", "tools.lifting_line_branches", "--out", str(tmp_path),
         "--chords", "4", "--alphas", "-2", "0", "2"],
        cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True)
    rows = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["rows"]
    assert len(rows) == 18
    assert all(row["accepted"] and row["condition_scaled"] < 10. for row in rows)
    assert max(row["gamma_delta_from_first_root"] or 0. for row in rows) < 1e-9
    assert all(row["top_streamwise_min"] > 0. and row["bottom_streamwise_min"] > 0.
               and row["paired_dot_min"] > 0. for row in rows)


@pytest.mark.parametrize("kutta", ["pressure", "potential_jump"])
def test_matched_area_closure_lift_stabilizes_with_galerkin(kutta):
    results = []
    for chord_panels in (4, 8, 16):
        wing = make_wing(chord_panels=chord_panels, span_panels=12, span=4.,
                         thickness=.12, wake_length=10., wake_panels=2)
        state = solve(wing, 4., kutta=kutta, pressure_recovery="area")
        assert state["nonlinear"]["success"]
        assert state["CL"] > 0. and state["CL_gamma"] > 0.
        assert state["jump_rms"] < 1e-12
        results.append(state)
    lifts = [state["CL"] for state in results]
    assert abs(lifts[2] - lifts[1]) < abs(lifts[1] - lifts[0])
    assert abs(results[-1]["CL"] - results[-1]["CL_gamma"]) / results[-1]["CL"] < .02
