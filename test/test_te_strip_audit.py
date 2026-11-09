"""The trailing-edge strip audit that guards the potential-jump closure."""

import numpy as np
import pytest

from panel.grass_mesh import build_grass_mesh, load_grass_wing
from panel.p1_lifting_g1 import make_wing
from panel.te_strip_audit import (SWEEP_LIMIT_DEG, STRIP_ASPECT_LIMIT,
                                  TIP_CHORD_RATIO_LIMIT, bounded_elliptic_sections,
                                  te_strip_diagnostics)


def _grass(sections, chord_panels=6, span_panels_per_segment=4):
    data = build_grass_mesh(sections, 1.0, chord_panels=chord_panels,
                            span_panels_per_segment=span_panels_per_segment)
    return load_grass_wing(data)


def _cusp_elliptic(n=9, sweep_fraction=0.97, chord_panels=6, span_panels=8):
    theta = np.linspace(0.0, 0.5 * np.pi * sweep_fraction, n)
    cst = np.zeros((n, 18))
    cst[:, :8] = 0.06
    cst[:, 8:16] = -0.06
    sections = dict(leading_edge_xyz=np.c_[np.zeros(n), 4.0 * np.sin(theta), np.zeros(n)],
                    chord=2.0 * np.cos(theta), twist_radians=np.zeros(n), cst=cst)
    return _grass(sections, chord_panels=chord_panels,
                  span_panels_per_segment=span_panels)


def test_rectangular_wing_passes_with_no_warnings():
    wing = make_wing(chord=2.0, span=8.0, span_panels=96, chord_panels=6, wake_length=5.)
    report = te_strip_diagnostics(wing)
    assert report["ok"] is True
    assert report["warnings"] == []
    assert report["max_sweep_deg"] == pytest.approx(0.0, abs=1e-9)
    # 96 span panels over a span of 8.0, uniformly spaced.
    assert report["min_strip_width"] == pytest.approx(8.0 / 96.0, rel=1e-9)
    assert report["max_strip_aspect"] < STRIP_ASPECT_LIMIT


def test_bounded_elliptic_tip_stays_below_the_sweep_limit():
    sections = bounded_elliptic_sections(sections=9, span=8.0, root_chord=2.0,
                                         tip_fraction=0.85, tip_chord=0.24)
    report = te_strip_diagnostics(_grass(sections))
    assert report["ok"] is True, report["warnings"]
    # Measured 53.8 deg for this planform; the limit is 60 deg.
    assert 50.0 < report["max_sweep_deg"] < 57.0
    # The tip caps are not TE stations, so the narrowest sampled chord is 0.2769, i.e. 13.8%
    # of the 2.0 root chord; the true tip chord is 0.24 (12%).
    assert 0.10 < report["tip_chord_ratio"] < 0.16


def test_cusp_elliptic_tip_is_flagged_by_sweep_and_strip_aspect():
    report = te_strip_diagnostics(_cusp_elliptic())
    kinds = {w["kind"] for w in report["warnings"]}
    assert report["ok"] is False
    assert "te_sweep" in kinds
    assert report["max_strip_aspect"] > 1000.0
    sweep = [w for w in report["warnings"] if w["kind"] == "te_sweep"][0]
    # Measured 74.0 deg at the tip for a 0.97 cut.
    assert sweep["value"] > 70.0
    assert sweep["limit"] == SWEEP_LIMIT_DEG
    assert "limit 60" in sweep["message"]
    assert sweep["station"] >= len(report["sweep_deg"]) - 3
    # A cusp tip warns on every strip near both tips; the report keeps three per kind.
    assert report["warning_counts"]["te_sweep"] > 3
    assert len([w for w in report["warnings"] if w["kind"] == "te_sweep"]) == 3
    assert sweep["total"] == report["warning_counts"]["te_sweep"]


def test_a_dense_but_bounded_tip_is_still_accepted():
    # Eight span panels per segment drive the strip aspect to 1613, which is not by itself a
    # failure: the measured cond(dp/dgamma) stays at 4.5e4 for this planform.
    sections = bounded_elliptic_sections(sections=9)
    report = te_strip_diagnostics(_grass(sections, span_panels_per_segment=8))
    assert report["ok"] is True, report["warnings"]
    assert report["max_strip_aspect"] > 1000.0


def test_warnings_follow_the_configured_limits():
    wing = _cusp_elliptic()
    assert te_strip_diagnostics(wing, sweep_limit=90.0, aspect_limit=1e9)["ok"] is True
    strict = te_strip_diagnostics(wing, sweep_limit=30.0)
    assert any(w["kind"] == "te_sweep" for w in strict["warnings"])


def test_cusp_tip_chord_ratio_warns_when_the_tip_is_degenerate():
    theta = np.linspace(0.0, 0.5 * np.pi * 0.99999, 9)
    cst = np.zeros((9, 18))
    cst[:, :8] = 0.06
    cst[:, 8:16] = -0.06
    sections = dict(leading_edge_xyz=np.c_[np.zeros(9), 4.0 * np.sin(theta), np.zeros(9)],
                    chord=2.0 * np.cos(theta), twist_radians=np.zeros(9), cst=cst)
    # Eight span panels put a TE station at chord 0.049, i.e. 2.4% of the root chord.
    report = te_strip_diagnostics(_grass(sections, span_panels_per_segment=8))
    cusp = [w for w in report["warnings"] if w["kind"] == "cusp_tip"]
    assert cusp, report["warnings"]
    assert cusp[0]["limit"] == TIP_CHORD_RATIO_LIMIT


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_the_command_line_tool_exits_on_the_audit_verdict(tmp_path):
    import json

    from tools.te_strip_audit import main

    assert main(["--demo", "bounded"]) == 0
    assert main(["--demo", "cusp"]) == 1
    sections = bounded_elliptic_sections(sections=9)
    saved = build_grass_mesh(sections, 1.0, chord_panels=6, span_panels_per_segment=4)
    path = tmp_path / "surface.json"
    path.write_text(json.dumps(_plain(saved)))
    assert main(["--mesh", str(path), "--json", str(tmp_path / "report.json")]) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["ok"] is True and report["wake_length"] > 0.0


def test_audit_does_not_modify_the_wing():
    wing = _cusp_elliptic()
    xyz = np.array(wing.xyz, copy=True)
    y = np.array(wing.te_pair_y, copy=True)
    te_strip_diagnostics(wing)
    assert np.array_equal(wing.xyz, xyz)
    assert np.array_equal(wing.te_pair_y, y)


def test_bounded_elliptic_sections_are_monotone_and_shaped():
    sections = bounded_elliptic_sections(sections=9)
    y = sections["leading_edge_xyz"][:, 1]
    assert np.all(np.diff(y) > 0.0)
    assert np.all(sections["chord"][:-1] >= sections["chord"][1:] - 1e-12)
    assert sections["cst"].shape == (9, 18)
    assert np.all(np.isfinite(sections["cst"]))

