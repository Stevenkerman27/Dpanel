"""Tests for the B3 protocol helpers (issue #92, gate B3).

The external codes are not run here: these tests pin the parts that decide the
numbers -- the full-span Trefftz drag, the linear fit, and the manifest.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from tools.issue92.b3_judge import _fit, _gamma_shape, _interp
from tools.issue92.trefftz_full import elliptic_self_test, trefftz_cd_full

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_full_span_trefftz_drag_recovers_the_elliptic_optimum():
    cd, theory = elliptic_self_test(201)
    assert cd == pytest.approx(theory, rel=1.5e-2)
    cd4, theory4 = elliptic_self_test(401)
    assert abs(cd4 / theory4 - 1.0) < 8e-3


def test_full_span_trefftz_is_converging_with_station_count():
    ratios = [elliptic_self_test(n)[0] / elliptic_self_test(n)[1] for n in (51, 101, 201, 401)]
    assert all(b > a for a, b in zip(ratios, ratios[1:]))


def test_full_span_trefftz_rejects_bad_input():
    with pytest.raises(ValueError):
        trefftz_cd_full([0.0, 1.0], [0.0, 0.0], area=1.0)
    with pytest.raises(ValueError):
        trefftz_cd_full([0.0, 0.5, 1.0], [0.0, 0.1], area=1.0)


def test_zero_circulation_gives_zero_drag():
    y = np.linspace(-1.0, 1.0, 9)
    cd, w = trefftz_cd_full(y, np.zeros_like(y), area=2.0)
    assert cd == 0.0
    assert np.allclose(w, 0.0)


def test_fit_recovers_a_known_line():
    alphas = [0.0, 2.0, 4.0, 6.0]
    values = [0.1 + 0.05 * a for a in alphas]
    fit = _fit(alphas, values)
    assert fit["slope_per_deg"] == pytest.approx(0.05, rel=1e-12)
    assert fit["intercept"] == pytest.approx(0.1, rel=1e-12)
    assert fit["max_residual"] < 1e-12


def test_gamma_shape_is_zero_for_identical_distributions():
    y = np.linspace(0.0, 1.0, 9)
    gamma = np.sin(np.pi * y)
    case = {"stations": [{"y": float(v), "chord": 1.0} for v in y]}
    dp_row = {"stations": np.r_[0.0, y, 1.0].tolist(),
              "gamma_stations": gamma.tolist()}
    vsp_row = {"lod": [{"alpha_deg": 4.0, "y": y.tolist(),
                        "cl": (2.0 * gamma).tolist(), "dspan": [0.0] * len(y)}]}
    shape = _gamma_shape(case, dp_row, vsp_row, alpha=4.0)
    assert shape["rms_relative"] == pytest.approx(0.0, abs=1e-12)


def test_manifest_is_consistent():
    doc = json.loads((ROOT / "docs/issue92/b3_cases.json").read_text(encoding="utf-8"))
    assert doc["alphas_deg"] == [0.0, 2.0, 4.0, 6.0]
    for name, case in doc["cases"].items():
        assert len(case["geometry_sha256"]) == 64
        int(case["geometry_sha256"], 16)
        ys = [s["y"] for s in case["stations"]]
        assert ys == sorted(ys) and len(ys) == len(set(ys)), name
        assert all(s["chord"] > 0.0 for s in case["stations"])
        assert case["sref"] > 0.0 and case["cref"] > 0.0


def test_interp_handles_unsorted_abscissae():
    xp = np.array([2.0, 0.0, 1.0])
    fp = np.array([2.0, 0.0, 1.0])
    assert _interp(xp, fp, np.array([0.5, 1.5])).tolist() == [0.5, 1.5]
