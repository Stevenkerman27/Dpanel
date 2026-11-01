"""Phase 0 (issue #92) guard rails: frozen fixtures, profiler, solver crossover.

The fixtures are the reference the PyTorch backend will be compared against, so any
digest drift must be loud rather than silent, and the crossover merge must stay a
plain union so per-backend processes can be run in isolation (a CUDA crash in one
process must not be able to destroy the CPU numbers).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.issue92 import fixtures, profile_production, solver_crossover  # noqa: E402


MANIFEST = ROOT / "docs/issue92/fixtures_manifest.json"
SMALL = ("g1_wing_rect", "g1_wing_camber", "g2_conventional", "g2_tail_low")


@pytest.fixture(scope="module")
def manifest():
    if not MANIFEST.exists():
        pytest.skip("fixtures manifest not built")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_the_battery_covers_single_and_multi_group_geometry(manifest):
    entries = manifest["fixtures"]
    assert {"g1_wing_rect", "g2_conventional", "g1_swept30",
            "g1_wingbody_cad_3x"} <= set(entries)
    for name, entry in entries.items():
        if entry["status"] != "ok":
            continue
        assert entry["digest"] and entry["n_wall_faces"] > 0
        assert (ROOT / "docs/issue92/fixtures" / (name + ".npz")).exists()


def test_small_fixture_digests_still_match_the_committed_manifest(manifest):
    for name in SMALL:
        arrays, entry = fixtures.load(name)
        per_array, digest = fixtures._digest(arrays)
        assert digest == entry["digest"]
        assert per_array == entry["array_sha256"]


def test_rebuilding_a_fixture_reproduces_its_digest(manifest, tmp_path, monkeypatch):
    expected = manifest["fixtures"]["g1_wing_rect"]
    every = fixtures.builders()
    monkeypatch.setattr(fixtures, "builders", lambda: {"g1_wing_rect": every["g1_wing_rect"]})
    built = fixtures.build(tmp_path)
    fresh = built["fixtures"]["g1_wing_rect"]
    assert fresh["digest"] == expected["digest"]
    assert fresh["array_sha256"] == expected["array_sha256"]


def test_a_tampered_fixture_is_refused(manifest, tmp_path):
    arrays_dir = tmp_path / "fixtures"
    arrays_dir.mkdir()
    arrays, entry = fixtures.load("g1_wing_rect")
    key = sorted(arrays)[0]
    arrays[key] = arrays[key].copy()
    arrays[key].reshape(-1)[0] = arrays[key].reshape(-1)[0] + 1e-6
    np.savez_compressed(arrays_dir / "g1_wing_rect.npz", **arrays)
    (tmp_path / "fixtures_manifest.json").write_text(
        json.dumps({"schema": "dpanel.issue92.fixtures.v1", "fixtures":
                    {"g1_wing_rect": entry}}), encoding="utf-8")
    with pytest.raises(ValueError, match="fixture digest drift"):
        fixtures.load("g1_wing_rect", arrays_dir=arrays_dir)


def test_wing_counters_expose_the_matrix_cost(manifest):
    from panel.p1_lifting_g1 import assemble

    entry = manifest["fixtures"]["g1_wing_rect"]
    wing = profile_production._wing_object("g1_wing_rect", entry)
    ops = assemble(wing)
    counters = profile_production._wing_counters(wing, ops)
    assert counters["n_wall_faces"] == entry["n_wall_faces"]
    assert counters["n_mu_dofs"] == wing.ndof_mu
    rows, columns = counters["matrix_shape"]
    # A holds the wall (Dirichlet trace) rows only; the Kutta rows are imposed as
    # constraints on the null space, not appended to A.
    assert rows == entry["n_wall_faces"]
    assert counters["n_kutta_rows"] > 0
    assert columns == counters["n_mu_dofs"] + counters["n_gamma_dofs"]
    assert 0.0 < counters["A_nnz_fraction"] <= 1.0
    assert counters["A_bytes_mb"] > 0.0


def test_crossover_merge_unions_backends_and_environments(tmp_path):
    left = {"environment": {"cpu": "x"}, "rows": [
        {"n": 192, "backends": {"cpu_lu": {"warm_median": 0.0004}}},
        {"n": 96, "backends": {"cpu_lu": {"warm_median": 0.0}}}]}
    right = {"environment": {"gpu": "y"}, "rows": [
        {"n": 192, "backends": {"cuda_lu": {"warm_median": 0.0005}}}]}
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(left), encoding="utf-8")
    b.write_text(json.dumps(right), encoding="utf-8")
    out = tmp_path / "merged.json"
    merged = solver_crossover._merge([a, b], out)
    assert [row["n"] for row in merged["rows"]] == [96, 192]
    assert set(merged["rows"][1]["backends"]) == {"cpu_lu", "cuda_lu"}
    assert merged["environment"] == {"cpu": "x", "gpu": "y"}
    assert json.loads(out.read_text(encoding="utf-8"))["rows"] == merged["rows"]


def test_crossover_verdict_reports_the_first_win():
    rows = [{"n": 192, "backends": {"cpu_lu": {"warm_median": 1.0},
                                    "torch_cpu": {"warm_median": 0.5}}},
            {"n": 385, "backends": {"cpu_lu": {"warm_median": 1.0},
                                    "cuda_lu": {"warm_median": 0.25}}},
            {"n": 385, "backends": {"cpu_lu": {"warm_median": 1.0},
                                    "cuda_lu": {"warm_median": 2.0}}}]
    verdict = solver_crossover._verdict(rows)
    assert verdict["torch_cpu_first_win_over_cpu_lu"] == {"n": 192, "speedup": 2.0}
    assert verdict["cuda_lu_first_win_over_cpu_lu"] == {"n": 385, "speedup": 4.0}
    assert verdict["largest_measured_speedup_cuda_lu"] == 0.5