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

# --- FP64 device probe (docs/issue92/gpu_fp64_probe.json) ------------------------
#
# The probe is produced by tools/issue92/gpu_fp64_probe.py in the CUDA interpreter, which
# the CPU-only suite cannot import. The tests therefore read the committed artifact and
# check the contract it has to satisfy: both devices measured, transfer separated from
# arithmetic, and the document carrying the honest range rather than a single speedup.

PROBE = ROOT / "docs/issue92/gpu_fp64_probe.json"
BASELINE_DOC = ROOT / "docs/ISSUE92_PHASE0_BASELINE.md"


@pytest.fixture(scope="module")
def probe():
    if not PROBE.exists():
        pytest.skip("FP64 device probe not measured on this machine")
    return json.loads(PROBE.read_text(encoding="utf-8"))


def test_fp64_probe_records_both_devices_with_verified_residuals(probe):
    env = probe["environment"]
    assert env["cuda_available"] is True
    assert env["gpu"]
    for device in ("cpu", "cuda"):
        rows = probe["kernels"]["matmul"][device]
        assert [row["n"] for row in rows] == [1024, 2048, 4096]
        assert all(row["tflops"] > 0.0 for row in rows)
        assert all(row["dtype"] == "float64" for row in rows)
    for n in ("1728", "3220"):
        block = probe["kernels"]["batched_solve"][n]
        assert set(block) == {"cpu", "cuda"}
        for device in ("cpu", "cuda"):
            rows = block[device]
            assert [row["rhs"] for row in rows] == [1, 16, 64]
            assert all(row["rel_residual"] < 1e-10 for row in rows)
            # one factorisation per matrix, reused by every right-hand side
            assert len({row["lu_factor_seconds"] for row in rows}) == 1


def test_fp64_probe_separates_transfer_from_arithmetic(probe):
    for n, row in probe["host_to_device"].items():
        expected_mb = round(int(n) ** 2 * 8 / 1048576.0, 1)
        assert row["matrix_mb"] == expected_mb
        assert row["host_to_device_seconds"] > 0.0
    pairwise = probe["kernels"]["pairwise"]
    for device in ("cpu", "cuda"):
        row = pairwise[device]
        assert row["interactions"] == row["n_targets"] * row["n_triangles"]
        assert row["ns_per_interaction"] > 0.0
    # the transfer must be small against the factorisation it precedes, which is the point
    # of the section: the CUDA lag is arithmetic, not PCIe
    n = "3220"
    transfer = probe["host_to_device"][n]["host_to_device_seconds"]
    factor = probe["kernels"]["batched_solve"][n]["cuda"][0]["lu_factor_seconds"]
    assert transfer < 0.5 * factor


def test_the_baseline_document_keeps_the_gpu_range_and_the_600x_caveat(probe):
    text = BASELINE_DOC.read_text(encoding="utf-8")
    assert "## 7b." in text
    assert "gpu_fp64_probe.json" in text
    # a single speedup would overstate a ratio that moved between runs
    assert "5-11x" in text
    # the kernel win is not the stage win
    assert "600x" in text
    assert probe["kernels"]["pairwise"]["cuda"]["ns_per_interaction"] < 3.0
