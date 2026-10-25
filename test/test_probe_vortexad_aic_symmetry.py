"""Self-tests for VortexAD mirror mismatch decomposition without VortexAD/JAX."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

file = Path(__file__).with_name("probe_vortexad_aic_symmetry.py")
spec = importlib.util.spec_from_file_location("probe_aic_symmetry", file)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def _toy():
    centers = np.array([
        [0.1, 1.0, 0.3], [0.1, -1.0, 0.3],
        [0.6, 1.5, 0.2], [0.6, -1.5, 0.2],
    ])
    normals = np.array([
        [0., 0.6, 0.8], [0., -0.6, 0.8],
        [0., 0., 1.], [0., 0., 1.],
    ])
    return centers, normals


def test_mirror_aic_symmetry_report_detects_diagonal_corruption():
    centers, normals = _toy()
    mirror, meta = audit.mirror_pairs(centers, normals)
    np.testing.assert_array_equal(mirror, [1, 0, 3, 2])
    assert meta["half_count"] == 2
    baseline = np.array([
        [2., 0.3, 0.1, 0.02],
        [0.3, 2., 0.02, 0.1],
        [0.1, 0.02, 3., 0.5],
        [0.02, 0.1, 0.5, 3.],
    ])
    result = audit.matrix_report(baseline, mirror)
    assert result["max_abs_delta"] == pytest.approx(0.0)
    broken = baseline.copy()
    broken[0, 0] += 4e-6
    r = audit.matrix_report(broken, mirror)
    assert r["max_abs_delta"] == pytest.approx(4e-6)
    assert r["max_diagonal_abs_delta"] == pytest.approx(4e-6)
    assert r["nonzero_gt_1e_12"] == 2
    assert audit.vector_report([1., 1., 2., 2.], mirror)["max_abs_delta"] == 0.0


def test_mirror_pairing_rejects_non_symmetric_normals():
    centers, normals = _toy()
    normals[1] = [0., 0., 1.]
    with pytest.raises(ValueError, match="normal mismatch"):
        audit.mirror_pairs(centers, normals)
