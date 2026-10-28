"""Adapter contract for the half-CAD G1 loader (issue #58 TE pairing defect)."""
import json
import os
from pathlib import Path
import numpy as np
import pytest

from panel.p1_lifting_g1 import from_dpanel_half_surface


HALF = Path(os.environ.get(
    "DPANEL_HALF_SURFACE",
    str(Path(__file__).resolve().parents[2] / "runs" / "half_cad_wingbody_v1" / "surface.json"),
))


pytestmark = pytest.mark.skipif(not HALF.exists(), reason="half-CAD fixture not available")


def test_te_pairs_use_upper_lower_faces_and_split_only_potential():
    data = json.loads(HALF.read_text(encoding="utf-8"))
    wing = from_dpanel_half_surface(str(HALF))
    up = np.asarray(data["upper_te_faces"], int)
    # One TE row per shared TE node, both banks, and tips excluded.
    assert len(wing.te_pairs) >= 2 * len(up) - 2
    assert np.all(wing.te_pairs[:, 0] != wing.te_pairs[:, 1])
    assert np.all(np.diff(wing.te_pair_y) >= 0.0)
    assert wing.te_pair_y.min() > wing.gamma_y.min()
    assert wing.te_pair_y.max() < wing.gamma_y.max()


def test_te_nodes_span_both_banks_and_cover_the_declared_span():
    wing = from_dpanel_half_surface(str(HALF))
    ys = wing.te_pair_y
    assert ys.min() < 0.0 < ys.max(), "mirrored TE must straddle the root"
    assert np.isclose(ys.min(), -ys.max(), atol=1e-9), "mirroring must be symmetric"
    # The TE rows must reach the wing span, not stop at the body junction.
    assert ys.max() > 0.9 * wing.gamma_y.max()


def test_duplicated_trace_ids_are_new_and_unique_per_node():
    wing = from_dpanel_half_surface(str(HALF))
    lower = wing.te_pairs[:, 1]
    assert len(np.unique(lower)) == len(lower)
    assert lower.min() >= len(np.unique(wing.dof_face[wing.face_bank == 1]))


def test_faces_must_share_a_te_node():
    data = json.loads(HALF.read_text(encoding="utf-8"))
    broken = dict(data)
    broken["lower_te_faces"] = np.roll(np.asarray(data["lower_te_faces"], int), 1).tolist()
    path = Path(__file__).resolve().parent / "broken_half_surface.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="share no node"):
            from_dpanel_half_surface(str(path))
    finally:
        path.unlink(missing_ok=True)