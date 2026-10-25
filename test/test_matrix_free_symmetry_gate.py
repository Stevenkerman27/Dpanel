"""Check XZ reflection keeps upper TE on upper, lower TE on lower.

These tests exercise the experimental matrix_free mirror_gate, without
running GMRES or requiring CAD files.
"""
from types import SimpleNamespace

import numpy as np

from panel.matrix_free import mirror_gate


def _symmetric_banks():
    return SimpleNamespace(
        center=np.array([
            [.3, 1., .2], [.3, -1., .2],
            [.3, 1., -.2], [.3, -1., -.2],
        ]),
        nf=4,
        station_y=np.array([-1., 0., 1.]),
        upper=np.array([0, 1]), lower=np.array([2, 3]),
        n_te=2,
    )


def test_xz_mirror_maps_upper_to_upper_and_lower_to_lower():
    gate = mirror_gate(_symmetric_banks(), freestream=np.array([1., 0., 0.]))
    assert gate["status"] == "ok", gate["reasons"]
    assert gate["te_edges_paired"] == 2
    assert gate["station_involution"]


def test_mismatched_lower_te_bank_is_blocked():
    case = _symmetric_banks()
    case.lower = np.array([2, 2])
    gate = mirror_gate(case)
    assert gate["status"] == "blocked"
    assert gate["te_edges_paired"] < 2


def test_on_plane_self_nearest_wall_panel_is_not_half_reduced():
    case = _symmetric_banks()
    case.center = np.concatenate((case.center, [[.5, 0., .3]]), axis=0)
    case.nf = 5
    gate = mirror_gate(case)
    assert gate["status"] == "blocked"
    assert gate["self_panels"] >= 1
    assert any("self-mirror" in reason for reason in gate["reasons"])
