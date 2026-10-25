"""Independent NumPy-only tests for `panel.symmetry`.

These fixtures are hand-built so every failure class is isolated: an exactly
mirrored surface (must pass), a surface whose mirrored vertex *sets* exist as
faces while the geometry does not mirror (must not be counted as paired), a
trailing-edge bank swap (the bug shape of the old gate), a centreline
straddling triangle and a true on-plane cap.
"""
import numpy as np
import pytest

from panel import symmetry


def plate(y_left=-1.0, upper=(0, 2), lower=(1, 3)):
    """Minimal lifting surface: two panels per half, mirrored across y=0.

    Right half: A=(0,1,0) B=(1,1,0) C=(0,1,0.1) D=(1,1,0.1); lower bank panel
    (A,B,D), upper bank panel (A,D,C); trailing edge A-D. The left half holds
    the reflections A'..D' at y=y_left with reversed winding, plus one mirrored
    wake triangle per half behind the trailing edge.
    """
    x = [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.1], [1.0, 1.0, 0.1]]
    ym = np.asarray(x) * np.array([1.0, y_left, 1.0])
    nodes = np.asarray(x + ym.tolist(), dtype=np.float64)
    wake = np.asarray([[0.2, 1.0, 0.0], [1.2, 1.0, 0.0], [0.2, 1.0, 0.1]])
    wake = np.vstack([nodes, wake, wake * np.array([1.0, y_left, 1.0])])
    return dict(
        schema="dpanel.planar_cut.v1",
        coordinates=wake,
        faces=np.asarray([[0, 3, 2], [0, 1, 3], [6, 7, 4], [7, 5, 4]], dtype=np.int64),
        wake_faces=np.asarray([[8, 9, 10], [13, 12, 11]], dtype=np.int64),
        gamma_stations=np.asarray([1.0, -1.0]),
        trailing_edges=np.asarray([[0, 3], [4, 7]], dtype=np.int64),
        upper_te_faces=np.asarray(list(upper), dtype=np.int64),
        lower_te_faces=np.asarray(list(lower), dtype=np.int64),
    )


def test_exactly_mirrored_plate_passes_every_layer():
    report = symmetry.analyze_surface(plate())
    assert report["status"] == "ok", report["blocking_layers"]
    nodes, faces, wake = report["nodes"], report["faces"], report["wake"]
    assert nodes["bijective"] and nodes["involution"] and nodes["unpaired"] == 0
    assert faces["paired"] == 4 and faces["winding_reversed"] == 4
    assert faces["spurious_vertex_set_hits"] == 0
    assert wake["paired"] == 2 and wake["involution"]
    te = report["trailing_edge"]
    assert te["te_edges_geometrically_mirrored"] == 2 and te["bank_mismatches"] == 0
    assert report["stations"]["involution"] and report["stations"]["paired"] == 2


def test_shifted_half_is_not_paired_even_though_vertex_sets_match():
    # y = -1.05 instead of -1.0: the mirrored vertex *sets* still exist as faces,
    # but no triangle is a geometric mirror. Set membership alone would report
    # 4/4 paired wall faces.
    report = symmetry.analyze_surface(plate(y_left=-1.05))
    faces = report["faces"]
    assert report["status"] == "blocked"
    assert faces["vertex_set_hits"] > 0
    assert faces["spurious_vertex_set_hits"] == faces["vertex_set_hits"]
    assert faces["paired"] == 0
    assert faces["worst_spurious_distance"] > 1e-3


def test_swapped_trailing_edge_banks_are_rejected():
    # mirror upper -> partner lower and mirror lower -> partner upper: the bug
    # shape of the old gate, which compared mirror(upper) against set(lower).
    report = symmetry.analyze_surface(plate(upper=(0, 3), lower=(1, 2)))
    te = report["trailing_edge"]
    assert report["status"] == "blocked"
    assert te["te_edges_geometrically_mirrored"] == 2
    assert te["te_edges_paired"] == 0 and te["bank_mismatches"] == 2
    assert "trailing_edge" in report["blocking_layers"]


def test_straddling_triangle_is_reported_separately_from_a_plane_cap():
    straddle = dict(
        schema="dpanel.planar_cut.v1",
        coordinates=np.asarray([[0.0, 0.5, 0.0], [1.0, 0.5, 0.0], [0.5, -0.5, 0.2]]),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
        wake_faces=np.zeros((0, 3), dtype=np.int64),
        gamma_stations=np.asarray([0.5, -0.5]),
        trailing_edges=np.asarray([[0, 1]], dtype=np.int64),
        upper_te_faces=np.asarray([0], dtype=np.int64),
        lower_te_faces=np.asarray([0], dtype=np.int64),
    )
    report = symmetry.analyze_surface(straddle)
    assert report["geometry"]["straddle_faces"] == 1
    assert report["geometry"]["plane_faces"] == 0
    assert not report["layers"]["no_straddle_faces"]

    cap = dict(straddle)
    cap["coordinates"] = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.2]])
    report = symmetry.analyze_surface(cap)
    assert report["geometry"]["plane_faces"] == 1
    assert report["geometry"]["straddle_faces"] == 0
    assert report["geometry"]["centroid_near_plane"] == 1
    assert not report["layers"]["no_plane_faces"]


def test_wake_pairing_is_tolerance_dependent_on_an_off_by_1e_3_mesh():
    surface = plate(y_left=-1.001)
    loose = symmetry.analyze_surface(surface, tol=1e-2)
    tight = symmetry.analyze_surface(surface, tol=1e-6)
    assert tight["nodes"]["unpaired"] > 0
    assert loose["wake"]["paired"] > tight["wake"]["paired"]
    assert tight["status"] == "blocked"
