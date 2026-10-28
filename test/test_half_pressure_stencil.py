"""Pressure LS rank recovery without reconnecting a physical wake/contact cut."""
import numpy as np
import pytest

from panel.pressure_stencil import adaptive_pressure_neighbors


def synthetic_line_to_surface():
    # A thin TE strip: at two rings face 0 can only see centers on one
    # spanwise line. A legal four-ring path reaches the second direction.
    centers = np.array([
        [0., 0., 0.], [1., 0., 0.], [2., 0., 0.],
        [3., 0., 0.], [3., 1., 0.], [4., 0., 0.],
    ])
    offsets = np.array([[-.04, -.04, 0.], [.08, -.04, 0.],
                        [-.04, .08, 0.]])
    triangles = centers[:, None]+offsets[None]
    adjacent = [
        {1}, {0, 2}, {1, 3}, {2, 4, 5}, {3}, {3}
    ]
    return centers, triangles, adjacent


def test_cut_preserving_rank_expansion_recovers_collinear_face():
    centers, tri, graph = synthetic_line_to_surface()
    neighbor, audit = adaptive_pressure_neighbors(
        centers, tri, graph, 1, initial_rings=2,
        max_rings=6, min_singular_ratio=1e-6)
    assert audit["initial_rank_deficient_faces"] == [0]
    assert audit["expanded_faces"] == [0]
    assert audit["selected_ring_depth"] == [4]
    assert audit["initial_singular_ratios"] == [0.]
    assert audit["final_singular_ratios"][0] > 1e-6
    np.testing.assert_array_equal(neighbor[0], [1, 2, 3, 4, 5])


def test_cut_isolates_one_dimensional_strip_and_fails_closed():
    centers, tri, graph = synthetic_line_to_surface()
    # The 3-4 connection is a physical contact cut; all remaining centers
    # reachable from face 0 are on the same line even at depth six.
    graph[3].remove(4)
    graph[4].remove(3)
    with pytest.raises(ValueError, match=r"face=0.*last_ratio=0.000e\+00"):
        adaptive_pressure_neighbors(
            centers, tri, graph, 1, initial_rings=2,
            max_rings=6, min_singular_ratio=1e-6)


def test_no_global_expansion_when_original_two_rings_are_full_rank():
    centers, tri, graph = synthetic_line_to_surface()
    centers[2, 1] = .15
    tri[2, :, 1] += .15
    neighbors, audit = adaptive_pressure_neighbors(
        centers, tri, graph, 1, initial_rings=2, max_rings=6)
    assert audit["expanded_faces"] == []
    assert audit["selected_ring_depth"] == [2]
    np.testing.assert_array_equal(neighbors[0], [1, 2])


def test_rank_tolerance_cannot_be_replaced_by_zero_or_pseudoinverse():
    centers, tri, graph = synthetic_line_to_surface()
    with pytest.raises(ValueError, match="configuration"):
        adaptive_pressure_neighbors(
            centers, tri, graph, 1, initial_rings=2,
            max_rings=4, min_singular_ratio=0)
    with pytest.raises(ValueError, match="max_rings=2"):
        adaptive_pressure_neighbors(
            centers, tri, graph, 1, initial_rings=2,
            max_rings=2, min_singular_ratio=1e-6)


def test_original_small_half_wing_stencils_need_no_adaptive_changes():
    from panel.run_half_native import load_half_surface
    from pathlib import Path
    source = Path(__file__).resolve().parents[1]/(
        "cases/panel_half/thick_wing_two_strip.json")
    baseline = load_half_surface(
        source, stencil_rings=2, max_stencil_rings=2)
    adaptive, report = load_half_surface(
        source, stencil_rings=2, max_stencil_rings=6,
        return_stencil_audit=True)
    np.testing.assert_array_equal(
        baseline["topology"].neighbor_ghost_ids,
        adaptive["topology"].neighbor_ghost_ids)
    assert not report["expanded_faces"]
    assert report["initial_stencil_rings"] == 2
    assert report["maximum_stencil_rings"] == 6


def test_expanding_one_face_keeps_all_other_two_ring_neighbors():
    # The adaptive row needs an extra BFS shell, but the other row must not
    # acquire any new source panels; only self-padding aligns JAX static width.
    centers = np.array([
        [0., 0., 0.], [10., 0., 0.], [1., 0., 0.],
        [2., 0., 0.], [2., 1., 0.], [3., 0., 0.],
        [11., 0., 0.], [10., 1., 0.],
    ])
    offsets = np.array([[-.04, -.04, 0.], [.08, -.04, 0.],
                        [-.04, .08, 0.]])
    tri = centers[:, None]+offsets[None]
    adjacent = [
        {2}, {6, 7}, {0, 3}, {2, 4, 5},
        {3}, {3}, {1}, {1},
    ]
    indices, info = adaptive_pressure_neighbors(
        centers, tri, adjacent, 2, initial_rings=2,
        max_rings=6, min_singular_ratio=1e-6)
    assert info["expanded_faces"] == [0]
    assert info["selected_ring_depth"] == [3, 2]
    np.testing.assert_array_equal(indices[0], [2, 3, 4, 5])
    np.testing.assert_array_equal(indices[1], [6, 7, 1, 1])
