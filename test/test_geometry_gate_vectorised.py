"""The vectorised bounding-box pass must decide exactly what _bounds_overlap did.

The coupled iteration used to test every wake/wall pair in Python (5.85M calls
on the G2 CAD body, 86% of an iteration). The replacement compares all pairs at
once, so its predicate is checked here against the reference function on random
boxes, including the tolerance cases that sit exactly on the boundary.
"""
import itertools

import numpy as np
import pytest

from panel.p1_coupled_material_wake import _bounds_candidates, _bounds_overlap


def _boxes(rng, count, scale=1.0):
    lower = rng.normal(scale=scale, size=(count, 3))
    extent = rng.random((count, 3))*scale
    return lower, lower+extent


@pytest.mark.parametrize("seed", range(4))
def test_mask_matches_the_reference_predicate(seed):
    rng = np.random.default_rng(seed)
    first_lower, first_upper = _boxes(rng, 40)
    second_lower, second_upper = _boxes(rng, 37)
    mask = _bounds_candidates(first_lower, first_upper, second_lower, second_upper, 0.0)
    for i, j in itertools.product(range(len(first_lower)), range(len(second_lower))):
        expected = _bounds_overlap(np.stack((first_lower[i], first_upper[i])),
                                   np.stack((second_lower[j], second_upper[j])), 0.0)
        assert bool(mask[i, j]) is expected


def test_tolerance_behaves_like_the_reference():
    lower = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    upper = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    gap = np.array([[1.5, 0.0, 0.0]])
    for tol in (0.0, 0.25, 0.49, 0.5, 0.51, 1.0):
        mask = _bounds_candidates(lower, upper, gap, gap+1.0, tol)
        expected = _bounds_overlap(np.stack((lower[0], upper[0])),
                                   np.stack((gap[0], gap[0]+1.0)), tol)
        assert bool(mask[0, 0]) is expected
        assert bool(mask[1, 0]) is expected


def test_a_pair_that_shares_only_a_corner_is_reported():
    # Separation on one axis is enough to be disjoint, so boxes that touch only
    # at a corner must be reported (the reference predicate does the same).
    first_lower, first_upper = np.zeros((1, 3)), np.ones((1, 3))
    second_lower, second_upper = np.ones((1, 3)), 2*np.ones((1, 3))
    assert bool(_bounds_candidates(first_lower, first_upper, second_lower, second_upper,
                                  0.0)[0, 0]) is True


def test_degenerate_and_single_box_inputs():
    point = np.zeros((1, 3))
    assert _bounds_candidates(point, point, point, point, 0.0).all()
    far = np.full((1, 3), 5.0)
    assert not _bounds_candidates(point, point, far, far, 0.0).any()


def test_scan_order_is_row_major():
    # The gate counts pairs in ascending (i, j) order; a mask read row by row
    # must therefore return ascending j for each i.
    lower, upper = _boxes(np.random.default_rng(11), 12)
    mask = _bounds_candidates(lower, upper, lower, upper, 0.0)
    for i in range(len(mask)):
        indices = np.nonzero(mask[i])[0]
        assert np.all(np.diff(indices) > 0)
        assert np.all(indices >= 0)
