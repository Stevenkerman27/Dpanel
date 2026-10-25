# Extracted from Stevenkerman27/GRASS-plane/test/test_panel3d_validation.py
# Source commit: aa6d81faf03c556db475d9edec6c025f90eb5fac
# SPDX-License-Identifier: Apache-2.0
# Modifications: package imports only.
"""Acceptance tests for the closed/oriented input contract of the Dirichlet solve."""
import pytest
import torch

from torch_panel.nonlifting import closed_body_potential, validate_closed_outward

VERTICES = torch.tensor(
    [[1.0, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=torch.float64,
)
FACES = torch.tensor(
    [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]],
    dtype=torch.long,
)
FREESTREAM = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)


def test_octahedron_is_accepted_and_symmetric():
    validate_closed_outward(VERTICES, FACES)
    points = torch.tensor([[2.0, 0, 0], [-2.0, 0, 0]], dtype=torch.float64)
    result = closed_body_potential(VERTICES, FACES, FREESTREAM, points)
    potential = result["perturbation_potential"]
    assert float(potential[0]) > 0.0
    assert torch.allclose(potential[0], -potential[1], atol=1e-12, rtol=1e-12)


def test_inward_winding_is_rejected():
    inward = FACES[:, [0, 2, 1]]
    with pytest.raises(ValueError, match="oriented inward"):
        validate_closed_outward(VERTICES, inward)
    with pytest.raises(ValueError, match="oriented inward"):
        closed_body_potential(VERTICES, inward, FREESTREAM)


def test_open_surface_is_rejected():
    with pytest.raises(ValueError, match="is not closed"):
        validate_closed_outward(VERTICES, torch.cat((FACES[:3], FACES[4:])))


def test_single_flipped_face_is_rejected():
    flipped = FACES.clone()
    flipped[3] = flipped[3][[0, 2, 1]]
    with pytest.raises(ValueError, match="inconsistently oriented"):
        validate_closed_outward(VERTICES, flipped)


def test_one_inward_component_among_two_is_rejected():
    # Two disjoint closed bodies: the second one wound inward must still be caught,
    # which is why the volume check is per component instead of on the total.
    shifted = VERTICES + torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)
    vertices = torch.cat((VERTICES, shifted))
    faces = torch.cat((FACES, FACES[:, [0, 2, 1]] + len(VERTICES)))
    with pytest.raises(ValueError, match="closed component 1 is oriented inward"):
        validate_closed_outward(vertices, faces)

    faces_outward = torch.cat((FACES, FACES + len(VERTICES)))
    validate_closed_outward(vertices, faces_outward)


def test_validation_keeps_geometry_gradients_alive():
    vertices = VERTICES.clone().requires_grad_(True)
    points = torch.tensor([[2.0, 0, 0]], dtype=torch.float64)
    potential = closed_body_potential(vertices, FACES, FREESTREAM, points)["perturbation_potential"]
    potential.sum().backward()
    assert vertices.grad is not None
    assert torch.isfinite(vertices.grad).all()
    assert float(vertices.grad.abs().sum()) > 0.0
