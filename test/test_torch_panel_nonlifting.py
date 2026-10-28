# Extracted from Stevenkerman27/GRASS-plane/test/test_panel3d_nonlifting.py
# Source commit: aa6d81faf03c556db475d9edec6c025f90eb5fac
# SPDX-License-Identifier: Apache-2.0
# Modifications: package imports only.
"""Nonlifting closed body benchmark: symmetric octahedron and gradients."""
import torch

from torch_panel.nonlifting import closed_body_potential


def test_octahedron_potential_and_geometry_gradient():
    vertices = torch.tensor(
        [[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.],
         [0., 0., 1.], [0., 0., -1.]],
        dtype=torch.float64, requires_grad=True,
    )
    faces = torch.tensor(
        [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
         [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]],
        dtype=torch.long,
    )
    evaluation_points = torch.tensor([[2., 0., 0.], [-2., 0., 0.]], dtype=torch.float64)
    output = closed_body_potential(
        vertices, faces, vertices.new_tensor([1., 0., 0.]), evaluation_points
    )
    assert torch.max(torch.abs(output["residual"])) < 1e-12
    potential = output["perturbation_potential"]
    assert potential[0] > 0
    assert torch.allclose(potential[0], -potential[1], atol=1e-12)
    (potential[0].square()).backward()
    assert torch.isfinite(vertices.grad).all()
    assert vertices.grad.abs().sum() > 0
