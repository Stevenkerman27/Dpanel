# Extracted from Stevenkerman27/GRASS-plane/test/test_panel3d_symmetry.py
# Source commit: aa6d81faf03c556db475d9edec6c025f90eb5fac
# SPDX-License-Identifier: Apache-2.0
# Modifications: package imports only.
"""XZ half-model image method equivalence and geometry derivatives."""
import torch
from torch_panel.kernels import triangle_influence
from torch_panel.symmetry import mirror_xz_triangles, symmetric_triangle_influence


def test_mirrored_half_octrahedron_matches_full_potential():
    """A closed octahedron cut on y=0 has an exact four-triangle half-model."""
    vertices = torch.tensor(
        [[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.],
         [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]],
        dtype=torch.float64, requires_grad=True,
    )
    faces = torch.tensor(
        [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
         [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]],
        dtype=torch.long,
    )
    faces_half = faces[[0, 1, 4, 5]]
    tri_half = vertices[faces_half]
    tri_full = vertices[faces]
    pts = vertices.new_tensor([[2.2, 0.4, 0.3], [1.8, -0.7, 0.8]])
    vel = vertices.new_tensor([1., 0., 0.])
    centres_half = tri_half.mean(1)
    centre_full = tri_full.mean(1)
    normals_half = torch.cross(tri_half[:, 1] - tri_half[:, 0],
                               tri_half[:, 2] - tri_half[:, 0], dim=-1)
    normals_half = torch.nn.functional.normalize(normals_half, dim=-1)
    normals_full = torch.cross(tri_full[:, 1] - tri_full[:, 0],
                               tri_full[:, 2] - tri_full[:, 0], dim=-1)
    normals_full = torch.nn.functional.normalize(normals_full, dim=-1)
    s_half = -(normals_half * vel).sum(-1)
    s_full = -(normals_full * vel).sum(-1)

    bh, dh = symmetric_triangle_influence(centres_half, tri_half, interior_diagonal=True)
    bf, df = triangle_influence(centre_full, tri_full, interior_diagonal=True)
    mh = torch.linalg.solve(dh, bh @ s_half)
    mf = torch.linalg.solve(df, bf @ s_full)
    ps, pd = symmetric_triangle_influence(pts, tri_half)
    fs, fd = triangle_influence(pts, tri_full)
    phi_half = pd @ mh - ps @ s_half
    phi_full = fd @ mf - fs @ s_full
    assert torch.allclose(phi_half, phi_full, atol=1e-12, rtol=1e-12)
    assert (dh @ mh - bh @ s_half).abs().max() < 1e-12
    phi_half.square().sum().backward()
    assert torch.isfinite(vertices.grad).all()
    assert vertices.grad.abs().sum() > 0


def test_mirror_reverses_winding_and_reproduces_reflected_normal():
    triangle = torch.tensor(
        [[[0.1, 0.1, 0.], [0.8, 0.3, 0.], [0.4, 0.2, 0.9]]],
        dtype=torch.float64,
    )
    twin = mirror_xz_triangles(triangle)
    def normals(tri):
        return torch.nn.functional.normalize(
            torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1), dim=-1
        )
    expected = normals(triangle) * triangle.new_tensor([1., -1., 1.])
    assert torch.allclose(normals(twin), expected, atol=1e-14)


def test_only_real_diagonal_gets_jump():
    tri = torch.tensor(
        [[[0., 0.3, 0.], [1., 0.3, 0.], [0., 0.9, 0.]]], dtype=torch.float64
    )
    p = tri.mean(1)
    _, direct = triangle_influence(p, tri, interior_diagonal=True)
    _, mirror = triangle_influence(p, mirror_xz_triangles(tri))
    _, sym = symmetric_triangle_influence(p, tri, interior_diagonal=True)
    assert torch.allclose(sym, direct + mirror, atol=1e-14)


def test_symmetry_plane_cap_is_rejected_instead_of_cancelling():
    import pytest
    triangle = torch.tensor(
        [[[0., 0., 0.], [1., 0., 0.], [0., 0., 1.]]],
        dtype=torch.float64,
    )
    point = torch.tensor([[0.3, 0.01, 0.3]], dtype=torch.float64)
    with pytest.raises(ValueError, match="symmetry-plane caps"):
        symmetric_triangle_influence(point, triangle)


def test_edge_on_plane_but_surface_off_plane_is_allowed():
    triangle = torch.tensor(
        [[[0., 0., 0.], [1., 0., 0.], [0., 0.6, 1.]]],
        dtype=torch.float64,
    )
    point = torch.tensor([[0.3, 0.2, 0.4]], dtype=torch.float64)
    source, doublet = symmetric_triangle_influence(point, triangle)
    assert torch.isfinite(source).all()
    assert torch.isfinite(doublet).all()
