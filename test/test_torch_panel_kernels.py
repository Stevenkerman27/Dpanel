# Extracted from Stevenkerman27/GRASS-plane/test/test_panel3d_kernels.py
# Source commit: aa6d81faf03c556db475d9edec6c025f90eb5fac
# SPDX-License-Identifier: Apache-2.0
# Modifications: package imports only.
"""Analytic triangle influence checks; requires only torch and pytest."""
import math

import pytest
import torch

from torch_panel.kernels import triangle_influence


def _triangle(*, grad=False):
    return torch.tensor(
        [[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]],
        dtype=torch.float64,
        requires_grad=grad,
    )


def test_source_against_independent_surface_quadrature():
    tri = _triangle()
    point = torch.tensor([[0.2, 0.2, 0.2]], dtype=torch.float64)
    source, doublet = triangle_influence(point, tri)
    # Independent numerical reference (2D quadrature on a 0.5-area triangle).
    from scipy.integrate import dblquad
    integral, error = dblquad(
        lambda y, x: 1.0 / math.sqrt((x - 0.2) ** 2 + (y - 0.2) ** 2 + 0.2 ** 2),
        0, 1, lambda x: 0, lambda x: 1 - x,
        epsabs=1e-9, epsrel=1e-9,
    )
    assert source.item() == pytest.approx(integral / (4 * math.pi), abs=2e-9)
    assert doublet.item() == pytest.approx(0.2367761161685525, abs=1e-12)


def test_normal_reversal_and_diagonal_jump():
    tri = _triangle()
    field = torch.tensor([[0.2, 0.2, 0.2]], dtype=torch.float64)
    s, d = triangle_influence(field, tri)
    rev_s, rev_d = triangle_influence(field, tri[:, [0, 2, 1]])
    assert torch.allclose(s, rev_s, atol=1e-12)
    assert torch.allclose(d, -rev_d, atol=1e-12)
    center = tri.mean(dim=1)
    source, doublet = triangle_influence(center, tri, interior_diagonal=True)
    assert torch.isfinite(source).all()
    assert doublet.item() == -0.5


def test_triangle_kernel_gradcheck():
    t = _triangle(grad=True)
    p = torch.tensor([[0.25, 0.2, 0.3]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda pt, face: triangle_influence(pt, face),
        (p, t),
        eps=1e-6, atol=5e-6, rtol=5e-4,
    )
