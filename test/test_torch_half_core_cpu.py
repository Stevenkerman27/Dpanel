"""Native Torch-only analytic kernel and implicit LU checks (no JAX/Gmsh)."""
import torch
from torch_panel.half_lifting import (
    ImplicitLUSolve, mirror_xz, triangle_potential_torch,
)


def test_analytic_kernel_diagonal_and_mirror_normal():
    tri = torch.tensor([[[0., .5, 0.], [1., .5, 0.], [.5, 1., 0.]]],
                       dtype=torch.float64, requires_grad=True)
    center = tri.mean(dim=1)
    single, doublet = triangle_potential_torch(
        center, tri, interior_diagonal=True)
    assert torch.isfinite(single).all()
    torch.testing.assert_close(doublet, doublet.new_tensor([[-.5]]),
                               atol=0, rtol=0)
    left = mirror_xz(tri)
    n1 = torch.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0], dim=-1)
    n2 = torch.cross(left[:, 1]-left[:, 0], left[:, 2]-left[:, 0], dim=-1)
    torch.testing.assert_close(n2, n1*n1.new_tensor([1., -1., 1.]),
                               atol=1e-14, rtol=1e-14)
    sample = center.detach()+center.new_tensor([[.1, .05, .4]])
    source, doublet = triangle_potential_torch(sample, tri)
    assert torch.isfinite(source).all() and torch.isfinite(doublet).all()


def test_lu_implicit_vjp_vs_native_nonsymmetric_system():
    torch.manual_seed(123)
    A = torch.randn(9, 9, dtype=torch.float64)
    A = A+4*torch.eye(9, dtype=torch.float64)
    b = torch.randn(9, dtype=torch.float64)
    seed = torch.randn(9, dtype=torch.float64)
    a, rhs = A.clone().requires_grad_(), b.clone().requires_grad_()
    state = ImplicitLUSolve.apply(a, rhs)
    da, db = torch.autograd.grad(seed@state, (a, rhs))
    a_ref, b_ref = A.clone().requires_grad_(), b.clone().requires_grad_()
    ref = torch.linalg.solve(a_ref, b_ref)
    da_ref, db_ref = torch.autograd.grad(seed@ref, (a_ref, b_ref))
    torch.testing.assert_close(state, ref, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(da, da_ref, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(db, db_ref, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(A.T@db, seed, atol=1e-11, rtol=1e-11)


def test_triangle_offsurface_geometry_gradient_finite_and_fd():
    tri = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]],
                       dtype=torch.float64)
    point = torch.tensor([[.3, .2, .4]], dtype=torch.float64,
                         requires_grad=True)
    def value(p):
        S, D = triangle_potential_torch(p, tri)
        return S.sum()+.31*D.sum()
    grad, = torch.autograd.grad(value(point), point)
    assert torch.isfinite(grad).all() and grad.abs().max() > 1e-8
    direction = torch.tensor([[.1, -.2, .3]], dtype=torch.float64)
    h = 1e-5
    fd = (value(point.detach()+h*direction)
          -value(point.detach()-h*direction))/(2*h)
    torch.testing.assert_close((grad*direction).sum(), fd,
                               atol=1e-8, rtol=1e-6)
