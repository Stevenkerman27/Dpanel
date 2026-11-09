"""Prescribed circulation must remain fixed while the wall potential is fitted."""
import numpy as np
import torch

from panel.p1_lifting_g1 import make_wing
from panel.p1_lifting_g2 import assemble_multi, combine_wings
from panel.p1_potential_jump_torch import affine_state


def test_prescribed_gamma_and_jump_are_both_satisfied():
    wing = make_wing(chord_panels=4, span_panels=6, wake_panels=2)
    mesh = combine_wings([wing])
    ops = assemble_multi(mesh)
    A, W, J = (torch.as_tensor(ops[k], dtype=torch.float64) for k in ("A", "W", "J"))
    nm, ng = A.shape[1], W.shape[1]
    weights = torch.sqrt(torch.as_tensor(ops["mesh"].areas) / np.mean(ops["mesh"].areas))
    B = weights[:, None] * torch.hstack((A, W))
    rhs = -weights * torch.as_tensor(ops["mesh"].centers)[:, 0]
    base, mapping = affine_state(B, rhs, J, nm, ng)
    gamma = torch.tensor([.08, .13, .16, .13, .08], dtype=torch.float64)
    state = torch.cat((base + mapping @ gamma, gamma))
    torch.testing.assert_close(state[nm:], gamma, rtol=0., atol=0.)
    torch.testing.assert_close(J @ state, torch.zeros(J.shape[0], dtype=J.dtype),
                               rtol=0., atol=1e-12)
    from panel.p1_rank_aware_torch import null_space
    basis = null_space(J[:, :nm])
    conditional_gradient = (B[:, :nm] @ basis).T @ (B @ state - rhs)
    assert float(conditional_gradient.abs().max()) < 1e-11


def test_legacy_fixed_gamma_driver_uses_the_same_feasible_state():
    from tools.issue71.fixed_gamma import solve_fixed_gamma
    wing = make_wing(chord_panels=3, span_panels=4, wake_panels=1)
    mesh = combine_wings([wing])
    ops = assemble_multi(mesh)
    gamma = np.array([.1, .15, .1])
    state = solve_fixed_gamma(mesh, ops, gamma)
    np.testing.assert_array_equal(state[ops["A"].shape[1]:], gamma)
    np.testing.assert_allclose(ops["J"] @ state, 0., atol=1e-12)
