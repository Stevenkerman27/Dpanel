"""Opt-in Torch kernel checkpoint must preserve analytic fields and VJP.

No claim of CUDA acceleration or lower peak memory is made by this test.
Performance must be measured in distinct GPU processes on the SAME fixture.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from torch_panel.half_lifting import (
    triangle_potential_torch, solve_half_torch, topology_to_torch,
)

jax = pytest.importorskip("jax")
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from panel.run_half_native import load_half_surface


SMALL = Path(__file__).resolve().parents[1]/(
    "cases/panel_half/thick_wing_two_strip.json")


def case(device="cpu"):
    reference = load_half_surface(SMALL)
    values = {"topology": topology_to_torch(reference["topology"], device=device)}
    for key in ("vertices", "station_y", "freestream", "density",
                "reference_area", "reference_chord", "reference_point"):
        values[key] = torch.tensor(np.asarray(reference[key]),
                                   dtype=torch.float64, device=device)
    return values


def full_loss(out):
    return (out["CL"]+.37*out["CM"][1]
            +.01*out["panel_forces_half"][:, 0].sum()
            +.002*out["Cp_raw_half"].square().sum())


def compare_half(checkpoint, device):
    inputs = case(device)
    x = inputs["vertices"].detach().clone().requires_grad_()
    y = inputs["station_y"].detach().clone().requires_grad_()
    f = inputs["freestream"].detach().clone().requires_grad_()
    inputs.update(vertices=x, station_y=y, freestream=f)
    out = solve_half_torch(
        **inputs, block_size=7, checkpoint_kernel=checkpoint)
    grads = torch.autograd.grad(full_loss(out), (x, y, f))
    return out, grads


def test_half_lu_checkpoint_preserves_fields_and_full_load_gradients():
    original, gold = compare_half(False, "cpu")
    restored, actual = compare_half(True, "cpu")
    for key in ("matrix_half", "rhs_half", "mu_half", "gamma_half",
                "wake_basis_half", "wake_slope_half", "te_basis_half",
                "Cp_raw_half", "panel_forces_half", "panel_moments_half",
                "component_forces", "component_moments",
                "force", "moment", "CL", "CD_pressure", "CM"):
        torch.testing.assert_close(
            restored[key], original[key], rtol=2e-11, atol=2e-12,
            msg=lambda message, key=key: f"{key}: {message}")
    for a, b in zip(actual, gold):
        torch.testing.assert_close(a, b, rtol=3e-9, atol=2e-9)
        assert torch.isfinite(a).all()


def test_checkpointed_rectangular_self_trace_does_not_swap_row_ids():
    # 5 target collocation points, block=2 means a final 1-row block.
    # Source panels are deliberately reordered to exercise global row IDs.
    xyz = torch.tensor([
        [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
        [[2., 0., 0.], [3., 0., 0.], [2., 1., 0.]],
        [[4., 0., 0.], [5., 0., 0.], [4., 1., 0.]],
        [[6., 0., 0.], [7., 0., 0.], [6., 1., 0.]],
        [[8., 0., 0.], [9., 0., 0.], [8., 1., 0.]],
    ], dtype=torch.float64)
    triangles = xyz.clone().requires_grad_()
    target = xyz.mean(1).detach().clone().requires_grad_()
    reference, D = triangle_potential_torch(
        target, triangles, interior_diagonal=True, block_size=2)
    got, D2 = triangle_potential_torch(
        target, triangles, interior_diagonal=True, block_size=2,
        checkpoint_blocks=True)
    torch.testing.assert_close(D2, D, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(torch.diag(D2), D.new_full((5,), -.5),
                               atol=0, rtol=0)
    # Use a smooth off-diagonal objective: exact self terms are constants.
    weight = D.new_tensor(np.arange(25).reshape(5,5)/37.)
    g1 = torch.autograd.grad((D*weight).sum()+reference.sum(),
                             (target, triangles), retain_graph=True)
    g2 = torch.autograd.grad((D2*weight).sum()+got.sum(),
                             (target, triangles))
    for a, b in zip(g1, g2):
        torch.testing.assert_close(a, b, rtol=5e-9, atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real CUDA device required")
def test_cuda_checkpoint_gradient_parity_without_cpu_fallback():
    original, grads1 = compare_half(False, "cuda")
    restored, grads2 = compare_half(True, "cuda")
    torch.testing.assert_close(restored["CL"], original["CL"],
                               rtol=2e-8, atol=2e-9)
    for a, b in zip(grads1, grads2):
        torch.testing.assert_close(a, b, rtol=5e-8, atol=1e-8)
