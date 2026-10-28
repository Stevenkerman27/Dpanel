"""Float64 half-native Torch lifting parity/implicit-adjoint regression.

Every JAX comparison uses the SAME frozen topology and geometry. Small
floating numerical differences are acceptable; source and Kutta conventions
and the derivatives must be physically and algebraically consistent.
CUDA is only tested when a real CUDA device is available.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from panel.run_half_native import load_half_surface
from panel.half_native import solve_half_fields
from torch_panel.half_lifting import (
    ImplicitLUSolve, solve_half_torch, topology_to_torch,
)

SAMPLE = Path(__file__).resolve().parents[1] / (
    "cases/panel_half/thick_wing_two_strip.json")


def case(device="cpu"):
    baseline = load_half_surface(SAMPLE)
    t = topology_to_torch(baseline["topology"], device=device)
    kwargs = dict(topology=t)
    for key in ("vertices", "station_y", "freestream", "density",
                "reference_area", "reference_chord", "reference_point"):
        kwargs[key] = torch.tensor(np.asarray(baseline[key]),
                                   dtype=torch.float64, device=device)
    return baseline, kwargs


def compare_fields(actual, reference, *, rtol=2e-6, atol=2e-8):
    fields = (
        "mu_half", "gamma_half", "sigma_half", "matrix_half", "rhs_half",
        "wake_basis_half", "wake_slope_half", "te_basis_half",
        "Cp_raw_half", "surface_velocity_half", "panel_normals_half",
        "panel_areas_half", "panel_forces_half", "panel_moments_half",
        "component_forces", "component_moments", "force", "moment",
        "CL", "CD_pressure", "CM", "tangent_determinant",
    )
    errors = {}
    for key in fields:
        a = actual[key].detach().cpu().numpy()
        b = np.asarray(reference[key])
        delta = a-b
        errors[key] = dict(
            max_abs=float(np.max(np.abs(delta))),
            relative_l2=float(np.linalg.norm(delta) /
                              max(np.linalg.norm(b), 1e-300)))
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol,
                                   err_msg=f"Field {key}")
    return errors


def objective(data):
    return (data["CL"]+.37*data["CM"][1]
            +.01*data["panel_forces_half"][:, 0].sum())


def test_torch_half_float64_fields_against_actual_jax_reference():
    jax_input, torch_input = case()
    actual = solve_half_torch(**torch_input, block_size=7)
    reference = solve_half_fields(**jax_input)
    diffs = compare_fields(actual, reference)
    assert max(d["relative_l2"] for d in diffs.values()) < 2e-5
    residual = actual["linear_residual_half"].detach().norm().item()
    norm = actual["rhs_half"].detach().norm().item()
    assert residual/max(norm, 1e-300) < 1e-11


def test_explicit_implicit_lu_adjoint_matches_torch_native():
    rng = np.random.default_rng(20261026)
    matrix = torch.tensor(rng.normal(size=(7, 7)), dtype=torch.float64)
    matrix += 4*torch.eye(7, dtype=torch.float64)
    rhs = torch.tensor(rng.normal(size=(7,)), dtype=torch.float64)
    seed = torch.tensor(rng.normal(size=(7,)), dtype=torch.float64)
    A1, b1 = matrix.clone().requires_grad_(), rhs.clone().requires_grad_()
    x1 = ImplicitLUSolve.apply(A1, b1)
    da, db = torch.autograd.grad(seed@x1, (A1, b1))
    A2, b2 = matrix.clone().requires_grad_(), rhs.clone().requires_grad_()
    x2 = torch.linalg.solve(A2, b2)
    da_ref, db_ref = torch.autograd.grad(seed@x2, (A2, b2))
    torch.testing.assert_close(x1, x2, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(da, da_ref, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(db, db_ref, atol=1e-11, rtol=1e-11)
    # A.T must be used because this augmented panel system is nonsymmetric.
    assert torch.linalg.vector_norm(matrix@db-seed) > 1e-3
    torch.testing.assert_close(matrix.T@db, seed, atol=1e-11, rtol=1e-11)


def test_torch_half_geometry_and_station_adjoint_vs_jax_and_fd():
    jax_input, args = case()
    x, y = args["vertices"], args["station_y"]
    dx = torch.stack((.01*x[:, 0], .02*x[:, 1], .01*x[:, 2]), dim=1)
    dy = .02*y
    theta = torch.tensor(0., dtype=torch.float64, requires_grad=True)

    def torch_loss(scale, implicit=True):
        return objective(solve_half_torch(
            **{**args, "vertices": x+scale*dx,
               "station_y": y+scale*dy}, block_size=7,
            implicit_adjoint=implicit))

    implicit = torch.autograd.grad(torch_loss(theta), theta)[0].item()
    torch_native = torch.autograd.grad(
        torch_loss(theta, implicit=False), theta)[0].item()
    np.testing.assert_allclose(implicit, torch_native, rtol=1e-8, atol=1e-9)

    xj, yj = jax_input["vertices"], jax_input["station_y"]
    dxj, dyj = jnp.asarray(dx.detach().numpy()), jnp.asarray(dy.detach().numpy())

    def jax_loss(scale):
        return objective(solve_half_fields(
            **{**jax_input, "vertices": xj+scale*dxj,
               "station_y": yj+scale*dyj}))

    jax_grad = float(jax.grad(jax_loss)(jnp.array(0.)))
    np.testing.assert_allclose(implicit, jax_grad, rtol=1e-5, atol=1e-8)
    for h in (1e-3, 1e-4, 1e-5):
        fd = float((torch_loss(h)-torch_loss(-h))/(2*h))
        np.testing.assert_allclose(implicit, fd, rtol=1e-5, atol=1e-8)


def test_torch_half_derivatives_include_refs_and_local_loads():
    _, inp = case()
    variables = ("vertices", "station_y", "freestream", "density",
                 "reference_area", "reference_chord", "reference_point")
    for key in variables:
        inp[key] = inp[key].clone().detach().requires_grad_()
    out = solve_half_torch(**inp, block_size=7)
    weights = torch.linspace(-.2, .4, len(out["panel_forces_half"]),
                             dtype=torch.float64)
    loss = objective(out)+.01*(weights[:, None]*out["panel_forces_half"]).sum()
    grads = torch.autograd.grad(loss, tuple(inp[k] for k in variables))
    for key, grad in zip(variables, grads):
        assert grad is not None and torch.isfinite(grad).all(), key
    assert grads[0].norm().item() > 1e-6
    assert grads[1].norm().item() > 1e-6


def test_half_rejects_sideslip_and_offplane_reference():
    _, args = case()
    args["freestream"] = torch.tensor([1., .1, .08], dtype=torch.float64)
    with pytest.raises(ValueError, match="sideslip"):
        solve_half_torch(**args)
    args["freestream"] = torch.tensor([1., 0., .08], dtype=torch.float64)
    args["reference_point"] = torch.tensor([.25, .03, 0.], dtype=torch.float64)
    with pytest.raises(ValueError, match="symmetry plane"):
        solve_half_torch(**args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not installed")
def test_cuda_native_half_fields_and_adjoint_against_jax_cpu():
    baseline, inp = case("cuda")
    inp["vertices"] = inp["vertices"].requires_grad_()
    out = solve_half_torch(**inp, block_size=16)
    reference = solve_half_fields(**baseline)
    compare_fields(out, reference, rtol=5e-5, atol=1e-7)
    grad, = torch.autograd.grad(objective(out), (inp["vertices"],))
    assert torch.isfinite(grad).all()
    assert grad.abs().max().item() > 1e-8
