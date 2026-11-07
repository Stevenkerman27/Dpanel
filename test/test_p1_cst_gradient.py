from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from panel.p1_cst_gradient import (cst_geometry, fixed_wake_cst, fixed_wake_sections,
                                   galerkin_operators, grass_section_geometry)
from panel.p1_galerkin_wall import assemble_galerkin_multi
from panel.p1_lifting_g2 import combine_wings, solve_multi
from tools.web.params import GRASS_DEFAULTS

pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "GRASS-plane" / "cst_airfoil_codec.py").is_file(),
    reason="GRASS-plane CST codec is not installed alongside this repository")


def code():
    return torch.tensor([.17] * 8 + [-.17] * 8 + [0., 0.], dtype=torch.float64)


def test_cst_weak_operators_match_production():
    wing, xyz = cst_geometry(code(), chord_panels=3, span_panels=3)
    model, _, _, A, W, rhs, mass, _ = galerkin_operators(wing, xyz)
    reference = assemble_galerkin_multi(replace(model, xyz=xyz.detach().numpy()))
    for actual, key in ((A, "A"), (W, "W"), (rhs, "b_xyz"), (mass, "M")):
        np.testing.assert_allclose(actual.detach().numpy(), reference[key], atol=2e-13, rtol=2e-13)
    production = solve_multi(combine_wings([replace(wing, xyz=xyz.detach().numpy())]), 4.)
    actual, info = fixed_wake_cst(code(), chord_panels=3, span_panels=3)
    np.testing.assert_allclose(actual.detach().numpy(),
                               [production["CL"], production["CD_pressure"]], atol=2e-9)
    assert info["stationarity"] < 1e-8


@pytest.mark.parametrize("index", [0, 8, 16])
def test_cst_gradient_matches_central_difference(index):
    value = code().requires_grad_(True)
    output, info = fixed_wake_cst(value, chord_panels=3, span_panels=3)
    derivative = torch.stack([torch.autograd.grad(v, value, retain_graph=True)[0][index]
                              for v in output]).detach().numpy()
    assert info["stationarity"] < 1e-8
    for step in (1e-4, 1e-5):
        delta = torch.zeros_like(value)
        delta[index] = step
        plus, _ = fixed_wake_cst(value.detach() + delta, chord_panels=3, span_panels=3)
        minus, _ = fixed_wake_cst(value.detach() - delta, chord_panels=3, span_panels=3)
        finite_difference = ((plus - minus) / (2 * step)).detach().numpy()
        np.testing.assert_allclose(derivative, finite_difference, atol=2e-7, rtol=2e-5)


def test_potential_jump_matches_production_solver():
    """The production closure and the autograd path must be the same equation.

    panel/p1_potential_jump_torch.py is shared by solve_multi (production) and the
    differentiable route, so a mismatch here means one of the two affine/Newton steps drifted.
    """
    wing, xyz = cst_geometry(code(), chord_panels=3, span_panels=3)
    production = solve_multi(combine_wings([replace(wing, xyz=xyz.detach().numpy())]), 4.,
                             kutta="potential_jump")
    assert production["kutta"] == "potential_jump"
    assert production["nonlinear"]["success"]
    assert production["nonlinear"]["relative_step"] < 1e-10
    actual, info = fixed_wake_cst(code(), chord_panels=3, span_panels=3, kutta="potential_jump")
    assert info["kutta"] == "potential_jump"
    np.testing.assert_allclose(actual.detach().numpy(),
                               [production["CL"], production["CD_pressure"]], atol=2e-9)


def test_potential_jump_production_handles_two_groups():
    """J is block-structured over groups; the closure must cover all circulations at once."""
    wing, xyz, _, _ = grass_section_geometry(GRASS_DEFAULTS, chord_panels=4,
                                             span_panels_per_segment=2)
    w = replace(wing, xyz=xyz.detach().numpy())
    mesh = combine_wings([w, w], origins=[(0., 0., 0.), (0., 6., 0.)])
    out = solve_multi(mesh, 4., kutta="potential_jump")
    assert len(mesh.groups) == 2
    assert out["nonlinear"]["success"]
    assert out["jump_rms"] <= 1e-13          # affine map satisfies J by construction
    assert out["pressure_kutta_rms"] <= 1e-10
    assert out["CL"] > 0


def test_open_trailing_edge_rejected():
    value = code()
    value[17] = .01
    with pytest.raises(ValueError, match="sharp-TE"):
        cst_geometry(value)


def test_cambered_section_gradient():
    value = torch.tensor([.19] * 8 + [-.15] * 8 + [.01, 0.],
                         dtype=torch.float64, requires_grad=True)
    output, _ = fixed_wake_cst(value, chord_panels=3, span_panels=3)
    derivative = torch.stack([torch.autograd.grad(v, value, retain_graph=True)[0][3]
                              for v in output]).detach().numpy()
    delta = torch.zeros_like(value)
    delta[3] = 1e-5
    plus, _ = fixed_wake_cst(value.detach() + delta, chord_panels=3, span_panels=3)
    minus, _ = fixed_wake_cst(value.detach() - delta, chord_panels=3, span_panels=3)
    np.testing.assert_allclose(derivative, ((plus - minus) / 2e-5).detach().numpy(),
                               atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize("index", [0, 8, 16])
def test_potential_jump_gradient_matches_central_difference(index):
    """One coefficient at a time.

    The pressure path is checked above with the same shape, but it took this separate case to
    catch the potential-jump adjoint: summing the per-coefficient adjoints without the upstream
    weights is only correct when every weight is one, so a gradient of (CL + CD) agreed with
    finite differences while dCL and dCD alone were wrong by percents.
    """
    value = code().requires_grad_(True)
    output, info = fixed_wake_cst(value, chord_panels=3, span_panels=3, kutta="potential_jump")
    assert info["kutta"] == "potential_jump" and info["success"]
    derivative = torch.stack([torch.autograd.grad(v, value, retain_graph=True)[0][index]
                              for v in output]).detach()
    step = 1e-5
    delta = torch.zeros_like(value)
    delta[index] = step
    plus, _ = fixed_wake_cst(value.detach() + delta, chord_panels=3, span_panels=3,
                             kutta="potential_jump")
    minus, _ = fixed_wake_cst(value.detach() - delta, chord_panels=3, span_panels=3,
                              kutta="potential_jump")
    np.testing.assert_allclose(derivative.numpy(), ((plus - minus) / (2 * step)).detach().numpy(),
                               atol=2e-6, rtol=2e-4)


def sections(parameters):
    c = code()
    return dict(leading_edge_xyz=torch.stack((torch.zeros(3, dtype=c.dtype), parameters[:3])),
                chord=torch.stack((parameters[6], parameters[3])),
                twist_radians=torch.stack((parameters[7], parameters[4])),
                cst=torch.stack((c, torch.cat((parameters[5:6], c[1:])))))


def test_grass_sections_forward_and_design_gradients():
    parameters = torch.tensor([.2, 2., .2, .8, .04, .19, 1., .02], dtype=torch.float64,
                              requires_grad=True)
    geometry = sections(parameters)
    output, info = fixed_wake_sections(geometry, chord_panels=3, span_panels_per_segment=2)
    derivatives = torch.stack([torch.autograd.grad(v, parameters, retain_graph=True)[0]
                               for v in output]).detach().numpy()
    wing, _, _, _ = grass_section_geometry(geometry, chord_panels=3, span_panels_per_segment=2)
    reference = solve_multi(combine_wings([wing]), 4.)
    np.testing.assert_allclose(output.detach().numpy(), [reference["CL"], reference["CD_pressure"]],
                               atol=2e-9)
    assert info["stationarity"] < 1e-8
    for i in range(len(parameters)):
        delta = torch.zeros_like(parameters)
        delta[i] = 1e-5
        plus, _, _, _ = grass_section_geometry(sections(parameters.detach() + delta),
                                                chord_panels=3, span_panels_per_segment=2)
        minus, _, _, _ = grass_section_geometry(sections(parameters.detach() - delta),
                                                 chord_panels=3, span_panels_per_segment=2)
        forward_plus = solve_multi(combine_wings([plus]), 4.)
        forward_minus = solve_multi(combine_wings([minus]), 4.)
        finite = np.array([forward_plus["CL"] - forward_minus["CL"],
                           forward_plus["CD_pressure"] - forward_minus["CD_pressure"]]) / 2e-5
        np.testing.assert_allclose(derivatives[:, i], finite,
                                   atol=2e-6, rtol=2e-4)


def test_three_sections_middle_chord_gradient():
    def geometry(chord):
        c = code()
        return dict(leading_edge_xyz=torch.tensor([[0., 0., 0.], [.12, 1., .05], [.4, 2., .2]],
                                                  dtype=torch.float64),
                    chord=torch.stack((chord * 0 + 1, chord, chord * 0 + .6)),
                    twist_radians=torch.tensor([0., .02, .05], dtype=torch.float64),
                    cst=torch.stack((c, c, c)))

    chord = torch.tensor(.8, dtype=torch.float64, requires_grad=True)
    output, _ = fixed_wake_sections(geometry(chord), chord_panels=3, span_panels_per_segment=1)
    derivative = torch.stack([torch.autograd.grad(v, chord, retain_graph=True)[0]
                              for v in output]).detach().numpy()
    plus, _ = fixed_wake_sections(geometry(chord.detach() + 1e-5),
                                  chord_panels=3, span_panels_per_segment=1)
    minus, _ = fixed_wake_sections(geometry(chord.detach() - 1e-5),
                                   chord_panels=3, span_panels_per_segment=1)
    np.testing.assert_allclose(derivative, ((plus - minus) / 2e-5).detach().numpy(),
                               atol=2e-6, rtol=2e-4)


def test_three_sections_formal_solver_centered_differences():
    from panel.grass_mesh import build_grass_mesh, reconstruct_grass_mesh

    data = dict(leading_edge_xyz=[[0., 0., 0.], [.12, 1., .05], [.4, 2., .2]],
                chord=[1., .8, .6], twist_radians=[0., .02, .05],
                cst=[code().tolist(), code().tolist(), code().tolist()])
    mesh = build_grass_mesh(data, 2.)
    geometry, wing, xyz, wake, area = reconstruct_grass_mesh(mesh)
    from panel.p1_cst_gradient import fixed_wake_geometry
    values, _ = fixed_wake_geometry(wing, xyz, wake, area, 4., 160)
    variables = (("leading_edge_xyz", (1, 0)), ("leading_edge_xyz", (1, 1)),
                 ("leading_edge_xyz", (1, 2)), ("chord", (1,)),
                 ("twist_radians", (1,)), ("cst", (1, 0)))
    for name, index in variables:
        derivative = [torch.autograd.grad(value, geometry[name], retain_graph=True)[0][index].item()
                      for value in values]
        results = []
        for direction in (-1, 1):
            changed = {key: value.detach().clone() for key, value in geometry.items()}
            changed[name][index] += direction * 1e-5
            moved, _, _, _ = grass_section_geometry(
                changed, chord_panels=mesh["chord_panels"],
                span_panels_per_segment=mesh["span_panels_per_segment"])
            result = solve_multi(combine_wings([moved]), 4.)
            results.append([result["CL"], result["CD_pressure"]])
        finite = (np.asarray(results[1]) - results[0]) / 2e-5
        np.testing.assert_allclose(derivative, finite, atol=2e-6, rtol=2e-4)


def test_ideal_drag_gradient_matches_production_and_finite_difference():
    from panel.drag_consistency_audit import ideal_trefftz_piecewise_linear
    from panel.p1_cst_gradient import fixed_wake_geometry

    parameters = torch.tensor([.2, 2., .2, .8, .04, .19, 1., .02], dtype=torch.float64,
                              requires_grad=True)
    wing, xyz, wake, area = grass_section_geometry(sections(parameters), chord_panels=3,
                                                  span_panels_per_segment=2)
    values, _ = fixed_wake_geometry(wing, xyz, wake, area, 4., 160, include_ideal_drag=True)
    derivatives = torch.autograd.grad(values[2], parameters)[0].detach().numpy()
    results = []
    for x in (parameters.detach(), parameters.detach() + torch.eye(8)[4] * 1e-5,
              parameters.detach() - torch.eye(8)[4] * 1e-5):
        moved, _, _, _ = grass_section_geometry(sections(x), chord_panels=3,
                                                  span_panels_per_segment=2)
        state = solve_multi(combine_wings([moved]), 4.)
        gamma = np.r_[0., state["group_results"][0]["gamma"], 0.]
        results.append(ideal_trefftz_piecewise_linear(
            moved.gamma_y, gamma, U=1., density=1., reference_area=moved.reference_area)["CDi_ideal"])
    assert values[2].item() == pytest.approx(results[0], abs=2e-9)
    assert derivatives[4] == pytest.approx((results[1] - results[2]) / 2e-5, abs=2e-6)
