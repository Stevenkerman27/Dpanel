"""Matched bank recovery: independent affine fields, cuts and geometry derivatives."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from panel.p1_lifting_g1 import make_wing
from panel.p1_lifting_g2 import combine_wings
from panel.p1_pressure_kutta_torch_driver import TorchPressureModel
from panel.p1_te_recovery import matched_te_velocity_maps


def fixture():
    wing = make_wing(chord_panels=4, span_panels=6, wake_panels=1)
    # A finite wedge with planar banks makes affine tangential reproduction exact.
    xyz = wing.xyz.copy()
    xyz[:, 2] = np.sign(xyz[:, 2]) * .1 * (1. - xyz[:, 0])
    return combine_wings([replace(wing, xyz=xyz)])


def test_area_recovery_accepts_a_single_interior_te_station():
    mesh = fixture()
    group = replace(mesh.groups[0], te_pairs=mesh.groups[0].te_pairs[:1])
    mesh = replace(mesh, groups=(group,))
    top, bottom, receipt = matched_te_velocity_maps(mesh, method="area")
    assert top.shape == bottom.shape == (1, 3, mesh.ndof_mu)
    assert len(receipt) == 2
    torch.testing.assert_close(top.sum(dim=2), torch.zeros((1, 3), dtype=top.dtype),
                               atol=1e-13, rtol=0.)


@pytest.mark.parametrize("method", ["area", "quadratic"])
def test_planar_affine_gradient_and_separate_bank_offsets(method):
    mesh = fixture()
    top, bottom, receipt = matched_te_velocity_maps(mesh, method=method)
    direction = np.array([.8, .13, -.2])
    # Restrict the check of independent gauges to adding a constant to every
    # lower-bank DOF that is absent from the upper-bank support.
    upper_dofs = set(mesh.dof_face[mesh.face_bank == 1].reshape(-1))
    lower_dofs = set(mesh.dof_face[mesh.face_bank == -1].reshape(-1))
    gauge = np.zeros(mesh.ndof_mu)
    gauge[list(lower_dofs - upper_dofs)] = 7.
    affine = np.zeros(mesh.ndof_mu)
    for face, dofs in enumerate(mesh.dof_face):
        affine[dofs] = mesh.xyz[mesh.faces[face]] @ direction
    for operator, bank in ((top, 1), (bottom, -1)):
        normal = np.array([.1, 0., float(bank)])
        normal /= np.linalg.norm(normal)
        expected = direction - (direction @ normal) * normal
        velocity = (operator @ torch.as_tensor(affine)).numpy()
        np.testing.assert_allclose(velocity, np.tile(expected, (len(velocity), 1)), atol=2e-12)
        torch.testing.assert_close(operator.sum(dim=2), torch.zeros_like(operator[:, :, 0]),
                                   atol=2e-12, rtol=0.)
    # The added lower-bank value never enters the upper reconstruction.
    torch.testing.assert_close(top @ torch.as_tensor(gauge), torch.zeros_like(top[:, :, 0]),
                               atol=1e-12, rtol=0.)
    assert all(r["bank"] in (1, -1) for r in receipt)


def test_matched_pressure_jacobian_and_geometry_gradient():
    mesh = fixture()
    xyz = torch.tensor(mesh.xyz, dtype=torch.float64, requires_grad=True)
    top, bottom, _ = matched_te_velocity_maps(mesh, xyz)
    state = torch.linspace(-.3, .8, mesh.ndof_mu + 5, dtype=torch.float64)
    grad = torch.zeros((len(mesh.faces), 3, mesh.ndof_mu), dtype=torch.float64)
    pairs = [([0], [1])] * len(mesh.groups[0].te_pairs)
    model = TorchPressureModel(grad, pairs, len(state), velocity_maps=(top, bottom))
    analytic = model.jacobian(state)
    step = 1e-6
    bump = torch.zeros_like(state)
    bump[mesh.groups[0].te_pairs[2, 0]] = step
    numeric = (model(state + bump) - model(state - bump)) / (2 * step)
    torch.testing.assert_close(analytic @ (bump / step), numeric, rtol=2e-7, atol=2e-7)
    value = model(state).sum()
    derivative = torch.autograd.grad(value, xyz)[0]
    direction = torch.zeros_like(xyz)
    point = mesh.groups[0].te_xyz[2]
    node = int(np.flatnonzero(np.all(mesh.xyz == point, axis=1))[0])
    direction[node, 2] = 1.
    values = []
    for sign in (-1., 1.):
        a, b, _ = matched_te_velocity_maps(mesh, xyz.detach() + sign * step * direction)
        candidate = TorchPressureModel(grad, pairs, len(state), velocity_maps=(a, b))
        values.append(candidate(state).sum())
    finite = (values[1] - values[0]) / (2 * step)
    assert abs(float(finite)) > 1e-5
    assert float((derivative * direction).sum()) == pytest.approx(float(finite), rel=2e-6, abs=2e-6)
