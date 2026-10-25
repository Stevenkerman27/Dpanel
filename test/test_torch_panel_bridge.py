"""First-order cross-framework checks on PR13's existing thick lifting wing."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
import jax.numpy as jnp

from panel.differentiable import solve_fields
from panel.torch_bridge import TorchPanelBridge, INPUT_NAMES, OUTPUT_NAMES
from test_differentiable_panel_fields import wing_inputs

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def case():
    source = wing_inputs()
    bridge = TorchPanelBridge(source["topology"])
    return source, bridge


def tensor_inputs(source, requires_grad=False):
    return tuple(torch.tensor(np.array(source[k]), dtype=torch.float64,
                              requires_grad=requires_grad) for k in INPUT_NAMES)


def weights_for(outputs):
    rng = np.random.default_rng(121)
    return tuple(rng.normal(size=tuple(v.shape)) for v in outputs)


def scalar_loss(outputs, weights):
    return sum((o * torch.as_tensor(w, dtype=torch.float64)).sum()
               for o, w in zip(outputs, weights))


def test_forward_and_all_seven_vjps_match_native_jax(case):
    source, bridge = case
    inputs = tensor_inputs(source, True)
    output = bridge(*inputs)
    weights = weights_for(output)
    native_inputs = tuple(source[k] for k in INPUT_NAMES)

    def native(*args):
        result = solve_fields(topology=source["topology"], **dict(zip(INPUT_NAMES, args)))
        return tuple(result[k] for k in OUTPUT_NAMES)

    values, back = jax.vjp(native, *native_inputs)
    expected_gradients = back(tuple(jnp.asarray(w) for w in weights))
    gradients = torch.autograd.grad(scalar_loss(output, weights), inputs)
    for actual, expected in zip(output, values):
        np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=2e-11, atol=2e-11)
    for name, actual, expected in zip(INPUT_NAMES, gradients, expected_gradients):
        assert torch.isfinite(actual).all(), name
        np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-9, atol=2e-9, err_msg=name)
    assert gradients[0].norm() > 0 and gradients[1].norm() > 0


def mapped_inputs(theta, baseline):
    """Torch-only geometry map with coherent station/wake motion; not CAD AD."""
    x, y, flow, rho, area, chord, ref = baseline
    changed = torch.stack((
        x[:, 0] * (1 + .03*theta[0]) + .02*theta[1]*x[:, 1]**2,
        x[:, 1] * (1 + .04*theta[2]) + .01*theta[3]*x[:, 1]**3,
        x[:, 2] * (1 + .04*theta[4]*(1+x[:, 1]**2)),
    ), dim=1)
    stations = y * (1 + .04*theta[2]) + .01*theta[3]*y**3
    return changed, stations, flow, rho, area, chord, ref


def test_torch_geometry_parameters_pass_gradcheck_and_three_step_fd(case):
    source, bridge = case
    baseline = tensor_inputs(source)
    weights = weights_for(bridge(*baseline))

    def objective(theta):
        return scalar_loss(bridge(*mapped_inputs(theta, baseline)), weights)

    theta = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(objective, (theta,), eps=1e-5, atol=1e-7, rtol=1e-4)
    gradient, = torch.autograd.grad(objective(theta), (theta,))
    direction = torch.tensor([.3, -.2, .1, .4, -.5], dtype=torch.float64)
    ad = (gradient * direction).sum().item()
    assert abs(ad) > 1e-6
    for h in (1e-3, 1e-4, 1e-5):
        fd = ((objective(theta.detach()+h*direction) - objective(theta.detach()-h*direction))/(2*h)).item()
        np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=1e-8)


def test_gradient_reaches_torch_neural_network_parameters(case):
    source, bridge = case
    baseline = tensor_inputs(source)
    model = torch.nn.Linear(2, 5, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.zero_()
    condition = torch.tensor([.7, -.3], dtype=torch.float64)
    theta = model(condition)
    result = bridge(*mapped_inputs(theta, baseline))
    loss = (result.CL - .4).square() + .03 * result.panel_moments.square().sum()
    loss.backward()
    assert model.weight.grad is not None
    assert torch.isfinite(model.weight.grad).all()
    assert model.weight.grad.norm() > 1e-6


def test_partial_outputs_noncontiguous_tensors_and_inference(case):
    source, bridge = case
    inputs = list(tensor_inputs(source, True))
    inputs[0] = inputs[0].T.contiguous().T
    assert not inputs[0].is_contiguous()
    result = bridge(*inputs)
    rho_gradient, = torch.autograd.grad(result.panel_forces[:, 2].sum(), (inputs[3],))
    torch.testing.assert_close(rho_gradient, result.panel_forces[:, 2].sum().detach()/inputs[3].detach(),
                               rtol=1e-10, atol=1e-10)
    with torch.no_grad():
        out = bridge(*inputs)
    assert all(not x.requires_grad for x in out)


def test_two_inflight_calls_and_repeated_backward_are_independent(case):
    source, bridge = case
    inputs = list(tensor_inputs(source))
    rho = inputs[3].clone().requires_grad_()
    one = bridge(*inputs[:3], rho, *inputs[4:])
    two = bridge(*inputs[:3], 2*rho, *inputs[4:])
    loss = one.panel_forces[:, 2].sum() + two.panel_forces[:, 2].sum()
    first, = torch.autograd.grad(loss, rho, retain_graph=True)
    second, = torch.autograd.grad(loss, rho)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first, loss.detach()/rho.detach(), rtol=1e-10, atol=1e-10)


def test_input_mutation_is_not_silently_used_for_backward(case):
    source, bridge = case
    inputs = list(tensor_inputs(source, True))
    result = bridge(*inputs)
    with torch.no_grad():
        inputs[0][0, 0] += .01
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        result.CL.backward()


def test_double_backward_is_explicitly_unsupported(case):
    source, bridge = case
    inputs = list(tensor_inputs(source, True))
    result = bridge(*inputs)
    grad, = torch.autograd.grad(result.panel_forces.square().sum(), inputs[3], create_graph=True)
    with pytest.raises(RuntimeError):
        torch.autograd.grad(grad, inputs[3])


def test_invalid_precision_shape_flow_and_geometry_are_rejected(case):
    source, bridge = case
    inputs = list(tensor_inputs(source))
    with pytest.raises(TypeError, match="float64"):
        bridge(inputs[0].float(), *inputs[1:])
    with pytest.raises(ValueError, match="scalar"):
        bridge(*inputs[:3], inputs[3].reshape(1), *inputs[4:])
    with pytest.raises(ValueError, match="nonzero freestream"):
        bridge(*inputs[:2], torch.zeros(3, dtype=torch.float64), *inputs[3:])
    nonplanar = inputs[0].clone(); nonplanar[-1, 2] += .1
    with pytest.raises(ValueError, match="planar"):
        bridge(nonplanar, *inputs[1:])
    y = inputs[1].clone(); y[2] += .1
    with pytest.raises(ValueError, match="strips"):
        bridge(inputs[0], y, *inputs[2:])
