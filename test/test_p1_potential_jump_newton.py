"""Root checks must test conditioning and correction, including zero-residual starts."""
import pytest
import torch

from panel.p1_potential_jump_torch import potential_jump_newton, solve_potential_jump


class QuadraticPressure:
    def __init__(self, factor=1., target=1.):
        self.factor, self.target = factor, target
        self.jacobian_calls = 0

    def __call__(self, state):
        return (state[:1] ** 2 - self.target) * self.factor

    def jacobian(self, state):
        self.jacobian_calls += 1
        row = torch.zeros((1, len(state)), dtype=state.dtype)
        row[0, 0] = 2. * state[0] * self.factor
        return row

    def reference_scale(self, state):
        return self.factor


@pytest.mark.parametrize("factor", [1e-12, 1., 1e12])
def test_damped_newton_is_invariant_to_pressure_units(factor):
    pressure = QuadraticPressure(factor=factor)
    trace = {}
    gamma, state, p, _, _, relative, condition = potential_jump_newton(
        pressure, torch.zeros(1, dtype=torch.float64),
        torch.ones((1, 1), dtype=torch.float64), torch.tensor([.1], dtype=torch.float64),
        1, max_nfev=20, diagnostics=trace)
    assert float(gamma[0]) == pytest.approx(1., abs=1e-12)
    assert relative < 1e-12
    assert trace["correction_relative"] < 1e-10
    assert trace["backtracks"] > 0
    assert condition == pytest.approx(1.)
    history = trace["history"]
    assert all(b["residual_relative"] <= a["residual_relative"]
               for a, b in zip(history, history[1:]))


def test_exact_root_still_checks_its_jacobian():
    pressure = QuadraticPressure()
    trace = {}
    result = potential_jump_newton(
        pressure, torch.zeros(1, dtype=torch.float64),
        torch.ones((1, 1), dtype=torch.float64), torch.ones(1, dtype=torch.float64),
        1, diagnostics=trace)
    assert pressure.jacobian_calls == 1
    assert result[-1] == 1.
    assert trace["correction_relative"] == 0.


def test_singular_zero_residual_root_is_refused():
    # mu[0]=Gamma from J; mu[1]=0 from B; p=mu[0]^2 has a singular zero at Gamma=0.
    B = torch.tensor([[0., 1., 0.]], dtype=torch.float64)
    J = torch.tensor([[1., 0., -1.]], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="ill-conditioned"):
        solve_potential_jump(
            B, torch.zeros(1, dtype=torch.float64), J, QuadraticPressure(target=0.),
            torch.zeros(1, dtype=torch.float64), nm=2)
