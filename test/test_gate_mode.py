"""Sampling-error thresholds must audit, not select which wake nodes may move."""
import numpy as np
import pytest


def generated_mesh():
    from panel.p1_lifting_g1 import make_wing
    from panel.p1_lifting_g2 import combine_wings
    return combine_wings([make_wing(chord_panels=4, span_panels=4, wake_panels=2)])


def test_unknown_gate_mode_is_rejected():
    from panel.p1_coupled_material_wake import solve_coupled_material_wake
    with pytest.raises(ValueError, match="gate_mode must be"):
        solve_coupled_material_wake(generated_mesh(), 5., iterations=0, gate_mode="sometimes")


@pytest.mark.parametrize("gate_mode", ["audit", "off"])
def test_audit_modes_never_freeze_a_node(gate_mode):
    from panel.p1_coupled_material_wake import solve_coupled_material_wake
    result = solve_coupled_material_wake(generated_mesh(), 5., kutta="signed_speed",
                                         iterations=1, gate_mode=gate_mode)
    assert result["gate_mode"] == gate_mode
    for row in result["history"]:
        for name, counts in row["wakes"].items():
            assert counts["eligibility"]["rejected"] == 0, name
            assert counts["eligibility"]["gate_mode"] == gate_mode


def test_off_skips_the_refined_quadrature_evaluation():
    from panel import velocity_profile as prof
    from panel.p1_coupled_material_wake import limits
    field = _constant_field()
    points = np.array([[.3, .1, 1.], [.2, -.2, 1.5]])
    normals = np.tile([0., 0., 1.], (2, 1))
    previous = prof.enable(True)
    try:
        prof.reset()
        audit = limits(field, points, normals, np.array([.2, .2]), eps_fraction=.05,
                       ladder=(1., .5), order=8, audit_quadrature=True)
        audited = prof.snapshot()["counters"]
        prof.reset()
        skipped = limits(field, points, normals, np.array([.2, .2]), eps_fraction=.05,
                         ladder=(1., .5), order=8, audit_quadrature=False)
        skipping = prof.snapshot()["counters"]
    finally:
        prof.enable(previous)
    assert audited.get("limits.refined_calls", 0) == 1
    assert skipping.get("limits.refined_calls", 0) == 0
    assert np.all(np.isfinite(skipped["mean"])) and np.all(np.isfinite(audit["mean"]))
    assert np.allclose(skipped["mean"], audit["mean"])


def _constant_field():
    class Uniform:
        method = "legacy_gauss"
        free = np.array([1., 0., 0.])

        def __call__(self, point, *, order):
            return np.array([1., 0., 0.])

        def evaluate_many(self, points, *, order):
            return np.tile([1., 0., 0.], (len(points), 1))

    return Uniform()
