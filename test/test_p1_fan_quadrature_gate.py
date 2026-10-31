"""Acceptance tests for the projection-fan quadrature gate (issue #84 task 1).

The defect these tests lock down: with area_quadrature='projection_fan' the fan ignored
the legacy Gauss order, so limits() compared a configuration with itself and reported
quadrature_error == 0. The gate now compares an explicit base fan against an explicit
refined fan, and a self-comparison is an error rather than a silent pass.
"""
import numpy as np
import pytest

from panel.p1_analytic_sheet_velocity import AnalyticSheetField
from panel.p1_coupled_material_wake import limits
from panel.p1_regularized_sheet_velocity import RegularizedP1SheetField


def fixture():
    xyz = np.array([[0., 0., 0.], [1.2, 0.1, 0.2], [0.2, 1.1, -0.1]])
    faces = np.array([[0, 1, 2]])
    values = np.array([[0.3, -0.2, 0.8]])
    return xyz, faces, values


def fan_field(base=(8, 2), refined=(32, 2), core=0.05):
    xyz, faces, values = fixture()
    return RegularizedP1SheetField([(xyz, faces, values)], core_radius=core,
                                   area_quadrature="projection_fan",
                                   fan_order=base[0], fan_strips=base[1],
                                   fan_refinement_order=refined[0],
                                   fan_refinement_strips=refined[1])


def sample_geometry():
    centroid = fixture()[0].mean(axis=0)
    points = np.stack([centroid + [0., 0., 0.01], centroid + [0.2, 0.1, 0.05]])
    normals = np.tile([0., 0., 1.], (len(points), 1))
    scales = np.array([0.1, 0.1])
    return points, normals, scales


def test_plan_records_base_and_refined_fan():
    field = fan_field()
    plan = field.quadrature_plan(48)
    assert plan["method"] == "projection_fan"
    assert plan["base"] == {"fan_order": 8, "fan_strips": 2}
    assert plan["refined"] == {"fan_order": 32, "fan_strips": 2}
    assert plan["legacy_order"] is None


def test_default_refinement_is_at_least_48_and_differs():
    xyz, faces, values = fixture()
    field = RegularizedP1SheetField([(xyz, faces, values)], core_radius=0.05,
                                    area_quadrature="projection_fan",
                                    fan_order=28, fan_strips=8)
    assert field.fan_refinement_order >= 48
    assert (field.fan_refinement_order != field.fan_order
            or field.fan_refinement_strips != field.fan_strips)
    base, refined = field.quadrature_arguments(48)
    assert base != refined


def test_identical_fan_pair_is_rejected():
    xyz, faces, values = fixture()
    with pytest.raises(ValueError, match="refinement must differ"):
        RegularizedP1SheetField([(xyz, faces, values)], core_radius=0.05,
                                area_quadrature="projection_fan",
                                fan_order=16, fan_strips=8,
                                fan_refinement_order=16,
                                fan_refinement_strips=8)


def test_gate_flags_a_low_order_base_and_passes_a_converged_one():
    points, normals, scales = sample_geometry()
    kwargs = dict(eps_fraction=0.05, ladder=np.array([1., 0.5]), order=48)
    coarse = limits(fan_field(base=(4, 2), refined=(48, 2)), points, normals, scales, **kwargs)
    fine = limits(fan_field(base=(32, 2), refined=(48, 2)), points, normals, scales, **kwargs)
    assert np.all(coarse["quadrature_error"] > 0.)
    assert np.all(coarse["quadrature_error"] > 1e-3)
    assert np.all(fine["quadrature_error"] < 1e-6)
    assert np.all(fine["quadrature_error"] < coarse["quadrature_error"])


def test_limits_rejects_a_self_comparison():
    class SelfComparable:
        method = "rm_regularized_p1"
        free = np.array([1., 0., 0.])

        def quadrature_plan(self, order):
            return dict(method=self.method, base=dict(order=order),
                        refined=dict(order=order), legacy_order=order)

        def quadrature_arguments(self, order):
            return dict(order=order), dict(order=order)

        def evaluate_many(self, points, *, order=48):
            return np.tile(self.free, (len(points), 1))

    field = SelfComparable()
    points, normals, scales = sample_geometry()
    with pytest.raises(ValueError, match="compare a configuration with itself"):
        limits(field, points, normals, scales, eps_fraction=0.05,
               ladder=np.array([1., 0.5]), order=48)


def test_analytic_gate_is_not_applicable_and_zero():
    xyz, faces, values = fixture()
    field = AnalyticSheetField([(xyz, faces, values)], freestream=(1., 0., 0.))
    points, normals, scales = sample_geometry()
    sample = limits(field, points, normals, scales, eps_fraction=0.05,
                    ladder=np.array([1., 0.5]), order=48)
    assert np.all(sample["quadrature_error"] == 0.)
    assert np.all(np.isfinite(sample["epsilon_error"]))


def test_legacy_gate_still_compares_n_with_double_order():
    xyz, faces, values = fixture()
    field = RegularizedP1SheetField([(xyz, faces, values)], core_radius=0.05)
    points, normals, scales = sample_geometry()
    sample = limits(field, points, normals, scales, eps_fraction=0.05,
                    ladder=np.array([1., 0.5]), order=16)
    h = 0.05*scales*0.5
    targets = np.concatenate((points+h[:, None]*normals, points-h[:, None]*normals))
    base = field.evaluate_many(targets, order=16)
    refined = field.evaluate_many(targets, order=32)
    mean = (base[:len(points)]+base[len(points):])/2
    refined_mean = (refined[:len(points)]+refined[len(points):])/2
    expected = np.linalg.norm(refined_mean-mean, axis=1)/np.linalg.norm(field.freestream)
    assert np.array_equal(sample["quadrature_error"], expected)


def test_scalar_and_batch_fan_refinement_agree():
    field = fan_field(base=(8, 2), refined=(16, 2))
    point = sample_geometry()[0][0]
    scalar = field(point, fan_order=16, fan_strips=2)
    batch = field.evaluate_many(point[None, :], fan_order=16, fan_strips=2)
    assert np.array_equal(scalar, batch[0])
    assert not np.array_equal(scalar, field(point, fan_order=8, fan_strips=2))
