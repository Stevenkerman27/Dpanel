"""End-to-end *small* Dpanel half JAX solve -> independent near/far audit.

Not an acceptance test of induced-drag physics: it checks that the audit
conserves the originally solved Cp panel force and respects frozen topology.
"""
import json
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from panel.half_native import solve_half_fields
from panel.run_half_native import load_half_surface
from panel.drag_consistency_audit import audit_saved_half_state


FIXTURE=Path(__file__).resolve().parents[1]/(
    "cases/panel_half/thick_wing_two_strip.json")


def test_same_small_frozen_half_mesh_near_pressure_is_reproducible():
    fixture=json.loads(FIXTURE.read_text(encoding="utf-8"))
    inputs=load_half_surface(FIXTURE)
    fields=solve_half_fields(**inputs)
    audit=audit_saved_half_state(fixture,fields)
    np.testing.assert_allclose(
        audit["near_pressure"]["CD_pressure"],
        np.asarray(fields["CD_pressure"]),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(
        audit["near_pressure"]["CL_pressure"],
        np.asarray(fields["CL"]),rtol=1e-12,atol=1e-12)
    assert audit["half_wall_count"]==len(inputs["topology"].faces)
    assert audit["ideal_trefftz_is_not_exact_finite_wake_momentum"]
    assert audit["wake_alignment"]["max_abs_Udotn_over_U"]>0
    assert np.isfinite(audit["ideal_trefftz"]["CDi_ideal"])
    assert not audit["physical_accuracy_accepted"]
