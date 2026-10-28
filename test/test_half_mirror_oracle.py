"""Full-versus-half oracle on exactly the SAME reflected discretization.

This does NOT compare against a separately generated Gmsh full-domain mesh.
The production half-native path continues to use only right-side unknowns.
"""
import json
from pathlib import Path

import jax
import numpy as np

from panel.verify_half_mirror import verify

jax.config.update("jax_enable_x64", True)

FIXTURE = Path(__file__).resolve().parents[1] / (
    "cases/panel_half/thick_wing_two_strip.json"
)


def test_native_half_matches_real_reflection_and_load_vjp():
    result = verify(FIXTURE, derivatives=True)
    assert result["status"] == "exact_mirror_discretization_verified"
    assert result["half_wall_panels"] == 20
    assert result["full_oracle_wall_panels"] == 40
    assert result["half_circulation_unknowns"] == 2
    assert result["full_oracle_circulation_unknowns"] == 3
    for name in ("half_vs_RAE_matrix", "half_vs_full_RHS",
                 "half_mu_vs_full_positive", "half_gamma_vs_full_positive",
                 "expanded_half_state_vs_full", "Cp_raw_right_vs_full",
                 "pressure_forces_right", "pressure_moments_right",
                 "full_force", "full_moment", "CL", "CM"):
        assert result[name]["max_abs"] < 1e-9, (name, result[name])
    checks = result["symmetric_geometry_station_derivatives"]
    assert len(checks) == 3
    for entry in checks:
        np.testing.assert_allclose(
            entry["half_ad"], entry["full_ad"], rtol=1e-7, atol=1e-8)


def test_contact_root_station_is_one_shared_mirror_orbit(tmp_path):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    # A short symmetric body contact can identify root and first inner
    # circulation station. The mirror full state still uses one root gamma
    # orbit, not two independently unconstrained left/right root values.
    fixture["gamma_station_dof"] = [0, 0, -1]
    updated = tmp_path / "half_contact_fixture.json"
    updated.write_text(json.dumps(fixture), encoding="utf-8")
    result = verify(updated, derivatives=False)
    assert result["half_circulation_unknowns"] == 1
    assert result["full_oracle_circulation_unknowns"] == 1
    assert result["expanded_half_state_vs_full"]["relative_l2"] < 1e-8
