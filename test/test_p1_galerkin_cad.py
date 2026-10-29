"""Fast topology/geometry smoke checks for committed CAD-derived half surfaces."""
import json
from pathlib import Path
import numpy as np
import pytest

from panel.p1_lifting_g1 import from_dpanel_half_surface

ROOT=Path(__file__).resolve().parents[1]
CASES={
 "elliptic_s075":ROOT/"cases/geom_robust/surfaces/elliptic_s075/surface.json",
 "conv_s075":ROOT/"cases/geom_robust/surfaces/conv_s075/surface.json",
 "swept30_s075":ROOT/"cases/geom_robust/surfaces/swept30_s075/surface.json",
 "bwb_s055":ROOT/"cases/geom_robust/surfaces/bwb_s055/surface.json",
}


@pytest.mark.parametrize("name",list(CASES))
def test_cad_half_surface_mirrors_with_finite_wake_and_consistent_gamma(name):
    path=CASES[name]
    raw=json.loads(path.read_text(encoding="utf-8"))
    wing=from_dpanel_half_surface(path)
    assert len(wing.faces)==2*len(raw["faces"])
    assert len(wing.wake_faces)==2*len(raw["wake_faces"])
    assert wing.dof_face.shape==wing.faces.shape
    assert wing.gamma_hat.shape==(len(wing.wake_xyz),len(wing.gamma_y)-2)
    assert wing.ndof_mu>=len(np.unique(wing.faces))
    assert len(wing.te_pairs)>0
    assert np.isfinite(wing.xyz).all()
    assert np.isfinite(wing.wake_xyz).all()
    assert np.isfinite(wing.gamma_hat).all()
    assert np.all(wing.gamma_y[1:]>wing.gamma_y[:-1])
    assert np.all(np.linalg.norm(np.cross(
        wing.xyz[wing.faces][:,1]-wing.xyz[wing.faces][:,0],
        wing.xyz[wing.faces][:,2]-wing.xyz[wing.faces][:,0]),axis=1)>1e-14)


def test_conv_gamma_basis_excludes_zero_support_inboard_contact_stations():
    wing=from_dpanel_half_surface(CASES["conv_s075"])
    # The actual exposed TE starts at the fuselage contact y ~= 0.25.
    # The CAD wake still spans y=0,0.2 for contact topology, but those stations
    # must not become independent Gamma unknowns in the lifting solve.
    assert np.min(wing.gamma_y[wing.gamma_y>0])==pytest.approx(0.24999647337329162,abs=1e-9)
    assert not np.any(np.isclose(np.abs(wing.gamma_y),0.0,atol=1e-10))
    assert not np.any(np.isclose(np.abs(wing.gamma_y),0.2,atol=1e-10))
    assert wing.gamma_hat.shape[1]==18
    assert np.min(wing.gamma_y)<-1.599 and np.max(wing.gamma_y)>1.599
    assert np.any(np.isclose(wing.gamma_y,-0.24999647337329162,atol=1e-9))
    assert np.any(np.isclose(wing.gamma_y,+0.24999647337329162,atol=1e-9))
    np.testing.assert_allclose(wing.gamma_y,-wing.gamma_y[::-1],rtol=0,atol=1e-12)


def test_other_cad_models_keep_their_original_gamma_stations():
    for name in ("elliptic_s075","swept30_s075","bwb_s055"):
        raw=json.loads(CASES[name].read_text(encoding="utf-8"))
        wing=from_dpanel_half_surface(CASES[name])
        expected=np.r_[-np.asarray(raw["gamma_stations"])[::-1][:-1],
                       np.asarray(raw["gamma_stations"])]
        np.testing.assert_allclose(wing.gamma_y,expected,rtol=0,atol=1e-12)


def test_loader_rejects_unmapped_independent_station_inboard_of_te(tmp_path):
    raw=json.loads(CASES["conv_s075"].read_text(encoding="utf-8"))
    raw["gamma_station_dof"][0]=99
    path=tmp_path/"bad_conv_surface.json"
    path.write_text(json.dumps(raw),encoding="utf-8")
    with pytest.raises(ValueError,match="not tied to the TE-root DOF"):
        from_dpanel_half_surface(path)


def test_cad_sweep_does_not_report_implausible_solution_as_success():
    from tools.issue71.galerkin_cad_sweep import status
    state={"nonlinear":None,"solution_status":"implausible_solution"}
    assert status(state,80)=="implausible_solution"
    assert status({"nonlinear":{"success":False,"nfev":80}},80)=="iteration_limit_not_converged"
    assert status({"nonlinear":None,"solution_status":"solver_success"},80)=="solver_success"
