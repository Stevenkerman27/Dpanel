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
