"""Pure geometry tests for the 3-D CAD trailing-edge contract."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cad_contact

def test_validate_te_polyline_accepts_nonplanar_curve():
    te = np.array([[1., -1., .2], [1.1, 0., .5], [1.2, 1., .1]])
    assert np.array_equal(cad_contact.validate_te_polyline(te), te)

def test_te_distance_projects_to_each_3d_segment():
    te = np.array([[0., 0., 0.], [1., 1., 1.], [2., 2., 0.]])
    points = np.array([[.5, .5, .5], [1., 1., 2.], [2., 2., 1.]])
    assert np.allclose(cad_contact._te_distance(points, te), [0., 1., np.sqrt(2./3.)])

def test_validate_te_polyline_rejects_duplicate_points():
    with pytest.raises(ValueError, match="distinct"):
        cad_contact.validate_te_polyline(np.array([[0., 0., 0.], [0., 0., 0.]]))

def test_validate_te_polyline_rejects_nonmonotone_material_station():
    with pytest.raises(ValueError, match="strictly increasing"):
        cad_contact.validate_te_polyline(np.array([[0., 0., 0.], [1., -1., .2]]))

def test_validate_te_polyline_rejects_nonfinite_coordinates():
    with pytest.raises(ValueError, match="finite"):
        cad_contact.validate_te_polyline(np.array([[0., 0., 0.], [1., 1., np.nan]]))
