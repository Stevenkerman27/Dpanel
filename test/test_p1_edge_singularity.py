import numpy as np
import pytest

from panel.p1_edge_singularity import (
    fluid_wedge_angle_from_solid, neumann_wedge_exponent,
    neumann_edge_mode, edge_enrichment_trace,
)


def test_thin_plate_crack_limit_has_inverse_square_root_velocity():
    omega=2*np.pi
    lam=neumann_wedge_exponent(omega)
    assert lam==pytest.approx(.5)
    a=neumann_edge_mode(np.array([1e-2,1e-4]),np.array([0.,0.]),omega)
    assert a["speed"][1]/a["speed"][0]==pytest.approx(10.)


def test_convex_ninety_degree_solid_corner_exponent():
    omega=fluid_wedge_angle_from_solid(np.pi/2)
    assert omega==pytest.approx(3*np.pi/2)
    assert neumann_wedge_exponent(omega)==pytest.approx(2/3)


def test_neumann_boundary_condition_is_satisfied_on_both_faces():
    omega=3*np.pi/2
    lam=neumann_wedge_exponent(omega)
    r=np.array([.01,.1,1.])
    a=neumann_edge_mode(r,np.zeros(3),omega)
    b=neumann_edge_mode(r,np.full(3,omega),omega)
    np.testing.assert_allclose(a["velocity_theta"],0.,atol=1e-14)
    np.testing.assert_allclose(b["velocity_theta"],0.,atol=1e-14)
    assert a["lambda"]==pytest.approx(lam)


def test_enrichment_trace_has_correct_sign_on_two_crack_faces():
    r=np.array([0.,.01,.25,1.])
    omega=2*np.pi
    np.testing.assert_allclose(edge_enrichment_trace(r,0,omega),r**.5)
    np.testing.assert_allclose(edge_enrichment_trace(r,1,omega),-r**.5)


def test_speed_singularity_exponent_matches_radial_derivative():
    omega=1.7*np.pi
    lam=np.pi/omega
    r=np.array([.02,.2,2.])
    out=neumann_edge_mode(r,np.full(3,.3),omega)
    expected=lam*r**(lam-1)
    np.testing.assert_allclose(out["speed"],expected,rtol=1e-13,atol=1e-13)


def test_invalid_wedge_geometry_is_rejected():
    with pytest.raises(ValueError):
        neumann_wedge_exponent(np.pi)
    with pytest.raises(ValueError):
        fluid_wedge_angle_from_solid(np.pi)
    with pytest.raises(ValueError):
        neumann_edge_mode([1.],[4*np.pi],[1.5*np.pi])
