"""Manufactured P1 Cp conditioning tests, not a full DPanel/CAD result."""
import numpy as np
import pytest
from panel.p1_cp_diagnostics import p1_surface_gradient


@pytest.mark.parametrize("width",[1.0,.1,.01,.001])
def test_affine_surface_potential_gradient_and_cp_exact_on_slivers(width):
    xyz=np.array([[0.,0.,0.],[26.,0.,0.],[26.,width,0.]])
    f=np.array([[0,1,2]]); U=np.array([1.,.2,0.])
    grad,q=p1_surface_gradient(xyz,f,f,xyz@U)
    np.testing.assert_allclose(grad[0],U,atol=3e-12,rtol=3e-12)
    assert 1-np.dot(grad[0],grad[0])==pytest.approx(1-np.dot(U,U),abs=1e-11)


def test_fixed_nodal_potential_error_amplifies_as_short_edge_shrinks():
    U=np.array([1.,.2,0.]); errors=[]
    for width in (.1,.01,.001):
        xyz=np.array([[0.,0.,0.],[26.,0.,0.],[26.,width,0.]])
        f=np.array([[0,1,2]]); values=xyz@U; values[2]+=1e-4
        grad,_=p1_surface_gradient(xyz,f,f,values)
        errors.append(abs((1-grad[0]@grad[0])-(1-U@U)))
    assert errors[0]<errors[1]<errors[2]
    assert errors[-1]/errors[0]>100


def test_neighbor_normal_equations_square_stencil_condition_number():
    # Illustrates why a P0 neighbor LS recovery can be sensitive to skinny stencils.
    x=np.array([[1.,0.],[1.,1e-7],[1.,-1e-7]])
    assert np.linalg.cond(x)>1e7
    assert np.linalg.cond(x.T@x)>1e14
