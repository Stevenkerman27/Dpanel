import numpy as np
import pytest

from panel.shed_group_blocks import te_weak_blocks,assemble_shed_groups,solve_reference


def test_gauss_weak_mass_is_not_product_of_aggregated_integrals():
    gauss=np.array([.5-np.sqrt(3)/6,.5+np.sqrt(3)/6])
    B=np.stack([1-gauss,gauss],axis=-1)[None]
    w=np.ones((1,2))*.5
    T,M,mass=te_weak_blocks(B,w,np.array([0]),np.array([1]),2)
    assert np.allclose(mass,[[1/3,1/6],[1/6,1/3]],atol=1e-15)
    assert np.allclose(M,[[2/3,1/3],[1/3,2/3]],atol=1e-15)
    assert np.allclose(T,[[1,-1],[1,-1]])
    integrated=np.einsum('eq,eqi->ei',w,B)
    assert not np.allclose(mass,integrated.T @ integrated)


def test_no_wake_is_regular_nonlifting_linear_algebra():
    D=np.array([[2,.25],[.125,3.]])
    b=np.array([1.,2.])
    A,r,s=assemble_shed_groups(D,b,[])
    assert np.array_equal(A,D)
    assert list(s)==['mu']
    x,err=solve_reference(A,r)
    assert err<1e-14 and np.allclose(A@x,b)


def systems():
    D=np.array([[3.,.2],[.1,2.]])
    b=np.array([1.,.5])
    G1=('wing',np.array([[.2],[.05]]),np.array([[.2,.01]]),np.array([[1.]]))
    G2=('htail',np.array([[.15],[.4]]),np.array([[.02,.25]]),np.array([[1.1]]))
    return D,b,G1,G2


def test_one_group_manual_legacy_block_layout():
    D,b,g1,_=systems()
    A,r,s=assemble_shed_groups(D,b,[g1])
    expect=np.block([[D,g1[1]],[-g1[2],g1[3]]])
    assert np.array_equal(A,expect)
    assert A.shape==(3,3) and s['wing']==slice(2,3)
    x,err=solve_reference(A,r)
    assert err < 1e-14


def test_two_groups_couple_through_shared_wall_state():
    D,b,g1,g2=systems()
    A,r,s=assemble_shed_groups(D,b,[g1,g2])
    assert A.shape==(4,4)
    assert A[s['wing'],s['htail']]==0
    assert A[s['htail'],s['wing']]==0
    x,err=solve_reference(A,r)
    assert err<1e-14 and abs(x[s['htail']][0])>1e-4
    Aone,rone,_=assemble_shed_groups(D,b,[g1])
    # A changed model with htail removed will NOT preserve wing & wall state.
    removed,_=solve_reference(Aone,rone)
    assert np.linalg.norm(removed-x[:3])>1e-5
    # Correct forced restriction has W2 * full gamma2 on the RHS.
    rone[:len(D)]-=g2[1]@x[s['htail']]
    keep,_=solve_reference(Aone,rone)
    assert np.allclose(keep,x[:3],rtol=1e-13,atol=1e-13)


def test_true_transpose_adjoint_identity_and_finite_difference():
    D,b,g1,g2=systems()
    A,r,_=assemble_shed_groups(D,b,[g1,g2])
    seed=np.array([.3,-.1,.2,.4])
    state,_=solve_reference(A,r)
    lam=np.linalg.solve(A.T,seed)
    db=np.array([.005,-.02,0,0])
    step=1e-5
    changed=np.linalg.solve(A,r+step*db)
    lhs=seed@((changed-state)/step)
    rhs=lam@db
    assert lhs==pytest.approx(rhs,abs=1e-9)


@pytest.mark.parametrize('bad',[('wing',np.empty((2,0)),np.empty((0,2)),np.empty((0,0))),
                                  ('wing',np.zeros((3,1)),np.zeros((1,2)),np.eye(1)),
                                  ('wing',np.ones((2,1)),np.zeros((1,3)),np.eye(1))])
def test_bad_group_dimensions_rejected(bad):
    D,b,*_=systems()
    with pytest.raises(ValueError):
        assemble_shed_groups(D,b,[bad])


def test_duplicate_names_rejected():
    D,b,g1,_=systems()
    with pytest.raises(ValueError,match='unique'):
        assemble_shed_groups(D,b,[g1,g1])


def test_unsupported_te_dof_and_same_bank_rejected():
    B=np.array([[[1.,0.],[1.,0.]]])
    with pytest.raises(ValueError,match='unsupported'):
        te_weak_blocks(B,np.ones((1,2)),np.array([0]),np.array([1]),2)
    with pytest.raises(ValueError,match='bank'):
        te_weak_blocks(np.ones((1,2,1)),np.ones((1,2)),np.array([0]),np.array([0]),2)
