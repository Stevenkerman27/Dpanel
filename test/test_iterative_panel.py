"""Exact operator/reduction/transpose/preconditioner/state-VJP regression."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from panel.kernel import triangle_potential
from panel.differentiable import solve_fields
from panel.iterative import (PanelOperator,make_plan,validate_plan,solve_state,
                             solve_fields_iterative)
from iterative_fixture import wing_inputs
jax.config.update('jax_enable_x64',True)


def build(half):
    inp=wing_inputs()
    plan=make_plan(inp['vertices'],inp['station_y'],inp['topology'],inp['freestream'],
                   block_size=7,preconditioner_block_size=8,symmetry=half)
    op=PanelOperator(inp['vertices'],inp['station_y'],inp['topology'],inp['freestream'],plan)
    return inp,plan,op


def dense_reduced(reference,plan):
    nf=len(plan.body_inverse)
    rows=np.r_[plan.body_rows,nf+plan.gamma_rows]
    inverse=np.r_[plan.body_inverse,len(plan.body_rows)+plan.gamma_inverse]
    E=np.eye(plan.shape[0])[inverse]
    return np.asarray(reference['matrix'])[rows]@E,np.asarray(reference['rhs'])[rows],E,rows


def test_rectangular_global_self_trace_matches_full_and_gradient():
    inp=wing_inputs();tri=inp['vertices'][inp['topology'].faces];c=tri.mean(1)
    S,D=triangle_potential(c,tri,True,block_size=0)
    ri=jnp.array([20,5,8,0]);ci=jnp.array([0,8,20,4,5])
    s,d=triangle_potential(c[ri],tri[ci],True,block_size=2,
                          target_panel_ids=ri,source_panel_ids=ci)
    np.testing.assert_allclose(s,S[ri[:,None],ci],atol=1e-14)
    np.testing.assert_allclose(d,D[ri[:,None],ci],atol=1e-14)
    g=jax.grad(lambda x:triangle_potential(x[ri],tri[ci],True,block_size=2,
        target_panel_ids=ri,source_panel_ids=ci)[0].sum())(c)
    assert np.isfinite(g).all()


@pytest.mark.parametrize('half',[False,True])
def test_exact_actions_transpose_and_lower_preconditioner(half):
    inp,plan,op=build(half);reference=solve_fields(**inp)
    A,b,E,rows=dense_reduced(reference,plan)
    n=len(b);x=jnp.array(np.random.default_rng(21).normal(size=n))
    y=jnp.array(np.random.default_rng(22).normal(size=n))
    np.testing.assert_allclose(op.rhs(),b,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(op.matvec(x),A@x,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(op.rmatvec(y),A.T@y,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(jnp.vdot(y,op.matvec(x)),jnp.vdot(op.rmatvec(y),x),rtol=1e-12,atol=1e-12)
    nb=op.nb;P=np.zeros_like(A)
    for ids in plan.near_blocks:P[np.ix_(ids,ids)]=A[np.ix_(ids,ids)]
    P[nb:,:]=A[nb:,:]
    pc,pct=op.preconditioner()
    np.testing.assert_allclose(pc(x),np.linalg.solve(P,x),rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(pct(y),np.linalg.solve(P.T,y),rtol=1e-11,atol=1e-11)
    if half:
        assert A.shape==(22,22) and reference['matrix'].shape==(43,43)
        full=np.r_[reference['mu'],reference['gamma']]
        np.testing.assert_allclose(E@np.linalg.solve(A,b),full,rtol=1e-10,atol=1e-11)


@pytest.mark.parametrize('half,preconditioned',[(False,False),(False,True),(True,True)])
def test_gmres_raw_loads_and_transpose_adjoint_against_lu(half,preconditioned):
    inp,plan,op=build(half)
    opts=dict(plan=plan,tolerance=1e-11,absolute_tolerance=1e-13,restart=25,
              max_cycles=15,preconditioned=preconditioned)
    got=jax.jit(lambda **kw:solve_fields_iterative(**kw,**opts))(**inp)
    ref=solve_fields(**inp)
    for k in ('mu','gamma','Cp_raw','panel_forces','panel_moments','CL','CM'):
        np.testing.assert_allclose(got[k],ref[k],rtol=1e-8,atol=1e-9,err_msg=k)
    b=op.rhs();seed=jnp.linspace(.2,1.,len(b))
    adj=jax.jit(jax.grad(lambda rhs:jnp.vdot(seed,solve_state(op,rhs,
        tolerance=1e-11,absolute_tolerance=1e-13,restart=25,max_cycles=15,
        preconditioned=preconditioned))))(b)
    np.testing.assert_allclose(op.rmatvec(adj),seed,rtol=1e-9,atol=1e-10)


@pytest.mark.parametrize('half',[False,True])
def test_implicit_geometry_gradient_and_station_chain_matches_lu_and_fd(half):
    inp,plan,op=build(half);x=inp['vertices'];y=inp['station_y']
    # Even x/z and odd y changes preserve physical reflection and wake plane.
    dx=jnp.stack((.03*x[:,0]+.02*x[:,1]**2,.05*x[:,1]**3,
                  .04*x[:,2]*(1+x[:,1]**2)),axis=1);dy=.05*y**3
    weight=jnp.array(np.random.default_rng(33).normal(size=(len(inp['topology'].faces),3)))
    def loss(a,method):
        changed={**inp,'vertices':x+a*dx,'station_y':y+a*dy}
        out=(solve_fields_iterative(**changed,plan=plan,tolerance=1e-12,
            absolute_tolerance=1e-14,restart=30,max_cycles=20,preconditioned=True)
             if method else solve_fields(**changed))
        return jnp.sum(weight*out['panel_forces'])+.2*jnp.sum(weight*out['panel_moments'])
    ad=jax.jit(jax.grad(lambda a:loss(a,True)))(jnp.array(0.))
    reference=jax.grad(lambda a:loss(a,False))(jnp.array(0.))
    np.testing.assert_allclose(ad,reference,rtol=2e-8,atol=1e-9)
    value=jax.jit(lambda a:loss(a,True))
    for h in (1e-3,1e-4,1e-5):
        fd=(value(h)-value(-h))/(2*h)
        np.testing.assert_allclose(ad,fd,rtol=3e-5,atol=1e-7)


def test_half_rejects_asymmetry_and_failed_gmres_is_not_accepted():
    inp,plan,op=build(True)
    with pytest.raises(ValueError,match='zero sideslip'):
        validate_plan(inp['vertices'],inp['station_y'],inp['topology'],[1.,.1,.1],plan)
    with pytest.raises(ValueError,match='symmetric'):
        validate_plan(inp['vertices'].at[1,0].add(.01),inp['station_y'],inp['topology'],inp['freestream'],plan)
    x=solve_state(op,op.rhs(),tolerance=1e-14,absolute_tolerance=0.,restart=1,max_cycles=1,preconditioned=False)
    assert np.isnan(np.asarray(x)).all()


def test_streamed_primal_has_no_full_wall_aic_array_in_jaxpr():
    inp,plan,op=build(False)
    program=jax.make_jaxpr(op.matvec)(jnp.zeros(plan.shape[0]))
    shapes=[]
    def visit(value):
        if hasattr(value,'jaxpr'):
            visit(value.jaxpr)
        elif hasattr(value,'eqns'):
            for eqn in value.eqns:
                for var in eqn.outvars:
                    aval=getattr(var,'aval',None)
                    if aval is not None and hasattr(aval,'shape'):shapes.append(tuple(aval.shape))
                for parameter in eqn.params.values():visit(parameter)
        elif isinstance(value,(tuple,list)):
            for item in value:visit(item)
        elif isinstance(value,dict):
            for item in value.values():visit(item)
    visit(program)
    assert (op.nf,op.nf) not in shapes
    assert plan.shape not in shapes
    # This is a traced-shape regression, NOT a peak-RSS/GPU-memory measurement.
