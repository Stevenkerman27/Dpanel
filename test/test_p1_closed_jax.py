import numpy as np
import pytest
import jax
import jax.numpy as jnp
from panel.p1_closed_jax import solve_nodal,make_velocity_evaluator
from panel.p1_closed_body import solve,surface_values,velocity_fd
from panel.run_p1_closed import icosphere,directions
jax.config.update('jax_enable_x64',True)


@pytest.mark.parametrize('level,q',[(0,1),(1,1),(0,3)])
def test_actual_JAX_solve_matches_numpy_full_rank_QR(level,q):
    v,f=icosphere(level);s=solve(v,f,samples_per_face=q)
    out=jax.jit(lambda x,u:solve_nodal(x,f,u,samples_per_face=q))(v,jnp.array([1.,0,0]))
    assert abs(np.asarray(out['mu'])-s['mu']).max()<3e-13
    assert abs(np.asarray(out['cp'])-surface_values(s)['cp']).max()<1e-12


def test_complete_geometry_VJP_matches_directional_finite_difference():
    v,f=icosphere(0);rng=np.random.default_rng(17)
    direction=rng.normal(size=v.shape)*.02;seed=rng.normal(size=len(v))
    loss=lambda x:jnp.vdot(solve_nodal(x,f,jnp.array([1.,.2,.1]))['mu'],jnp.asarray(seed))
    evaluate=jax.jit(loss);gradient=jax.jit(jax.grad(loss))(v)
    predicted=float(jnp.sum(gradient*direction))
    discrepancies=[]
    for h in (2e-4,1e-4,5e-5):
        fd=(float(evaluate(v+h*direction))-float(evaluate(v-h*direction)))/(2*h)
        discrepancies.append(abs(fd-predicted))
    assert max(discrepancies)<2e-8
    assert np.isfinite(gradient).all()


def test_pressure_loss_geometry_VJP_includes_source_and_area_weights():
    v,f=icosphere(0);rng=np.random.default_rng(9);d=rng.normal(size=v.shape)*.01
    def loss(x):
        o=solve_nodal(x,f,jnp.array([1.,.1,.2]))
        return jnp.sum(o['cp']**2*o['areas'])
    grad=jax.jit(jax.grad(loss))(v);val=jax.jit(loss)
    h=1e-4;fd=(float(val(v+h*d))-float(val(v-h*d)))/(2*h)
    assert float(jnp.sum(grad*d))==pytest.approx(fd,rel=1e-6,abs=1e-8)


@pytest.mark.parametrize('order',[0,1])
def test_exterior_spatial_AD_agrees_with_fourth_order_FD(order):
    v,f=icosphere(1);s=solve(v,f,order=order);p=directions(16)*1.02
    adv=np.asarray(make_velocity_evaluator(s)(p));fdv=velocity_fd(s,p,step=1e-4)
    assert abs(adv-fdv).max()<2e-8


def test_JAX_memory_budget_and_x64_guard():
    v,f=icosphere(3)
    with pytest.raises(MemoryError):solve_nodal(v,f,[1,0,0])
    v,f=icosphere(0)
    before=jax.config.x64_enabled
    try:
        jax.config.update('jax_enable_x64',False)
        with pytest.raises(ValueError,match='float64'):solve_nodal(v,f,[1,0,0])
    finally:
        jax.config.update('jax_enable_x64',before)
