"""Tests on a closed thick wing and actual lifting wake, not a nonlifting toy."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from panel.differentiable import PanelTopology, solve_fields, validate_configuration, linear_solve
jax.config.update('jax_enable_x64', True)


def wing_inputs():
    # Full closed diamond-section wing: 5 stations, caps, shared sharp TE nodes.
    ys = np.linspace(-1., 1., 5)
    ring = np.array([[0.,0.,0.],[.5,0.,.1],[1.,0.,0.],[.5,0.,-.1]])
    xyz = np.concatenate([ring+[0.,y,0.] for y in ys]).tolist()
    faces=[];upper=[];lower=[];te=[]
    for i in range(len(ys)-1):
        for j in range(4):
            a,b,c,d = 4*i+j,4*i+(j+1)%4,4*(i+1)+j,4*(i+1)+(j+1)%4
            faces += [[a,b,d],[a,d,c]]
        te.append([4*i+2,4*(i+1)+2])
        upper.append(8*i+2);lower.append(8*i+5)
    for end in (0,4):
        cap=len(xyz);xyz.append([.5,float(ys[end]),0.])
        for j in range(4):
            f=[cap,4*end+j,4*end+(j+1)%4]
            faces.append(f if end==4 else f[::-1])
    downstream=[]
    for y in ys:
        downstream.append(len(xyz));xyz.append([4.,float(y),0.])
    wake=[]
    for i in range(4):
        wake += [[4*i+2,downstream[i],downstream[i+1]], [4*i+2,downstream[i+1],4*(i+1)+2]]
    faces=np.array(faces);xyz=np.array(xyz)
    adjacency=[set() for f in faces]; edge_map={}
    for i,f in enumerate(faces):
        for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):
            edge_map.setdefault(tuple(sorted((a,b))), []).append(i)
    assert all(len(v)==2 for v in edge_map.values())
    for edge,(a,b) in edge_map.items():
        if list(edge) not in te:
            adjacency[a].add(b);adjacency[b].add(a)
    # Freeze a topology-only two-ring stencil; no geometry/solution in selection.
    neigh=[]
    for i,one in enumerate(adjacency):
        neigh.append(sorted((one | set().union(*(adjacency[j] for j in one))) - {i}))
    width=max(map(len,neigh)); neighbors=np.array([r+[i]*(width-len(r)) for i,r in enumerate(neigh)])
    hat=np.zeros((5,3));hat[1:4]=np.eye(3)
    topology=PanelTopology(*(jnp.asarray(x) for x in (
        faces,np.array(wake),np.array(te),np.array(upper),np.array(lower),hat,
        np.repeat(np.arange(4),2),np.repeat(np.arange(4)[:,None],2,axis=1),neighbors,
        np.full(len(faces),10),
    )))
    return dict(vertices=jnp.array(xyz),station_y=jnp.array(ys),topology=topology,
                freestream=jnp.array([1.,0.,.08]),density=jnp.array(1.225),
                reference_area=jnp.array(2.),reference_chord=jnp.array(1.),
                reference_point=jnp.array([.25,0.,0.]))


def outputs(inp):
    return solve_fields(**inp)


def test_lifting_state_raw_loads_and_linear_adjoint():
    inputs=wing_inputs();validate_configuration(inputs['vertices'],inputs['station_y'],inputs['topology'])
    out=jax.jit(solve_fields)(**inputs)
    assert np.isfinite(np.asarray(out['Cp_raw'])).all()
    assert float(jnp.linalg.norm(out['gamma'])) > 1e-3
    assert float(out['CL']) > 0
    np.testing.assert_allclose(out['linear_residual'],0,atol=1e-11)
    np.testing.assert_allclose(out['component_forces'].sum(axis=0),out['force'],atol=1e-12)
    np.testing.assert_allclose(out['component_moments'].sum(axis=0),out['moment'],atol=1e-12)
    A=out['matrix'];b=out['rhs'];g=jnp.linspace(.2,1.,len(b))
    adj=jax.grad(lambda rhs:jnp.dot(g,linear_solve(A,rhs)))(b)
    np.testing.assert_allclose(A.T@adj,g,rtol=1e-11,atol=1e-11)


def test_local_force_moment_vjp_to_vertices_and_moving_stations():
    inp=wing_inputs()
    nf=inp['topology'].faces.shape[0]
    rng=np.random.default_rng(19)
    w=jnp.array(rng.normal(size=(nf,3)));z=jnp.array(rng.normal(size=(nf,3)))
    def local(x,y):
        out=outputs({**inp,'vertices':x,'station_y':y})
        return out['panel_forces'],out['panel_moments']
    (force,moment),back=jax.vjp(local,inp['vertices'],inp['station_y'])
    gx,gy=back((w,z))
    assert np.isfinite(np.asarray(gx)).all() and np.isfinite(np.asarray(gy)).all()
    # Compatible nonlinear span motion + sweep + local thickness: cut remains z=0.
    x=inp['vertices'];y=inp['station_y']
    dx=jnp.stack((.03*x[:,0]+.02*x[:,1]**2, .05*x[:,1]**3, .04*x[:,2]*(1+x[:,1]**2)),axis=1)
    dy=.05*y**3
    ad=float(jnp.sum(gx*dx)+jnp.sum(gy*dy))
    assert abs(ad)>1e-6
    def loss(x,y):
        f,m=local(x,y);return jnp.sum(f*w)+jnp.sum(m*z)
    for h in (1e-3,1e-4,1e-5):
        validate_configuration(x+h*dx,y+h*dy,inp['topology'])
        fd=float((loss(x+h*dx,y+h*dy)-loss(x-h*dx,y-h*dy))/(2*h))
        np.testing.assert_allclose(fd,ad,rtol=2e-5,atol=1e-8)
    # Vertex-only change, with no change to the wake or station graph.
    node=1;axis=2;h=1e-5
    xp=x.at[node,axis].add(h);xm=x.at[node,axis].add(-h)
    fd=float((loss(xp,y)-loss(xm,y))/(2*h))
    np.testing.assert_allclose(fd,float(gx[node,axis]),rtol=2e-5,atol=1e-8)


def test_density_speed_references_and_coordinate_scale_laws():
    inp=wing_inputs();out=outputs(inp)
    fast=outputs({**inp,'freestream':3*inp['freestream'],'density':2*inp['density']})
    np.testing.assert_allclose(fast['Cp_raw'],out['Cp_raw'],rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(fast['panel_forces'],18*out['panel_forces'],rtol=1e-10,atol=1e-10)
    scale=3.;shift=jnp.array([4.,-.6,.2])
    scaled=outputs({**inp,'vertices':scale*inp['vertices']+shift,
        'station_y':scale*inp['station_y']+shift[1],
        'reference_point':scale*inp['reference_point']+shift,
        'reference_area':scale**2*inp['reference_area'],'reference_chord':scale*inp['reference_chord']})
    np.testing.assert_allclose(scaled['Cp_raw'],out['Cp_raw'],rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(scaled['panel_forces'],scale**2*out['panel_forces'],rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(scaled['panel_moments'],scale**3*out['panel_moments'],rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(scaled['CL'],out['CL'],rtol=1e-10,atol=1e-10)
    grad=jax.grad(lambda p: outputs({**inp,'reference_point':p})['moment'][1])(inp['reference_point'])
    force=np.asarray(out['force'])
    np.testing.assert_allclose(grad,[force[2],0.,-force[0]],rtol=1e-10,atol=1e-10)
    area_grad=jax.grad(lambda a: outputs({**inp,'reference_area':a})['CL'])(inp['reference_area'])
    np.testing.assert_allclose(area_grad,-out['CL']/inp['reference_area'],rtol=1e-10)


def test_invalid_strip_or_nonplanar_wake_is_not_silently_projected():
    inp=wing_inputs();x=inp['vertices'];y=inp['station_y'];t=inp['topology']
    with pytest.raises(ValueError,match='planar'):
        validate_configuration(x.at[-1,2].add(.1),y,t)
    with pytest.raises(ValueError,match='strips'):
        validate_configuration(x,y.at[2].add(.1),t)


def test_operating_point_gradients_obey_incompressible_load_homogeneity():
    inp=wing_inputs()
    w=jnp.array(np.random.default_rng(71).normal(size=(len(inp['topology'].faces),3)))
    def objective(velocity,rho):
        return jnp.sum(outputs({**inp,'freestream':velocity,'density':rho})['panel_forces']*w)
    value,(gv,grho)=jax.value_and_grad(objective,argnums=(0,1))(inp['freestream'],inp['density'])
    np.testing.assert_allclose(grho,value/inp['density'],rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(jnp.dot(gv,inp['freestream']),2*value,rtol=1e-10,atol=1e-10)
    assert np.isfinite(np.asarray(gv)).all()


def test_matches_unchanged_legacy_continuous_function_on_same_lifting_mesh():
    """Extract only the real legacy function AST; do not execute its CLI/imports."""
    import ast
    from pathlib import Path
    from panel.kernel import triangle_potential
    path=Path(__file__).resolve().parents[1]/'panel'/'solve.py'
    nodes=ast.parse(path.read_text(encoding='utf-8')).body
    function=next(x for x in nodes if isinstance(x,ast.FunctionDef) and x.name=='solve_model')
    inp=wing_inputs();t=inp['topology'];x=inp['vertices'];y=inp['station_y']
    nf=len(t.faces);ng=t.hat_map.shape[1]
    wi=t.wake_interval
    fraction=(x[t.wake_faces].mean(axis=1)[:,1]-y[wi])/(y[wi+1]-y[wi])
    basis=(1-fraction[:,None])*t.hat_map[wi]+fraction[:,None]*t.hat_map[wi+1]
    slope=(t.hat_map[wi+1]-t.hat_map[wi])/(y[wi+1]-y[wi])[:,None]
    gauss=np.array([.5-np.sqrt(3)/6,.5+np.sqrt(3)/6])
    ty=x[t.te_edges[:,0],1,None]*(1-gauss)+x[t.te_edges[:,1],1,None]*gauss
    ti=t.te_interval;tf=(ty-y[ti])/(y[ti+1]-y[ti])
    te_basis=(1-tf[...,None])*t.hat_map[ti]+tf[...,None]*t.hat_map[ti+1]
    incidence=np.zeros((len(t.te_edges),nf))
    incidence[np.arange(len(t.te_edges)),np.asarray(t.upper_te)]=1.
    incidence[np.arange(len(t.te_edges)),np.asarray(t.lower_te)]=-1.
    namespace=dict(jnp=jnp,xyz=x,faces=t.faces,wake_faces=t.wake_faces,
        wake_basis=basis,wake_slope=slope,te_edges=t.te_edges,te_basis=te_basis,
        incidence=incidence,gamma_count=ng,neighbor_ids=t.neighbors,
        reference=inp['reference_point'],sref=2.,chord=1.,
        wing_mask=np.ones(nf,dtype=bool),body_mask=np.zeros(nf,dtype=bool),
        seam_faces=np.array([],dtype=int),triangle_potential=triangle_potential)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
    legacy=namespace['solve_model']
    for alpha,bulge,span in ((5.,0.,1.),(4.,.1,1.1)):
        old=legacy(jnp.array([alpha,bulge,span]))
        changed=x.at[:,1].multiply(span)
        changed=changed.at[:,2].set(x[:,2]*(1+bulge*jnp.exp(-(x[:,1]/.5)**2)))
        angle=np.deg2rad(alpha)
        new=outputs({**inp,'vertices':changed,'station_y':y*span,
            'freestream':jnp.array([np.cos(angle),0.,np.sin(angle)]),'density':jnp.array(2.)})
        for key in ('matrix','rhs','mu','gamma','Cp_raw','panel_forces','panel_moments'):
            np.testing.assert_allclose(new[key],old[key],rtol=2e-11,atol=2e-11,err_msg=key)
