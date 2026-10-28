import numpy as np
import pytest
from panel.p1_closed_body import (prepare_mesh, influence_block, collocation, assemble,
                                  solve, surface_values, potential, velocity_fd)
from panel.run_p1_closed import icosphere, directions, sphere_velocity


def test_real_shared_node_p1_solve_not_p0_interpolation():
    v,f=icosphere(1);out=solve(v,f);d=out['diagnostics']
    assert out['mu'].shape==(42,)
    assert d['n_equations']==80 and d['matrix_rank']==42
    assert not d['gauge_applied']
    assert d['constant_density_identity_error']<2e-14
    assert d['constant_mode_response_norm']==pytest.approx(np.sqrt(80))
    assert d['qr_stationarity_relative_residual']<1e-13
    assert 1e-5<d['weighted_bie_relative_residual']<.01
    assert np.linalg.norm(out['mu']-.5*v[:,0])>1e-4  # actually solved, not injected truth
    assert np.sqrt(np.mean((out['mu']-.5*v[:,0])**2))<.02


@pytest.mark.parametrize('level',[0,1,2])
def test_partition_rank_and_prescribed_nodal_manufactured_rhs(level):
    v,f=icosphere(level);m=prepare_mesh(v,f);ops=assemble(m,samples_per_face=3)
    A,w=ops['A'],ops['weights'];u=.3+v@np.array([.17,-.28,.51])
    Q,R=np.linalg.qr(w[:,None]*A,mode='reduced')
    recovered=np.linalg.solve(R,Q.T@(w*(A@u)))
    assert abs(recovered-u).max()<2e-12
    assert abs(A@np.ones(len(v))+1).max()<3e-14
    assert np.linalg.svd(R,compute_uv=False)[-1]>1e-4


def test_independent_p1_Duffy_influence_moments():
    from numpy.polynomial.legendre import leggauss
    v,f=icosphere(0);m=prepare_mesh(v,f);q=np.array([[1.3,.2,.4]])
    S,D0,D1=influence_block(q,m)
    z,w=leggauss(36);z=(z+1)/2;w=w/2
    for j in (0,3,17):
        tri=m.triangles[j];u=z[:,None];b=(1-u)*z[None,:]
        B=np.stack(np.broadcast_arrays(1-u-b,u,b),axis=-1)
        x=np.einsum('ija,ak->ijk',B,tri);r=q[0]-x;R=np.linalg.norm(r,axis=-1)
        wg=(w[:,None]*w[None,:])*(1-u)*2*m.areas[j]
        d=np.einsum('ijk,k->ij',r,m.normals[j])/(4*np.pi*R**3)
        exact=np.sum((wg*d)[:,:,None]*B,axis=(0,1))
        source=np.sum(wg/(4*np.pi*R))
        assert np.max(abs(exact-D1[0,j]))<1e-11
        assert abs(source-S[0,j])<1e-11
        assert abs(exact.sum()-D0[0,j])<1e-11


def test_strict_face_interior_trace_and_kernel_partition():
    v,f=icosphere(0);m=prepare_mesh(v,f)
    x=np.array([.2,.3,.5])@m.triangles[0]
    S,D,D1=influence_block(x[None],m,self_faces=np.array([0]))
    assert D[0,0]==-.5
    assert np.allclose(D1[0,0],-.5*np.array([.2,.3,.5]),atol=1e-14)
    assert np.max(abs(D1.sum(axis=2)-D))<1e-14
    with pytest.raises(ValueError):
        influence_block(m.vertices[[0]],m,self_faces=np.array([0]))


def test_source_and_constant_doublet_match_original_JAX_kernel():
    import jax;import jax.numpy as jnp
    jax.config.update('jax_enable_x64',True)
    from panel.kernel import triangle_potential
    v,f=icosphere(1);m=prepare_mesh(v,f)
    S,D,P= influence_block(m.centers,m,self_faces=np.arange(len(f)))
    oldS,oldD=triangle_potential(jnp.asarray(m.centers),jnp.asarray(m.triangles),interior_diagonal=True)
    assert np.max(abs(np.asarray(oldS)-S))<2e-14
    assert np.max(abs(np.asarray(oldD)-D))<2e-14
    oldmu=np.linalg.solve(np.asarray(oldD),np.asarray(oldS)@(-m.normals[:,0]))
    control=solve(v,f,order=0)
    assert np.max(abs(oldmu-control['mu']))<2e-13


def test_unused_wake_coordinates_are_never_wall_unknowns():
    v,f=icosphere(0);extra=np.r_[v,[[10,20,30],[90,4,3]]]
    a=solve(v,f);b=solve(extra,f)
    assert b['mesh'].audit['ignored_unreferenced_vertices']==2
    assert len(b['mu'])==12 and np.allclose(a['mu'],b['mu'])


@pytest.mark.parametrize('bad',['open','inward','flip','duplicate','indices','degenerate','budget','noninteger'])
def test_reject_illegal_mesh_without_invented_gauge(bad):
    v,f=icosphere(0);kw={}
    if bad=='open': f=f[:-1]
    if bad=='inward': f=f[:,[0,2,1]]
    if bad=='flip': f[0]=f[0,::-1]
    if bad=='duplicate': f=np.r_[f,f[:1]]
    if bad=='indices': f[0,0]=999
    if bad=='degenerate': v[f[0,2]]=v[f[0,1]]
    if bad=='budget': kw['max_nodes']=4
    if bad=='noninteger': f=f.astype(float)
    with pytest.raises((ValueError,MemoryError)):prepare_mesh(v,f,**kw)


def test_tetrahedron_closed_rank_without_corner_coefficient():
    v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
    f=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
    s=solve(v,f,samples_per_face=3)
    assert s['diagnostics']['matrix_rank']==4
    assert s['diagnostics']['gauge_applied'] is False
    assert s['diagnostics']['constant_density_identity_error']<1e-14


def test_shape_scale_rotation_and_flow_superposition():
    v,f=icosphere(0);u=np.array([.4,-.2,.7])
    R=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    a=solve(v,f,u);b=solve(2*v@R.T+[3,-2,1],f,u@R.T)
    assert np.max(abs(b['mu']-2*a['mu']))<1e-13
    assert np.max(abs(surface_values(b)['cp']-surface_values(a)['cp']))<1e-13
    c=solve(v,f,[1,.2,-.3]);d=solve(v,f,u+np.array([1,.2,-.3]))
    assert np.max(abs(a['mu']+c['mu']-d['mu']))<2e-13


def test_sphere_nearfield_p1_improves_same_mesh_p0_and_converges():
    p=directions(24)*1.02;truth=sphere_velocity(p);cp=1-np.sum(truth**2,axis=1)
    errors=[]
    for level in (1,2):
        v,f=icosphere(level);s=solve(v,f);control=solve(v,f,order=0)
        vel=velocity_fd(s,p,step=1e-4);old=velocity_fd(control,p,step=1e-4)
        err=np.sqrt(np.mean((1-np.sum(vel**2,axis=1)-cp)**2))
        err0=np.sqrt(np.mean((1-np.sum(old**2,axis=1)-cp)**2))
        assert err<err0*.25
        errors.append(err)
        half=velocity_fd(s,p,step=5e-5)
        assert abs(vel-half).max()<1e-7
    assert errors[1]<errors[0]*.6


def test_independent_full_field_not_projected_tangent_zero_flux():
    v,f=icosphere(1);s=solve(v,f);m=s['mesh']
    p=m.centers[:10]+1e-3*np.sqrt(m.areas[:10,None])*m.normals[:10]
    vel=velocity_fd(s,p,step=1e-6)
    # LS boundary discretization has an actual nonzero defect; do not hide it
    # by measuring the explicitly tangential pressure-reconstruction velocity.
    flux=np.sum(vel*m.normals[:10],axis=1)
    assert np.max(abs(flux))>1e-5
    assert np.isfinite(flux).all()


def test_small_sphere_surface_force_symmetry_not_accuracy_proof():
    v,f=icosphere(1);s=solve(v,f);q=surface_values(s,reference_area=np.pi)
    assert np.linalg.norm(q['force_coefficient'])<1e-12
    assert abs(s['diagnostics']['source_flux'])<1e-13


def test_samples_and_memory_fail_before_assembly():
    v,f=icosphere(0);m=prepare_mesh(v,f)
    with pytest.raises(ValueError):collocation(m,2)
    with pytest.raises(MemoryError):assemble(m,max_matrix_entries=10)
    with pytest.raises(ValueError):solve(v,f,[0,0,0])


def test_cli_fresh_directory_and_overwrite_refusal(tmp_path):
    import os,subprocess,sys,json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1];out=tmp_path/'case'
    env=dict(os.environ,PYTHONPATH=str(root),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    cmd=[sys.executable,'-m','panel.run_p1_closed','--levels','0','--samples','12',
         '--offsets','.02','--out',str(out)]
    first=subprocess.run(cmd,cwd=root,env=env,capture_output=True,text=True)
    assert first.returncode==0,first.stderr
    data=json.loads((out/'report.json').read_text())
    assert data['cases'][0]['methods']['P1']['diagnostics']['n_unknowns']==12
    assert data['physical_accuracy_accepted'] is False
    before=(out/'report.json').read_bytes()
    second=subprocess.run(cmd,cwd=root,env=env,capture_output=True,text=True)
    assert second.returncode!=0
    assert (out/'report.json').read_bytes()==before
