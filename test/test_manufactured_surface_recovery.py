import builtins
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from panel import manufactured_surface_recovery as m
from panel import pressure_recovery_ab as r


def half_sphere(level=1):
    x = [np.array(p, float) for p in ((0,1,0),(1,0,0),(0,0,1),(-1,0,0),(0,0,-1))]
    faces = [[0, 1+i, 1+(i+1)%4] for i in range(4)]
    for face in faces:
        a, b, c = (x[i] for i in face)
        if np.dot(np.cross(b-a, c-a), (a+b+c)/3) < 0:
            face[1], face[2] = face[2], face[1]
    for _ in range(level):
        midpoint, new = {}, []
        def mid(i, j):
            key = tuple(sorted((i,j)))
            if key not in midpoint:
                v = (x[i]+x[j])/2
                midpoint[key] = len(x)
                x.append(v/np.linalg.norm(v))
            return midpoint[key]
        for a, b, c in faces:
            ab, bc, ca = mid(a,b), mid(b,c), mid(c,a)
            new += [[a,ab,ca],[ab,b,bc],[ca,bc,c],[ab,bc,ca]]
        faces = new
    return np.asarray(x), np.asarray(faces, dtype=np.int64)


def neighbors_for_full(faces, nf):
    edge_map = {}
    for i, face in enumerate(faces):
        for a, b in zip(face, np.roll(face, -1)):
            edge_map.setdefault(tuple(sorted((int(a),int(b)))), []).append(i)
    adjacency = [set() for _ in faces]
    for group in edge_map.values():
        for i in group:
            adjacency[i].update(j for j in group if j != i)
    lists = []
    for i in range(nf):
        seen, front = {i}, {i}
        for _ in range(2):
            front = set().union(*(adjacency[j] for j in front))-seen
            seen.update(front)
        lists.append(sorted(seen-{i}))
    width = max(map(len, lists))
    return np.array([ls+[i]*(width-len(ls)) for i, ls in enumerate(lists)], dtype=np.int64)


def synthetic_inputs(level=1):
    x, f = half_sphere(level)
    nf = len(f)
    fixture = dict(schema='dpanel.half_planar_cut.v1', coordinates=x.tolist(), faces=f.tolist(),
                   cut_edges=[], trailing_edges=[], upper_te_faces=[0], lower_te_faces=[1],
                   freestream=[1.,0.,.1], moment_reference=[.1,0.,0.], reference_chord=1.,
                   reference_area=2., density=1.2, component_ids=[20 if i%2 else 30 for i in range(nf)])
    # Artificial TE face-group IDs here only test report grouping, NOT a wake/Kutta case.
    xx, ff, ee = r.mirror_half_wall(x, f, [])
    ids = neighbors_for_full(ff, nf)
    mu = np.sin(x[f].mean(axis=1)[:,0])*.2
    control = r.affine_face_gradient(xx,ff,np.r_[mu,mu],ids)
    cp = r.surface_cp(control, r.mesh_geometry(x,f)[-1], fixture['freestream'])
    _, _, patches = r.cut_vertex_gradient(xx,ff,np.r_[mu,mu],ee,max_rings=4)
    return fixture, ids, mu, cp, patches


def write_synthetic_run(path):
    path.mkdir()
    fixture, ids, mu, cp, patches = synthetic_inputs()
    raw = (json.dumps(fixture)+'\n').encode()
    (path/'surface_input.json').write_bytes(raw)
    source_sha = hashlib.sha256(raw).hexdigest()
    state = dict(mu_half=mu, gamma_half=np.array([.1]), sigma_half=np.zeros_like(mu),
                 Cp_raw_half=cp, neighbor_ids=ids, surface_sha256=np.array(source_sha),
                 true_relative_residual=np.array(0.))
    np.savez(path/'frozen_state.npz', **state)
    np.savez(path/'recovered_fields.npz', original_cp=cp, candidate_cp=cp)
    (path/'patches.json').write_text(json.dumps(patches))
    report = dict(status='postprocessing_comparison_not_physical_validation',
                  source_sha256=source_sha, frozen_state_hash=m._state_hash(state), max_rings=4)
    (path/'report.json').write_text(json.dumps(report))
    return fixture, state


@pytest.mark.parametrize('mode', m.MODES)
def test_harmonic_even_potential_and_gradient_fd(mode):
    rng = np.random.default_rng(55)
    x = rng.normal(size=(17,3))
    args = dict(speed=2.3,chord=.7,origin=[.2,0.,-.1],amplitude=.1)
    phi, g = m.known_potential(x,mode,**args)
    mirrored, gm = m.known_potential(x*m.REFLECT,mode,**args)
    np.testing.assert_allclose(phi,mirrored,atol=1e-15)
    np.testing.assert_allclose(gm,g*m.REFLECT,atol=1e-15)
    h = 1e-5
    for j in range(3):
        e = np.eye(3)[j]*h
        fd = (m.known_potential(x+e,mode,**args)[0]-m.known_potential(x-e,mode,**args)[0])/(2*h)
        np.testing.assert_allclose(fd,g[:,j],rtol=1e-8,atol=3e-11)
    assert np.trace(m.polynomial_coefficients(mode)[2]) == 0


def test_triangle_rule_integrates_all_monomials_through_degree_four():
    tri = np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
    pts,w = m.triangle_rule(tri,3)
    for p in range(5):
        for q in range(5-p):
            value = np.dot(w,pts[0,:,0]**p*pts[0,:,1]**q)
            exact = 2*math.factorial(p)*math.factorial(q)/math.factorial(p+q+2)
            assert value == pytest.approx(exact,abs=2e-15)
    assert np.all(w>0)


def test_known_quadratic_force_and_moment_and_centroid_bias():
    tri = np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
    pts,w = m.triangle_rule(tri)
    a=.1; rho=2.; flow=np.array([1.,0.,0.])
    _,g=m.known_potential(pts,'quadratic_x2_z2',speed=1.,chord=1.,origin=[0.,0.,0.],amplitude=a)
    cp=1-np.sum((flow+g)**2,axis=2)
    f,mo=m.face_loads(pts,cp,w,np.array([[0.,0.,1.]]),np.array([.5]),flow,rho,np.zeros(3))
    mean_cp=-4*a/3-4*a*a/6
    mean_xcp=-4*a/6-4*a*a/10
    assert f[0,2] == pytest.approx(-.5*mean_cp,abs=1e-14)
    assert mo[0,1] == pytest.approx(.5*mean_xcp,abs=1e-14)
    cp_center=1-(1+2*a/3)**2
    assert abs(cp_center-mean_cp) > .002


def test_p1_is_exact_for_ambient_affine_on_curved_facets():
    x,f=half_sphere(2)
    tri=x[f]; n=r.mesh_geometry(x,f)[-1]
    b=np.array([.2,.5,-.1]); values=tri@b+9.
    g=m.p1_gradient(tri,values)
    expected=m.project_tangent(np.broadcast_to(b,n.shape),n)
    np.testing.assert_allclose(g,expected,rtol=1e-12,atol=1e-13)


def test_planar_affine_both_operators_and_gauge():
    x=np.array([[i/3,j/3,0.] for j in range(4) for i in range(4)])
    f=[]
    for j in range(3):
        for i in range(3):
            a=j*4+i; f.extend([[a,a+1,a+5],[a,a+5,a+4]])
    f=np.array(f); center=x[f].mean(axis=1)
    mu=center@np.array([.3,.7,0.])
    ids=np.tile(np.arange(len(f)),(len(f),1))
    for offset in (0.,17.):
        face=r.affine_face_gradient(x,f,mu+offset,ids)
        node,_,_=r.cut_vertex_gradient(x,f,mu+offset,[],max_rings=4)
        np.testing.assert_allclose(face,np.tile([.3,.7,0.],(len(f),1)),atol=1e-13)
        np.testing.assert_allclose(node,face,atol=1e-12)


def test_cut_is_retained_for_known_jump():
    x=np.array([[i/3,j/3,0.] for j in range(4) for i in range(4)])
    f=[]
    for j in range(3):
        for i in range(3):
            a=j*4+i; f.extend([[a,a+1,a+5],[a,a+5,a+4]])
    f=np.array(f); centers=x[f].mean(axis=1)
    cuts=np.array([[2,6],[6,10],[10,14]],dtype=int)
    mu=centers@np.array([.3,.7,0.])+4.*(centers[:,0]>2/3)
    split,_,_=r.cut_vertex_gradient(x,f,mu,cuts,max_rings=4)
    mixed,_,_=r.cut_vertex_gradient(x,f,mu,[],max_rings=4)
    np.testing.assert_allclose(split,np.tile([.3,.7,0.],(len(f),1)),atol=1e-12)
    assert np.max(np.abs(mixed-split)) > 1.


def test_analytic_p1_rotates_and_translates_covariantly():
    x,f=half_sphere(1); tri=x[f]
    val=tri@np.array([.3,.2,-.1])
    g=m.p1_gradient(tri,val)
    angle=.42
    rot=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
    moved=tri@rot.T+[3.,-1.,2.]
    np.testing.assert_allclose(m.p1_gradient(moved,val),g@rot.T,atol=1e-13)


def test_mirror_force_and_axial_moment_match_explicit_full():
    rng=np.random.default_rng(8)
    x=rng.normal(size=(7,3)); x[:,1]=np.abs(x[:,1])
    f=rng.normal(size=(7,3)); p=np.array([.3,0.,-.2])
    moment=np.cross(x-p,f)
    force_full=f+f*m.REFLECT
    moment_full=moment+np.cross(x*m.REFLECT-p,f*m.REFLECT)
    rows=m.coefficient_rows(f,moment,flow=np.array([2.,0.,.1]),density=1.2,area_ref=3.,chord=.7)
    qS=.5*1.2*(4.01)*3
    np.testing.assert_allclose(rows[:,0],force_full@np.array([2.,0.,.1])/np.sqrt(4.01)/qS)
    np.testing.assert_allclose(rows[:,2],moment_full[:,1]/(qS*.7),atol=1e-15)


def test_analyze_constant_affine_oracle_closure_and_no_mutation():
    fixture,ids,mu,cp,patches=synthetic_inputs()
    before=json.dumps(fixture,sort_keys=True); old=ids.copy()
    report,arrays=m.analyze(fixture,ids,max_rings=4,origin=[0.,0.,0.],expected_patches=patches)
    assert not report['physical_accuracy_accepted']
    assert before==json.dumps(fixture,sort_keys=True)
    np.testing.assert_array_equal(old,ids)
    for mode in m.MODES:
        for name,data in report['modes'][mode]['operators'].items():
            assert data['groups']['all']['decomposition_max_abs'] < 1e-13
            assert data['cp_error_identity_max_abs'] < 1e-12
            if mode=='constant' or (mode.startswith('linear') and name=='exact_vertex_p1_reference'):
                assert data['groups']['all']['gradient_max_over_U'] < 1e-12
    # Not all operators must reproduce ambient affine restrictions on curved stencils.
    assert report['modes']['linear_x']['operators']['affine_qr']['groups']['all']['gradient_rms_over_U'] > 1e-5


def test_similarity_scaling_preserves_dimensionless_results():
    fixture,ids,_,_,patches=synthetic_inputs()
    a,_=m.analyze(fixture,ids,max_rings=4,origin=[0.,0.,0.])
    moved=json.loads(json.dumps(fixture)); scale=3.; translation=np.array([2.,0.,-1.])
    moved['coordinates']=(np.array(moved['coordinates'])*scale+translation).tolist()
    moved['moment_reference']=(np.array(moved['moment_reference'])*scale+translation).tolist()
    moved['reference_chord']*=scale; moved['reference_area']*=scale**2
    moved['freestream']=(np.array(moved['freestream'])*7.).tolist(); moved['density']*=4
    b,_=m.analyze(moved,ids,max_rings=4,origin=translation)
    for mode in m.MODES:
        for name in a['modes'][mode]['operators']:
            aa=a['modes'][mode]['operators'][name]['groups']['all']
            bb=b['modes'][mode]['operators'][name]['groups']['all']
            assert aa['gradient_rms_over_U']==pytest.approx(bb['gradient_rms_over_U'],abs=1e-13)
            for key in aa['coefficients']:
                for metric in aa['coefficients'][key]:
                    assert aa['coefficients'][key][metric]==pytest.approx(bb['coefficients'][key][metric],abs=2e-13)


@pytest.mark.parametrize('bad', ['sideslip','origin','neighbors','reference','patch'])
def test_invalid_configuration_refused(bad):
    fixture,ids,_,_,patches=synthetic_inputs()
    origin=[0.,0.,0.]
    if bad=='sideslip': fixture['freestream'][1]=.1
    if bad=='origin': origin[1]=.1
    if bad=='neighbors': ids=ids.astype(float)
    if bad=='reference': fixture['density']=-1
    if bad=='patch': patches[0]['depth']+=1
    with pytest.raises(ValueError):
        m.analyze(fixture,ids,max_rings=4,origin=origin,expected_patches=patches)


def test_cli_replay_is_backend_free_and_deterministic(tmp_path,monkeypatch):
    root=tmp_path/'source'; write_synthetic_run(root)
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.split('.')[0] in ('jax','torch','numba','scipy','gmsh'):
            raise AssertionError('Backend imported during read-only manufactured audit')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    inputs={p.name:m._sha(p) for p in root.iterdir()}
    assert m.main(['--recovery-run',str(root),'--out',str(tmp_path/'out')])==0
    assert m.main(['--recovery-run',str(root),'--out',str(tmp_path/'repeat')])==0
    assert inputs=={p.name:m._sha(p) for p in root.iterdir()}
    with np.load(tmp_path/'out'/'manufactured_fields.npz') as a, np.load(tmp_path/'repeat'/'manufactured_fields.npz') as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key],b[key])
    with pytest.raises(FileExistsError):
        m.run(root,tmp_path/'out')


@pytest.mark.parametrize('target', ['surface','state','cp','residual'])
def test_tampering_refused_before_output(tmp_path,target):
    root=tmp_path/'source'; write_synthetic_run(root)
    if target=='surface':
        j=json.loads((root/'surface_input.json').read_text()); j['density']*=1.01
        (root/'surface_input.json').write_text(json.dumps(j))
    elif target in ('state','residual'):
        with np.load(root/'frozen_state.npz') as data: d={k:data[k].copy() for k in data.files}
        if target=='state': d['mu_half'][0]+=.01
        else: d['true_relative_residual']=np.array(1e-3)
        np.savez(root/'frozen_state.npz',**d)
    else:
        with np.load(root/'recovered_fields.npz') as data: d={k:data[k].copy() for k in data.files}
        d['original_cp'][0]+=.01; np.savez(root/'recovered_fields.npz',**d)
    with pytest.raises(ValueError): m.run(root,tmp_path/'rejected')
    assert not (tmp_path/'rejected').exists()


def test_resource_and_rank_guards(tmp_path):
    root=tmp_path/'source'; fixture,_=write_synthetic_run(root)
    with pytest.raises(MemoryError): m.run(root,tmp_path/'oversize',max_half_panels=1)
    x=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    with pytest.raises(ValueError,match='rank deficient'):
        r.cut_vertex_gradient(x,np.array([[0,1,2]]),[1.],[],max_rings=0)


def test_polynomial_invalid_arguments():
    args=dict(speed=1.,chord=1.,origin=[0.,0.,0.],amplitude=.1)
    with pytest.raises(ValueError): m.known_potential(np.zeros((1,3)),'unknown',**args)
    args['chord']=0.
    with pytest.raises(ValueError): m.known_potential(np.zeros((1,3)),'linear_x',**args)
    with pytest.raises(ValueError): m.triangle_rule(np.zeros((1,3,3)),0)
