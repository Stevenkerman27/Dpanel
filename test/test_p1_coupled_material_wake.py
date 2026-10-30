"""B2 mathematical/unit tests plus explicit full-repository integration tests."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from panel.p1_coupled_material_wake import (
    TotalVelocity, GeometryGates, triangle_data, limits, material_paths,
    streamline_target, area_diagnostics, convex_overlap, geometry_failure,
    snapshot_hash, solve_coupled_material_wake,
)


def small_field():
    xyz = np.array([[0., 0., 0.], [1., .1, .2], [.2, .9, -.1]])
    faces = np.array([[0, 1, 2]])
    group = NS(name='wake', wake_xyz=xyz+[1.4, -.1, .3], wake_faces=faces,
               gamma_hat=np.array([[1., 0.], [.4, .6], [0., 1.]]))
    mesh = NS(xyz=xyz, faces=faces, dof_face=faces, groups=(group,))
    state = dict(mu=np.array([.2, -.1, .4]), gamma=np.array([.3, -.2]), alpha_deg=5.)
    return mesh, state


def independent_area_velocity(point, tri, values, order=60):
    """Area quadrature of grad[ integral mu n.r/(4*pi*r^3) dS ]."""
    q, w = np.polynomial.legendre.leggauss(order)
    q, w = (q+1)/2, w/2
    u, v = np.meshgrid(q, q, indexing='ij')
    weights = w[:, None]*w[None, :]*(1-u)
    bary = np.stack(((1-u)*(1-v), u, (1-u)*v), axis=-1)
    source = bary@tri
    mu = bary@values
    r = point-source
    rn = np.linalg.norm(r, axis=-1)
    av = np.cross(tri[1]-tri[0], tri[2]-tri[0])
    integrand = av/rn[..., None]**3 - 3*(r@av)[..., None]*r/rn[..., None]**5
    return np.sum((weights*mu)[..., None]*integrand, axis=(0, 1))/(4*np.pi)


def test_total_velocity_matches_independent_linear_area_integral():
    mesh, state = small_field()
    field = TotalVelocity(mesh, state)
    point = np.array([.3, -.4, 1.7])
    expected = field.free + independent_area_velocity(point, mesh.xyz, state['mu'])
    g = mesh.groups[0]
    expected += independent_area_velocity(point, g.wake_xyz, g.gamma_hat@state['gamma'])
    np.testing.assert_allclose(field(point), expected, atol=2e-12, rtol=2e-12)


def test_multiple_groups_contribute_once_not_independent_lift_superposition():
    mesh, state = small_field()
    p = np.array([.2, -.5, 2.])
    one = TotalVelocity(mesh, state)
    mesh2 = NS(**vars(mesh))
    mesh2.groups = (mesh.groups[0], mesh.groups[0])
    state2 = dict(state, gamma=np.tile(state['gamma'], 2))
    two = TotalVelocity(mesh2, state2)
    wallmesh = NS(**vars(mesh), )
    wallmesh.groups = ()
    wall = TotalVelocity(wallmesh, dict(state, gamma=np.array([])))
    np.testing.assert_allclose(two(p)-wall(p), 2*(one(p)-wall(p)), atol=1e-14)


def test_material_strength_gradient_recomputed_not_carried_in_global_axes():
    mesh, state = small_field()
    g = mesh.groups[0]
    old = TotalVelocity(mesh, state)
    g.wake_xyz = g.wake_xyz*np.array([1., 2., 3.])
    new = TotalVelocity(mesh, state)
    tri, _n, _a, shape, _e = triangle_data(g.wake_xyz, g.wake_faces)
    nodal = g.gamma_hat@state['gamma']
    np.testing.assert_allclose(new.grad_face[-1], nodal[g.wake_faces][0]@shape[0])
    assert not np.allclose(new.grad_face[-1], old.grad_face[-1])
    recovered = new.mu_face[-1] + (tri[0]-tri[0].mean(axis=0))@new.grad_face[-1]
    np.testing.assert_allclose(recovered, nodal, atol=1e-14)


class UniformField:
    free = np.array([1., 0., 0.])
    # TotalVelocity records the velocity method in the sampling kernel; the mock
    # field advertises the production default so the coupling diagnostics take
    # the analytic-edge branch.
    method = 'analytic_edges'
    def __call__(self, p, *, order):
        return self.free.copy()


@pytest.mark.parametrize('ladder', [(1., 0.), (1., -1.), (1., 1.), (1., np.nan), (1.,)])
def test_invalid_epsilon_ladders_rejected(ladder):
    with pytest.raises(ValueError):
        limits(UniformField(), np.zeros((1, 3)), np.array([[0., 0., 1.]]), np.ones(1),
               eps_fraction=.05, ladder=ladder, order=48)


def test_constant_field_limits_exact_and_quadrature_audited():
    v = limits(UniformField(), np.zeros((1, 3)), np.array([[0., 0., 1.]]), np.ones(1),
               eps_fraction=.05, ladder=(1., .5), order=48)
    np.testing.assert_array_equal(v['mean'], [[1., 0., 0.]])
    assert v['epsilon_error'][0] == v['quadrature_error'][0] == 0


def path_fixture():
    xyz = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.],
                    [0., 1., 0.], [1., 1., 0.], [2., 1., 0.]])
    faces = np.array([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]])
    mat = NS(reference_xyz=xyz.copy(), roots=np.array([0, 3]), used=np.arange(6),
             predecessor=np.array([-1, 0, 1, -1, 3, 4]), order=np.array([0, 3, 1, 4, 2, 5]),
             panel_area=triangle_data(xyz, faces)[2])
    return xyz, faces, mat


def test_aligned_uniform_wake_is_fixed_point_not_permanent_convection():
    xyz, _f, mat = path_fixture()
    target = streamline_target(xyz, mat, np.tile([1., 0., 0.], (6, 1)), np.ones(6, bool))
    np.testing.assert_array_equal(target, xyz)
    s, r = material_paths(mat)
    np.testing.assert_array_equal(s, [0, 1, 2, 0, 1, 2])
    np.testing.assert_array_equal(r, [0, 0, 0, 3, 3, 3])


def test_negative_vx_has_no_axial_division_and_roots_stay_fixed():
    xyz, _f, mat = path_fixture()
    target = streamline_target(xyz, mat, np.tile([-1., .2, .3], (6, 1)), np.ones(6, bool))
    assert np.isfinite(target).all()
    np.testing.assert_array_equal(target[mat.roots], xyz[mat.roots])
    assert np.all(np.abs(target[1]-xyz[1]) > 0)


def test_ineligible_nodes_are_not_silently_marched():
    xyz, _f, mat = path_fixture()
    target = streamline_target(xyz, mat, np.tile([1., 1., 1.], (6, 1)), np.zeros(6, bool))
    np.testing.assert_array_equal(target, xyz)


def test_area_weighting_relative_bands_and_identity_residual_not_jump():
    normal = np.tile([0., 0., 1.], (2, 1))
    plus = np.array([[2., 0., 1.], [2., 0., 3.]])
    minus = np.array([[1., 0., 1.], [1., 0., 3.]])
    sample = dict(plus=plus, minus=minus, mean=(plus+minus)/2,
                  epsilon_error=np.zeros(2), quadrature_error=np.zeros(2), eps=np.ones(2)*.001)
    report = area_diagnostics(sample, normal, np.array([1., 3.]), np.array([.2, .8]))
    assert report['overall']['slip_rms'] == pytest.approx(np.sqrt(7))
    assert report['overall']['identity_residual_rms'] == 0
    assert report['overall']['dcp_rms'] == 3
    assert sum(b['area_fraction'] for b in report['bands']) == 1
    assert report['bands'][1]['faces'] == 0


def test_convex_overlap_detects_swept_crossing_and_coplanar_contact():
    tri = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    swept = np.vstack((tri+[0, 0, -1.], tri+[0, 0, 1.]))
    assert convex_overlap(swept, tri, 1e-10)
    assert convex_overlap(tri, tri+[.1, .1, 0.], 1e-10)
    assert not convex_overlap(swept, tri+[3., 3., 0.], 1e-10)
    # AABB overlaps, triangles do not: exercise in-plane SAT axes.
    assert not convex_overlap(tri, np.array([[.8, .8, 0.], [1.8, .8, 0.], [.8, 1.8, 0.]]), 1e-10)


def geometry_fixture():
    xyz, faces, mat = path_fixture()
    group = NS(name='wake', wake_xyz=xyz, wake_faces=faces)
    wall = np.array([[-3., 0., -3.], [-2., 0., -3.], [-3., 1., -3.]])
    mesh = NS(groups=(group,), xyz=wall, faces=np.array([[0, 1, 2]]), reference_chord=1.)
    return mesh, {'wake': mat}, {'wake': xyz.copy()}


def test_geometry_rejects_degeneracy_without_raising_or_partial_commit():
    mesh, mats, coordinates = geometry_fixture()
    coordinates['wake'][4] = coordinates['wake'][1]
    gate = GeometryGates(max_step_fraction=100.)
    assert geometry_failure(mesh, coordinates, mats, gate) == 'area'
    assert not np.array_equal(mesh.groups[0].wake_xyz, coordinates['wake'])


@pytest.mark.parametrize('change,reason', [('root', 'root'), ('step', 'step'),
                                         ('attachment', 'attachment'), ('nonfinite', 'nonfinite')])
def test_geometry_refusal_reasons(change, reason):
    mesh, mats, coordinates = geometry_fixture()
    if change == 'root':
        coordinates['wake'][0, 2] = .1
    elif change == 'step':
        coordinates['wake'][2, 2] = .9
    elif change == 'attachment':
        coordinates['wake'][1, 0] = -1.
    else:
        coordinates['wake'][1, 2] = np.nan
    assert geometry_failure(mesh, coordinates, mats, GeometryGates()) == reason


def test_snapshot_hash_covers_geometry_and_strength_without_zip_metadata():
    arrays = dict(mu=np.array([1., 2.]), xyz=np.zeros((3, 3)))
    assert snapshot_hash(arrays) == snapshot_hash(dict(reversed(list(arrays.items()))))
    other = {k: v.copy() for k, v in arrays.items()}
    other['xyz'][0, 0] = 1.
    assert snapshot_hash(arrays) != snapshot_hash(other)
    with pytest.raises(ValueError):
        snapshot_hash(dict(bad=np.array([object()], dtype=object)))


def generated_mesh():
    from panel.p1_lifting_g1 import make_wing
    from panel.p1_lifting_g2 import combine_wings
    return combine_wings([make_wing(chord_panels=4, span_panels=4, wake_panels=2)])


def test_integration_phase_a_kernel_matches_total_p1_field():
    from panel.wake_velocity import sheet_velocity_sum
    mesh, state = small_field()
    field = TotalVelocity(mesh, state)
    for p in (np.array([.2, -.3, 1.4]), np.array([2., .5, -.8])):
        reference = field.free-sheet_velocity_sum(p, field.triangles, field.mu_face, field.grad_face)
        np.testing.assert_allclose(field(p), reference, atol=2e-12, rtol=2e-12)


def test_integration_same_total_potential_gradient_and_fixed_wake_iteration_zero():
    from panel.p1_steady_free_wake import _potentials_at
    from panel.p1_lifting_g2 import solve_multi
    mesh = generated_mesh()
    state = solve_multi(mesh, 5., kutta='signed_speed')
    field = TotalVelocity(mesh, state)
    point, h = np.array([.3, -.4, 2.]), 2e-5
    fd = np.array([(_potentials_at(mesh, state, (point+h*e)[None])[0]
                   -_potentials_at(mesh, state, (point-h*e)[None])[0])/(2*h) for e in np.eye(3)])
    np.testing.assert_allclose(field(point), fd, atol=2e-7, rtol=2e-7)
    result = solve_coupled_material_wake(mesh, 5., kutta='signed_speed', iterations=0)
    assert max(result['initial_equivalence'].values()) <= 1e-10
    for key in ('mu', 'gamma'):
        np.testing.assert_allclose(result['state'][key], state[key], atol=1e-9, rtol=1e-9)
    assert not result['physical_accuracy_accepted'] and not result['converged']


def test_integration_accepted_geometry_is_resolved_and_material_labels_frozen(monkeypatch):
    import panel.p1_coupled_material_wake as b2
    import panel.p1_lifting_g2 as g2
    mesh = generated_mesh()
    seen = []
    original = g2.solve_multi
    def tracked(model, *args, **kwargs):
        seen.append(model.groups[0].wake_xyz.copy())
        return original(model, *args, **kwargs)
    monkeypatch.setattr(g2, 'solve_multi', tracked)
    # Manufactured smooth flow isolates coupling/geometry from the singular-PV research question.
    class SmoothField(UniformField):
        def __init__(self, model, state, *, method='analytic_edges', core_radius=None, **kwargs):
            self.free = np.array([1., 0., .02])
            self.method = method
    monkeypatch.setattr(b2, 'TotalVelocity', SmoothField)
    result = b2.solve_coupled_material_wake(mesh, 5., kutta='signed_speed', iterations=1)
    assert result['history'][0]['outgoing_update']['accepted']
    assert len(seen) == 2  # canonical iteration 0 reused; accepted iteration 1 re-solved
    np.testing.assert_array_equal(seen[-1], result['mesh'].groups[0].wake_xyz)
    assert not np.array_equal(seen[0], seen[-1])
    np.testing.assert_array_equal(mesh.groups[0].gamma_hat, result['mesh'].groups[0].gamma_hat)
    check = original(result['mesh'], 5., kutta='signed_speed')
    np.testing.assert_allclose(check['gamma'], result['state']['gamma'], atol=1e-9, rtol=1e-9)
    assert not result['converged']


def test_geometry_fold_wall_crossing_and_intersection_budget():
    mesh, mats, coordinates = geometry_fixture()
    x = coordinates['wake'][:, 0].copy()
    coordinates['wake'][:, 0] = x*np.cos(np.deg2rad(80))
    coordinates['wake'][:, 2] = x*np.sin(np.deg2rad(80))
    assert geometry_failure(mesh, coordinates, mats, GeometryGates(max_step_fraction=10.)) == 'fold'
    mesh, mats, coordinates = geometry_fixture()
    mesh.xyz = np.array([[.5, -1., .1], [3., -1., .1], [.5, 3., .1]])
    coordinates['wake'][[1, 2, 4, 5], 2] = .2
    assert geometry_failure(mesh, coordinates, mats, GeometryGates()) == 'wall'
    mesh, mats, coordinates = geometry_fixture()
    mesh.xyz = np.array([[.1, .8, 0.], [.2, .8, 0.], [.1, .9, 0.]])
    mesh.faces = np.array([[0, 1, 2], [0, 1, 2]])
    assert geometry_failure(mesh, coordinates, mats, GeometryGates(max_collision_pairs=1)) == 'intersection_budget'


def test_cross_wake_swept_intersection_is_rejected():
    mesh, mats, coordinates = geometry_fixture()
    base = mesh.groups[0]
    raised = NS(name='raised', wake_xyz=base.wake_xyz+[0, 0, .1], wake_faces=base.wake_faces)
    mesh.groups = (base, raised)
    m = NS(**vars(mats['wake']))
    m.reference_xyz = raised.wake_xyz.copy()
    mats['raised'] = m
    coordinates['raised'] = raised.wake_xyz.copy()
    coordinates['raised'][[1, 2, 4, 5], 2] = -.1
    assert geometry_failure(mesh, coordinates, mats, GeometryGates()) == 'intersection'


def mock_coupling(monkeypatch, field):
    """Test orchestration without pretending mocked solves are physics validation."""
    import sys
    import types
    import panel.p1_coupled_material_wake as b2
    mesh, mats, _ = geometry_fixture()
    mesh.dof_face = mesh.faces
    mesh.groups[0].gamma_hat = np.ones((6, 1))
    calls = []
    def solve(model, *args, **kwargs):
        calls.append(model.groups[0].wake_xyz.copy())
        return dict(alpha_deg=5., mu=np.zeros(3), gamma=np.ones(1), cp=np.zeros(1),
                    CL=.2, CL_gamma=.21, CD_pressure=.01, CM_y=-.03, wall_rms=.001,
                    jump_rms=0., pressure_kutta_rms=.001, signed_kutta_rms=.002,
                    linear_constrained_rank=3, linear_singular_min=.1,
                    nonlinear=None, group_results=[])
    ops = dict(A=np.zeros((1, 3)), W=np.zeros((1, 1)),
               J=np.array([[1., 0., 0., 0.]]), K=np.array([[0., 1., 0., 0.]]),
               grad=np.zeros((1, 3, 4)))
    g2 = types.ModuleType('panel.p1_lifting_g2')
    g2.assemble_multi = lambda model: ops
    g2.solve_multi = solve
    old = types.ModuleType('panel.p1_steady_free_wake')
    old.prepare_wake_material = lambda group: mats[group.name]
    old.reassemble_wake_only = lambda model, static: dict(static)
    def with_xyz(model, coordinates):
        result = NS(**vars(model))
        group = NS(**vars(model.groups[0]))
        group.wake_xyz = coordinates[group.name].copy()
        result.groups = (group,)
        return result
    old.with_wake_xyz = with_xyz
    monkeypatch.setitem(sys.modules, 'panel.p1_lifting_g2', g2)
    monkeypatch.setitem(sys.modules, 'panel.p1_steady_free_wake', old)
    monkeypatch.setattr(b2, 'TotalVelocity', lambda model, state, **kw: field)
    return mesh, calls


def test_mock_coupling_resolves_accepted_geometry_and_snapshots_are_aligned(monkeypatch):
    f = UniformField()
    f.free = np.array([1., 0., .02])
    mesh, calls = mock_coupling(monkeypatch, f)
    snapshots = []
    def save(it, model, state, record, arrays):
        snapshots.append((arrays['g0_xyz'].copy(), snapshot_hash(arrays), record['state_sha256']))
    result = solve_coupled_material_wake(mesh, iterations=1, save_callback=save)
    assert result['history'][0]['outgoing_update']['accepted']
    assert len(calls) == 2 and len(snapshots) == 2
    assert not np.array_equal(calls[0], calls[-1])
    np.testing.assert_array_equal(calls[-1], result['mesh'].groups[0].wake_xyz)
    assert all(actual == recorded for _, actual, recorded in snapshots)
    assert not result['converged']


@pytest.mark.parametrize('failure', ['convergence', 'speed', 'wall'])
def test_mock_coupling_cannot_report_frozen_nodes_as_converged(monkeypatch, failure):
    class BadField(UniformField):
        def __call__(self, p, *, order):
            if failure == 'convergence':
                return np.array([1., 0., np.log(abs(p[2])+1e-30)])
            if failure == 'speed':
                return np.zeros(3)
            return np.array([1., 0., .02])
    mesh, calls = mock_coupling(monkeypatch, BadField())
    if failure == 'wall':
        monkeypatch.setattr('panel.p1_coupled_material_wake.geometry_failure', lambda *a: 'wall')
    result = solve_coupled_material_wake(mesh, iterations=2)
    assert result['status'].startswith('blocked_')
    assert not result['converged'] and not result['kinematic_candidate']
    assert len(calls) == 1  # initial canonical solve only
    update = result['history'][0]['outgoing_update']
    assert not update['accepted'] and update['reason_counts'][failure] > 0


def frozen_sample():
    mesh, state = small_field()
    g = mesh.groups[0]
    field = TotalVelocity(mesh, state)
    tri, normal, area, _shape, edges = triangle_data(g.wake_xyz, g.wake_faces)
    normals = np.zeros_like(g.wake_xyz)
    scale = np.full(len(normals), np.inf)
    for k in range(3):
        np.add.at(normals, g.wake_faces[:, k], normal*area[:, None])
        np.minimum.at(scale, g.wake_faces[:, k], edges.min(axis=1))
    free = np.arange(3)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    kwargs = dict(eps_fraction=.05, ladder=(1., .5), order=48)
    face = limits(field, tri.mean(axis=1), normal, edges.min(axis=1), **kwargs)
    nodes = limits(field, g.wake_xyz, normals, scale, **kwargs)
    arrays = dict(mu=state['mu'], gamma=state['gamma'], alpha_deg=np.asarray(state['alpha_deg']),
                  wall_xyz=mesh.xyz, wall_faces=mesh.faces, wall_dof_face=mesh.dof_face,
                  g0_xyz=g.wake_xyz, g0_faces=g.wake_faces, g0_gamma_hat=g.gamma_hat,
                  g0_free_nodes=free, sampling_kernel=np.asarray(field.method),
                  sampling_eps_fraction=np.asarray(.05),
                  sampling_ladder=np.asarray([1., .5]), sampling_order=np.asarray(48))
    arrays.update({'g0_face_'+k: v for k, v in face.items()})
    arrays.update({'g0_node_'+k: v for k, v in nodes.items()})
    return arrays


def test_frozen_field_replay_is_bitwise_and_rejects_input_tampering():
    from panel.p1_coupled_material_wake import replay_snapshot
    arrays = frozen_sample()
    digest = snapshot_hash(arrays)
    result = replay_snapshot(arrays, digest)
    assert result['bitwise_equal']
    assert max(result['max_differences'].values()) == 0
    arrays['mu'][0] += .01
    with pytest.raises(ValueError, match='SHA mismatch'):
        replay_snapshot(arrays, digest)


def test_analytic_total_velocity_does_not_depend_on_legacy_gauss_order():
    mesh, state = small_field()
    field = TotalVelocity(mesh, state)
    assert field.method == "analytic_edges"
    p = np.array([.38, -.02, .024])
    np.testing.assert_array_equal(field(p, order=48), field(p, order=96))
    legacy = TotalVelocity(mesh, state, method="legacy_gauss")
    far = np.array([.3, -.4, 1.7])
    np.testing.assert_allclose(field(far), legacy(far, order=96), rtol=1e-10, atol=1e-10)


def test_initial_equivalence_is_operator_gate_not_nonlinear_path(monkeypatch):
    from panel.p1_lifting_g2 import make_conventional_fixture
    from panel.p1_coupled_material_wake import solve_coupled_material_wake
    import panel.p1_lifting_g2 as g2
    mesh = generated_mesh()
    original = g2.solve_multi
    calls = []
    def tracked(model, *args, **kwargs):
        calls.append(kwargs.get("operators"))
        return original(model, *args, **kwargs)
    monkeypatch.setattr(g2, "solve_multi", tracked)
    run = solve_coupled_material_wake(mesh, 5., kutta="signed_speed", iterations=0)
    assert len(calls) == 1  # cannot follow two different nonlinear termination paths
    assert run["initial_equivalence"]["W_relative"] < 1e-12
    assert run["initial_equivalence"]["CL"] == 0.


def test_high_aspect_260_analytic_is_not_unresolved_48_point_gauss():
    # This controls the SAME linear doublet strength, geometry and field point;
    # only the edge integration differs. No viscous core is introduced.
    tri = np.array([[0., 0., 0.], [26., 0., 0.], [26., .1, 0.]])
    mesh = NS(xyz=tri, faces=np.array([[0, 1, 2]]),
              dof_face=np.array([[0, 1, 2]]), groups=())
    state = dict(mu=np.array([.1, .2, .3]), gamma=np.array([]), alpha_deg=0.)
    p = tri.mean(axis=0) + np.array([0., 0., .000625])
    analytic = TotalVelocity(mesh, state, method="analytic_edges")
    legacy = TotalVelocity(mesh, state, method="legacy_gauss")
    exact_value = analytic(p, order=48)
    np.testing.assert_array_equal(exact_value, analytic(p, order=96))
    # The original 48/96 global-edge Gauss evaluations both miss a near-edge
    # contribution; increasing the fixed order once does not certify a PV.
    assert np.linalg.norm(legacy(p, order=48)-exact_value) > .5
    assert np.linalg.norm(legacy(p, order=96)-exact_value) > .5


def test_unconverged_initial_pressure_kutta_is_distinct_from_later_failure(monkeypatch):
    import sys
    field = UniformField()
    mesh, calls = mock_coupling(monkeypatch, field)
    g2 = sys.modules["panel.p1_lifting_g2"]
    original = g2.solve_multi
    def blocked(*args, **kwargs):
        state = original(*args, **kwargs)
        state["nonlinear"] = dict(success=False, nfev=80,
                                  message="maximum evaluations reached")
        return state
    monkeypatch.setattr(g2, "solve_multi", blocked)
    result = solve_coupled_material_wake(mesh, iterations=4)
    assert result["status"] == "blocked_initial_nonlinear_solve"
    assert result["iterations"] == 0
    assert result["history"][0]["nonlinear_failure_phase"] == "initial_fixed_wake"
    assert result["history"][0]["outgoing_update"] is None
    assert not result["converged"]



def test_rm_regularized_total_velocity_matches_all_p1_parts_and_keeps_gradient():
    from panel.p1_regularized_sheet_velocity import RegularizedP1SheetField
    mesh, state = small_field()
    core = 0.11
    field = TotalVelocity(mesh, state, method="rm_regularized_p1", core_radius=core)
    wall = (mesh.xyz, mesh.faces, np.asarray(state["mu"])[mesh.dof_face])
    group = mesh.groups[0]
    wake_nodal = group.gamma_hat @ state["gamma"]
    wake = (group.wake_xyz, group.wake_faces, wake_nodal[group.wake_faces])
    oracle = RegularizedP1SheetField([wall, wake], freestream=field.free,
                                     core_radius=core, quadrature_order=16)
    for point in (np.array([0.3, -0.4, 1.7]),
                  group.wake_xyz[group.wake_faces[0]].mean(axis=0)):
        np.testing.assert_allclose(field(point, order=16), oracle(point, order=16),
                                   rtol=1e-13, atol=1e-14)
    assert np.linalg.norm(field.grad_face) > 0


def test_rm_regularized_total_velocity_requires_explicit_core():
    mesh, state = small_field()
    with pytest.raises(ValueError, match="explicit positive core_radius"):
        TotalVelocity(mesh, state, method="rm_regularized_p1")
    with pytest.raises(ValueError, match="explicit positive core_radius"):
        TotalVelocity(mesh, state, method="rm_regularized_p1", core_radius=0.0)
    # The existing default remains the unregularized analytic P1 evaluator.
    default = TotalVelocity(mesh, state)
    explicit = TotalVelocity(mesh, state, method="analytic_edges")
    point = np.array([0.3, -0.4, 1.7])
    np.testing.assert_array_equal(default(point), explicit(point))


def test_projection_fan_options_fail_closed():
    mesh, state = small_field()
    # An unknown quadrature name must not fall through to the legacy path.
    with pytest.raises(ValueError, match="area_quadrature must be"):
        TotalVelocity(mesh, state, method="rm_regularized_p1", core_radius=0.1,
                      area_quadrature="bogus")
    # Requesting the fan while the analytic evaluator is selected can never be honoured.
    with pytest.raises(ValueError, match="requires method='rm_regularized_p1'"):
        TotalVelocity(mesh, state, method="analytic_edges",
                      area_quadrature="projection_fan")
    # Fan controls are validated even before the regularized field is built.
    with pytest.raises(ValueError, match="fan_order must be"):
        TotalVelocity(mesh, state, method="rm_regularized_p1", core_radius=0.1,
                      area_quadrature="projection_fan", fan_order=1)
    with pytest.raises(ValueError, match="fan_strips must be"):
        TotalVelocity(mesh, state, method="rm_regularized_p1", core_radius=0.1,
                      area_quadrature="projection_fan", fan_strips=0)
    # The legacy default is untouched.
    legacy = TotalVelocity(mesh, state, method="rm_regularized_p1", core_radius=0.1)
    assert legacy.area_quadrature == "legacy_duffy"


def test_rm_regularized_g1_iteration_zero_preserves_unregularized_strength_solve():
    from panel.p1_lifting_g2 import solve_multi
    mesh = generated_mesh()
    canonical = solve_multi(mesh, 5., kutta="signed_speed")
    result = solve_coupled_material_wake(
        mesh, 5., kutta="signed_speed", iterations=0,
        velocity_method="rm_regularized_p1", core_radius=0.03, order=12)
    for key in ("mu", "gamma", "cp"):
        np.testing.assert_allclose(result["state"][key], canonical[key],
                                   atol=1e-10, rtol=1e-10)
    row = result["history"][0]
    assert row["velocity_method"] == "rm_regularized_p1"
    assert row["core_radius"] == pytest.approx(0.03)
    assert row["strength_solve_kernel"] == "unregularized_P1_BEM_operators"
    assert result["kernel"] == "RM_regularized_P1_edge_plus_interior_gradient"
    assert not result["physical_accuracy_accepted"] and not result["converged"]



def generated_g2_mesh():
    from panel.p1_lifting_g1 import make_wing
    from panel.p1_lifting_g2 import combine_wings
    main = make_wing(chord_panels=4, span_panels=4, wake_panels=2)
    tail = make_wing(chord_panels=4, span_panels=4, span=1.8, chord=.55,
                     camber=0., wake_panels=2)
    return combine_wings([main, tail], names=["main_wing", "raised_tail"],
                         origins=[(0., 0., 0.), (1.6, 0., .35)],
                         reference_chord=main.chord, reference_point=main.reference_point)


def test_rm_regularized_g2_iteration_zero_preserves_all_group_strengths():
    from panel.p1_lifting_g2 import solve_multi
    mesh = generated_g2_mesh()
    canonical = solve_multi(mesh, 5., kutta="signed_speed")
    result = solve_coupled_material_wake(
        mesh, 5., kutta="signed_speed", iterations=0,
        velocity_method="rm_regularized_p1", core_radius=0.03, order=12)
    for key in ("mu", "gamma", "cp"):
        np.testing.assert_allclose(result["state"][key], canonical[key],
                                   atol=1e-10, rtol=1e-10)
    assert len(result["state"]["group_results"]) == 2
    assert set(result["history"][0]["wakes"]) == {"main_wing", "raised_tail"}
    assert result["history"][0]["core_radius"] == pytest.approx(0.03)
    assert not result["physical_accuracy_accepted"] and not result["converged"]



def test_frozen_snapshot_replay_can_override_velocity_without_resolving_strengths():
    mesh = generated_mesh()
    saved = {}
    def capture(iteration, current, state, record, arrays):
        saved.update({key: np.asarray(value).copy() for key, value in arrays.items()})
    result = solve_coupled_material_wake(
        mesh, 5., kutta="signed_speed", iterations=0, save_callback=capture)
    sha = result["history"][0]["state_sha256"]
    replay_same = __import__(
        "panel.p1_coupled_material_wake", fromlist=["replay_snapshot"]).replay_snapshot(
            saved, sha)
    assert replay_same["replay_mode"] == "reproduce_recorded_samples"
    assert replay_same["bitwise_equal"]
    replay_rm = __import__(
        "panel.p1_coupled_material_wake", fromlist=["replay_snapshot"]).replay_snapshot(
            saved, sha, velocity_method="rm_regularized_p1",
            core_radius=0.03, quadrature_order=24)
    assert replay_rm["replay_mode"] == "alternate_velocity_on_frozen_state"
    assert replay_rm["evaluation_finite"]
    assert replay_rm["evaluated_method"] == "rm_regularized_p1"
    assert replay_rm["evaluated_core_radius"] == pytest.approx(0.03)
    assert replay_rm["quadrature_order"] == 24
    assert replay_rm["strength_solve_kernel"] == "frozen_unregularized_P1_BEM_state"



def test_total_velocity_projection_fan_matches_direct_regularized_field():
    from panel.p1_regularized_sheet_velocity import RegularizedP1SheetField
    mesh, state = small_field()
    core = 0.08
    field = TotalVelocity(mesh, state, method="rm_regularized_p1",
                          core_radius=core, area_quadrature="projection_fan",
                          fan_order=24, fan_strips=8)
    wall = (mesh.xyz, mesh.faces, np.asarray(state["mu"])[mesh.dof_face])
    group = mesh.groups[0]
    nodal = group.gamma_hat @ state["gamma"]
    wake = (group.wake_xyz, group.wake_faces, nodal[group.wake_faces])
    oracle = RegularizedP1SheetField(
        [wall, wake], freestream=field.free, core_radius=core,
        area_quadrature="projection_fan", fan_order=24, fan_strips=8)
    point = np.array([0.3, -0.4, 1.7])
    np.testing.assert_allclose(field(point), oracle(point), rtol=1e-13, atol=1e-14)
    assert field.area_quadrature == "projection_fan"
    assert field.fan_order == 24 and field.fan_strips == 8


def test_solve_callback_snapshot_records_projection_fan_controls():
    mesh = generated_mesh()
    captured = {}
    def save(iteration, current, state, record, arrays):
        captured["record"] = record
        captured["arrays"] = {k: np.asarray(v).copy() for k, v in arrays.items()}
    result = solve_coupled_material_wake(
        mesh, 5., kutta="signed_speed", iterations=0,
        velocity_method="rm_regularized_p1", core_radius=0.04,
        area_quadrature="projection_fan", fan_order=24, fan_strips=8,
        save_callback=save)
    assert captured["record"]["area_quadrature"] == "projection_fan"
    assert captured["record"]["fan_order"] == 24
    assert captured["record"]["fan_strips"] == 8
    assert str(captured["arrays"]["sampling_area_quadrature"]) == "projection_fan"
    assert int(captured["arrays"]["sampling_fan_order"]) == 24
    assert int(captured["arrays"]["sampling_fan_strips"]) == 8
    assert result["area_quadrature"] == "projection_fan"


def test_snapshot_replay_overrides_projection_fan_and_preserves_old_snapshot_default():
    from panel.p1_coupled_material_wake import replay_snapshot
    mesh = generated_mesh()
    captured = {}
    def save(iteration, current, state, record, arrays):
        captured.update({k: np.asarray(v).copy() for k, v in arrays.items()})
    result = solve_coupled_material_wake(
        mesh, 5., kutta="signed_speed", iterations=0,
        velocity_method="rm_regularized_p1", core_radius=0.04,
        save_callback=save)
    sha = result["history"][0]["state_sha256"]
    fan = replay_snapshot(captured, sha, area_quadrature="projection_fan",
                          fan_order=24, fan_strips=8)
    assert fan["evaluated_area_quadrature"] == "projection_fan"
    assert fan["fan_order"] == 24 and fan["fan_strips"] == 8
    # Legacy snapshot without the new keys remains legacy_duffy by default.
    old = dict(captured)
    for key in ("sampling_area_quadrature", "sampling_fan_order", "sampling_fan_strips"):
        old.pop(key, None)
    old_sha = __import__("panel.p1_coupled_material_wake",
                         fromlist=["snapshot_hash"]).snapshot_hash(old)
    replay = replay_snapshot(old, old_sha)
    assert replay["evaluated_area_quadrature"] == "legacy_duffy"
