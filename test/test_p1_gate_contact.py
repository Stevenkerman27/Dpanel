"""T5: declared-contact corridor, per-node step caps, and gate-mode safety."""
import numpy as np
import pytest

from panel.p1_geometry_contact import classify_new_triangle, classify_swept_wall, declared_contact_pairs
from panel.p1_coupled_material_wake import clip_step_update

TOL = 1e-6
WALL = np.array([[-.5, 0., 0.], [1.5, 0., 0.], [.5, .8, 0.]])
WALL_BOTH = np.array([[-.5, -.6, 0.], [1.5, -.6, 0.], [.5, .8, 0.]])
WAKE = np.array([[0., 0., 0.], [1., 0., 0.], [.5, -.2, 0.]])
SHARED = np.array([[0., 0., 0.], [1., 0., 0.]])
GRAZE = 1e-3


def swept(old, new):
    return np.concatenate((np.asarray(old, float), np.asarray(new, float)), axis=0)


def test_the_identity_configuration_only_touches_along_the_shared_edge():
    cls = classify_swept_wall(swept(WAKE, WAKE), WALL, tol=TOL, declared='edge',
                              declared_region=SHARED, graze_tol=GRAZE)
    assert cls.kind == 'contact' and not cls.violation, cls


def test_a_small_departure_from_the_shared_vertex_stays_in_the_corridor():
    new = WAKE.copy()
    new[1, 2] -= 5e-4
    cls = classify_swept_wall(swept(WAKE, new), WALL, tol=TOL, declared='edge',
                              declared_region=SHARED, graze_tol=GRAZE)
    assert cls.kind == 'contact' and not cls.violation, cls


def test_the_same_departure_without_a_corridor_is_a_violation():
    new = WAKE.copy()
    new[1, 2] -= 5e-4
    cls = classify_swept_wall(swept(WAKE, new), WALL, tol=TOL, declared='edge',
                              declared_region=SHARED, graze_tol=0.)
    assert cls.violation, cls


def test_a_deep_push_through_the_attachment_is_still_refused():
    new = WAKE.copy()
    new[1, 2] -= 4e-3
    cls = classify_swept_wall(swept(WAKE, new), WALL, tol=TOL, declared='edge',
                              declared_region=SHARED, graze_tol=GRAZE)
    assert cls.violation and cls.depth > 0.9*4e-3, cls


def test_an_off_simplex_penetration_is_refused_even_when_declared():
    # Third vertex is 0.2 away from the declared edge but inside the wall
    # footprint of WALL_BOTH, so the corridor must not excuse it.
    new = WAKE.copy()
    new[:, 2] -= 4e-3
    cls = classify_new_triangle(new, WALL_BOTH, tol=TOL, declared='edge',
                                declared_region=SHARED, graze_tol=GRAZE)
    assert cls.violation and cls.depth > 0.9*4e-3, cls
    assert classify_new_triangle(new, WALL_BOTH, tol=TOL, declared=None).violation


def test_an_undeclared_pair_does_not_get_a_corridor():
    new = WAKE.copy()
    new[1, 2] -= 5e-4
    cls = classify_swept_wall(swept(WAKE, new), WALL, tol=TOL, declared=None,
                              declared_region=SHARED, graze_tol=GRAZE)
    assert cls.violation


def test_declared_pairs_use_vertex_identity_not_coordinates():
    wall = np.array([[0, 1, 2], [2, 1, 3]])
    wake = np.array([[0, 1, 4]])
    pairs = declared_contact_pairs([wake], wall, group_labels=['main_wing'])
    assert pairs[('main_wing', 0, 0)] == ('edge', (0, 1))
    assert pairs[('main_wing', 0, 1)] == ('vertex', (1,))
    # Only ids matter: a group label is preserved and face order is stable.
    assert declared_contact_pairs([wake], wall, group_labels=['main_wing']) == pairs


def test_clip_caps_every_node_and_pins_the_roots():
    old = np.zeros((3, 3))
    target = np.array([[1., 0., 0.], [0., 2., 0.], [0., 0., .5]])
    caps = np.array([.1, .1, np.inf])
    new, ledger = clip_step_update(old, target, caps, roots=[2], reference=np.zeros((3, 3)))
    assert ledger['clipped'] == 2 and ledger['clipped_fraction'] == pytest.approx(2/3)
    moved = np.linalg.norm(new-old, axis=1)
    assert moved[0] <= .1 and moved[1] <= .1
    assert np.allclose(new[2], 0.)
    assert ledger['raw_max'] == pytest.approx(2.) and ledger['accepted_max'] <= .1


def test_clip_keeps_the_accepted_step_strictly_inside_the_gate_threshold():
    scale = np.array([.02, .02, .02])
    gates_max = .25
    old = np.zeros((3, 3))
    target = np.array([[.031, 0., 0.], [0., .017, 0.], [0., 0., .009]])
    new, ledger = clip_step_update(old, target, gates_max*scale)
    step = np.linalg.norm(new-old, axis=1)/scale
    assert np.all(step < gates_max), step
    assert np.all(step > 0.99*gates_max)


def _exploding_limits(field, points, normals, scales, **kw):
    n = len(points)
    mean = np.tile([1., 0., 0.], (n, 1))
    return dict(mean=mean, plus=mean+[0., 0., .01], minus=mean-[0., 0., .01],
                epsilon_error=np.full(n, .9), quadrature_error=np.zeros(n),
                eps=np.full(n, .02), normal_jump=np.zeros(n), tangent_jump=np.zeros(n))


@pytest.mark.parametrize("gate_mode", ["audit", "safe"])
def test_audit_advances_rejected_nodes_while_safe_freezes_them(monkeypatch, gate_mode):
    from panel import p1_coupled_material_wake as m
    from panel.p1_lifting_g1 import make_wing
    from panel.p1_lifting_g2 import combine_wings
    monkeypatch.setattr(m, 'limits', _exploding_limits)
    mesh = combine_wings([make_wing(chord_panels=3, span_panels=3, wake_panels=2)])
    result = m.solve_coupled_material_wake(mesh, 5., kutta='signed_speed', iterations=1,
                                           gate_mode=gate_mode)
    rows = [row for name, row in result['history'][0]['wakes'].items()
            if name.startswith('lifting')]
    assert rows
    for row in rows:
        block = row['eligibility']
        assert block['convergence'] > 0
        assert block['gate_mode'] == gate_mode
        if gate_mode == 'audit':
            # Every sampling-rejected node is advanced, and that is reported.
            assert block['advanced_despite_audit'] > 0
            assert block['rejected'] == 0
            assert block['production_safety'] == 'sampling_not_gated'
        else:
            assert block['advanced_despite_audit'] == 0
            assert block['rejected'] == block['convergence']
            assert block['production_safety'] == 'sampling_converged'

def _audit_fixture():
    from types import SimpleNamespace as NS
    from panel.p1_coupled_material_wake import GeometryGates, triangle_data
    xyz = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.],
                    [0., 1., 0.], [1., 1., 0.], [2., 1., 0.]])
    faces = np.array([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]])
    mat = NS(reference_xyz=xyz.copy(), roots=np.array([0, 3]), used=np.arange(6),
             predecessor=np.array([-1, 0, 1, -1, 3, 4]), order=np.array([0, 3, 1, 4, 2, 5]),
             panel_area=triangle_data(xyz, faces)[2])
    group = NS(name='wake', wake_xyz=xyz, wake_faces=faces)
    wall = np.array([[-.5, -.5, 0.], [2.5, -.5, 0.], [1.25, 1.5, 0.]])
    mesh = NS(groups=(group,), xyz=wall, faces=np.array([[0, 1, 2]]), reference_chord=1.)
    coordinates = {'wake': xyz.copy()}
    coordinates['wake'][[1, 2, 4, 5], 2] += 1e-6
    return mesh, {'wake': mat}, coordinates, GeometryGates()


def test_an_inherited_crossing_is_attributed_not_silently_excused(monkeypatch):
    from panel import p1_coupled_material_wake as m
    from panel.p1_geometry_contact import ContactClass

    def fake_classify(*a, **kw):
        return ContactClass('transverse', 1e-3, 0., kw.get('declared'), True)

    mesh, mats, coordinates, gates = _audit_fixture()
    monkeypatch.setattr(m, 'classify_swept_wall', fake_classify)
    monkeypatch.setattr(m, 'classify_new_triangle', fake_classify)
    audit = m.geometry_audit(mesh, coordinates, mats, gates)
    assert audit['reason'] is None, audit['reason']
    contacts = audit['detail']['contacts']
    assert contacts['pre_existing'] > 0 and contacts['violations_count'] == 0
    assert contacts['pre_existing_pairs']


def test_a_deeper_or_new_crossing_still_blocks(monkeypatch):
    from panel import p1_coupled_material_wake as m
    from panel.p1_geometry_contact import ContactClass

    def deep(*a, **kw):
        return ContactClass('transverse', 5e-3, 0., kw.get('declared'), True)

    def clean(*a, **kw):
        return ContactClass('disjoint', 0., 0., kw.get('declared'), False)

    mesh, mats, coordinates, gates = _audit_fixture()
    monkeypatch.setattr(m, 'classify_swept_wall', deep)
    monkeypatch.setattr(m, 'classify_new_triangle', clean)
    audit = m.geometry_audit(mesh, coordinates, mats, gates)
    assert audit['reason'] == 'wall'
    assert audit['detail']['contacts']['violations_count'] > 0
    assert audit['detail']['contacts']['pre_existing'] == 0

    def mild(*a, **kw):
        return ContactClass('transverse', 1e-3, 0., kw.get('declared'), True)

    monkeypatch.setattr(m, 'classify_swept_wall', deep)
    monkeypatch.setattr(m, 'classify_new_triangle', mild)
    # Inherited but now DEEPER than the old crossing: still blocking.
    from panel.p1_geometry_contact import ContactClass as C
    monkeypatch.setattr(m, 'classify_swept_wall',
                        lambda *a, **kw: C('transverse', 1e-2, 0., kw.get('declared'), True))
    audit = m.geometry_audit(mesh, coordinates, mats, gates)
    assert audit['reason'] == 'wall'

def test_depth_is_judged_per_point_not_over_the_whole_hull():
    # TE-style geometry: the sheet pivots about the shared edge, so one corner
    # swings 0.6 chord below the wall plane while its projection falls outside
    # the wall footprint.  The only offending point is the crossing point next
    # to the shared edge, whose own depth is zero, so the hull-wide maximum
    # depth (0.6) must not convict it.
    old = np.array([[0., 0., 0.], [1., 0., 0.], [.3, .02, .02]])
    new = np.array([[0., 0., 0.], [1., 0., 0.], [.3, -.03, -.6]])
    cls = classify_swept_wall(swept(old, new), WALL, tol=TOL, declared='edge',
                              declared_region=SHARED, graze_tol=0.05)
    assert cls.kind == 'contact' and not cls.violation, cls
    assert cls.depth > 0.5, cls
    tight = classify_swept_wall(swept(old, new), WALL, tol=TOL, declared='edge',
                                declared_region=SHARED, graze_tol=0.)
    assert tight.violation, tight


def test_a_deep_vertex_inside_the_footprint_is_still_a_violation():
    old = np.array([[0., 0., 0.], [1., 0., 0.], [.3, .02, .02]])
    deep = np.array([[0., 0., 0.], [1., 0., 0.], [.3, .02, -.6]])
    cls = classify_swept_wall(swept(old, deep), WALL, tol=TOL, declared='edge',
                              declared_region=SHARED, graze_tol=0.05)
    assert cls.violation and cls.depth > 0.5, cls
