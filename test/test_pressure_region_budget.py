import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from panel.pressure_region_budget import (
    budget, face_loads, fixed_radius_masks, geometry, normal_spread,
    run, sha, state_hash, triangle_distance_bounds, QUANTITIES,
)


def fixture():
    # Two horizontal triangles, one vertical: nonzero drag and moment budgets.
    return dict(schema='dpanel.half_planar_cut.v1',
        coordinates=[[0, .2, 0], [1, .2, 0], [0, 1.2, 0],
                     [2, .2, .1], [3, .2, .1], [2, 1.2, .1],
                     [4, .2, 0], [4, 1.2, 0], [4, .2, 1]],
        faces=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        component_ids=[20, 20, 30], freestream=[1., 0., .1],
        density=1.225, reference_area=4., reference_chord=1.,
        moment_reference=[.25, 0., 0.])


def cp():
    return np.array([-.3, .4, -.2]), np.array([-.5, .1, -.7])


def make_saved_run(path):
    path.mkdir()
    fx = fixture()
    cp0, cp1 = cp()
    (path/'surface_input.json').write_text(json.dumps(fx), encoding='utf-8')
    surface_sha = sha(path/'surface_input.json')
    state = dict(mu_half=np.ones(3), gamma_half=np.ones(1), sigma_half=np.ones(3),
                 Cp_raw_half=cp0, neighbor_ids=np.zeros((3, 1), dtype=np.int64),
                 surface_sha256=np.array(surface_sha))
    np.savez(path/'frozen_state.npz', **state)
    np.savez(path/'recovered_fields.npz', original_cp=cp0, candidate_cp=cp1)
    (path/'patches.json').write_text('[]', encoding='utf-8')
    loads = {}
    for name, c in [('original', cp0), ('cut_vertex_affine_candidate', cp1)]:
        values, _, _ = face_loads(fx, c)
        loads[name] = dict(zip(QUANTITIES, values.sum(axis=0).tolist()))
    report = dict(status='postprocessing_comparison_not_physical_validation',
                  source_sha256=surface_sha, frozen_state_hash=state_hash(state), loads=loads)
    (path/'report.json').write_text(json.dumps(report), encoding='utf-8')
    return fx


def test_distance_interior_edges_and_vertices():
    tri = np.array([[[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]]])
    for p, expected in [([.5, .5, 3.], 3.), ([1., -2., 0.], 2.), ([-3., -4., 0.], 5.)]:
        low, high = triangle_distance_bounds(tri, p)
        assert low[0] == pytest.approx(expected)
        assert high[0] == pytest.approx(np.linalg.norm(tri[0]-p, axis=1).max())


def test_sphere_crossing_with_all_vertices_outside():
    tri = np.array([[[-2., -2., 0.], [2., -2., 0.], [0., 3., 0.]]])
    lo, hi = triangle_distance_bounds(tri, [0, 0, 0])
    masks = fixed_radius_masks(lo, hi, .1)
    assert masks['crossing'].tolist() == [True]
    assert np.linalg.norm(tri[0], axis=1).min() > .1


def test_partitions_and_signed_gross_closure():
    result, arrays = budget(fixture(), *cp(), [], [0., .2, 0.], [.1, 1., 10.])
    assert arrays['region_ids'].shape == (3, 3)
    for partition in result['partitions']:
        parts = list(partition['regions'].values())
        assert sum(p['half_faces'] for p in parts) == 3
        assert sum(p['area_fraction'] for p in parts) == pytest.approx(1)
        for q in QUANTITIES:
            for key in ('original', 'candidate', 'delta', 'sum_abs_delta'):
                assert sum(p['quantities'][q][key] for p in parts) == pytest.approx(
                    result['whole']['quantities'][q][key], abs=1e-14)
        for key in ('original_force_N','candidate_force_N','delta_force_N',
                    'original_moment_Nm','candidate_moment_Nm','delta_moment_Nm'):
            np.testing.assert_allclose(np.sum([p[key] for p in parts], axis=0), result['whole'][key])
    assert result['physical_accuracy_accepted'] is False


def test_exact_reflected_force_and_axial_moment():
    fx = fixture()
    cp0, _ = cp()
    _, force, moment = face_loads(fx, cp0)
    x, f, tri, av, _, _ = geometry(fx)
    image = tri[:, [0, 2, 1]]*[1., -1., 1.]
    full = np.concatenate([tri, image])
    vectors = np.cross(full[:, 1]-full[:, 0], full[:, 2]-full[:, 0])/2
    flow = np.array(fx['freestream'])
    direct = -.5*fx['density']*(flow@flow)*np.tile(cp0, 2)[:, None]*vectors
    moments = np.cross(full.mean(axis=1)-fx['moment_reference'], direct)
    np.testing.assert_allclose(force.sum(axis=0), direct.sum(axis=0), atol=1e-15)
    np.testing.assert_allclose(moment.sum(axis=0), moments.sum(axis=0), atol=1e-15)


def test_similarity_and_reference_units():
    fx = fixture()
    original, _ = budget(fx, *cp(), [], [0., .2, 0.], [.1, 1., 10.])
    factor = 7.
    fx['coordinates'] = (np.array(fx['coordinates'])*factor).tolist()
    fx['moment_reference'] = (np.array(fx['moment_reference'])*factor).tolist()
    fx['reference_area'] *= factor**2
    fx['reference_chord'] *= factor
    changed, arrays = budget(fx, *cp(), [], [0., .2*factor, 0.], [.1, 1., 10.])
    for q in QUANTITIES:
        assert changed['whole']['quantities'][q]['delta'] == pytest.approx(
            original['whole']['quantities'][q]['delta'])
    for a, b in zip(original['partitions'], changed['partitions']):
        for region in ('inside', 'crossing', 'outside'):
            assert a['regions'][region]['half_faces'] == b['regions'][region]['half_faces']


def test_rotation_and_translation_preserving_symmetry():
    fx = fixture()
    angle = .47
    rot = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                    [-np.sin(angle), 0., np.cos(angle)]])
    translation = np.array([2., 0., -3.])
    point = np.array([0., .2, 0.])
    before, arr0 = budget(fx, *cp(), [], point, [.2, 1.1])
    fx['coordinates'] = (np.array(fx['coordinates'])@rot.T+translation).tolist()
    fx['freestream'] = (np.array(fx['freestream'])@rot.T).tolist()
    fx['moment_reference'] = (np.array(fx['moment_reference'])@rot.T+translation).tolist()
    after, arr1 = budget(fx, *cp(), [], point@rot.T+translation, [.2, 1.1])
    np.testing.assert_allclose(arr0['original_coefficients'], arr1['original_coefficients'], atol=1e-15)
    np.testing.assert_array_equal(arr0['region_ids'], arr1['region_ids'])


def test_zero_delta_has_no_relative_percentage():
    cp0, _ = cp()
    r, _ = budget(fixture(), cp0, cp0, [], [0, .2, 0], [.1])
    for q in QUANTITIES:
        assert r['whole']['quantities'][q]['sum_abs_delta'] == 0
        assert r['whole']['quantities'][q]['share_of_global_abs_delta'] is None


def test_area_weighted_cp_and_no_inplace_edits():
    cp0, cp1 = cp()
    old0, old1 = cp0.copy(), cp1.copy()
    cp0.setflags(write=False)
    cp1.setflags(write=False)
    r, arrays = budget(fixture(), cp0, cp1, [], [0., .2, 0.], [.1, .2])
    area = arrays['area_half']
    assert r['whole']['cp_delta_area_rms'] == pytest.approx(
        np.sqrt(np.sum(area*(cp1-cp0)**2)/area.sum()))
    np.testing.assert_array_equal(cp0, old0)
    np.testing.assert_array_equal(cp1, old1)


def test_crease_indicator_distinct_from_rank_diagnostic():
    fx = fixture()
    fx['coordinates'] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    fx['faces'] = [[0, 1, 2], [0, 3, 1]]
    fx['component_ids'] = [20, 20]
    patches = [{'fan_faces':[0, 1], 'support_faces':[0, 1], 'singular_ratio':1.}]
    edge, patch = normal_spread(fx, patches)
    np.testing.assert_allclose(edge, [90., 90.])
    np.testing.assert_allclose(patch, [45., 45.])


def test_growing_pointwise_peak_can_have_decreasing_integral():
    # Counterexample to inferring load divergence from Cp peaks alone.
    fx = fixture()
    fx['faces'] = [[0, 1, 2]]
    fx['component_ids'] = [30]
    peaks, loads = [], []
    base = np.array(fx['coordinates'])
    for h in (1., .5, .25):
        fx['coordinates'] = (base*h).tolist()
        values, _, _ = face_loads(fx, np.array([-1/h]))
        peaks.append(1/h)
        loads.append(abs(values[0, 1]))
    assert peaks[2] > peaks[1] > peaks[0]
    assert loads[2] < loads[1] < loads[0]


@pytest.mark.parametrize('error', ['nan', 'density', 'sideslip', 'radius', 'index', 'degenerate'])
def test_invalid_inputs_are_rejected(error):
    fx, (cp0, cp1), radii = fixture(), cp(), [.1]
    if error == 'nan': cp1[0] = np.nan
    if error == 'density': fx['density'] = 0.
    if error == 'sideslip': fx['freestream'][1] = .01
    if error == 'radius': radii = [.2, .1]
    if error == 'index': fx['faces'][0][0] = 100
    if error == 'degenerate': fx['faces'][0] = [0, 0, 0]
    with pytest.raises(ValueError):
        budget(fx, cp0, cp1, [], [0, .2, 0], radii)


def test_saved_run_roundtrip_and_overwrite_refusal(tmp_path):
    saved = tmp_path/'recovery'
    make_saved_run(saved)
    hashes = {p.name:sha(p) for p in saved.iterdir()}
    output = tmp_path/'out'
    result = run(saved, output, [0, .2, 0], [.1, 1.])
    assert result['input_sha256'] == hashes
    assert hashes == {p.name:sha(p) for p in saved.iterdir()}
    assert result['regional_fields_sha256'] == sha(output/'regional_fields.npz')
    with pytest.raises(FileExistsError): run(saved, output, [0, .2, 0], [.1])


def test_tampered_snapshot_is_rejected(tmp_path):
    saved = tmp_path/'recovery'
    make_saved_run(saved)
    with np.load(saved/'frozen_state.npz') as a:
        state = {k:a[k].copy() for k in a.files}
    state['mu_half'][0] += 1
    np.savez(saved/'frozen_state.npz', **state)
    with pytest.raises(ValueError, match='identity'):
        run(saved, tmp_path/'bad', [0, .2, 0], [.1])
    assert not (tmp_path/'bad').exists()


def test_tampered_candidate_total_is_rejected(tmp_path):
    saved = tmp_path/'recovery'
    make_saved_run(saved)
    cp0, cp1 = cp()
    np.savez(saved/'recovered_fields.npz', original_cp=cp0, candidate_cp=cp1*2)
    with pytest.raises(ValueError, match='closure'):
        run(saved, tmp_path/'bad', [0, .2, 0], [.1])


def test_cli_has_no_jax_dependency(tmp_path):
    saved = tmp_path/'recovery'
    make_saved_run(saved)
    cwd = Path(__file__).resolve().parents[1]
    code = """
import builtins
old_import = builtins.__import__
def checked(name, *a, **kw):
    if name.split('.')[0] in ('jax','torch','gmsh','numba'):
        raise AssertionError('Unexpected numerical backend: '+name)
    return old_import(name, *a, **kw)
builtins.__import__ = checked
from panel.pressure_region_budget import main
main()
"""
    command = [sys.executable, '-c', code, '--recovery-run', str(saved),
               '--out', str(tmp_path/'cli'), '--feature-point', '0', '.2', '0']
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path/'cli'/'report.json').read_text())['physical_accuracy_accepted'] is False
