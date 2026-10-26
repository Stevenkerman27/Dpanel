"""Run against the actual repository checkout, including its saved small fixture.

These tests require the real half_native/run_half_native modules, not substitutes.
Missing JAX is an explicit skip, not physical or numerical acceptance.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest


def test_real_small_half_fixture_regression_and_refinement(tmp_path):
    pytest.importorskip('jax')
    from panel.run_gamma_refinement import run
    root=Path(__file__).resolve().parents[1]
    fixture=root/'cases/panel_half/thick_wing_two_strip.json'
    before=fixture.read_bytes()
    report=run(fixture,tmp_path/'nested',factors=(1,2,4),comparison_subdivisions=8)
    assert report['status']=='nested_gamma_experiment_completed_not_physical_validation'
    assert report['physical_accuracy_accepted'] is False
    assert report['source_surface_sha256']==hashlib.sha256(before).hexdigest()
    assert fixture.read_bytes()==before
    assert len(report['levels'])==3
    ng=[v['free_gamma_dofs'] for v in report['levels']]
    assert ng[0] < ng[1] < ng[2]
    for v in report['levels']:
        assert v['true_residual']['true_relative_residual'] < 1e-10
        assert v['old_space_embedding']['wake_action']['max_abs'] < 5e-11
    for item in report['levels'][0]['original_regression'].values():
        assert item['max_abs'] < 5e-11
    with np.load(tmp_path/'nested/factor_1.npz') as a, np.load(tmp_path/'nested/factor_4.npz') as b:
        np.testing.assert_array_equal(a['pressure_neighbor_ids'],b['pressure_neighbor_ids'])
        np.testing.assert_array_equal(a['sigma_half'],b['sigma_half'])
        assert len(b['wake_triangles']) > len(a['wake_triangles'])
        assert float(b['gamma_station_values'][-1]) == 0
    with pytest.raises(FileExistsError):run(fixture,tmp_path/'nested')


def test_piece_budget_failure_writes_report_without_default_changes(tmp_path):
    pytest.importorskip('jax')
    from panel.run_gamma_refinement import run
    root=Path(__file__).resolve().parents[1]
    fixture=root/'cases/panel_half/thick_wing_two_strip.json'
    with pytest.raises(MemoryError,match='piece budget'):
        run(fixture,tmp_path/'rejected',max_pieces=1)
    r=json.loads((tmp_path/'rejected/report.json').read_text())
    assert r['status']=='failed' and not r['physical_accuracy_accepted']
