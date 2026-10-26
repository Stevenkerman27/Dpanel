"""Full-checkout tests; missing JAX is a skip, never physical acceptance."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jax.config.update('jax_enable_x64', True)

from panel.run_half_native import load_half_surface
from panel.half_native import solve_half_fields
from panel.compare_pressure_recovery import compare, state_hash

ROOT = Path(__file__).resolve().parents[1]
SURFACE = ROOT/'cases/panel_half/thick_wing_two_strip.json'


def test_existing_half_solver_control_and_immutable_state():
    fixture = json.loads(SURFACE.read_text(encoding='utf-8'))
    inputs = load_half_surface(SURFACE)
    result = jax.jit(solve_half_fields)(**inputs)
    state = {k:np.asarray(result[k]).copy() for k in
             ('mu_half','gamma_half','sigma_half','Cp_raw_half')}
    state['neighbor_ids'] = np.asarray(inputs['topology'].neighbor_ghost_ids)
    before = state_hash(state)
    report, fields, patches = compare(fixture,state,max_rings=4)
    assert before == state_hash(state)
    assert report['baseline_cp_max_abs_error'] < 1e-10
    assert report['physical_accuracy_accepted'] is False
    assert np.isfinite(fields['candidate_cp']).all()
    assert len(patches) > 0


def test_cli_snapshot_replay_and_wrong_surface_rejection(tmp_path):
    first, replay = tmp_path/'first', tmp_path/'replay'
    cmd = [sys.executable,'-m','panel.compare_pressure_recovery',
           '--surface',str(SURFACE),'--max-rings','4']
    subprocess.run(cmd+['--out',str(first)],cwd=ROOT,check=True)
    subprocess.run(cmd+['--out',str(replay),'--snapshot',str(first/'frozen_state.npz')],
                   cwd=ROOT,check=True)
    a=json.loads((first/'report.json').read_text())
    b=json.loads((replay/'report.json').read_text())
    assert a['frozen_state_hash'] == b['frozen_state_hash']
    assert a['loads'] == b['loads']
    with np.load(first/'recovered_fields.npz') as fa, np.load(replay/'recovered_fields.npz') as fb:
        for key in fa.files:
            np.testing.assert_array_equal(fa[key],fb[key])
    changed=json.loads(SURFACE.read_text())
    changed['density'] *= 1.01
    wrong=tmp_path/'wrong.json'
    wrong.write_text(json.dumps(changed))
    proc=subprocess.run([sys.executable,'-m','panel.compare_pressure_recovery',
        '--surface',str(wrong),'--out',str(tmp_path/'bad'),
        '--snapshot',str(first/'frozen_state.npz')],cwd=ROOT,capture_output=True,text=True)
    assert proc.returncode != 0
    bad=json.loads((tmp_path/'bad/report.json').read_text())
    assert bad['status'] == 'failed' and 'SHA' in bad['error']
