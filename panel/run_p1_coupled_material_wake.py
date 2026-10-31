"""Run issue #65 B2; JSON/NPZ evidence is retained even when the run fails."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import os

import numpy as np

from .p1_coupled_material_wake import solve_coupled_material_wake, snapshot_hash, replay_snapshot
from .velocity_profile import enable as _profile_enable, reset as _profile_reset
from .velocity_profile import write as _profile_write


def _git(*args):
    try:
        return subprocess.check_output(['git', *args], cwd=Path(__file__).resolve().parents[1],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('g1', 'g2'), default='g1')
    parser.add_argument('--main-half-surface', type=Path)
    parser.add_argument('--alpha', type=float, default=5.)
    parser.add_argument('--kutta', choices=('pressure', 'signed_speed'), default='pressure')
    parser.add_argument('--chord-panels', type=int, default=4)
    parser.add_argument('--span-panels', type=int, default=4)
    parser.add_argument('--wake-panels', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=4)
    parser.add_argument('--relaxation', type=float, default=.15)
    parser.add_argument('--eps-fraction', type=float, default=.05)
    parser.add_argument('--ladder', type=float, nargs='+', default=[1., .5, .25, .125])
    parser.add_argument('--quadrature-order', type=int, default=48,
                        help='Legacy edge Gauss order; RM-P1 uses this as Duffy area order; ignored by analytic edges')
    parser.add_argument('--velocity-method',
                        choices=('analytic_edges', 'legacy_gauss', 'rm_regularized_p1'),
                        default='analytic_edges',
                        help='Default analytic P1; rm_regularized_p1 is opt-in and requires --core-radius')
    parser.add_argument('--core-radius', type=float, default=None,
                        help='Explicit Rosenhead–Moore core length in geometry units; required only for rm_regularized_p1')
    parser.add_argument('--area-quadrature',
                        choices=('legacy_duffy', 'projection_fan', 'projection_fan_adaptive'),
                        default='legacy_duffy',
                        help='RM-P1 area integration; the fan modes are opt-in')
    parser.add_argument('--near-ratio', type=float, default=2.0,
                        help='Near/far split: near when the point-to-triangle distance is at most this multiple of the circumradius')
    parser.add_argument('--adaptive-far-order', type=int, default=8)
    parser.add_argument('--adaptive-far-levels', type=int, default=1)
    parser.add_argument('--adaptive-budget', type=float, default=2e-4,
                        help='Relative a-posteriori error budget of the adaptive area quadrature')
    parser.add_argument('--adaptive-max-escalations', type=int, default=2)
    parser.add_argument('--fan-order', type=int, default=16)
    parser.add_argument('--fan-strips', type=int, default=8)
    parser.add_argument('--fan-refinement-order', type=int, default=None,
                        help='Projection-fan gate refinement order; default max(48, 2*fan_order)')
    parser.add_argument('--fan-refinement-strips', type=int, default=None,
                        help='Projection-fan gate refinement strips; default fan_strips')
    parser.add_argument('--gate-mode', choices=('enforce', 'audit', 'off'), default='enforce',
                        help="Sampling-error thresholds: 'enforce' freezes nodes that fail them "
                             "(legacy, distorts the sheet), 'audit' records them without freezing, "
                             "'off' also skips the refined quadrature evaluation")
    parser.add_argument('--epsilon-tol', type=float, default=.02)
    parser.add_argument('--quadrature-tol', type=float, default=.002)
    parser.add_argument('--max-nfev', type=int, default=80)
    parser.add_argument('--replay-snapshot', type=Path)
    parser.add_argument('--expected-state-sha')
    parser.add_argument('--replay-velocity-method',
                        choices=('analytic_edges', 'legacy_gauss', 'rm_regularized_p1'),
                        default=None, help='Override only the sampled velocity kernel when replaying a frozen state')
    parser.add_argument('--replay-core-radius', type=float, default=None,
                        help='Explicit RM core radius for frozen-state replay')
    parser.add_argument('--replay-quadrature-order', type=int, default=None,
                        help='Override area quadrature order for frozen-state replay')
    parser.add_argument('--replay-area-quadrature',
                        choices=('legacy_duffy', 'projection_fan', 'projection_fan_adaptive'),
                        default=None)
    parser.add_argument('--replay-near-ratio', type=float, default=None)
    parser.add_argument('--replay-adaptive-far-order', type=int, default=None)
    parser.add_argument('--replay-adaptive-far-levels', type=int, default=None)
    parser.add_argument('--replay-adaptive-budget', type=float, default=None)
    parser.add_argument('--replay-adaptive-max-escalations', type=int, default=None)
    parser.add_argument('--replay-fan-order', type=int, default=None)
    parser.add_argument('--replay-fan-strips', type=int, default=None)
    parser.add_argument('--replay-fan-refinement-order', type=int, default=None)
    parser.add_argument('--replay-fan-refinement-strips', type=int, default=None)
    parser.add_argument('--profile-json', type=Path, default=None,
                        help='Opt-in kernel profiling counters/timers (issue #84 P0); inert when omitted')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    if args.profile_json is not None:
        _profile_reset()
        _profile_enable(True)
    parameters = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    receipt = dict(schema='dpanel.issue65.b2.v3', command=sys.argv if argv is None else argv,
                   code_sha=_git('rev-parse', 'HEAD'), git_status=_git('status', '--porcelain'),
                   python=sys.version, platform=platform.platform(), numpy=np.__version__,
                   thread_environment={k: os.environ.get(k) for k in
                       ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
                   parameters=parameters, physical_accuracy_accepted=False, snapshots=[])
    receipt['implementation_sha256'] = hashlib.sha256(
        Path(__file__).with_name('p1_coupled_material_wake.py').read_bytes()).hexdigest()
    _write_json(args.out/'receipt.json', receipt)
    try:
        if args.replay_snapshot is not None:
            if not args.expected_state_sha:
                raise ValueError('--expected-state-sha is required for frozen-state replay')
            with np.load(args.replay_snapshot, allow_pickle=False) as archive:
                arrays = {k: archive[k] for k in archive.files}
            replay = replay_snapshot(
                arrays, args.expected_state_sha,
                velocity_method=args.replay_velocity_method,
                core_radius=args.replay_core_radius,
                quadrature_order=args.replay_quadrature_order,
                area_quadrature=args.replay_area_quadrature,
                near_ratio=args.replay_near_ratio,
                adaptive_far_order=args.replay_adaptive_far_order,
                adaptive_far_levels=args.replay_adaptive_far_levels,
                adaptive_budget=args.replay_adaptive_budget,
                adaptive_max_escalations=args.replay_adaptive_max_escalations,
                fan_order=args.replay_fan_order,
                fan_strips=args.replay_fan_strips,
                fan_refinement_order=args.replay_fan_refinement_order,
                fan_refinement_strips=args.replay_fan_refinement_strips)
            _write_json(args.out/'replay.json', replay)
            replay_ok = (replay['bitwise_equal']
                         if replay['replay_mode'] == 'reproduce_recorded_samples'
                         else replay['evaluation_finite'])
            receipt.update(exit_code=0 if replay_ok else 3,
                           status=replay['replay_mode'],
                           replay_method=replay['evaluated_method'],
                           replay_core_radius=replay['evaluated_core_radius'],
                           replay_quadrature_order=replay['quadrature_order'])
            return receipt['exit_code']
        import scipy
        from .p1_lifting_g1 import make_wing, from_dpanel_half_surface
        from .p1_lifting_g2 import combine_wings
        receipt['scipy'] = scipy.__version__
        if args.main_half_surface is not None:
            if not args.main_half_surface.is_file():
                raise FileNotFoundError('real CAD fixture is absent: '+str(args.main_half_surface))
            receipt['input_sha256'] = hashlib.sha256(args.main_half_surface.read_bytes()).hexdigest()
            main_wing = from_dpanel_half_surface(args.main_half_surface)
            receipt['cad_wake_panels_note'] = 'CAD adapter owns wake discretisation; --wake-panels applies to generated wings only'
        else:
            main_wing = make_wing(chord_panels=args.chord_panels, span_panels=args.span_panels,
                                  span=4., chord=1., camber=.02, wake_panels=args.wake_panels)
        wings, names, origins = [main_wing], ['main_wing'], [(0., 0., 0.)]
        if args.case == 'g2':
            wings.append(make_wing(chord_panels=args.chord_panels, span_panels=args.span_panels,
                                   span=1.8, chord=.55, camber=0., wake_panels=args.wake_panels))
            names.append('raised_tail')
            origins.append((1.6, 0., .35))
        mesh = combine_wings(wings, names=names, origins=origins,
                             reference_chord=main_wing.chord,
                             reference_point=main_wing.reference_point)
        receipt.update(reference_area=mesh.reference_area, reference_chord=mesh.reference_chord,
                       reference_point=mesh.reference_point.tolist(), group_names=names,
                       wall_faces=len(mesh.faces), wake_faces=[len(g.wake_faces) for g in mesh.groups])
        def save(iteration, current, state, record, arrays):
            stem = 'iteration_%03d' % iteration
            np.savez_compressed(args.out/(stem+'.npz'), **arrays)
            if snapshot_hash(arrays) != record['state_sha256']:
                raise RuntimeError('snapshot changed during save')
            _write_json(args.out/(stem+'.json'), record)
            receipt['snapshots'].append(dict(npz=stem+'.npz', report=stem+'.json',
                                             state_sha256=record['state_sha256']))
            _write_json(args.out/'receipt.json', receipt)
        result = solve_coupled_material_wake(mesh, args.alpha, kutta=args.kutta,
                   iterations=args.iterations, relaxation=args.relaxation,
                   eps_fraction=args.eps_fraction, ladder=args.ladder,
                   order=args.quadrature_order, epsilon_tol=args.epsilon_tol,
                   quadrature_tol=args.quadrature_tol, max_nfev=args.max_nfev,
                   gate_mode=args.gate_mode,
                   velocity_method=args.velocity_method, core_radius=args.core_radius,
                   area_quadrature=args.area_quadrature,
                near_ratio=args.near_ratio,
                adaptive_far_order=args.adaptive_far_order,
                adaptive_far_levels=args.adaptive_far_levels,
                adaptive_budget=args.adaptive_budget,
                adaptive_max_escalations=args.adaptive_max_escalations,
                   fan_order=args.fan_order, fan_strips=args.fan_strips,
                   fan_refinement_order=args.fan_refinement_order,
                   fan_refinement_strips=args.fan_refinement_strips,
                   save_callback=save)
        report = {k: v for k, v in result.items() if k not in ('mesh', 'state')}
        report['negative_pressure_drag_warning'] = (
            'CD_pressure is a signed near-field pressure integral subject to '
            'large cancellation and is NOT a validated induced/total drag estimate'
            if any(row['CD_pressure'] < 0 for row in result['history']) else None)
        report['physical_validation_note'] = (
            'No steady-wake claim; a finite epsilon result with rejected nodes '
            'does not certify a principal-value limit or wall/Kutta consistency')
        _write_json(args.out/'report.json', report)
        # A blocked research run is valid evidence, NOT a successful integration run.
        receipt['exit_code'] = 2 if result['status'].startswith('blocked_') else 0
        receipt['status'] = result['status']
        print(result['status']+'; physical_accuracy_accepted=false')
    except Exception as exc:
        receipt.update(exit_code=1, status='execution_error', error=repr(exc))
        (args.out/'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print(str(exc), file=sys.stderr)
    finally:
        receipt['seconds'] = time.perf_counter()-started
        if args.profile_json is not None:
            _profile_enable(False)
            _profile_write(args.profile_json, status=receipt.get('status'),
                           wall_seconds=receipt['seconds'], parameters=parameters)
        _write_json(args.out/'receipt.json', receipt)
    return receipt['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
