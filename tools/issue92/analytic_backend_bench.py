"""End-to-end throughput and parity of the analytic wake evaluator backends.

The closed form used to be evaluated one target per Python call (backend 'loop'). This
tool measures the level the coupled loop actually calls - TotalVelocity.evaluate_many on
the real G2 CAD geometry (872 wall faces, 794 wake triangles) - for:

    loop            per-target reference path (production default)
    numpy/batch     the same closed form, batched over blocks of targets
    torch/cpu       the batched kernel on CPU
    torch/cuda      the batched kernel on the GPU (opt-in)

Parity is always against 'loop'. The target set mimics production sampling: free wake
nodes displaced along the local chord by up to half a chord, so targets sit near the
sheet where the closed form has its largest gradient.

Declared acceptance before the measurement (THRESHOLDS):

  * parity  numpy:                max relative error == 0 (exact float64 port)
  * parity  torch CPU / CUDA:     max relative error <= 1e-12 / 1e-11
  * speed   numpy batch:          >= 1.3x the loop
  * speed   torch CUDA:           >= 5x the loop on >= 512 targets
  * decision equivalence:         the sampling gate accepts/rejects the same targets
                                  and epsilon_error agrees within 1e-9

Two secondary quantities are reported but not thresholded, because the maximum per-target
relative error is dominated by targets where |V| nearly cancels:

  * relative error with a floor of 1% of the mean speed (RELATIVE_FLOOR_FRACTION)
  * the projected time of one production update (18,992 targets)

Usage (a Python that has torch):

    python -m tools.issue92.analytic_backend_bench \
        --surface runs/cases/wingbody/half_cad_wingbody_v1/surface.json \
        --out docs/issue92/analytic_batch_bench.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panel.p1_coupled_material_wake import TotalVelocity, limits  # noqa: E402
from panel.p1_lifting_g1 import from_dpanel_half_surface, make_wing  # noqa: E402
from panel.p1_lifting_g2 import combine_wings  # noqa: E402
from panel.p1_projection_duffy import torch_available  # noqa: E402

THRESHOLDS = {"numpy_max_relative_error": 0.0,
              "torch_cpu_max_relative_error": 1e-12,
              "torch_cuda_max_relative_error": 1e-11,
              "numpy_min_speedup": 1.3,
              "cuda_min_speedup": 5.0,
              "cuda_min_targets": 512,
              "gate_epsilon_error_tolerance": 1e-9}
RELATIVE_FLOOR_FRACTION = 0.01
PRODUCTION_TARGETS = 18992
GATE_TARGETS = 16
CONFIGS = (("loop", "loop", "cpu", None),
           ("numpy/chunk64", "numpy", "cpu", 64),
           ("numpy/chunk256", "numpy", "cpu", 256),
           ("torch/cpu/chunk64", "torch", "cpu", 64),
           ("torch/cpu/chunk256", "torch", "cpu", 256),
           ("torch/cuda/chunk64", "torch", "cuda", 64),
           ("torch/cuda/chunk256", "torch", "cuda", 256))


def default_surface():
    """The CAD fixture lives in the main checkout, not in the worktree."""
    for base in (ROOT, ROOT.parents[1]):
        candidate = base/'runs/cases/wingbody/half_cad_wingbody_v1/surface.json'
        if candidate.is_file():
            return candidate
    return ROOT/'runs/cases/wingbody/half_cad_wingbody_v1/surface.json'


def g2_mesh(surface):
    main = from_dpanel_half_surface(Path(surface))
    tail = make_wing(chord_panels=4, span_panels=4, span=1.8, chord=.55, camber=0.,
                     wake_panels=3)
    return combine_wings([main, tail], names=['main_wing', 'raised_tail'],
                         origins=[(0., 0., 0.), (1.6, 0., .35)],
                         reference_chord=main.chord, reference_point=main.reference_point)


def fabricated_state(mesh, alpha_deg=5.):
    """A well-formed state; the closed form costs the same for any strength values."""
    centroid = mesh.xyz[mesh.dof_face].mean(axis=1)
    mu = 0.01*np.sin(3.*centroid[:, 0]) * np.cos(2.*centroid[:, 1])
    gamma = np.concatenate([0.02*np.cos(np.arange(g.gamma_hat.shape[1]))
                            for g in mesh.groups])
    return dict(mu=mu, gamma=gamma, alpha_deg=alpha_deg)


def targets(mesh, count, seed=92):
    rng = np.random.default_rng(seed)
    nodes = np.concatenate([g.wake_xyz for g in mesh.groups])
    base = nodes[rng.integers(0, len(nodes), size=count)]
    offsets = np.stack([rng.uniform(-0.1, 0.5, count), rng.uniform(-0.15, 0.15, count),
                        rng.uniform(-0.15, 0.15, count)], axis=1)
    return base + offsets


def build_total_velocity(mesh, state, backend, device, chunk):
    kwargs = dict(analytic_backend=backend, analytic_device=device)
    if chunk is not None:
        kwargs['analytic_chunk'] = chunk
    return TotalVelocity(mesh, state, **kwargs)


def time_backend(field, points, repeats):
    best, values = None, None
    for _ in range(repeats):
        started = time.perf_counter()
        values = field.evaluate_many(points)
        elapsed = time.perf_counter()-started
        best = elapsed if best is None else min(best, elapsed)
    return best, np.asarray(values, dtype=float)


def gate_agreement(mesh, state, config, points):
    """The decision the coupled loop makes: does the gate accept the same targets?"""
    normals = np.tile([0., 0., 1.], (len(points), 1))
    scales = np.full(len(points), .1)
    kwargs = dict(eps_fraction=.05, ladder=(1., .5, .25), order=48, tag='bench')
    out = {}
    for label, (backend, device, chunk) in (('loop', ('loop', 'cpu', None)),
                                            (config[0], (config[1], config[2], config[3]))):
        field = build_total_velocity(mesh, state, backend, device, chunk)
        out[label] = limits(field, points, normals, scales, **kwargs)
    reference, candidate = out['loop'], out[config[0]]
    tol = THRESHOLDS['gate_epsilon_error_tolerance']
    return dict(reference='loop', candidate=config[0], targets=int(len(points)),
                epsilon_tolerance=.02, quadrature_tolerance=.002,
                max_epsilon_error_difference=float(
                    np.abs(candidate['epsilon_error']-reference['epsilon_error']).max()),
                max_plus_difference=float(np.abs(candidate['plus']-reference['plus']).max()),
                decisions_identical=bool(np.array_equal(
                    candidate['epsilon_error'] <= .02, reference['epsilon_error'] <= .02)),
                within_tolerance=bool(np.abs(
                    candidate['epsilon_error']-reference['epsilon_error']).max() <= tol),
                reference_epsilon_errors=[float(x) for x in reference['epsilon_error']])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--surface', type=Path, default=default_surface())
    parser.add_argument('--targets', type=int, default=1024)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--out', type=Path,
                        default=ROOT/'docs/issue92/analytic_batch_bench.json')
    args = parser.parse_args(argv)
    if args.targets < THRESHOLDS["cuda_min_targets"]:
        parser.error('--targets must be >= ' + str(THRESHOLDS["cuda_min_targets"]))

    mesh = g2_mesh(args.surface)
    state = fabricated_state(mesh)
    points = targets(mesh, args.targets)
    have_torch = torch_available()

    rows, reference, uref = [], None, None
    for label, backend, device, chunk in CONFIGS:
        if backend == 'torch' and not have_torch:
            rows.append(dict(config=label, status='skipped', reason='torch unavailable'))
            continue
        try:
            field = build_total_velocity(mesh, state, backend, device, chunk)
            seconds, values = time_backend(field, points, args.repeats)
        except Exception as error:  # a backend that cannot run must not abort the sweep
            rows.append(dict(config=label, status='failed', error=repr(error)[:300]))
            continue
        if reference is None:
            reference = values
            uref = float(np.linalg.norm(reference, axis=1).mean())
        delta = np.abs(values-reference)
        scale = np.maximum(np.abs(reference), 1e-300)
        floored = np.maximum(np.abs(reference), RELATIVE_FLOOR_FRACTION*uref)
        rows.append(dict(config=label, status='ok', backend=backend, device=device,
                         chunk=chunk, seconds=seconds, targets=int(len(points)),
                         ms_per_target=1000.*seconds/len(points),
                         max_absolute_error=float(delta.max()),
                         max_relative_error=float((delta/scale).max()),
                         max_floored_relative_error=float((delta/floored).max()),
                         mean_speed=uref))

    ok = [r for r in rows if r.get('status') == 'ok']
    base = ok[0]
    for row in ok:
        row['speedup_vs_loop'] = base['seconds']/row['seconds']
        row['projected_seconds_18992'] = row['ms_per_target']*PRODUCTION_TARGETS/1000.

    checks = {}
    for row in ok:
        name, kind = row['config'], row['backend']
        if kind == 'numpy':
            checks[name+':parity'] = bool(
                row['max_relative_error'] <= THRESHOLDS['numpy_max_relative_error'])
        elif kind == 'torch' and row['device'] == 'cpu':
            checks[name+':parity'] = bool(
                row['max_relative_error'] <= THRESHOLDS['torch_cpu_max_relative_error'])
        elif kind == 'torch' and row['device'] == 'cuda':
            checks[name+':parity'] = bool(
                row['max_relative_error'] <= THRESHOLDS['torch_cuda_max_relative_error'])
            checks[name+':speedup'] = bool(
                row['speedup_vs_loop'] >= THRESHOLDS['cuda_min_speedup'])
    np_rows = [r for r in ok if r['backend'] == 'numpy']
    if np_rows:
        checks['numpy:min_speedup'] = bool(
            max(r['speedup_vs_loop'] for r in np_rows) >= THRESHOLDS['numpy_min_speedup'])

    best = max(ok, key=lambda r: r['speedup_vs_loop'])
    gate_config = (best['config'], best['backend'], best['device'], best['chunk'])
    try:
        gate = gate_agreement(mesh, state, gate_config,
                              points[:min(GATE_TARGETS, len(points))])
        checks['gate:decisions_identical'] = gate['decisions_identical']
        checks['gate:within_tolerance'] = gate['within_tolerance']
    except Exception as error:
        gate = dict(status='unavailable', error=repr(error)[:300])
        checks['gate:decisions_identical'] = False
        checks['gate:within_tolerance'] = False

    report = dict(tool='tools/issue92/analytic_backend_bench.py',
                  generated=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  python=platform.python_version(),
                  torch_available=bool(have_torch),
                  geometry=dict(surface=str(args.surface),
                                wall_faces=int(len(mesh.dof_face)),
                                wake_triangles=int(sum(len(g.wake_faces)
                                                       for g in mesh.groups)),
                                groups=[g.name for g in mesh.groups]),
                  targets=int(len(points)), repeats=int(args.repeats),
                  relative_floor_fraction=RELATIVE_FLOOR_FRACTION,
                  production_targets=PRODUCTION_TARGETS,
                  thresholds=THRESHOLDS, configs=rows, gate=gate, checks=checks,
                  passed=all(checks.values()) if checks else False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print('geometry: wall %d, wake triangles %d, targets %d'
          % (len(mesh.dof_face), sum(len(g.wake_faces) for g in mesh.groups), len(points)))
    for row in rows:
        if row.get('status') != 'ok':
            print('%-20s %s %s' % (row['config'], row['status'],
                                   row.get('reason', row.get('error'))))
            continue
        print('%-20s %8.3f s  %7.3f ms/target  speedup %5.2fx  maxrel %.3e  floored %.3e  proj %.1f s'
              % (row['config'], row['seconds'], row['ms_per_target'],
                 row['speedup_vs_loop'], row['max_relative_error'],
                 row['max_floored_relative_error'], row['projected_seconds_18992']))
    print('gate:', json.dumps({k: v for k, v in gate.items()
                               if k != 'reference_epsilon_errors'}))
    print('checks:', json.dumps(checks), 'passed:', report['passed'])
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
