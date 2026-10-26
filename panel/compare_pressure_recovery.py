"""Freeze one half-panel solution, compare only the pressure recovery operator.

Example: python -m panel.compare_pressure_recovery --surface SAVED/surface.json
         --out runs/NEW_recovery --max-rings 4 --max-half-panels 1600
Use --snapshot OLD/frozen_state.npz to reuse exactly that state without JAX.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from .pressure_recovery_ab import (
    mesh_geometry, mirror_half_wall, affine_face_gradient,
    cut_vertex_gradient, surface_cp,
)
from .drag_consistency_audit import pressure_integral_audit


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for key in ('mu_half', 'gamma_half', 'sigma_half', 'Cp_raw_half', 'neighbor_ids'):
        array = np.ascontiguousarray(state[key])
        digest.update(key.encode())
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def compare(fixture, state, *, max_rings):
    """No solve occurs here; both algorithms consume the identical immutable mu."""
    x = np.asarray(fixture['coordinates'], dtype=np.float64)
    f = np.asarray(fixture['faces'], dtype=np.int64)
    nf = len(f)
    mu = np.asarray(state['mu_half'], dtype=np.float64)
    original = np.asarray(state['Cp_raw_half'], dtype=np.float64)
    if mu.shape != (nf,) or original.shape != (nf,) or not np.isfinite(original).all():
        raise ValueError('Snapshot/mesh shape mismatch or nonfinite Cp')
    cuts = np.vstack((np.asarray(fixture['cut_edges'], dtype=np.int64).reshape(-1,2),
                      np.asarray(fixture['trailing_edges'], dtype=np.int64).reshape(-1,2)))
    xx, ff, ee = mirror_half_wall(x, f, cuts)
    full_mu = np.r_[mu, mu]
    normal = mesh_geometry(x, f)[-1]
    baseline_gradient = affine_face_gradient(xx, ff, full_mu, state['neighbor_ids'])
    baseline_cp = surface_cp(baseline_gradient, normal, fixture['freestream'])
    control_error = float(np.max(np.abs(baseline_cp-original)))
    if not np.allclose(baseline_cp, original, rtol=2e-6, atol=2e-8):
        raise ValueError(f'Baseline QR failed to reproduce original Cp: max_abs={control_error}')
    gradient, corners, patches = cut_vertex_gradient(
        xx, ff, full_mu, ee, max_rings=max_rings)
    candidate_cp = surface_cp(gradient[:nf], normal, fixture['freestream'])
    kwargs = dict(density=fixture['density'], reference_area=fixture['reference_area'],
                  reference_chord=fixture['reference_chord'],
                  reference_point=fixture['moment_reference'],
                  component_ids=fixture['component_ids'])
    loads = {
        name: pressure_integral_audit(x, f, cp, fixture['freestream'], **kwargs)
        for name, cp in (('original', original), ('affine_qr_control', baseline_cp),
                         ('cut_vertex_affine_candidate', candidate_cp))
    }
    mirror_error = float(np.max(np.abs(gradient[nf:]-gradient[:nf]*[1.,-1.,1.])))
    mirror_ok = np.allclose(gradient[nf:], gradient[:nf]*[1.,-1.,1.], rtol=2e-6, atol=2e-8)
    if not mirror_ok:
        raise ValueError(f'Candidate mirror equivariance failed: max_abs={mirror_error}')
    result = dict(
        status='postprocessing_comparison_not_physical_validation',
        physical_accuracy_accepted=False,
        solver_equations_changed=False,
        baseline_cp_max_abs_error=control_error,
        candidate_vs_original_cp_max_abs=float(np.max(np.abs(candidate_cp-original))),
        candidate_mirror_gradient_max_abs=mirror_error,
        loads=loads,
        patch_count=len(patches),
        max_patch_depth=max(p['depth'] for p in patches),
        min_patch_singular_ratio=min(p['singular_ratio'] for p in patches),
        max_patch_faces=max(len(p['support_faces']) for p in patches),
        component_names='Unverified: retain numeric CAD component IDs',
        wall_normal_residual='Not evaluated: surface recovery is not the direct analytic field',
        selection_policy='Do not select by closeness to ideal Trefftz or by smaller CD',
    )
    fields = dict(original_cp=original, affine_qr_cp=baseline_cp,
                  candidate_cp=candidate_cp, candidate_gradient_half=gradient[:nf],
                  candidate_corner_mu_half=corners[:nf])
    return result, fields, patches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--surface', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--max-half-panels', type=int, default=1600)
    parser.add_argument('--max-rings', type=int, default=4)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a new output directory; preserve all prior experiments')
    if args.max_half_panels < 1 or args.max_rings < 0:
        raise ValueError('Invalid resource/patch budget')
    args.out.mkdir(parents=True)
    report = dict(status='started', physical_accuracy_accepted=False,
                  Python=sys.version, NumPy=np.__version__, platform=platform.platform(),
                  max_rings=args.max_rings, max_half_panels=args.max_half_panels)
    start = time.perf_counter()
    try:
        raw_surface = args.surface.read_bytes()
        source_sha = hashlib.sha256(raw_surface).hexdigest()
        fixture = json.loads(raw_surface)
        if fixture.get('schema') != 'dpanel.half_planar_cut.v1':
            raise ValueError('Require a validated half-planar-cut surface')
        nf = len(fixture['faces'])
        if nf > args.max_half_panels:
            raise MemoryError('Explicitly raise max-half-panels after reviewing resource budget')
        report.update(source_sha256=source_sha, half_wall_panels=nf)
        (args.out/'surface_input.json').write_bytes(raw_surface)
        module_dir = Path(__file__).resolve().parent
        report['panel_source_sha256'] = {p.name: sha(p) for p in sorted(module_dir.glob('*.py'))}
        if args.snapshot is not None:
            with np.load(args.snapshot, allow_pickle=False) as data:
                state = {k: data[k].copy() for k in data.files}
            if str(state['surface_sha256']) != source_sha:
                raise ValueError('Snapshot surface SHA does not match --surface')
            provenance = json.loads(str(state['provenance_json']))
            report.update(state_origin='reused immutable snapshot',
                          snapshot_sha256=sha(args.snapshot), solver_provenance=provenance)
        else:
            import jax
            jax.config.update('jax_enable_x64', True)
            from .run_half_native import load_half_surface
            from .half_native import solve_half_fields
            inputs = load_half_surface(args.surface)
            out = jax.jit(solve_half_fields)(**inputs)
            out['CL'].block_until_ready()
            state = {k: np.asarray(out[k]).copy() for k in
                     ('mu_half','gamma_half','sigma_half','Cp_raw_half')}
            state['neighbor_ids'] = np.asarray(inputs['topology'].neighbor_ghost_ids).copy()
            denominator = float(np.linalg.norm(np.asarray(out['rhs_half'])))
            if denominator <= 0:
                raise ValueError('Zero RHS needs an explicitly defined absolute residual criterion')
            residual = float(np.linalg.norm(np.asarray(out['linear_residual_half']))/denominator)
            state['true_relative_residual'] = np.asarray(residual)
            provenance = dict(JAX=jax.__version__, source_sha256=source_sha,
                              panel_source_sha256=report['panel_source_sha256'])
            state['provenance_json'] = np.asarray(json.dumps(provenance))
            state['surface_sha256'] = np.asarray(source_sha)
            report.update(state_origin='one existing half-native LU solve, then frozen',
                          solver_provenance=provenance)
        residual = float(state['true_relative_residual'])
        if not np.isfinite(residual) or residual > 1e-10:
            raise ValueError(f'Frozen solve fails true residual gate: {residual}')
        report['true_relative_residual'] = residual
        before = state_hash(state)
        report['frozen_state_hash'] = before
        np.savez_compressed(args.out/'frozen_state.npz', **state)
        for array in state.values():
            array.setflags(write=False)
        result, fields, patches = compare(fixture, state, max_rings=args.max_rings)
        if state_hash(state) != before:
            raise AssertionError('A/B altered the frozen potential state')
        np.savez_compressed(args.out/'recovered_fields.npz', **fields)
        write_json(args.out/'patches.json', patches)
        report.update(result)
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                      elapsed_seconds=time.perf_counter()-start)
        write_json(args.out/'report.json', report)
        raise
    report['elapsed_seconds'] = time.perf_counter()-start
    write_json(args.out/'report.json', report)
    print(json.dumps({k:report[k] for k in ('status','half_wall_panels',
                     'baseline_cp_max_abs_error','candidate_vs_original_cp_max_abs',
                     'physical_accuracy_accepted')}, indent=2))


if __name__ == '__main__':
    main()
