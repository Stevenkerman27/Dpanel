"""Opt-in nested-Gamma experiment; fixed wall, pressure operator and wake support.

python -m panel.run_gamma_refinement --surface SAVED/surface.json --out runs/NEW
No production module is edited. New states are research data, not certified loads.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
from .gamma_refinement import (
    refine_space, basis_at, split_wake, wake_basis, weak_te, wake_operator,
    nested_checks, solve_augmented, relative_error,
)


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _guard(a, b, name):
    if not np.allclose(a, b, rtol=5e-10, atol=5e-12):
        raise ArithmeticError(f'1x regression failed for {name}: {relative_error(a, b)}')
    return relative_error(a, b)


def run(surface, out, *, factors=(1, 2, 4), max_half_panels=1500,
        max_pieces=20000, block=32, max_pairs=1000000, comparison_subdivisions=32):
    surface, out = Path(surface), Path(out)
    if out.exists():
        raise FileExistsError('Use a new output directory; preserve prior experiments')
    factors = tuple(factors)
    if (not factors or factors[0] != 1 or sorted(set(factors)) != list(factors)
            or any(f not in (1, 2, 4) for f in factors)
            or comparison_subdivisions < max(factors)
            or comparison_subdivisions % max(factors) != 0
            or min(max_half_panels, max_pieces, block, max_pairs) < 1):
        raise ValueError('Use ordered factors starting at 1 from {1,2,4} and positive budgets')
    raw = surface.read_bytes()
    fixture = json.loads(raw)
    if fixture.get('schema') != 'dpanel.half_planar_cut.v1':
        raise ValueError('Require validated half-planar-cut geometry')
    nf = len(fixture['faces'])
    if nf > max_half_panels:
        raise MemoryError('Explicit half-wall budget exceeded before any solve/import')
    y0 = np.asarray(fixture['gamma_stations'], float)
    if 2*(len(y0)-1)*comparison_subdivisions+1 > 4001:
        raise MemoryError('Fixed ideal-Trefftz comparison grid exceeds 4001 stations')
    out.mkdir(parents=True)
    report = dict(status='started', physical_accuracy_accepted=False,
                  source_surface_sha256=hashlib.sha256(raw).hexdigest(),
                  wall_panels=nf, factors=list(factors), levels=[],
                  fixed=['wall coordinates/faces', 'pressure stencil and formula',
                         'freestream and reference values', 'physical wake shape/length',
                         'TE geometry/banks and contact constraints'],
                  changed='Gamma trial/test space; mu and Gamma are re-solved',
                  geometry_vjp_supported=False)
    started = time.perf_counter()
    try:
        import jax
        import jax.numpy as jnp
        import scipy
        jax.config.update('jax_enable_x64', True)
        from .run_half_native import load_half_surface
        from .half_native import solve_half_fields, _half_loads
        from .drag_consistency_audit import (
            pressure_integral_audit, ideal_trefftz_piecewise_linear,
        )
        report['environment'] = dict(python=platform.python_version(),
            numpy=np.__version__, scipy=scipy.__version__, jax=jax.__version__,
            device=str(jax.devices()[0]))
        module_dir = Path(__file__).resolve().parent
        report['source_sha256'] = {p.name: _sha(p) for p in module_dir.glob('*.py')}
        (out/'surface_input.json').write_bytes(raw)
        inputs = load_half_surface(surface)
        reference = jax.jit(solve_half_fields)(**inputs)
        reference['CL'].block_until_ready()
        old_A = np.asarray(reference['matrix_half'])
        old_rhs = np.asarray(reference['rhs_half'])
        body, rhs = old_A[:nf, :nf].copy(), old_rhs[:nf].copy()
        x = np.asarray(inputs['vertices'])
        t = inputs['topology']
        faces, wake_faces = np.asarray(t.faces), np.asarray(t.wake_faces)
        original_wake = x[wake_faces]
        centers = x[faces].mean(axis=1)
        ends = x[np.asarray(t.te_edges)]
        up, lo = np.asarray(t.upper_te), np.asarray(t.lower_te)
        flow = np.asarray(inputs['freestream'])
        speed, chord = float(np.linalg.norm(flow)), float(inputs['reference_chord'])
        origin = np.asarray(inputs['reference_point'])
        density, area = float(inputs['density']), float(inputs['reference_area'])
        dofs0 = np.asarray(fixture['gamma_station_dof'], dtype=np.int64)
        comparison_y = np.concatenate([
            np.linspace(a, b, comparison_subdivisions+1)[:-1]
            for a, b in zip(y0[:-1], y0[1:])]+[y0[-1:]])
        report['ideal_comparison'] = dict(
            subdivisions_per_original_interval=comparison_subdivisions,
            full_station_count=2*len(comparison_y)-1,
            interpretation='Same quadrature grid at every level; ideal model, NOT physical truth')
        recover = jax.jit(_half_loads)
        old_te = None
        for factor in factors:
            tic = time.perf_counter()
            space = refine_space(y0, dofs0, factor)
            pieces, parents, interval, geom = split_wake(
                original_wake, space.stations, max_pieces=max_pieces)
            te = weak_te(ends, up, lo, nf, space)
            W = wake_operator(centers, pieces, interval, space,
                              block=block, max_pairs=max_pairs)
            if factor == 1:
                old_te = te
                _guard(W, old_A[:nf, nf:], 'original wake block')
                _guard(-te['K']/te['measure'][:, None], old_A[nf:, :nf], 'Kutta wall block')
                _guard(te['M']/te['measure'][:, None], old_A[nf:, nf:], 'Kutta mass block')
            identities = nested_checks(old_A[:nf, nf:], old_te, W, te, space.prolongation)
            mu, gamma, residual = solve_augmented(body, rhs, W, te)
            fields = recover(inputs['vertices'], t, inputs['freestream'], inputs['density'],
                             inputs['reference_area'], inputs['reference_chord'],
                             inputs['reference_point'], jnp.asarray(mu))
            fields['CL'].block_until_ready()
            cp = np.asarray(fields['Cp_raw_half'])
            regression = None
            if factor == 1:
                regression = dict(mu=_guard(mu, reference['mu_half'], 'mu'),
                                  gamma=_guard(gamma, reference['gamma_half'], 'gamma'),
                                  Cp=_guard(cp, reference['Cp_raw_half'], 'Cp'),
                                  CL=_guard(fields['CL'], reference['CL'], 'CL'),
                                  CD=_guard(fields['CD_pressure'], reference['CD_pressure'], 'CD'),
                                  CM=_guard(fields['CM'], reference['CM'], 'CM'))
            loads = pressure_integral_audit(x, faces, cp, flow, density=density,
                reference_area=area, reference_chord=chord, reference_point=origin,
                component_ids=fixture['component_ids'])
            _guard(loads['CD_pressure'], fields['CD_pressure'], 'pressure reintegration CD')
            _guard(loads['CM_y'], np.asarray(fields['CM'])[1], 'pressure reintegration CM')
            g_sample = basis_at(comparison_y, space)@gamma
            full_y, full_g = np.r_[-comparison_y[:0:-1], comparison_y], np.r_[g_sample[:0:-1], g_sample]
            ideal = ideal_trefftz_piecewise_linear(full_y, full_g, U=speed,
                                                   density=density, reference_area=area)
            jump = mu[up]-mu[lo]
            misfit = te['sample_hat']@gamma-jump[te['sample_edge']]
            te_l2 = float(np.sqrt(np.average(misfit**2, weights=te['sample_weight']))/(speed*chord))
            gamma_nodes = space.hat@gamma
            slopes = np.diff(gamma_nodes)/np.diff(space.stations)
            record = dict(factor=factor, free_gamma_dofs=len(gamma), stations=len(space.stations),
                wake_integration=geom, true_residual=residual, old_space_embedding=identities,
                original_regression=regression, loads=loads,
                component_names='Numeric CAD IDs only; no unverified name mapping',
                ideal_trefftz_common_grid=ideal,
                te_jump_l2_over_Uc=te_l2,
                te_jump_note='L2 projection of discrete wall potential jump, NOT DeltaCp or physical Kutta validation',
                gamma_min=float(gamma_nodes.min()), gamma_max=float(gamma_nodes.max()),
                slope_max_abs=float(np.max(np.abs(slopes))),
                mass_condition_number=float(np.linalg.cond(te['M'])),
                te_length_relative_error=te['length_relative_error'],
                elapsed_seconds=time.perf_counter()-tic)
            export_basis, export_slope = wake_basis(pieces, interval, space)
            np.savez_compressed(out/f'factor_{factor}.npz', mu_half=mu, gamma_half=gamma,
                sigma_half=np.asarray(reference['sigma_half']),
                wake_doublet_half=export_basis@gamma, wake_slope_half=export_slope@gamma,
                source_surface_sha256=np.asarray(report['source_surface_sha256']),
                gamma_stations=space.stations, gamma_station_dof=space.dofs,
                gamma_station_values=gamma_nodes, Cp_raw_half=cp,
                wake_triangles=pieces, wake_parent=parents, wake_interval=interval,
                prolongation=space.prolongation, pressure_neighbor_ids=np.asarray(t.neighbor_ghost_ids),
                panel_forces_half=np.asarray(fields['panel_forces_half']),
                panel_moments_half=np.asarray(fields['panel_moments_half']))
            report['levels'].append(record)
            _write(out/'report.json', report)
        if surface.read_bytes() != raw:
            raise RuntimeError('Input surface changed during experiment')
        report['status'] = 'nested_gamma_experiment_completed_not_physical_validation'
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                      elapsed_seconds=time.perf_counter()-started)
        _write(out/'report.json', report)
        raise
    report['elapsed_seconds'] = time.perf_counter()-started
    _write(out/'report.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--surface', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--factors', default='1,2,4')
    p.add_argument('--max-half-panels', type=int, default=1500)
    p.add_argument('--max-pieces', type=int, default=20000)
    p.add_argument('--block', type=int, default=32)
    p.add_argument('--max-pairs', type=int, default=1000000)
    p.add_argument('--comparison-subdivisions', type=int, default=32)
    a = p.parse_args()
    result = run(a.surface, a.out, factors=tuple(int(f) for f in a.factors.split(',')),
                 max_half_panels=a.max_half_panels, max_pieces=a.max_pieces,
                 block=a.block, max_pairs=a.max_pairs,
                 comparison_subdivisions=a.comparison_subdivisions)
    print(json.dumps({k: result[k] for k in ('status', 'wall_panels', 'physical_accuracy_accepted')}))


if __name__ == '__main__':
    main()
