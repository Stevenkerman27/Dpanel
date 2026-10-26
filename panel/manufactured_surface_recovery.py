"""Known-potential verification on frozen PR29 wall geometry and stencils.

NumPy only. No panel solve, wake change, geometry repair, or pressure clipping.
The reference is the restriction of a known polynomial to the SAME flat faces,
NOT a CAD reference or an exact solution of the aircraft boundary-value problem.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from . import pressure_recovery_ab as recovery

REFLECT = np.array([1., -1., 1.])
AXIAL = -REFLECT
MODES = ('constant', 'linear_x', 'linear_z', 'quadratic_xz',
         'quadratic_x2_z2', 'quadratic_y2_z2')
INPUT_FILES = ('surface_input.json', 'frozen_state.npz',
               'recovered_fields.npz', 'patches.json', 'report.json')
STATE_KEYS = ('mu_half', 'gamma_half', 'sigma_half', 'Cp_raw_half', 'neighbor_ids')


def polynomial_coefficients(mode):
    """All modes are harmonic in ambient space and even under y reflection."""
    b, h, c = np.zeros(3), np.zeros((3, 3)), 0.
    if mode == 'constant':
        c = 1.
    elif mode == 'linear_x':
        b[0] = 1.
    elif mode == 'linear_z':
        b[2] = 1.
    elif mode == 'quadratic_xz':
        h[0, 2] = h[2, 0] = 1.
    elif mode == 'quadratic_x2_z2':
        h[0, 0], h[2, 2] = 2., -2.
    elif mode == 'quadratic_y2_z2':
        h[1, 1], h[2, 2] = 2., -2.
    else:
        raise ValueError(f'Unknown prescribed potential: {mode}')
    return c, b, h


def known_potential(points, mode, *, speed, chord, origin, amplitude):
    """phi=a*U*c*f((x-origin)/c); gradient=a*U*grad(f), in SI units."""
    points, origin = np.asarray(points, float), np.asarray(origin, float)
    if (points.ndim < 2 or points.shape[-1] != 3 or origin.shape != (3,)
            or not np.isfinite(points).all() or not np.isfinite(origin).all()
            or not np.isfinite([speed, chord, amplitude]).all()
            or min(speed, chord, amplitude) <= 0):
        raise ValueError('Finite points/origin and positive U/c/amplitude required')
    c, b, h = polynomial_coefficients(mode)
    scaled = (points-origin)/chord
    phi = amplitude*speed*chord*(c + scaled@b
            + .5*np.einsum('...i,ij,...j->...', scaled, h, scaled))
    grad = amplitude*speed*(b + scaled@h)
    return phi, grad


def project_tangent(vector, normal):
    return vector-np.sum(vector*normal, axis=-1)[..., None]*normal


def p1_gradient(triangles, corner_values):
    """Exact P1 gradient from known vertex values; no cell-to-node recovery."""
    tri, values = np.asarray(triangles, float), np.asarray(corner_values, float)
    if (tri.ndim != 3 or tri.shape[1:] != (3, 3)
            or values.shape != tri.shape[:2] or not np.isfinite(tri).all()
            or not np.isfinite(values).all()):
        raise ValueError('Finite triangles and three values per triangle required')
    cross = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    norm = np.linalg.norm(cross, axis=1)
    if np.any(norm <= 0):
        raise ValueError('Degenerate P1 triangle')
    normal = cross/norm[:, None]
    shifted = values-values[:, :1]
    answer = np.zeros((len(tri), 3))
    for k in range(3):
        edge = tri[:, (k+2) % 3]-tri[:, (k+1) % 3]
        answer += shifted[:, k, None]*np.cross(normal, edge)/norm[:, None]
    return answer


def triangle_rule(triangles, order=3):
    """Positive Duffy tensor Gauss rule, weights normalized to unit face area.

    A degree-d polynomial maps to u degree <=d+1 and v degree <=d.
    Three Gauss points/axis therefore integrate force (degree 2) and moment
    (degree 3) exactly for the present quadratic potentials on flat triangles.
    """
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError('Positive integer quadrature order required')
    tri = np.asarray(triangles, float)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3) or not np.isfinite(tri).all():
        raise ValueError('Finite triangles required')
    t, w = np.polynomial.legendre.leggauss(order)
    t, w = (t+1)/2, w/2
    u, v = np.meshgrid(t, t, indexing='ij')
    wu, wv = np.meshgrid(w, w, indexing='ij')
    bary = np.column_stack(((1-u.ravel())*(1-v.ravel()),
                            u.ravel(), (1-u.ravel())*v.ravel()))
    weights = (2*(1-u)*wu*wv).ravel()
    return np.einsum('qk,fkc->fqc', bary, tri), weights


def face_loads(points, cp, weights, normals, areas, flow, density, reference_point):
    """Integrate the supplied face samples, retaining each half-face F and M."""
    points, cp, weights = (np.asarray(a, float) for a in (points, cp, weights))
    if (points.ndim != 3 or points.shape[-1] != 3 or cp.shape != points.shape[:2]
            or weights.shape != (points.shape[1],) or not np.isfinite(cp).all()
            or not np.isfinite(points).all() or np.any(weights <= 0)
            or not np.isclose(weights.sum(), 1., atol=1e-13, rtol=1e-13)):
        raise ValueError('Finite samples and positive normalized weights required')
    q = .5*density*float(np.dot(flow, flow))
    df = -q*cp[:, :, None]*normals[:, None, :]*areas[:, None, None]
    force = np.einsum('fqk,q->fk', df, weights)
    moment = np.einsum('fqk,q->fk', np.cross(points-reference_point, df), weights)
    return force, moment


def coefficient_rows(force, moment, *, flow, density, area_ref, chord):
    """Exact polar/axial mirror sum; symmetric flow and on-plane moment origin."""
    flow = np.asarray(flow, float)
    speed = float(np.linalg.norm(flow))
    if (flow.shape != (3,) or not np.isfinite(flow).all()
            or not np.isfinite([density, area_ref, chord]).all()
            or min(speed, density, area_ref, chord) <= 0
            or abs(flow[1]) > 1e-12*speed):
        raise ValueError('Positive references and zero-sideslip flow required')
    drag = flow/speed
    lift = np.cross(drag, [0., 1., 0.])
    lift /= np.linalg.norm(lift)
    full_force, full_moment = force*(1+REFLECT), moment*(1+AXIAL)
    qs = .5*density*speed**2*area_ref
    return np.column_stack((full_force@drag/qs, full_force@lift/qs,
                            full_moment[:, 1]/(qs*chord)))


def error_summary(mask, area, grad_error, cp_error, recovered, center, integrated):
    """Absolute/normalized errors only: near-zero exact loads are not divisors."""
    if not np.any(mask):
        return None
    rg = grad_error[mask]
    e = cp_error[mask]
    pieces = {'recovery': recovered-center,
              'centroid_sampling': center-integrated,
              'total': recovered-integrated}
    out = dict(face_count=int(mask.sum()),
               area_fraction=float(area[mask].sum()/area.sum()),
               gradient_rms_over_U=float(np.sqrt(np.average(
                   np.sum(rg**2, axis=1), weights=area[mask]))),
               gradient_max_over_U=float(np.linalg.norm(rg, axis=1).max()),
               cp_rms=float(np.sqrt(np.average(e*e, weights=area[mask]))),
               cp_max_abs=float(np.abs(e).max()),
               coefficients={})
    for j, name in enumerate(('CD', 'CL', 'CM_y')):
        data = dict(recovered=float(recovered[mask, j].sum()),
                    exact_centroid=float(center[mask, j].sum()),
                    exact_integrated=float(integrated[mask, j].sum()))
        for label, difference in pieces.items():
            selected = difference[mask, j]
            data[label+'_delta'] = float(selected.sum())
            data[label+'_sum_abs_delta'] = float(np.abs(selected).sum())
        out['coefficients'][name] = data
    closure = pieces['total']-pieces['recovery']-pieces['centroid_sampling']
    out['decomposition_max_abs'] = float(np.max(np.abs(closure[mask])))
    return out


def _patch_signature(patches):
    return [(p['vertex'], p['fan_faces'], p['support_faces'], p['depth'])
            for p in patches]


def analyze(fixture, neighbor_ids, *, max_rings, origin, amplitude=.1,
            expected_patches=None):
    """No solve. Prescribed values replace state inputs, never mutate them."""
    x, f, tri, centers, av, area, normal = recovery.mesh_geometry(
        fixture['coordinates'], fixture['faces'])
    nf = len(f)
    flow = np.asarray(fixture['freestream'], float)
    point = np.asarray(fixture['moment_reference'], float)
    origin = np.asarray(origin, float)
    chord, rho, sref = (float(fixture[k]) for k in
                       ('reference_chord', 'density', 'reference_area'))
    speed = float(np.linalg.norm(flow))
    if (point.shape != (3,) or origin.shape != (3,)
            or not np.isfinite(point).all() or not np.isfinite(origin).all()
            or abs(point[1]) > 1e-12*chord or abs(origin[1]) > 1e-12*chord):
        raise ValueError('Potential origin and moment reference must be on y=0')
    # Also validates speed and units before any computation.
    coeff = dict(flow=flow, density=rho, area_ref=sref, chord=chord)
    coefficient_rows(np.zeros((nf, 3)), np.zeros((nf, 3)), **coeff)
    ids = np.asarray(neighbor_ids)
    if (ids.ndim != 2 or len(ids) != nf or not np.issubdtype(ids.dtype, np.integer)
            or ids.size == 0 or ids.min() < 0 or ids.max() >= 2*nf):
        raise ValueError('Frozen right-face rows / full-mirror neighbor IDs required')
    cuts = np.vstack((np.asarray(fixture['cut_edges'], dtype=np.int64).reshape(-1, 2),
                      np.asarray(fixture['trailing_edges'], dtype=np.int64).reshape(-1, 2)))
    xx, ff, ee = recovery.mirror_half_wall(x, f, cuts)
    full_centers = xx[ff].mean(axis=1)
    tags = np.asarray(fixture['component_ids'])
    if tags.shape != (nf,) or not np.issubdtype(tags.dtype, np.integer):
        raise ValueError('Numeric component IDs required')
    te = np.zeros(nf, bool)
    for key in ('upper_te_faces', 'lower_te_faces'):
        bank = np.asarray(fixture[key])
        if (bank.ndim != 1 or not np.issubdtype(bank.dtype, np.integer)
                or (len(bank) and (bank.min() < 0 or bank.max() >= nf))):
            raise ValueError('Invalid real trailing-edge bank IDs')
        te[bank] = True
    masks = {'all': np.ones(nf, bool), 'te_adjacent': te, 'not_te_adjacent': ~te}
    masks.update({'component_'+str(int(tag)): tags == tag for tag in np.unique(tags)})
    full_normals = np.vstack((normal, normal*REFLECT))
    stencil_angles = np.degrees(np.arccos(np.clip(
        np.sum(full_normals[ids]*normal[:, None], axis=2), -1., 1.))).max(axis=1)
    result = dict(status='manufactured_recovery_verified_not_physical_validation',
                  physical_accuracy_accepted=False, new_flow_solve=False,
                  reference='analytic restriction to the same flat triangles; not CAD/flow truth',
                  origin=origin.tolist(), amplitude=amplitude, modes={},
                  component_names='unverified; numeric CAD IDs only',
                  quadrature='3x3 Duffy Gauss, cross-checked against 4x4',
                  half_faces=nf, half_wall_area=float(area.sum()),
                  full_mirrored_wall_area=float(2*area.sum()),
                  max_baseline_stencil_normal_angle_deg=float(stencil_angles.max()))
    arrays = dict(half_centers=centers, half_areas=area, half_normals=normal,
                  component_ids=tags, te_adjacent_mask=te,
                  baseline_stencil_normal_angle_deg=stencil_angles)
    p_args = dict(speed=speed, chord=chord, origin=origin, amplitude=amplitude)
    for mode in MODES:
        full_mu, _ = known_potential(full_centers, mode, **p_args)
        _, ambient = known_potential(centers, mode, **p_args)
        exact_grad = project_tangent(ambient, normal)
        exact_cp = recovery.surface_cp(exact_grad, normal, flow)
        exact_velocity = project_tangent(np.broadcast_to(flow, normal.shape), normal)+exact_grad
        corner_mu, _ = known_potential(tri, mode, **p_args)
        gradients = dict(affine_qr=recovery.affine_face_gradient(xx, ff, full_mu, ids))
        node_gradient, _, patches = recovery.cut_vertex_gradient(
            xx, ff, full_mu, ee, max_rings=max_rings)
        if expected_patches is not None and _patch_signature(patches) != _patch_signature(expected_patches):
            raise ValueError('Prescribed field changed the saved geometry-only patch plan')
        gradients['cut_vertex_candidate'] = node_gradient[:nf]
        gradients['exact_vertex_p1_reference'] = p1_gradient(tri, corner_mu)
        cforce, cmoment = face_loads(centers[:, None], exact_cp[:, None], np.ones(1),
                                    normal, area, flow, rho, point)
        center_coeff = coefficient_rows(cforce, cmoment, **coeff)
        quadrature = []
        for order in (3, 4):
            qpoints, weights = triangle_rule(tri, order)
            _, qgradient = known_potential(qpoints, mode, **p_args)
            velocity = project_tangent(qgradient+flow, normal[:, None])
            qcp = 1-np.sum(velocity**2, axis=2)/speed**2
            qforce, qmoment = face_loads(qpoints, qcp, weights, normal, area, flow, rho, point)
            quadrature.append((qforce, qmoment))
        if any(not np.allclose(a, b, rtol=2e-11, atol=1e-12)
               for a, b in zip(quadrature[0], quadrature[1])):
            raise AssertionError('Polynomial force/moment quadrature failed')
        integrated_coeff = coefficient_rows(*quadrature[0], **coeff)
        mode_report = dict(operators={},
                           quadrature_force_max_abs=float(np.max(np.abs(quadrature[0][0]-quadrature[1][0]))),
                           quadrature_moment_max_abs=float(np.max(np.abs(quadrature[0][1]-quadrature[1][1]))))
        prefix = mode+'__'
        arrays.update({prefix+'exact_gradient': exact_grad, prefix+'exact_centroid_cp': exact_cp,
                       prefix+'exact_centroid_coefficients': center_coeff,
                       prefix+'exact_integrated_coefficients': integrated_coeff,
                       prefix+'known_face_mu_half': full_mu[:nf],
                       prefix+'exact_integrated_force_half': quadrature[0][0],
                       prefix+'exact_integrated_moment_half': quadrature[0][1]})
        for name, gradient in gradients.items():
            cp = recovery.surface_cp(gradient, normal, flow)
            error_v = gradient-exact_grad
            linear = -2*np.sum(exact_velocity*error_v, axis=1)/speed**2
            square = -np.sum(error_v**2, axis=1)/speed**2
            identity_error = float(np.max(np.abs(cp-exact_cp-linear-square)))
            if not np.allclose(cp-exact_cp, linear+square, rtol=1e-10, atol=1e-12):
                raise AssertionError('Cp velocity-error decomposition failed')
            force, moment = face_loads(centers[:, None], cp[:, None], np.ones(1),
                                       normal, area, flow, rho, point)
            rows = coefficient_rows(force, moment, **coeff)
            groups = {label: error_summary(mask, area, error_v/speed, cp-exact_cp,
                                           rows, center_coeff, integrated_coeff)
                      for label, mask in masks.items()}
            if any(g is not None and g['decomposition_max_abs'] > 1e-11 for g in groups.values()):
                raise AssertionError('Recovery/sampling/load decomposition failed')
            mode_report['operators'][name] = dict(
                groups=groups, cp_error_identity_max_abs=identity_error,
                linear_cp_error_area_rms=float(np.sqrt(np.average(linear**2, weights=area))),
                square_cp_error_area_rms=float(np.sqrt(np.average(square**2, weights=area))))
            for suffix, value in dict(gradient=gradient, cp=cp, cp_error_linear=linear,
                                      cp_error_square=square, coefficient_rows=rows,
                                      force_half=force, moment_half=moment).items():
                arrays[prefix+name+'__'+suffix] = value
        result['modes'][mode] = mode_report
    return result, arrays


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _state_hash(state):
    digest = hashlib.sha256()
    for key in STATE_KEYS:
        array = np.ascontiguousarray(state[key])
        digest.update(key.encode())
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_frozen_run(directory, *, max_half_panels=2200):
    directory = Path(directory)
    hashes = {name: _sha(directory/name) for name in INPUT_FILES}
    fixture = json.loads((directory/'surface_input.json').read_text(encoding='utf-8'))
    report = json.loads((directory/'report.json').read_text(encoding='utf-8'))
    patches = json.loads((directory/'patches.json').read_text(encoding='utf-8'))
    if (fixture.get('schema') != 'dpanel.half_planar_cut.v1'
            or report.get('status') != 'postprocessing_comparison_not_physical_validation'
            or report.get('source_sha256') != hashes['surface_input.json']):
        raise ValueError('Need a successful PR29 run with matching surface SHA')
    if (not isinstance(max_half_panels, int) or max_half_panels < 1
            or len(fixture['faces']) > max_half_panels):
        raise MemoryError('Explicit half-panel resource guard exceeded')
    with np.load(directory/'frozen_state.npz', allow_pickle=False) as data:
        state = {name: data[name].copy() for name in data.files}
    with np.load(directory/'recovered_fields.npz', allow_pickle=False) as data:
        cp_original = data['original_cp'].copy()
    if (str(state['surface_sha256']) != hashes['surface_input.json']
            or _state_hash(state) != report.get('frozen_state_hash')
            or not np.array_equal(cp_original, state['Cp_raw_half'])):
        raise ValueError('Frozen state hash/Cp does not match the saved PR29 run')
    residual = float(state['true_relative_residual'])
    if not np.isfinite(residual) or residual > 1e-10 or residual < 0:
        raise ValueError('Original frozen solve residual is not accepted')
    for value in state.values():
        value.setflags(write=False)
    # Control: reuse the saved production stencil on the original state first.
    x, f, *_ = recovery.mesh_geometry(fixture['coordinates'], fixture['faces'])
    cuts = np.vstack((np.asarray(fixture['cut_edges'], int).reshape(-1, 2),
                      np.asarray(fixture['trailing_edges'], int).reshape(-1, 2)))
    xx, ff, _ = recovery.mirror_half_wall(x, f, cuts)
    gradient = recovery.affine_face_gradient(xx, ff, np.r_[state['mu_half'], state['mu_half']],
                                            state['neighbor_ids'])
    normals = recovery.mesh_geometry(x, f)[-1]
    cp_control = recovery.surface_cp(gradient, normals, fixture['freestream'])
    if not np.allclose(cp_control, cp_original, rtol=2e-6, atol=2e-8):
        raise ValueError('Saved production field is not reproduced by the QR control')
    return fixture, state, patches, report, hashes, float(np.max(np.abs(cp_control-cp_original)))


def run(directory, out, *, origin=(0., 0., 0.), amplitude=.1, max_half_panels=2200):
    out, directory = Path(out), Path(directory)
    if out.exists():
        raise FileExistsError('Use a new output directory; preserve old evidence')
    fixture, state, patches, old, hashes, cp_error = load_frozen_run(
        directory, max_half_panels=max_half_panels)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(status='started', physical_accuracy_accepted=False,
                  python=platform.python_version(), numpy=np.__version__,
                  input_sha256=hashes, original_frozen_state_hash=_state_hash(state),
                  source_sha256=hashes['surface_input.json'],
                  source_module_sha256={Path(__file__).name: _sha(__file__),
                                        Path(recovery.__file__).name: _sha(recovery.__file__)},
                  baseline_original_cp_max_abs_error=cp_error,
                  original_true_relative_residual=float(state['true_relative_residual']))
    try:
        result, arrays = analyze(fixture, state['neighbor_ids'], max_rings=int(old['max_rings']),
                                 origin=origin, amplitude=amplitude, expected_patches=patches)
        if hashes != {name: _sha(directory/name) for name in INPUT_FILES}:
            raise AssertionError('Input files changed during the audit')
        report.update(result)
        np.savez_compressed(out/'manufactured_fields.npz', **arrays)
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        raise
    report['elapsed_seconds'] = time.perf_counter()-started
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--potential-origin', nargs=3, type=float, default=(0., 0., 0.))
    parser.add_argument('--amplitude', type=float, default=.1)
    parser.add_argument('--max-half-panels', type=int, default=2200)
    args = parser.parse_args(argv)
    result = run(args.recovery_run, args.out, origin=args.potential_origin,
                 amplitude=args.amplitude, max_half_panels=args.max_half_panels)
    print(json.dumps({key: result[key] for key in ('status', 'half_faces',
                     'physical_accuracy_accepted', 'elapsed_seconds')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
