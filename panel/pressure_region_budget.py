"""Read-only regional load differences from PR29 frozen recovery artifacts.

No JAX, solve, remesh, pressure clipping, or replacement load is performed.
Each physical-radius partition contains ALL panels, including a separate
sphere-crossing bucket. A pointwise peak is not an integral convergence test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

REFLECT = np.array([1., -1., 1.])
AXIAL_REFLECT = -REFLECT
QUANTITIES = ('CD_pressure', 'CL_pressure', 'CM_y')
STATE_KEYS = ('mu_half', 'gamma_half', 'sigma_half', 'Cp_raw_half', 'neighbor_ids')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for key in STATE_KEYS:
        a = np.ascontiguousarray(state[key])
        digest.update(key.encode())
        digest.update(str((a.dtype.str, a.shape)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def geometry(fixture):
    x = np.asarray(fixture['coordinates'], dtype=float)
    f = np.asarray(fixture['faces'])
    if (x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all()
            or f.ndim != 2 or f.shape[1] != 3 or len(f) == 0
            or not np.issubdtype(f.dtype, np.integer)
            or f.min() < 0 or f.max() >= len(x)):
        raise ValueError('Invalid finite coordinates / integer wall faces')
    tri = x[f]
    av = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])/2
    area = np.linalg.norm(av, axis=1)
    if np.any(area <= 0) or not np.isfinite(area).all():
        raise ValueError('Degenerate wall triangle')
    if np.any(x[f, 1] < 0):
        raise ValueError('Require an already validated positive-y half wall')
    return x, f, tri, av, area, av/area[:, None]


def face_loads(fixture, cp):
    """Each row includes its exact mirror's force and axial moment."""
    _, f, tri, av, _, _ = geometry(fixture)
    cp = np.asarray(cp, dtype=float)
    u = np.asarray(fixture['freestream'], dtype=float)
    p = np.asarray(fixture['moment_reference'], dtype=float)
    rho, sref, cref = (float(fixture[k]) for k in
                       ('density', 'reference_area', 'reference_chord'))
    if (cp.shape != (len(f),) or not np.isfinite(cp).all()
            or u.shape != (3,) or p.shape != (3,)
            or not np.isfinite(u).all() or not np.isfinite(p).all()
            or not np.isfinite([rho, sref, cref]).all()
            or min(rho, sref, cref) <= 0 or np.linalg.norm(u) <= 0
            or abs(u[1]) > 1e-12*np.linalg.norm(u) or abs(p[1]) > 1e-12*cref):
        raise ValueError('Need finite Cp, positive SI references, and symmetric flow/origin')
    q = .5*rho*np.dot(u, u)
    force = -q*cp[:, None]*av
    moment = np.cross(tri.mean(axis=1)-p, force)
    force = force + force*REFLECT
    moment = moment + moment*AXIAL_REFLECT
    drag = u/np.linalg.norm(u)
    lift = np.cross(drag, [0., 1., 0.])
    values = np.column_stack((force@drag/(q*sref), force@lift/(q*sref),
                              moment[:, 1]/(q*sref*cref)))
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite load arithmetic')
    return values, force, moment


def triangle_distance_bounds(triangles, point):
    """Exact minimum and maximum Euclidean distances on flat triangles.

    Minimum: interior perpendicular projection or one of three edge segments.
    Maximum: a vertex. This detects sphere crossings even if ALL vertices lie
    outside the sphere; centroid/vertex-only distance tests cannot do that.
    """
    tri = np.asarray(triangles, dtype=float)
    p = np.asarray(point, dtype=float)
    if (tri.ndim != 3 or tri.shape[1:] != (3, 3) or p.shape != (3,)
            or not np.isfinite(tri).all() or not np.isfinite(p).all()):
        raise ValueError('Finite triangles and point required')
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    e, v = b-a, c-a
    cross = np.cross(e, v)
    norm2 = np.sum(cross*cross, axis=1)
    if np.any(norm2 <= 0):
        raise ValueError('Degenerate distance triangle')
    relative = p-a
    # Barycentric coordinates of the orthogonal projection, using cross
    # products rather than subtraction of nearly equal Gram determinants.
    beta = np.sum(np.cross(relative, v)*cross, axis=1)/norm2
    gamma = np.sum(np.cross(e, relative)*cross, axis=1)/norm2
    inside = (beta >= 0) & (gamma >= 0) & (beta+gamma <= 1)
    plane2 = np.sum(relative*cross, axis=1)**2/norm2
    minimum2 = np.where(inside, plane2, np.inf)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        segment = tri[:, j]-tri[:, i]
        fraction = np.sum((p-tri[:, i])*segment, axis=1)/np.sum(segment**2, axis=1)
        closest = tri[:, i]+np.clip(fraction, 0, 1)[:, None]*segment
        minimum2 = np.minimum(minimum2, np.sum((closest-p)**2, axis=1))
    maximum2 = np.max(np.sum((tri-p)**2, axis=2), axis=1)
    return np.sqrt(minimum2), np.sqrt(maximum2)


def fixed_radius_masks(minimum, maximum, radius):
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Positive radius required')
    inside = maximum <= radius
    outside = minimum > radius
    return {'inside': inside, 'crossing': ~(inside | outside), 'outside': outside}


def normal_spread(fixture, patches):
    """Geometry-only indicators, NOT proof of a singularity or a failed fit."""
    _, f, _, _, area, normal = geometry(fixture)
    nf = len(f)
    edges = {}
    for i, face in enumerate(f):
        for a, b in zip(face, np.roll(face, -1)):
            edges.setdefault(tuple(sorted((int(a), int(b)))), []).append(i)
    edge_angle = np.zeros(nf)
    for faces in edges.values():
        if len(faces) > 2:
            raise ValueError('Nonmanifold edge in geometry audit')
        if len(faces) == 2:
            i, j = faces
            angle = np.degrees(np.arccos(np.clip(normal[i]@normal[j], -1, 1)))
            edge_angle[[i, j]] = np.maximum(edge_angle[[i, j]], angle)
    # PR29 orders virtual faces as right, then its exact reflection.
    normals = np.vstack((normal, normal*REFLECT))
    areas = np.r_[area, area]
    patch_angle = np.zeros(nf)
    for patch in patches:
        fan = np.asarray(patch['fan_faces'])
        support = np.asarray(patch['support_faces'])
        for ids in (fan, support):
            if (ids.ndim != 1 or not len(ids) or not np.issubdtype(ids.dtype, np.integer)
                    or ids.min() < 0 or ids.max() >= 2*nf):
                raise ValueError('Invalid full-mirror patch face indices')
        n = np.sum(normals[fan]*areas[fan, None], axis=0)
        if np.linalg.norm(n) <= 1e-12*areas[fan].sum():
            raise ValueError('Ambiguous patch reference normal')
        n /= np.linalg.norm(n)
        angle = float(np.degrees(np.arccos(np.clip(normals[support]@n, -1, 1))).max())
        np.maximum.at(patch_angle, fan % nf, angle)
    return edge_angle, patch_angle


def summarize(mask, area, cp0, cp1, values0, values1, force0, force1, moment0,
              moment1, edge_angle, patch_angle, global_abs_delta):
    count = int(mask.sum())
    selected_area = float(area[mask].sum())
    dcp = cp1-cp0
    delta = values1-values0
    result = dict(half_faces=count, full_mirrored_area=2*selected_area,
                  area_fraction=selected_area/float(area.sum()),
                  cp_delta_area_rms=float(np.sqrt(np.average(dcp[mask]**2,
                                             weights=area[mask]))) if count else None,
                  cp_delta_abs_max=float(np.abs(dcp[mask]).max()) if count else None,
                  adjacent_normal_angle_max_deg=float(edge_angle[mask].max()) if count else None,
                  patch_normal_spread_max_deg=float(patch_angle[mask].max()) if count else None,
                  original_force_N=force0[mask].sum(axis=0).tolist(),
                  candidate_force_N=force1[mask].sum(axis=0).tolist(),
                  delta_force_N=(force1-force0)[mask].sum(axis=0).tolist(),
                  original_moment_Nm=moment0[mask].sum(axis=0).tolist(),
                  candidate_moment_Nm=moment1[mask].sum(axis=0).tolist(),
                  delta_moment_Nm=(moment1-moment0)[mask].sum(axis=0).tolist(),
                  quantities={})
    for k, name in enumerate(QUANTITIES):
        dv = delta[mask, k]
        result['quantities'][name] = dict(
            original=float(values0[mask, k].sum()), candidate=float(values1[mask, k].sum()),
            delta=float(dv.sum()), positive_delta=float(dv[dv > 0].sum()),
            negative_delta=float(dv[dv < 0].sum()), sum_abs_delta=float(np.abs(dv).sum()),
            original_sum_abs=float(np.abs(values0[mask, k]).sum()),
            candidate_sum_abs=float(np.abs(values1[mask, k]).sum()),
            share_of_global_abs_delta=(float(np.abs(dv).sum()/global_abs_delta[k])
                                       if global_abs_delta[k] else None))
    return result


def budget(fixture, original_cp, candidate_cp, patches, point, radii):
    _, f, tri, _, area, _ = geometry(fixture)
    radii = np.asarray(radii, dtype=float)
    if (radii.ndim != 1 or not len(radii) or not np.isfinite(radii).all()
            or np.any(radii <= 0) or np.any(np.diff(radii) <= 0)):
        raise ValueError('Strictly increasing positive r/Cref required')
    cp0, cp1 = np.asarray(original_cp, dtype=float), np.asarray(candidate_cp, dtype=float)
    v0, f0, m0 = face_loads(fixture, cp0)
    v1, f1, m1 = face_loads(fixture, cp1)
    lo, hi = triangle_distance_bounds(tri, point)
    edge_angle, patch_angle = normal_spread(fixture, patches)
    absolute = np.abs(v1-v0).sum(axis=0)
    tags = np.asarray(fixture['component_ids'])
    if tags.shape != (len(f),) or not np.issubdtype(tags.dtype, np.integer):
        raise ValueError('Need numeric component IDs, not guessed names')
    def summary(mask):
        return summarize(mask, area, cp0, cp1, v0, v1, f0, f1, m0, m1,
                         edge_angle, patch_angle, absolute)
    all_faces = np.ones(len(f), dtype=bool)
    whole = summary(all_faces)
    partitions, mask_rows = [], []
    for r in radii:
        masks = fixed_radius_masks(lo, hi, r*float(fixture['reference_chord']))
        if not np.all(sum(mask.astype(int) for mask in masks.values()) == 1):
            raise AssertionError('Partition must contain every face exactly once')
        parts = {name: summary(mask) for name, mask in masks.items()}
        for name, mask in masks.items():
            parts[name]['components'] = {str(int(t)):summary(mask & (tags == t))
                                         for t in np.unique(tags)}
        closure = {}
        for name in QUANTITIES:
            closure[name] = {}
            for key in ('original', 'candidate', 'delta', 'sum_abs_delta'):
                subtotal = sum(p['quantities'][name][key] for p in parts.values())
                total = whole['quantities'][name][key]
                if not np.isclose(subtotal, total, rtol=1e-12, atol=1e-13):
                    raise AssertionError('Regional loads fail whole-load closure')
                closure[name][key] = abs(subtotal-total)
        vector_closure = {}
        for key in ('original_force_N', 'candidate_force_N', 'delta_force_N',
                    'original_moment_Nm', 'candidate_moment_Nm', 'delta_moment_Nm'):
            subtotal = np.sum([p[key] for p in parts.values()], axis=0)
            total = np.asarray(whole[key])
            # Componentwise vector closure, not only a small projected residual.
            scale = max(float(np.max(np.abs(total))), 1.)
            if not np.allclose(subtotal, total, rtol=1e-12, atol=1e-13*scale):
                raise AssertionError('Regional force/moment closure failed')
            vector_closure[key] = float(np.max(np.abs(subtotal-total)))
        partitions.append(dict(radius_over_chord=float(r), regions=parts,
                               coefficient_closure_absolute=closure,
                               force_moment_closure_absolute=vector_closure))
        mask_rows.append(np.where(masks['inside'], 0, np.where(masks['crossing'], 1, 2)))
    report = dict(schema='dpanel.pressure_region_budget.v1',
                  status='read_only_partition_not_physical_validation',
                  physical_accuracy_accepted=False, feature_point=np.asarray(point).tolist(),
                  reference_chord=float(fixture['reference_chord']),
                  classification='exact triangle/sphere bounds; crossing faces kept separate',
                  whole=whole, partitions=partitions,
                  components={str(int(t)):summary(tags == t) for t in np.unique(tags)},
                  scope='No panel excluded from full loads. No singularity or convergence claim.')
    arrays = dict(original_cp=cp0, candidate_cp=cp1, delta_cp=cp1-cp0,
                  area_half=area, centers=tri.mean(axis=1), component_ids=tags,
                  original_coefficients=v0, candidate_coefficients=v1, delta_coefficients=v1-v0,
                  min_distance=lo, max_distance=hi,
                  adjacent_normal_angle_deg=edge_angle, patch_normal_spread_deg=patch_angle,
                  radius_over_chord=radii, region_ids=np.asarray(mask_rows, dtype=np.int8))
    return report, arrays


def run(recovery_dir, out_dir, point, radii):
    root, target = Path(recovery_dir), Path(out_dir)
    if target.exists():
        raise FileExistsError('Use a new output directory')
    names = ('surface_input.json', 'frozen_state.npz', 'recovered_fields.npz',
             'patches.json', 'report.json')
    hashes = {n: sha(root/n) for n in names}
    fixture = json.loads((root/'surface_input.json').read_text(encoding='utf-8'))
    previous = json.loads((root/'report.json').read_text(encoding='utf-8'))
    if (fixture.get('schema') != 'dpanel.half_planar_cut.v1'
            or previous.get('status') != 'postprocessing_comparison_not_physical_validation'
            or previous.get('source_sha256') != hashes['surface_input.json']):
        raise ValueError('Need a successful PR29 run with a matching surface SHA')
    with np.load(root/'frozen_state.npz', allow_pickle=False) as saved:
        state = {k: saved[k] for k in saved.files}
    with np.load(root/'recovered_fields.npz', allow_pickle=False) as saved:
        cp0, cp1 = saved['original_cp'].copy(), saved['candidate_cp'].copy()
    if (str(state['surface_sha256']) != hashes['surface_input.json']
            or state_hash(state) != previous['frozen_state_hash']
            or not np.array_equal(cp0, state['Cp_raw_half'])):
        raise ValueError('Frozen state identity mismatch')
    patches = json.loads((root/'patches.json').read_text(encoding='utf-8'))
    report, arrays = budget(fixture, cp0, cp1, patches, point, radii)
    # Independent force projection must reproduce both already-saved totals.
    for key, method in (('original', 'original'), ('candidate', 'cut_vertex_affine_candidate')):
        for q in QUANTITIES:
            actual = report['whole']['quantities'][q][key]
            expected = previous['loads'][method][q]
            if not np.isclose(actual, expected, rtol=1e-11, atol=1e-12):
                raise ValueError(f'Original report closure fails: {key}/{q}')
    if any(sha(root/n) != hashes[n] for n in names):
        raise ValueError('Input changed during audit')
    report.update(input_sha256=hashes, frozen_state_hash=previous['frozen_state_hash'],
                  code_sha256=sha(__file__), Python=platform.python_version(),
                  NumPy=np.__version__, selected_point_is_user_declared=True,
                  input_hash_note='Recorded at audit time; not a retroactive signature of PR29 files')
    target.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(target/'regional_fields.npz', **arrays)
    report['regional_fields_sha256'] = sha(target/'regional_fields.npz')
    (target/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n',
                                     encoding='utf-8')
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recovery-run', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--feature-point', nargs=3, type=float, required=True,
                   help='Fixed CAD feature xyz in the SAME coordinate units as surface_input')
    p.add_argument('--radii-over-chord', default='0.025,0.05,0.1,0.2')
    args = p.parse_args(argv)
    result = run(args.recovery_run, args.out, args.feature_point,
                 [float(v) for v in args.radii_over_chord.split(',')])
    print(json.dumps({'status':result['status'], 'whole':result['whole']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
