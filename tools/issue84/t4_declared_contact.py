"""T4: does excluding declared trailing-edge contact unblock the audit geometry update?

The production wall gate sweeps (old U new) triangles and rejects any overlap with
the wall. A wake triangle that shares an edge with a wall facet is a declared
attachment, not a collision, yet its swept hull always contains that contact.
This script re-runs the G2 case with such pairs excluded and records every
rejected trial, including which wake/wall pair fired.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import panel.p1_coupled_material_wake as m
from tools.issue84.wake_node_rejection import build_case

OUT = Path("docs/issue84/t4_declared_contact.json")
calls = []
DECLARED: set = set()


def declared_pairs(mesh):
    """Wake triangle/wall facet pairs that share a vertex in the reference geometry."""
    pairs = set()
    wall = mesh.xyz[mesh.faces]
    keys = []
    for group in mesh.groups:
        for k in range(len(group.wake_faces)):
            keys.append((group.name, k, {_key(v) for v in group.wake_xyz[group.wake_faces[k]]}))
    for name, k, coords in keys:
        for j, tri in enumerate(wall):
            if coords & {_key(v) for v in tri}:
                pairs.add((name, k, int(j)))
    return pairs


def _key(v):
    return (round(float(v[0]), 9), round(float(v[1]), 9), round(float(v[2]), 9))


def geometry_failure(mesh, candidates, materials, gates):
    tol = gates.tolerance_fraction * mesh.reference_chord
    swept, meta = [], []
    for group in mesh.groups:
        old, new = group.wake_xyz, candidates[group.name]
        mat = materials[group.name]
        if not np.isfinite(new).all():
            return 'nonfinite'
        if not np.array_equal(new[mat.roots], mat.reference_xyz[mat.roots]):
            return 'root'
        _s, root_ids = m.material_paths(mat)
        free = np.setdiff1d(mat.used, mat.roots)
        ref = mat.reference_xyz[free] - mat.reference_xyz[root_ids[free]]
        if np.any(np.einsum('ij,ij->i', new[free] - mat.reference_xyz[root_ids[free]], ref) <= 0):
            return 'attachment'
        tri0, n0, a0, _g0, e0 = m.triangle_data(old, group.wake_faces)
        try:
            tri1, n1, a1, _g1, _e1 = m.triangle_data(new, group.wake_faces)
        except ValueError:
            return 'area'
        scale = np.full(len(old), np.inf)
        for k in range(3):
            np.minimum.at(scale, group.wake_faces[:, k], e0.min(axis=1))
        if np.any(np.linalg.norm(new[mat.used] - old[mat.used], axis=1)
                  > gates.max_step_fraction * scale[mat.used]):
            return 'step'
        ratio = np.r_[a1 / a0, a1 / mat.panel_area]
        if np.any((ratio < 1 / gates.max_area_ratio) | (ratio > gates.max_area_ratio)):
            return 'area'
        if np.any(np.einsum('fi,fi->f', n0, n1) < np.cos(np.deg2rad(gates.max_fold_deg))):
            return 'fold'
        shrink = 1e-8
        t0 = (1 - shrink) * tri0 + shrink * tri0.mean(axis=1)[:, None]
        t1 = (1 - shrink) * tri1 + shrink * tri1.mean(axis=1)[:, None]
        for k in range(len(t0)):
            swept.append(np.concatenate((t0[k], t1[k]), axis=0))
            meta.append((group.name, int(k), {_key(v) for v in old[group.wake_faces[k]]}))
    wall = mesh.xyz[mesh.faces]
    wall_keys = [{_key(v) for v in tri} for tri in wall]
    pairs = skipped = 0
    for i, hull in enumerate(swept):
        for j, tri in enumerate(wall):
            if not m._bounds_overlap(hull, tri, tol):
                continue
            if (meta[i][0], meta[i][1], j) in DECLARED:
                skipped += 1
                continue
            pairs += 1
            if pairs > gates.max_collision_pairs:
                return 'intersection_budget'
            if m.convex_overlap(hull, tri, tol):
                calls.append(dict(reason='wall', group=meta[i][0], wake_triangle=meta[i][1],
                                  wall_triangle=int(j), skipped_declared=skipped))
                return 'wall'
    calls.append(dict(reason=None, skipped_declared=skipped))
    return None


def recorder(iteration, current, state, record, arrays):
    update = record.get('outgoing_update') or {}
    rows.append(dict(iteration=int(iteration),
                     CL=float(state['CL']), CL_gamma=float(state['CL_gamma']),
                     CD_pressure=float(state['CD_pressure']), CM_y=float(state['CM_y']),
                     map_residual_chord=record.get('map_residual_chord'),
                     accepted=update.get('accepted'), relaxation=update.get('relaxation'),
                     moved=update.get('moved_nodes'),
                     displacement_rms=update.get('displacement_rms'),
                     trial_reasons=update.get('trial_reasons'),
                     counters=record.get('wakes') and
                     {k: dict(root=v['eligibility']['root'], eligible=v['eligibility']['eligible'],
                              rejected=v['eligibility']['rejected'],
                              convergence=v['eligibility']['convergence'])
                      for k, v in record['wakes'].items()}))


rows = []
mesh, alpha = build_case(["D:/3D/Projects/panel/runs/cases/wingbody/half_cad_wingbody_v1/surface.json"],
                         case="g2", alpha_deg=5.0)
DECLARED.update(declared_pairs(mesh))
print("declared contact pairs", len(DECLARED))
original = m.geometry_failure
m.geometry_failure = geometry_failure
result = m.solve_coupled_material_wake(mesh, alpha, kutta='signed_speed', iterations=2,
                                       max_backtracks=12, gate_mode='audit',
                                       save_callback=recorder)
m.geometry_failure = original
payload = dict(schema='dpanel.issue84.t4_declared_contact.v1', case='g2', alpha_deg=5.0,
               kutta='signed_speed', gate_mode='audit', max_backtracks=12,
               status=result['status'], seconds=float(result['seconds']),
               declared_pairs=len(DECLARED),
               rows=rows, trials=calls, partial=False)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print('status', result['status'], 'seconds', round(result['seconds'], 1))
for row in rows:
    print(row['iteration'], 'CL', round(row['CL'], 6), 'CDp', round(row['CD_pressure'], 6),
          'acc', row['accepted'], 'relax', row['relaxation'], 'moved', row['moved'],
          'rms', None if row['map_residual_chord'] is None else round(row['map_residual_chord'], 3))
print('trials', [c['reason'] for c in calls])
for c in calls:
    if c['reason'] == 'wall':
        print('wall hit', c['group'], c['wake_triangle'], 'vs wall', c['wall_triangle'],
              'skipped', c['skipped_declared'])
print('wrote', OUT)
