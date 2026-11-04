"""T4: how far does the audit update get once declared trailing-edge contact is excluded?"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import panel.p1_coupled_material_wake as m
from tools.issue84.wake_node_rejection import build_case

import sys

MIN_SHARED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/issue84/t4_declared_sequence.json")
DECLARED = set()
calls = []
rows = []


def _key(v):
    return (round(float(v[0]), 9), round(float(v[1]), 9), round(float(v[2]), 9))


def declared_pairs(mesh):
    """(group, wake triangle, wall facet, shared vertex count) on the reference geometry."""
    wall = mesh.xyz[mesh.faces]
    wall_keys = [{_key(v) for v in tri} for tri in wall]
    pairs = {}
    for group in mesh.groups:
        for k, face in enumerate(group.wake_faces):
            coords = {_key(v) for v in group.wake_xyz[face]}
            for j, keys in enumerate(wall_keys):
                shared = len(coords & keys)
                if shared >= MIN_SHARED:
                    pairs[(group.name, int(k), int(j))] = shared
    return pairs


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
            meta.append((group.name, int(k), old[group.wake_faces[k]].mean(axis=0)))
    wall = mesh.xyz[mesh.faces]
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
                                  wall_triangle=int(j), centroid=[float(v) for v in meta[i][2]],
                                  skipped_declared=int(skipped)))
                return 'wall'
    calls.append(dict(reason=None, skipped_declared=int(skipped)))
    return None


def recorder(iteration, current, state, record, arrays):
    update = record.get('outgoing_update') or {}
    rows.append(dict(iteration=int(iteration),
                     CL=float(state['CL']), CL_gamma=float(state['CL_gamma']),
                     CD_pressure=float(state['CD_pressure']), CM_y=float(state['CM_y']),
                     pressure_kutta_rms=float(state.get('pressure_kutta_rms', np.nan)),
                     wall_rms=float(state.get('wall_rms', np.nan)),
                     map_residual_chord=record.get('map_residual_chord'),
                     accepted=update.get('accepted'), relaxation=update.get('relaxation'),
                     moved=update.get('moved_nodes'),
                     displacement_rms=update.get('displacement_rms'),
                     trial_reasons=update.get('trial_reasons'),
                     wakes={k: dict(root=v['eligibility']['root'],
                                    eligible=v['eligibility']['eligible'],
                                    rejected=v['eligibility']['rejected'],
                                    convergence=v['eligibility']['convergence'])
                            for k, v in record['wakes'].items()}))


mesh, alpha = build_case(["D:/3D/Projects/panel/runs/cases/wingbody/half_cad_wingbody_v1/surface.json"],
                         case="g2", alpha_deg=5.0)
pairs = declared_pairs(mesh)
DECLARED.update(pairs)
edge_pairs = sum(1 for v in pairs.values() if v >= 2)
vertex_pairs = sum(1 for v in pairs.values() if v == 1)
print('declared pairs', len(pairs), 'edge', edge_pairs, 'vertex', vertex_pairs)
original = m.geometry_failure
m.geometry_failure = geometry_failure
result = m.solve_coupled_material_wake(mesh, alpha, kutta='signed_speed', iterations=6,
                                       max_backtracks=12, gate_mode='audit',
                                       save_callback=recorder)
m.geometry_failure = original
payload = dict(schema='dpanel.issue84.t4_declared_sequence.v1', min_shared=MIN_SHARED, case='g2', alpha_deg=5.0,
               kutta='signed_speed', gate_mode='audit', max_backtracks=12, iterations=6,
               status=result['status'], seconds=float(result['seconds']),
               declared_pairs=dict(total=len(pairs), edge=edge_pairs, vertex=vertex_pairs),
               rows=rows, trials=calls, partial=False)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print('status', result['status'], 'seconds', round(result['seconds'], 1))
for row in rows:
    print(row['iteration'], 'CL', round(row['CL'], 6), 'CDp', round(row['CD_pressure'], 6),
          'acc', row['accepted'], 'relax', row['relaxation'], 'moved', row['moved'],
          'map', None if row['map_residual_chord'] is None else round(row['map_residual_chord'], 3))
print('wrote', OUT)
