"""T5-D: new wall-contact gate and per-node step control A/B on the G2 CAD case.

Conditions:
  legacy_gate_global     boolean SAT wall gate without declared contact, global backtracking (pre-T5)
  new_gate_global        signed contact gate, global backtracking only
  new_gate_per_node      signed contact gate, per-node step caps (production path)
  legacy_gate_per_node   boolean SAT gate, per-node step caps
"""
from __future__ import annotations

import json
import argparse
import time
from pathlib import Path

import numpy as np

import panel.p1_coupled_material_wake as m
from tools.issue84.wake_node_rejection import build_case

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("condition", nargs="?", default="new_gate_per_node")
parser.add_argument("iterations", nargs="?", type=int, default=6)
parser.add_argument("out", nargs="?", type=Path,
                    default=Path("runs/studies/wake_material_graph/g2_t5/t5_gate_ab.json"))
parser.add_argument("step_fraction", nargs="?", type=float, default=.25)
parser.add_argument("--surface", type=Path,
                    default=ROOT / "runs/cases/wingbody/half_cad_wingbody_v1/surface.json")
args = parser.parse_args()
SURFACE = [args.surface]
ITERATIONS, OUT = args.iterations, args.out
CONDITION, STEP_FRACTION = args.condition, args.step_fraction


def legacy_audit(mesh, candidates, materials, gates, *, declared=None, step_scale=None):
    """Pre-T5 gate: boolean SAT, no declared contact, identical order of checks."""
    tol = gates.tolerance_fraction*mesh.reference_chord
    detail = dict(contacts=dict(tested=0, declared=0, undeclared=0, pairs_tested=0,
                                kinds={}, violations=[], violations_count=0,
                                worst_depth=0.), groups={}, legacy_wall_gate=True)
    swept, vertex_sets = [], []
    for gi, group in enumerate(mesh.groups):
        old, new = group.wake_xyz, np.asarray(candidates[group.name], float)
        mat = materials[group.name]
        if not np.isfinite(new).all():
            return dict(reason='nonfinite', detail=detail)
        if not np.array_equal(new[mat.roots], mat.reference_xyz[mat.roots]):
            return dict(reason='root', detail=detail)
        _s, root_ids = m.material_paths(mat)
        free = np.setdiff1d(mat.used, mat.roots)
        ref = mat.reference_xyz[free]-mat.reference_xyz[root_ids[free]]
        if np.any(np.einsum('ij,ij->i', new[free]-mat.reference_xyz[root_ids[free]], ref) <= 0):
            return dict(reason='attachment', detail=detail)
        tri0, n0, a0, _g0, e0 = m.triangle_data(old, group.wake_faces)
        try:
            tri1, n1, a1, _g1, _e1 = m.triangle_data(new, group.wake_faces)
        except ValueError:
            return dict(reason='area', detail=detail)
        scale = np.full(len(old), np.inf)
        for k in range(3):
            np.minimum.at(scale, group.wake_faces[:, k], e0.min(axis=1))
        over = (np.linalg.norm(new[mat.used]-old[mat.used], axis=1)
                > gates.max_step_fraction*scale[mat.used])
        if np.any(over):
            detail['groups'][group.name] = dict(step_nodes=[int(v) for v in mat.used[over]][:20])
            return dict(reason='step', detail=detail)
        ratio = np.r_[a1/a0, a1/mat.panel_area]
        if np.any((ratio < 1/gates.max_area_ratio) | (ratio > gates.max_area_ratio)):
            return dict(reason='area', detail=detail)
        if np.any(np.einsum('fi,fi->f', n0, n1) < np.cos(np.deg2rad(gates.max_fold_deg))):
            return dict(reason='fold', detail=detail)
        shrink = 1e-8
        t0 = (1-shrink)*tri0 + shrink*tri0.mean(axis=1)[:, None]
        t1 = (1-shrink)*tri1 + shrink*tri1.mean(axis=1)[:, None]
        swept.extend(np.concatenate((t0, t1), axis=1))
        vertex_sets.extend([{(gi, int(n)) for n in f} for f in group.wake_faces])
    wall = mesh.xyz[mesh.faces]
    pairs = 0
    for i, hull in enumerate(swept):
        for j, tri in enumerate(wall):
            if not m._bounds_overlap(hull, tri, tol):
                continue
            pairs += 1
            if pairs > gates.max_collision_pairs:
                return dict(reason='intersection_budget', detail=detail)
            detail['contacts']['tested'] += 1
            detail['contacts']['undeclared'] += 1
            if m.convex_overlap(hull, tri, tol):
                detail['contacts']['kinds']['sat_overlap'] = 1
                detail['contacts']['violations_count'] += 1
                if len(detail['contacts']['violations']) < 20:
                    detail['contacts']['violations'].append(dict(wake_face=int(i),
                                                                wall_face=int(j),
                                                                kind='sat_overlap'))
                detail['contacts']['pairs_tested'] = pairs
                return dict(reason='wall', detail=detail)
    for i, hull in enumerate(swept):
        for j in range(i):
            if vertex_sets[i] & vertex_sets[j]:
                continue
            if not m._bounds_overlap(hull, swept[j], tol):
                continue
            pairs += 1
            if pairs > gates.max_collision_pairs:
                return dict(reason='intersection_budget', detail=detail)
            if m.convex_overlap(hull, swept[j], tol):
                return dict(reason='intersection', detail=detail)
    detail['contacts']['pairs_tested'] = pairs
    return dict(reason=None, detail=detail)


def identity_clip(old, target, caps, *, roots=None, reference=None, unit='map_arclength'):
    old = np.asarray(old, float)
    new = np.asarray(target, float).copy()
    if roots is not None:
        roots = np.asarray(roots, int)
        new[roots] = np.asarray(reference, float)[roots]
    delta = np.linalg.norm(new-old, axis=1)
    return new, dict(mode="none", unit=unit, clipped=0, clipped_ids=[], clipped_fraction=0.,
                     raw_max=float(delta.max()) if len(delta) else 0.,
                     raw_rms=float(np.sqrt(np.mean(delta**2))) if len(delta) else 0.,
                     accepted_max=float(delta.max()) if len(delta) else 0.,
                     accepted_rms=float(np.sqrt(np.mean(delta**2))) if len(delta) else 0.,
                     cap_min=None)


rows = []


def recorder(iteration, current, state, record, arrays):
    update = record.get('outgoing_update') or {}
    gate = update.get('gate') or {}
    contacts = gate.get('contacts') or {}
    row = dict(iteration=int(iteration), material_graph=record['material_graph'],
               CL=float(state['CL']), CL_gamma=float(state['CL_gamma']),
               CD_pressure=float(state['CD_pressure']), CM_y=float(state['CM_y']),
               pressure_kutta_rms=float(state.get('pressure_kutta_rms', np.nan)),
               wall_rms=float(state.get('wall_rms', np.nan)),
               nonlinear=state.get('nonlinear'), map_residual_chord=record.get('map_residual_chord'),
               kinematic_candidate=record.get('kinematic_candidate'),
               accepted=update.get('accepted'), relaxation=update.get('relaxation'),
               control=update.get('control'), global_backtracks=update.get('global_backtracks'),
               moved=update.get('moved_nodes'), displacement_rms=update.get('displacement_rms'),
               displacement_max=update.get('displacement_max'),
               trial_reasons=update.get('trial_reasons'),
               reason_counts=update.get('reason_counts'),
               clip=update.get('clip'), control_nodes=update.get('control_nodes'),
               step_scale=update.get('step_scale'),
               trial_details=[dict(backtrack=t['backtrack'], reason=t['reason'],
                                   scale=t.get('scale'),
                                   kinds=(t.get('gate') or {}).get('contacts', {}).get('kinds'),
                                   violations_count=(t.get('gate') or {}).get('contacts', {}).get('violations_count'),
                                   worst_depth=(t.get('gate') or {}).get('contacts', {}).get('worst_depth'),
                                   violations=list((t.get('gate') or {}).get('contacts', {}).get('violations') or [])[:5])
                              for t in (update.get('trial_details') or [])],
               contacts=dict(kinds=contacts.get('kinds'), violations_count=contacts.get('violations_count'),
                             worst_depth=contacts.get('worst_depth'),
                             declared=contacts.get('declared'), undeclared=contacts.get('undeclared'),
                             pairs_tested=contacts.get('pairs_tested'),
                             tested=contacts.get('tested'),
                             pre_existing=contacts.get('pre_existing'),
                             pre_existing_pairs=list(contacts.get('pre_existing_pairs') or [])[:5],
                             violations=list(contacts.get('violations') or [])[:10]),
               wakes={k: dict(root=v['eligibility']['root'], eligible=v['eligibility']['eligible'],
                               truncation_nodes=v['eligibility'].get('truncation_nodes'),
                              rejected=v['eligibility']['rejected'],
                              convergence=v['eligibility']['convergence'],
                              advanced_despite_audit=v['eligibility'].get('advanced_despite_audit'))
                      for k, v in record['wakes'].items()})
    rows.append(row)
    print(iteration, 'CL', round(row['CL'], 6), 'CDp', round(row['CD_pressure'], 6),
          'acc', row['accepted'], 'ctrl', row['control'], 'relax', row['relaxation'],
          'moved', row['moved'], 'map', None if row['map_residual_chord'] is None
          else round(row['map_residual_chord'], 3), flush=True)


mesh, alpha = build_case(SURFACE, case="g2", alpha_deg=5.0)
if CONDITION.endswith("_geometric"):
    # Pre-change step unit: cap = max_step_fraction * shortest incident edge.
    _base_scale = m.wake_step_scale

    def geometric_scale(wake_xyz, wake_faces, material, **kw):
        out = _base_scale(wake_xyz, wake_faces, material, **kw)
        out['scale'] = np.asarray(out['geometric'], float).copy()
        out['truncation'] = np.zeros(len(out['scale']), bool)
        return out
    m.wake_step_scale = geometric_scale
if CONDITION.startswith("legacy_gate"):
    m.geometry_audit = legacy_audit
if "per_node" in CONDITION:
    clip = m.clip_step_update
else:
    clip = identity_clip
m.clip_step_update = clip
started = time.perf_counter()
_policy = 'freeze' if 'frozen' in CONDITION else 'advance'
result = m.solve_coupled_material_wake(mesh, alpha, kutta='signed_speed', iterations=ITERATIONS,
                                       max_backtracks=12, gate_mode='audit',
                                       gates=m.GeometryGates(max_step_fraction=STEP_FRACTION),
                                       truncation_policy=_policy, save_callback=recorder)
payload = dict(schema='dpanel.issue84.t5_gate_ab.v1', condition=CONDITION, case='g2',
               alpha_deg=5.0, kutta='signed_speed', gate_mode='audit', max_backtracks=12,
               iterations=ITERATIONS, truncation_policy=_policy, step_fraction=STEP_FRACTION,
               material_graph=rows[0]['material_graph'],
               status=result['status'], seconds=float(result['seconds']),
               kinematic_candidate=result['kinematic_candidate'], rows=rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(payload, indent=2, sort_keys=True)+chr(10), encoding="utf-8")
print('status', result['status'], 'seconds', round(time.perf_counter()-started, 1))
print('wrote', OUT)
