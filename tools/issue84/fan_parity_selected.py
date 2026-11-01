"""Parity and reference stability for the selected projection-fan configs (issue84 A5)."""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panel.p1_coupled_material_wake import TotalVelocity  # noqa: E402
from panel.p1_lifting_g2 import assemble_multi, solve_multi  # noqa: E402
from tools.issue84.projection_gate_set import load_points  # noqa: E402
from tools.issue84.wake_node_rejection import build_case  # noqa: E402

ROOT = Path(r'D:\3D\Projects\panel')
SURFACE = ROOT / 'runs' / 'half_cad_wingbody_v1' / 'surface.json'
CSVS = [('quad', ROOT / 'runs' / 'issue84' / 'wake_rejection_g2_rm48.csv'),
        ('eps', ROOT / 'runs' / 'issue84' / 'wake_rejection_g2_analytic.csv')]

mesh, alpha = build_case([SURFACE], case='g2', alpha_deg=5.0)
ops = assemble_multi(mesh)
state = solve_multi(mesh, alpha, kutta='pressure', operators=ops)

points = []
for label, path in CSVS:
    for row in load_points(path):
        points.append(np.asarray(row['xyz'], float))
points = np.asarray(points)
print('points', len(points), flush=True)


def field(config, ref=None):
    order, strips = config
    if ref is not None:
        order, strips = ref
        return TotalVelocity(mesh, state, method='rm_regularized_p1', core_radius=0.01,
                             area_quadrature='projection_fan', fan_order=order, fan_strips=strips)
    return TotalVelocity(mesh, state, method='rm_regularized_p1', core_radius=0.01,
                         area_quadrature='projection_fan', fan_order=order, fan_strips=strips)


report = {'points': len(points)}
for config in ((16, 8), (24, 8), (32, 8)):
    name = '%d/%d' % config
    fan = field(config)
    started = time.perf_counter()
    batch = np.asarray(fan.evaluate_many(points), float)
    batch_seconds = time.perf_counter() - started
    scalar = np.stack([np.asarray(fan(point), float) for point in points])
    chunk1 = np.asarray(fan._regularized.evaluate_many(points, chunk_size=1), float)
    report[name] = {
        'batch_seconds': batch_seconds,
        'batch_vs_scalar_max_abs': float(np.max(np.abs(batch - scalar))),
        'chunk1_max_abs': float(np.max(np.abs(chunk1 - batch))),
    }
    print(name, json.dumps(report[name]), flush=True)

samples = [0, 7, 14, 29, 36, 51, 74, 108]
stability = []
for index in samples:
    point = points[index]
    ref16 = np.asarray(field((48, 16))(point), float)
    ref64 = np.asarray(field((48, 64))(point), float)
    stability.append({'index': int(index), 'rel': float(np.linalg.norm(ref16 - ref64) / np.linalg.norm(ref64)),
                      'norm': float(np.linalg.norm(ref64))})
report['reference_stability'] = {'reference': '48/16', 'finer': '48/64', 'rows': stability,
                                 'rel_max': max(row['rel'] for row in stability)}
print('stability', json.dumps(report['reference_stability']), flush=True)

out = Path(r'D:\3D\Projects\panel\runs\fix_master\docs\issue84\fan_parity_selected.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
print('wrote', out)
