"""Wide-h ladder probe: is the rejected node error 'h too coarse' or an edge singularity?"""
import argparse, csv, json, sys
import numpy as np

from tools.issue84.wake_node_rejection import build_case
from panel.p1_lifting_g2 import assemble_multi, solve_multi
from panel.p1_coupled_material_wake import TotalVelocity

ap = argparse.ArgumentParser()
ap.add_argument('--main-half-surface', required=True)
ap.add_argument('--csv', required=True)
ap.add_argument('--nodes', type=int, nargs='+', required=True)
ap.add_argument('--group', default='main_wing')
ap.add_argument('--order', type=int, default=48)
ap.add_argument('--eps-fraction', type=float, default=.05)
ap.add_argument('--out', default='runs/issue84/ladder_probe.json')
a = ap.parse_args()

rows = {int(r['node']): r for r in csv.DictReader(open(a.csv))
        if r['group'] == a.group}
mesh, alpha = build_case([a.main_half_surface])
ops = assemble_multi(mesh)
state = solve_multi(mesh, alpha, kutta='pressure', operators=ops)
field = TotalVelocity(mesh, state, method='analytic_edges')
group = [g for g in mesh.groups if g.name == a.group][0]
uref = float(np.linalg.norm(field.free))
factors = [8., 4., 2., 1., .5, .25, .125, .0625, .03125]

out = {}
for node in a.nodes:
    r = rows[node]
    p = np.array([float(r['xyz_x']), float(r['xyz_y']), float(r['xyz_z'])])
    n = np.array([float(r['nx']), float(r['ny']), float(r['nz'])])
    scale = float(r['scale'])
    means, hs = [], []
    for f in factors:
        h = a.eps_fraction*scale*f
        up, um = field(p+h*n, order=a.order), field(p-h*n, order=a.order)
        means.append((up+um)/2)
        hs.append(h)
    means = np.asarray(means)
    incs = [float(np.linalg.norm(means[i]-means[i-1])/uref) for i in range(1, len(means))]
    out[node] = dict(
        recorded_epsilon_error=float(r['epsilon_error']),
        eligible=r['eligible'], reasons=r['reasons'], xyz=p.tolist(),
        boundary=r['on_sheet_boundary'], near_tip=r['near_tip'], scale=scale,
        aspect_max=float(r['local_aspect_max']), speed=float(r['speed']),
        offsets=[float(v) for v in hs], means=[v.tolist() for v in means],
        increments=incs, speed_at_each=[float(np.linalg.norm(v)) for v in means])
json.dump(out, open(a.out, 'w'), indent=2)
for node, d in out.items():
    print('node', node, 'rec %.4f' % d['recorded_epsilon_error'], 'boundary', d['boundary'],
          'tip', d['near_tip'], 'scale %.4g' % d['scale'], 'aspect %.1f' % d['aspect_max'])
    print('  h      ', ' '.join('%.3e' % h for h in d['offsets']))
    print('  |V|    ', ' '.join('%.6f' % v for v in d['speed_at_each']))
    print('  incr   ', ' '.join('%.2e' % v for v in d['increments']))
