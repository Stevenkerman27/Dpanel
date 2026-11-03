import json, sys, numpy as np
from panel.p1_coupled_material_wake import GeometryGates, geometry_audit, declared_contact_pairs
from panel.p1_steady_free_wake import prepare_wake_material
from tools.issue84.wake_node_rejection import build_case

SURFACE = ["D:/3D/Projects/panel/runs/cases/wingbody/half_cad_wingbody_v1/surface.json"]
mesh, alpha = build_case(SURFACE, case="g2")
materials = {g.name: prepare_wake_material(g) for g in mesh.groups}
declared = declared_contact_pairs([g.wake_faces for g in mesh.groups], mesh.faces,
                                  group_labels=[g.name for g in mesh.groups])
candidates = {g.name: np.array(g.wake_xyz, float) for g in mesh.groups}
audit = geometry_audit(mesh, candidates, materials, GeometryGates(), declared=declared)
c = audit['detail']['contacts']
print('reason', audit['reason'], 'declared_pairs', len(declared), 'kinds', c['kinds'],
      'pairs_tested', c['pairs_tested'], 'viol', c['violations_count'],
      'declared', c['declared'], 'undeclared', c['undeclared'], 'worst', c['worst_depth'])
for v in c['violations'][:8]:
    print(json.dumps(v, sort_keys=True))
    g = [x for x in mesh.groups if x.name == v['group']][0]
    w = np.asarray(g.wake_xyz, float)[g.wake_faces[v['wake_face']]]
    t = np.asarray(mesh.xyz, float)[mesh.faces[v['wall_face']]]
    print('  wake', np.round(w, 6).tolist())
    print('  wall', np.round(t, 6).tolist())
