
import numpy as np
from tools.issue84.wake_node_rejection import build_case
from panel.p1_geometry_contact import declared_contact_pairs
mesh, alpha = build_case(['D:/3D/Projects/panel/runs/half_cad_wingbody_v1/surface.json'], case='g2', alpha_deg=5.0)
g = mesh.groups[0]
decl = declared_contact_pairs([gg.wake_faces for gg in mesh.groups], mesh.faces,
                              group_labels=[gg.name for gg in mesh.groups])
for wf in (300, 684, 685):
    tri = g.wake_xyz[g.wake_faces[wf]]
    print('wake', wf, np.round(tri, 5).tolist())
for j in (18, 19, 415):
    t = mesh.xyz[mesh.faces[j]]
    n = np.cross(t[1]-t[0], t[2]-t[0]); a = 0.5*np.linalg.norm(n)
    print('wall', j, 'area', round(float(a), 6), 'normal', np.round(n/np.linalg.norm(n), 3).tolist(),
          'verts', np.round(t, 5).tolist())
hits = [(k, v) for k, v in decl.items() if k[1] in (300, 684, 685)]
print('declared entries for those wake faces:')
for k, v in hits:
    print('  wake', k[1], 'wall', k[2], 'kind', v[0], 'shared', v[1])
print('total declared', len(decl))
