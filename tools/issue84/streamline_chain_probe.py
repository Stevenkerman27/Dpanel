from __future__ import annotations
import numpy as np
import panel.p1_coupled_material_wake as m
from tools.issue84.wake_node_rejection import build_case

mesh, alpha = build_case(["D:/3D/Projects/panel/runs/cases/wingbody/half_cad_wingbody_v1/surface.json"], case="g2", alpha_deg=5.0)
g = mesh.groups[0]
from panel.p1_steady_free_wake import prepare_wake_material
mat = prepare_wake_material(g)
ref = np.asarray(mat.reference_xyz, float)
used = np.asarray(mat.used, int)
print("ptp ref x/y/z", np.ptp(ref[used, 0]), np.ptp(ref[used, 1]), np.ptp(ref[used, 2]))
print("used", len(used), "roots", len(mat.roots), "faces", g.wake_faces.shape)
ds = np.array([np.linalg.norm(ref[n]-ref[mat.predecessor[n]]) if mat.predecessor[n] >= 0 else 0.
               for n in used])
print("ds pct", np.percentile(ds, [0, 50, 90, 99, 100]), "count>1", int((ds > 1).sum()))
for n in (744, 379, 759):
    if n < len(ref):
        p = int(mat.predecessor[n])
        print(n, "pred", p, "ref[n]", np.round(ref[n], 4), "ref[p]", np.round(ref[p], 4),
              "ds", round(float(ds[list(used).index(n)]) if n in used else -1, 4))
edges = np.column_stack([np.linalg.norm(ref[g.wake_faces[:,0]]-ref[g.wake_faces[:,1]], axis=1),
                         np.linalg.norm(ref[g.wake_faces[:,1]]-ref[g.wake_faces[:,2]], axis=1),
                         np.linalg.norm(ref[g.wake_faces[:,0]]-ref[g.wake_faces[:,2]], axis=1)])
print("edge len pct", np.percentile(edges, [0, 50, 90, 100]))
