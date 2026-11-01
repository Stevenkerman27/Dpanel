"""Item C part 2b: quadrature-order error as absolute |dv| / U (U=|U_inf|=1.0046),
against an order-192 reference, same frozen G2 state, core 0.01.
Absolute scale avoids the cancellation-inflated relative metric of part 2.
"""

import os
from pathlib import Path

ROOT = Path(os.environ.get("PANEL_ROOT", Path.cwd()))
import json, time
import numpy as np
from tools.issue84.wake_node_rejection import build_case
from panel.p1_lifting_g2 import assemble_multi, solve_multi
from panel.p1_coupled_material_wake import TotalVelocity

mesh, alpha = build_case([str(ROOT / "runs/half_cad_wingbody_v1/surface.json")])
ops = assemble_multi(mesh); state = solve_multi(mesh, alpha, kutta="pressure", operators=ops)
snap = np.load(str(ROOT / "runs/issue84/g2_ref/iteration_000.npz"))
state["mu"] = snap["mu"]; state["gamma"] = snap["gamma"]
core = 0.01
field = TotalVelocity(mesh, state, method="rm_regularized_p1", core_radius=core)
U = float(np.linalg.norm(field.free))
g = [gg for gg in mesh.groups if gg.name == "main_wing"][0]
V = np.asarray(g.wake_xyz, float); F = np.asarray(g.wake_faces, int)
tri = V[F]; centers = tri.mean(axis=1)
nrm = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]); nrm /= np.linalg.norm(nrm, axis=1)[:, None]
face_idx = np.linspace(0, len(F)-1, 12).astype(int)
ratios = (0.3, 1.0, 3.0, 10.0, 30.0)
orders = (8, 16, 32, 48, 96)
ref_order = 192
out = {"core_radius": core, "U_inf": U, "ref_order": ref_order, "metric": "max|dv|/U over 12 points",
       "orders": list(orders), "bands": {}}
for r in ratios:
    pts = np.asarray([centers[f] + r*core*nrm[f] for f in face_idx])
    t = time.time(); ref = field.evaluate_many(pts, order=ref_order); tr = time.time()-t
    row = {}
    for o in orders:
        t = time.time(); v = field.evaluate_many(pts, order=o); dt = time.time()-t
        err = np.linalg.norm(v-ref, axis=1)/U
        row[str(o)] = {"max": float(err.max()), "median": float(np.median(err)), "seconds": dt}
    out["bands"]["r/core=%.1f" % r] = row
    print("r/core %5.1f | o8 %.2e o16 %.2e o32 %.2e o48 %.2e o96 %.2e | ref192 %.1fs"
          % (r, row["8"]["max"], row["16"]["max"], row["32"]["max"], row["48"]["max"],
             row["96"]["max"], tr))
json.dump(out, open(str(ROOT / "runs/issue84/rm_order_bands.json"), "w"), indent=2)
print("done")
