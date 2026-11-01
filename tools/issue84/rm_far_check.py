"""Item C part 3b: far-field agreement between the regularized (core 0.01) and the
unregularized analytic_edges field, on points that are far from every sheet panel
(so neither the core nor the near-field quadrature matters).
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
rm = TotalVelocity(mesh, state, method="rm_regularized_p1", core_radius=core)
an = TotalVelocity(mesh, state, method="analytic_edges")
U = float(np.linalg.norm(an.free))
g = [gg for gg in mesh.groups if gg.name == "main_wing"][0]
V = np.asarray(g.wake_xyz, float); F = np.asarray(g.wake_faces, int)
centers = V[F].mean(axis=1)
rng = np.random.default_rng(11)
out = {"core_radius": core, "U_inf": U, "note": "points with min distance > D from every wake panel centroid",
       "bands": {}}
for D in (0.5, 1.0, 3.0):
    pts = []
    while len(pts) < 12:
        p = rng.uniform([-6, -4, -3], [7, 4, 3])
        if np.min(np.linalg.norm(centers-p, axis=1)) > D:
            pts.append(p)
    pts = np.asarray(pts)
    t = time.time(); va = an.evaluate_many(pts, order=48); ta = time.time()-t
    row = {"min_distance": D, "analytic_seconds": ta, "n": len(pts)}
    for o in (48, 96):
        t = time.time(); vr = rm.evaluate_many(pts, order=o); dt = time.time()-t
        d = np.linalg.norm(vr-va, axis=1)/U
        row[str(o)] = {"max": float(d.max()), "median": float(np.median(d)), "seconds": dt}
    out["bands"]["d>%.1f" % D] = row
    print("d>%.1f | rm48 %.2e  rm96 %.2e  (U=%.4f, an %.2fs, rm48 %.1fs)"
          % (D, row["48"]["max"], row["96"]["max"], U, ta, row["48"]["seconds"]))
json.dump(out, open(str(ROOT / "runs/issue84/rm_far_check.json"), "w"), indent=2)
