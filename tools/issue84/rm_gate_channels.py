"""Item C part 4: does the PRODUCTION gate accept the RM path at order 48?

limits() (panel/p1_coupled_material_wake.py:151-214) reports two channels:
  epsilon_error    - change of the +-h mean along the epsilon ladder
  quadrature_error - order vs 2*order difference at the smallest h
For analytic_edges the second channel is 0 (NOT APPLICABLE). With
rm_regularized_p1 it becomes active, so order 48 is only certified if that
channel stays below quadrature_tol. This script evaluates both channels on a
node sample taken from the analytic_edges diagnosis (rejected vs eligible).
"""

import os
from pathlib import Path

ROOT = Path(os.environ.get("PANEL_ROOT", Path.cwd()))
import json, time
import numpy as np
from tools.issue84.wake_node_rejection import (build_case, triangle_data,
                                               prepare_wake_material, material_paths)
from panel.p1_lifting_g2 import assemble_multi, solve_multi
from panel.p1_coupled_material_wake import TotalVelocity, limits

REJ = [56, 546, 626, 407, 452, 422, 543, 544, 545, 542, 759, 3]
OK = [59, 3, 12, 21, 33, 40, 61, 77, 91, 104, 118, 131]

mesh, alpha = build_case([str(ROOT / "runs/half_cad_wingbody_v1/surface.json")])
ops = assemble_multi(mesh); state = solve_multi(mesh, alpha, kutta="pressure", operators=ops)
snap = np.load(str(ROOT / "runs/issue84/g2_ref/iteration_000.npz"))
state["mu"] = snap["mu"]; state["gamma"] = snap["gamma"]

g = [gg for gg in mesh.groups if gg.name == "main_wing"][0]
xyz = np.asarray(g.wake_xyz, float)
tri, normal, area, _shape, edges = triangle_data(g.wake_xyz, g.wake_faces)
mat = prepare_wake_material(g)
normals = np.zeros_like(xyz); scales = np.full(len(xyz), np.inf)
for k in range(3):
    np.add.at(normals, g.wake_faces[:, k], normal*area[:, None])
    np.minimum.at(scales, g.wake_faces[:, k], edges.min(axis=1))
normals /= np.linalg.norm(normals, axis=1)[:, None]
used = set(int(v) for v in mat.used)

core = 0.01
rm = TotalVelocity(mesh, state, method="rm_regularized_p1", core_radius=core)
out = {"core_radius": core, "epsilon_tol": 0.02, "quadrature_tol": 0.002, "rows": []}
for label, ids in (("analytic_rejected", REJ), ("analytic_eligible", OK)):
    ids = [i for i in dict.fromkeys(ids) if i in used and np.isfinite(scales[i])]
    pts, nrm, sc = xyz[ids], normals[ids], scales[ids]
    for order in (16, 48, 96):
        t = time.time()
        s = limits(rm, pts, nrm, sc, eps_fraction=0.05, ladder=(1., .5, .25, .125), order=order)
        eps = np.asarray(s["epsilon_error"]); q = np.asarray(s["quadrature_error"])
        row = {"class": label, "n": len(ids), "order": order, "node_ids": ids,
               "eps_max": float(eps.max()), "quad_max": float(q.max()),
               "eps_fail": int((eps > 0.02).sum()), "quad_fail": int((q > 0.002).sum()),
               "epsilon_error": [float(x) for x in eps],
               "quadrature_error": [float(x) for x in q],
               "seconds": time.time()-t}
        out["rows"].append(row)
        print("%-18s order %3d | eps max %.3e fail %2d/%d | quad max %.3e fail %2d/%d | %.1fs"
              % (label, order, row["eps_max"], row["eps_fail"], len(ids),
                 row["quad_max"], row["quad_fail"], len(ids), row["seconds"]), flush=True)
json.dump(out, open(str(ROOT / "runs/issue84/rm_gate_channels.json"), "w"), indent=2)