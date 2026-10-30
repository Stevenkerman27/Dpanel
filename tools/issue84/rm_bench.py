"""Item C: RM-P1 scalar vs batched evaluation, quadrature-order validity by distance band.

Same frozen G2 state, same point set, same core.  Reports wall clock, peak RSS,
max abs agreement between the scalar loop and the batched path, and the
quadrature-order error of the total velocity vector against an order-96
reference, per distance band.
"""

import os
from pathlib import Path

ROOT = Path(os.environ.get("PANEL_ROOT", Path.cwd()))
import ctypes, json, time
import numpy as np
from ctypes import wintypes

from tools.issue84.wake_node_rejection import build_case
from panel.p1_lifting_g2 import assemble_multi, solve_multi
from panel.p1_coupled_material_wake import TotalVelocity


def peak_rss_mb():
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    info = PMC(); info.cb = ctypes.sizeof(PMC)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    h = k32.GetCurrentProcess()
    psapi.GetProcessMemoryInfo(ctypes.c_void_p(h), ctypes.byref(info), ctypes.sizeof(info))
    return info.PeakWorkingSetSize/1048576.0


t0 = time.time()
mesh, alpha = build_case([str(ROOT / "runs/half_cad_wingbody_v1/surface.json")])
ops = assemble_multi(mesh)
fresh = solve_multi(mesh, alpha, kutta="pressure", operators=ops)
snap = np.load(str(ROOT / "runs/issue84/g2_ref/iteration_000.npz"))
state = dict(fresh)
frozen_ok = (len(snap["mu"]) == len(fresh["mu"]) and len(snap["gamma"]) == len(fresh["gamma"])
             and np.allclose(snap["mu"], fresh["mu"], rtol=0, atol=0)
             and np.allclose(snap["gamma"], fresh["gamma"], rtol=0, atol=0))
if frozen_ok:
    state["mu"] = snap["mu"]; state["gamma"] = snap["gamma"]
print("solved in %.1f s; frozen snapshot bit-identical: %s" % (time.time()-t0, frozen_ok))

core = 0.01
field = TotalVelocity(mesh, state, method="rm_regularized_p1", core_radius=core)
g = [gg for gg in mesh.groups if gg.name == "main_wing"][0]
V = np.asarray(g.wake_xyz, float); F = np.asarray(g.wake_faces, int)
tri = V[F]
centers = tri.mean(axis=1)
normals = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
normals /= np.linalg.norm(normals, axis=1)[:, None]
wall_tri = np.asarray(mesh.xyz, float)[np.asarray(mesh.faces, int)]
wall_c = wall_tri.mean(axis=1)

rng = np.random.default_rng(20261030)
face_idx = np.linspace(0, len(F)-1, 8).astype(int)
near, bands = [], []
for h in (1e-2, 1e-3, 1e-4):
    for f in face_idx:
        near.append(centers[f] + h*normals[f]); bands.append(h)
near = np.asarray(near); bands = np.asarray(bands)

node_idx = np.linspace(0, len(V)-1, 40).astype(int)
sheet = V[node_idx]

far = []
while len(far) < 40:
    p = rng.uniform([-6, -4, -3], [7, 4, 3])
    if np.min(np.linalg.norm(wall_c-p, axis=1)) > .5:
        far.append(p)
far = np.asarray(far)

points = np.vstack([near, sheet, far])
labels = (["near"]*len(near) + ["sheet"]*len(sheet) + ["far"]*len(far))

res = {"core_radius": core, "faces": int(len(field.triangles)),
       "points": int(len(points)), "frozen_snapshot_bit_identical": bool(frozen_ok),
       "scalar_loop": {}, "batched": {}, "order_sweep": {}, "agreement": {}}

res["peak_after_solve_mb"] = peak_rss_mb()
t = time.time(); bat = field.evaluate_many(points, order=48); t_b48 = time.time()-t
res["peak_after_batch_mb"] = peak_rss_mb()
t = time.time(); sca = np.stack([field(p, order=48) for p in points]); t_s48 = time.time()-t
res["peak_after_scalar_mb"] = peak_rss_mb()
t = time.time(); ref = field.evaluate_many(points, order=96); t_b96 = time.time()-t
res["peak_after_order96_mb"] = peak_rss_mb()
print("peak RSS: solve %.1f | +batch48 %.1f | +scalar48 %.1f | +order96 %.1f MB"
      % (res["peak_after_solve_mb"], res["peak_after_batch_mb"],
         res["peak_after_scalar_mb"], res["peak_after_order96_mb"]))
res["batched"] = dict(order=48, seconds=t_b48, ms_per_point=1000*t_b48/len(points))
res["scalar_loop"] = dict(order=48, seconds=t_s48, ms_per_point=1000*t_s48/len(points))
res["batched"]["order96_seconds"] = t_b96
res["agreement"] = dict(max_abs=float(np.max(np.abs(sca-bat))),
                        rel_l2=float(np.linalg.norm(sca-bat)/np.linalg.norm(bat)),
                        speedup=t_s48/t_b48)
print("batch order48 %.2f s (%.1f ms/pt) | scalar %.2f s (%.1f ms/pt) | speedup %.2fx | max|diff| %.3e"
      % (t_b48, 1000*t_b48/len(points), t_s48, 1000*t_s48/len(points), t_s48/t_b48,
         np.max(np.abs(sca-bat))))
print("order96 reference %.2f s" % t_b96)

sweep = {}
for order in (8, 16, 32, 48):
    t = time.time(); v = field.evaluate_many(points, order=order); dt = time.time()-t
    err = np.linalg.norm(v-ref, axis=1)
    scale = np.linalg.norm(ref, axis=1)
    sweep[order] = dict(seconds=dt, ms_per_point=1000*dt/len(points),
                        max_rel=float(np.max(err/scale)),
                        rel_l2=float(np.linalg.norm(v-ref)/np.linalg.norm(ref)))
    for name in ("near", "sheet", "far"):
        m = np.asarray(labels) == name
        sweep[order][name+"_max_rel"] = float(np.max(err[m]/scale[m]))
    print("order %2d: %.2f s | max rel %.3e | near %.3e sheet %.3e far %.3e"
          % (order, dt, sweep[order]["max_rel"], sweep[order]["near_max_rel"],
             sweep[order]["sheet_max_rel"], sweep[order]["far_max_rel"]))
res["order_sweep"] = sweep
res["peak_rss_mb"] = peak_rss_mb()

# per-band order validity: which order meets 1e-3 / 1e-4 relative error
labels_arr = np.asarray(labels)
band_of_point = np.full(len(points), np.nan)
near_mask = labels_arr == "near"
band_of_point[near_mask] = bands
for band in (1e-2, 1e-3, 1e-4):
    m = near_mask & (band_of_point == band)
    v48 = bat[m]; v96 = ref[m]
    rel = np.max(np.linalg.norm(v48-v96, axis=1)/np.linalg.norm(v96, axis=1))
    res.setdefault("near_band_order_check", {})["h=%.0e" % band] = float(rel)
    print("band h=%.0e: order48 vs order96 max rel %.3e" % (band, rel))
json.dump(res, open(str(ROOT / "runs/issue84/rm_bench.json"), "w"), indent=2)
print("peak RSS %.1f MB, total %.1f s" % (res["peak_rss_mb"], time.time()-t0))
