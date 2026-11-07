"""Cross-evaluate P1 sheet and ribbon velocities on one saved wake geometry."""
import argparse
import json
import os
import platform
from pathlib import Path
import sys
import time

import numpy as np
import torch

from panel.p1_filament_free_wake import wake_velocity
from panel.p1_lifting_g1 import make_wing
from panel.p1_lifting_g2 import combine_wings, solve_multi
from panel.p1_steady_free_wake import _potentials_at, mean_wake_velocity, with_wake_xyz
from panel.shed_groups import _segment_triangle_t


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--snapshot", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--chord-panels", type=int, default=10)
parser.add_argument("--span-panels", type=int, default=12)
parser.add_argument("--wake-panels", type=int, default=8)
parser.add_argument("--length", type=float, default=10.)
parser.add_argument("--cores", type=float, nargs="+", default=[.025, .05, .1])
parser.add_argument("--subdivisions", type=int, nargs="+", default=[2, 4, 8])
parser.add_argument("--probes", type=float, nargs="+", default=[.055])
parser.add_argument("--audit-stencils", action="store_true")
parser.add_argument("--compare-unguarded", action="store_true",
                    help="also record the original unchecked probe for diagnosis")
parser.add_argument("--order", type=int, default=3)
parser.add_argument("--blend", type=float, default=1.,
                    help="interpolate snapshot geometry from initial wake and recompute wall")
args = parser.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)
if not 0 <= args.blend <= 1:
    raise ValueError("blend must be between zero and one")
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
wing = make_wing(chord_panels=args.chord_panels, span_panels=args.span_panels,
                 wake_panels=args.wake_panels, wake_length=args.length)
mesh = combine_wings([wing], reference_chord=1., reference_area=4.,
                     reference_point=np.array([.25, 0., 0.]))
with np.load(args.snapshot) as data:
    xyz = data["xyz"] if "xyz" in data else data["xyz_0"]
    faces = data["faces"] if "faces" in data else data["faces_0"]
    if not np.array_equal(faces, mesh.groups[0].wake_faces):
        raise ValueError("snapshot topology does not match mesh")
    if "gamma_hat" in data and not np.array_equal(data["gamma_hat"], mesh.groups[0].gamma_hat):
        raise ValueError("snapshot circulation basis does not match mesh")
    xyz=mesh.groups[0].wake_xyz+args.blend*(xyz-mesh.groups[0].wake_xyz)
    mesh = with_wake_xyz(mesh, {mesh.groups[0].name: xyz})
    if "mu" in data and args.blend==1.:
        state = dict(alpha_deg=4., mu=data["mu"], gamma=data["gamma"])
        state_source = "snapshot"
    else:
        state = solve_multi(mesh, 4., kutta="pressure", max_nfev=80,
                            wall_formulation="galerkin", galerkin_order=args.order)
        state_source = "recomputed Galerkin wall and pressure Kutta"
        gamma_difference = float(np.max(np.abs(state["gamma"]-data["gamma"]))) if "gamma" in data else None
group = mesh.groups[0]
tri = group.wake_xyz[group.wake_faces]
normal = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
area = np.linalg.norm(normal, axis=1)/2
normal /= np.linalg.norm(normal, axis=1)[:, None]
points = tri.mean(axis=1)
sheets = []
for probe in args.probes:
    start = time.perf_counter()
    all_values, all_summary = mean_wake_velocity(mesh, state, probe_fraction=probe,
                                step_fraction=probe*.009/.055)
    values = all_values[group.name]
    slip = np.einsum("ij,ij->i", values, normal)
    record = dict(probe_fraction=probe, seconds=time.perf_counter()-start,
                       slip_rms=float(np.sqrt(np.mean(slip**2))),
                       slip_max=float(np.max(np.abs(slip))),
                       slip_p95=float(np.percentile(np.abs(slip), 95)),
                       max_slip_face=int(np.argmax(np.abs(slip))),
                       probe_adjusted_faces=all_summary[group.name]["probe_adjusted_faces"],
                       probe_max_halvings=all_summary[group.name]["probe_max_halvings"])
    if args.compare_unguarded:
        length=np.sqrt(area)
        h=probe*length
        step=probe*.009/.055*length
        centers=points[:,None,None,:]+np.array([-1.,1.])[None,:,None,None]*h[:,None,None,None]*normal[:,None,None,:]
        stencil=centers[:,:,:,None,:]+np.array([-1.,1.])[None,None,None,:,None]*step[:,None,None,None,None]*np.eye(3)[None,None,:,None,:]
        potential=_potentials_at(mesh,state,stencil.reshape(-1,3)).reshape(len(points),2,3,2)
        unguarded=((potential[:,:,:,1]-potential[:,:,:,0])/(2*step[:,None,None])).mean(axis=1)
        unguarded_slip=np.einsum("ij,ij->i",unguarded,normal)
        record["unguarded_slip_rms"]=float(np.sqrt(np.mean(unguarded_slip**2)))
        record["unguarded_slip_max"]=float(np.max(np.abs(unguarded_slip)))
    if args.audit_stencils:
        low = tri.min(axis=1)
        high = tri.max(axis=1)
        hits = []
        for face in range(len(tri)):
            for side in (-1, 1):
                center = points[face]+side*probe*np.sqrt(area[face])*normal[face]
                for axis in range(3):
                    step = np.eye(3)[axis]*probe*.009/.055*np.sqrt(area[face])
                    a = center-step
                    b = center+step
                    candidates = np.flatnonzero(np.all(low <= np.maximum(a, b)+1e-9, axis=1)
                                                & np.all(high >= np.minimum(a, b)-1e-9, axis=1))
                    for other in candidates:
                        if other == face:
                            continue
                        hit = _segment_triangle_t(a, b, tri[other], 1e-9)
                        if hit is not None and 1e-9 < hit < 1-1e-9:
                            hits.append([int(face), int(other), side, axis])
        record["requested_stencil_crossings"] = hits
    sheets.append(record)
    if probe == args.probes[0]:
        sheet = values
        sheet_slip = slip
rows = []
for core in args.cores:
    for subdivisions in args.subdivisions:
        start = time.perf_counter()
        velocity = wake_velocity(mesh, state, points, core, subdivisions)
        seconds = time.perf_counter()-start
        slip = np.einsum("ij,ij->i", velocity, normal)
        delta = velocity-sheet
        rows.append(dict(core=core, subdivisions=subdivisions, seconds=seconds,
                         slip_rms=float(np.sqrt(np.mean(slip**2))),
                         slip_max=float(np.max(np.abs(slip))),
                         slip_difference_rms=float(np.sqrt(np.mean((slip-sheet_slip)**2))),
                         velocity_difference_rms=float(np.sqrt(np.mean(np.sum(delta**2, axis=1))))))
report = dict(command=[sys.executable, *sys.argv], platform=platform.platform(),
              python=sys.version, numpy=np.__version__, torch=torch.__version__,
              threads={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
              state_source=state_source, geometry_blend=args.blend, targets=len(points),
              recomputed_gamma_max_difference=gamma_difference if state_source.startswith("recomputed") else None,
              sheet=sheets,
              ribbons=rows,
              note="Same geometry, wall state, face centers and normals; ribbon differences use the first listed sheet probe. Differing near-sheet models, one cold timing per evaluator. Not an accuracy or full-solver benchmark.")
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(dict(state_source=state_source, sheet=report["sheet"], ribbons=rows)), flush=True)
