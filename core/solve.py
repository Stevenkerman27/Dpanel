"""Low-speed constant-source prototype; exterior traces and physical mass constraint."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
from numba import set_num_threads
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve, solve
from scipy.linalg.lapack import dgecon
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from threadpoolctl import threadpool_limits

from triangle_kernel import influence

parser = argparse.ArgumentParser()
parser.add_argument("mesh", type=Path)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--flow", type=float, nargs=3, default=[1., 0., 0.])
parser.add_argument("--rotate-flow", action="store_true")
parser.add_argument("--alpha", type=float, help="Override flow by (cos(alpha),0,sin(alpha))")
parser.add_argument("--wake-length", type=float, default=50., help="Fixed wake length in Cref units")
parser.add_argument("--mass", choices=["constrain", "report"], default="constrain")
parser.add_argument("--off-surface", action="store_true", help="Independent points within each flat face")
parser.add_argument("--ncpu", type=int, default=8)
args = parser.parse_args()
set_num_threads(args.ncpu)
threadpool_limits(args.ncpu)
warnings.simplefilter("error", LinAlgWarning)
started = time.perf_counter()
args.out.mkdir(parents=True, exist_ok=True)
data = np.load(args.mesh, allow_pickle=False)
xyz = data["xyz"].copy()
tri = data["tri"]
tags = data["tags"]
assert xyz.ndim == 2 and xyz.shape[1] == 3 and np.isfinite(xyz).all()
assert tri.ndim == 2 and tri.shape[1] == 3 and np.issubdtype(tri.dtype, np.integer)
assert tri.min() >= 0 and tri.max() < len(xyz)
assert len(np.unique(np.sort(tri, axis=1), axis=0)) == len(tri), "Duplicate triangles"
origin = (xyz.max(axis=0) + xyz.min(axis=0)) / 2.
length = float(np.max(np.ptp(xyz, axis=0)) / 2.)
assert length > 0.
xyz = (xyz-origin) / length
panels = np.ascontiguousarray(xyz[tri])
centers = panels.mean(axis=1)
area_vectors = np.cross(panels[:,1]-panels[:,0], panels[:,2]-panels[:,0]) / 2.
area = np.linalg.norm(area_vectors, axis=1)
assert np.all(area > 0.), "Degenerate triangles"
normals = area_vectors / area[:,None]
count = len(tri)
edges = np.vstack([tri[:,[0,1]], tri[:,[1,2]], tri[:,[2,0]]])
sorted_edges = np.sort(edges, axis=1)
unique_edges, inverse, counts = np.unique(sorted_edges, axis=0, return_inverse=True, return_counts=True)
assert np.all(counts == 2), "Mesh must be a closed two-manifold"
order = np.argsort(inverse).reshape(-1, 2)
assert np.all(edges[order[:,0]] == edges[order[:,1],::-1]), "Inconsistent face orientation"
face_ids = np.tile(np.arange(count), 3)
adjacency = coo_matrix((np.ones(len(order)), (face_ids[order[:,0]], face_ids[order[:,1]])), shape=(count,count))
components, labels = connected_components(adjacency, directed=False)
volume = np.bincount(labels, weights=np.einsum("ij,ij->i", centers, area_vectors)/3.)
assert np.all(volume > 0.), "Every connected component must be outward-oriented"
flow = np.asarray(args.flow, dtype=float)
if args.alpha is not None:
    alpha = np.deg2rad(args.alpha)
    flow = np.array([np.cos(alpha), 0., np.sin(alpha)])
if args.rotate_flow:
    flow = data["rotation"] @ flow
flow /= np.linalg.norm(flow)
assert np.isfinite(flow).all()
sref = float(data["sref"]) / length**2 if "sref" in data else np.pi * (float(data["scale"])/length)**2
trace = np.arange(count, dtype=np.int64)
print(f"Assembling {count} triangles, {components} closed components", flush=True)
potential, velocity_kernel = influence(centers, panels, normals, trace, 0)
del potential
assert np.isfinite(velocity_kernel).all(), "Singular kernel evaluation"
matrix = np.einsum("ijk,ik->ij", velocity_kernel, normals)
rhs = -normals @ flow
wake_velocity = np.empty((count,0,3))
wake_edges = data["wake_edges"].copy() if "wake_edges" in data else np.empty((0,2),dtype=np.int64)
wake_count = len(wake_edges)
if wake_count:
    assert args.wake_length > 0. and abs(flow[1]) < 1e-12
    # This experiment covers straight spanwise trailing edges only.
    wake_edges = np.sort(wake_edges, axis=1)
    assert len(np.unique(wake_edges,axis=0)) == wake_count
    reverse = xyz[wake_edges[:,0],1] > xyz[wake_edges[:,1],1]
    wake_edges[reverse] = wake_edges[reverse,::-1]
    delta = xyz[wake_edges[:,1]]-xyz[wake_edges[:,0]]
    assert np.max(abs(delta[:,[0,2]])) < 1e-10 and np.all(delta[:,1] > 0.)
    tangent_span = delta/np.linalg.norm(delta,axis=1)[:,None]
    lift_axis = np.cross(flow, [0.,1.,0.])
    upper, lower = [], []
    edge_lookup = {tuple(edge):index for index,edge in enumerate(unique_edges)}
    for edge in wake_edges:
        pair = order[edge_lookup[tuple(sorted(edge))]]
        faces = face_ids[pair]
        lift_normals = normals[faces] @ lift_axis
        assert lift_normals.min() < 0. < lift_normals.max(), "TE must join upper and lower faces"
        upper.append(faces[np.argmax(lift_normals)])
        lower.append(faces[np.argmin(lift_normals)])
    upper, lower = np.array(upper), np.array(lower)
    tangent_upper = np.cross(tangent_span, normals[upper])
    tangent_lower = np.cross(tangent_span, normals[lower])
    tangent_upper *= np.sign(tangent_upper@flow)[:,None]
    tangent_lower *= np.sign(tangent_lower@flow)[:,None]
    assert np.all(np.linalg.norm(tangent_upper,axis=1) > .99)
    assert np.all(np.linalg.norm(tangent_lower,axis=1) > .99)
    near0, near1 = xyz[wake_edges[:,0]], xyz[wake_edges[:,1]]
    extrusion = args.wake_length*float(data["cref"])/length*flow
    far0, far1 = near0+extrusion, near1+extrusion
    wake_panels = np.stack([np.stack([near0,far0,far1],axis=1),
                            np.stack([near0,far1,near1],axis=1)],axis=1).reshape(-1,3,3)
    wake_normals = np.cross(wake_panels[:,1]-wake_panels[:,0],wake_panels[:,2]-wake_panels[:,0])
    wake_normals /= np.linalg.norm(wake_normals,axis=1)[:,None]
    p, wk = influence(centers,wake_panels,wake_normals,np.full(count,-1,dtype=np.int64),1)
    wake_velocity = wk.reshape(count,wake_count,2,3).sum(axis=2)
    del p, wk
    assert np.isfinite(wake_velocity).all()
    rhs = np.column_stack([rhs, -np.einsum("ijk,ik->ij",wake_velocity,normals)])
matrix_norm = np.linalg.norm(matrix, ord=1)
lu, piv = lu_factor(matrix)
rcond, info = dgecon(lu, matrix_norm)
assert info == 0 and np.isfinite(rcond) and rcond > 0.
sigma = lu_solve((lu,piv), rhs)
unconstrained_sigma = sigma.copy()
constraint = np.zeros((components, count))
for component in range(components):
    mask = labels == component
    constraint[component,mask] = area[mask] / area[mask].sum()
unconstrained_flux = constraint @ sigma
if args.mass == "constrain":
    # Weighted constrained least squares via LU solves, without forming A.T @ A.
    dual = lu_solve((lu,piv), constraint.T, trans=1)
    correction = lu_solve((lu,piv), dual / (area/area.mean())[:,None])
    sigma -= correction @ solve(constraint @ correction, constraint @ sigma)
gamma = np.empty(0)
kutta_max, kutta_condition = None, None
if wake_count:
    kutta_body = (np.einsum("ijk,ik->ij",velocity_kernel[upper],tangent_upper)
                  -np.einsum("ijk,ik->ij",velocity_kernel[lower],tangent_lower))
    kutta_wake = (np.einsum("ijk,ik->ij",wake_velocity[upper],tangent_upper)
                  -np.einsum("ijk,ik->ij",wake_velocity[lower],tangent_lower))
    kutta_rhs = -(tangent_upper-tangent_lower)@flow
    kutta_matrix = kutta_wake + kutta_body@sigma[:,1:]
    singular = np.linalg.svd(kutta_matrix,compute_uv=False)
    assert singular[-1] > np.finfo(float).eps*wake_count*singular[0], "Rank-deficient Kutta system"
    kutta_condition = float(singular[0]/singular[-1])
    gamma = solve(kutta_matrix,kutta_rhs-kutta_body@sigma[:,0])
    sigma = sigma[:,0]+sigma[:,1:]@gamma
    unconstrained_sigma = unconstrained_sigma[:,0]+unconstrained_sigma[:,1:]@gamma
    unconstrained_flux = constraint @ unconstrained_sigma
velocity = flow + np.einsum("ijk,j->ik", velocity_kernel, sigma) + np.einsum("ijk,j->ik",wake_velocity,gamma)
if wake_count:
    kutta_residual = (np.einsum("ij,ij->i",velocity[upper],tangent_upper)
                      -np.einsum("ij,ij->i",velocity[lower],tangent_lower))
    kutta_max = float(np.max(abs(kutta_residual)))
del velocity_kernel
residual = np.einsum("ij,ij->i", velocity, normals)
cp = 1. - np.sum(velocity**2, axis=1)
panel_force = -cp[:,None] * area_vectors / sref
force = panel_force.sum(axis=0)
component_force = {str(int(tag)): panel_force[tags == tag].sum(axis=0).tolist() for tag in np.unique(tags)}
off_rms, off_max = None, None
if args.off_surface:
    off_sum, off_peak = 0., 0.
    for bary in ([.6,.2,.2], [.2,.6,.2], [.2,.2,.6]):
        points = np.einsum("ijk,j->ik", panels, bary)
        for start in range(0, count, 128):
            stop = min(start+128, count)
            p, kernel = influence(points[start:stop], panels, normals, trace[start:stop], 0)
            v = flow + np.einsum("ijk,j->ik", kernel, sigma)
            if wake_count:
                wp, wk = influence(points[start:stop],wake_panels,wake_normals,
                                   np.full(stop-start,-1,dtype=np.int64),1)
                v += np.einsum("ijk,j->ik",wk,np.repeat(gamma,2))
            r = np.einsum("ij,ij->i", v, normals[start:stop])
            off_sum += float(np.sum(area[start:stop]*r*r))
            off_peak = max(off_peak, float(np.max(np.abs(r))))
    off_rms = np.sqrt(off_sum/(3.*area.sum()))
    off_max = off_peak
    del p, kernel
result = dict(
    mesh=str(args.mesh.resolve()), mesh_sha256=hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
    source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                   for name in ["solve.py", "triangle_kernel.py"]},
    method="constant triangular source, exterior Neumann trace; fixed doublet wake with linear tangential Kutta",
    mass_mode=args.mass, ncpu=args.ncpu, triangles=count, closed_components=int(components),
    coordinate_origin=origin.tolist(), coordinate_length=length, flow=flow.tolist(),
    density=1., dynamic_pressure=.5, speed=1., sref_physical=sref*length**2,
    rcond_1_estimate=float(rcond), condition_1_estimate=float(1./rcond),
    unconstrained_normalized_source_flux=unconstrained_flux.tolist(),
    normalized_source_flux=(constraint@sigma).tolist(),
    signed_source_integral_physical=np.bincount(labels, weights=area*sigma*length**2).tolist(),
    collocation_rms=float(np.sqrt(np.sum(area*residual**2)/area.sum())),
    collocation_max=float(np.max(np.abs(residual))), off_collocation_rms=off_rms,
    off_collocation_max=off_max, cp_min=float(cp.min()), cp_max=float(cp.max()),
    pressure_force_coefficient=force.tolist(), component_pressure_force_coefficient=component_force,
    cp_definition="1-|V_total|^2; no normal removal, smoothing or clipping",
    force_definition="-sum(Cp * outward_normal * area)/Sref, all components use same panel force",
    elapsed_seconds=time.perf_counter()-started,
    wake_strips=wake_count, wake_length_cref=args.wake_length if wake_count else None,
    kutta_max=kutta_max, kutta_condition_2=kutta_condition,
    pressure_CL=float(force@np.cross(flow,[0.,1.,0.])), pressure_CD=float(force@flow),
    wake_CL=float(2.*np.sum(gamma*delta[:,1])/sref) if wake_count else None,
    boundary_gates=dict(mass_flux=bool(np.max(abs(constraint@sigma)) <= 1e-12),
                        centroid_rms=bool(np.sqrt(np.average(residual**2,weights=area)) <= 1e-6),
                        centroid_max=bool(np.max(abs(residual)) <= 1e-5)),
    validation_scope="Closed-body reference validation is separate; lifting formulation is experimental and unaccepted",
)
np.savez_compressed(args.out/"surface.npz", centers=centers*length+origin, normals=normals,
                    area=area*length**2, sigma=sigma, unconstrained_sigma=unconstrained_sigma,
                    velocity=velocity, cp=cp, panel_force_coefficient=panel_force, tags=tags,
                    normal_residual=residual, flow=flow, gamma=gamma, wake_edges=wake_edges)
(args.out/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps(result, indent=2), flush=True)
