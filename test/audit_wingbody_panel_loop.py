"""Independent NumPy/Numba audit of the saved wing/body panel experiment.

The field velocity uses Dpanel's independent analytic kernel, NOT the velocity
used to construct Cp. Wall errors stay visible and never enter acceptance as
the identically-zero Qn from the Dirichlet surface reconstruction.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
parser.add_argument("--ncpu", type=int, default=8)
args = parser.parse_args()
args.run = args.run.resolve()
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = str(args.ncpu)
os.environ["TMP"] = os.environ["TEMP"] = os.environ["TMPDIR"] = str(args.run / "tmp")
os.environ["NUMBA_CACHE_DIR"] = str(args.run / "numba_cache")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
import numpy as np
from numba import set_num_threads
from threadpoolctl import threadpool_limits
from triangle_kernel import influence

set_num_threads(args.ncpu)
threadpool_limits(args.ncpu)
started = time.perf_counter()
fixture = json.loads((args.run / "surface_input.json").read_text(encoding="utf-8"))
result = json.loads((args.run / "result.json").read_text(encoding="utf-8"))
data = np.load(args.run / "fields.npz", allow_pickle=False)
xyz, faces = data["coordinates"], data["faces"]
triangles = np.ascontiguousarray(xyz[faces])
centers = triangles.mean(axis=1)
area_vector = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]) / 2
areas = np.linalg.norm(area_vector, axis=1)
normals = area_vector / areas[:, None]
assert np.allclose(centers, data["panel_center"][0], atol=1e-13)
assert np.allclose(normals, data["panel_normal"][0], atol=1e-13)
l_axis, m_axis = data["panel_x_dir"][0], data["panel_y_dir"][0]
mu, sigma = data["mu"].reshape(-1), data["sigma"].reshape(-1)
tags = data["component_ids"]
alpha = np.deg2rad(result["parameters"]["alpha_deg"])
flow = np.array([np.cos(alpha), 0., np.sin(alpha)])
edges = np.asarray(fixture["trailing_edges"])
upper, lower = fixture["upper_te_faces"], fixture["lower_te_faces"]
adjacency = [set(row) for row in fixture["neighbors"]]
for i, j in zip(upper, lower):
    adjacency[i].remove(j)
    adjacency[j].remove(i)
stencil_reports = []
recovered = None
for rings in (1, result["pressure_reconstruction"]["rings"]):
    gradient = np.empty((len(faces), 2))
    worst_ratio, bad_count, errors = 1., 0, []
    for i in range(len(faces)):
        reached, frontier = {i}, {i}
        for ring in range(rings):
            frontier = set().union(*(adjacency[j] for j in frontier)) - reached
            reached.update(frontier)
        neighbors = sorted(reached - {i})
        delta = centers[neighbors] - centers[i]
        matrix = np.column_stack((delta @ l_axis[i], delta @ m_axis[i]))
        singular = np.linalg.svd(matrix, compute_uv=False)
        ratio = singular[-1] / singular[0]
        worst_ratio = min(worst_ratio, ratio)
        if ratio < 1e-10:
            bad_count += 1
            gradient[i] = np.nan
            continue
        # QR-based least squares is independent of the CSDL 2x2 formula.
        q, r = np.linalg.qr(matrix, mode="reduced")
        gradient[i] = np.linalg.solve(r, q.T @ (mu[neighbors] - mu[i]))
        affine = matrix @ np.array([0.37, -0.21])
        affine_recovered = np.linalg.solve(r, q.T @ affine)
        errors.append(float(np.max(abs(affine_recovered - [0.37, -0.21]))))
    stencil_reports.append(dict(rings=rings, rank_deficient_panels=bad_count,
                                minimum_singular_value_ratio=float(worst_ratio),
                                affine_reproduction_max_abs=max(errors)))
    if rings == result["pressure_reconstruction"]["rings"]:
        recovered = gradient
assert stencil_reports[-1]["rank_deficient_panels"] == 0
gradient_error = float(np.max(abs(recovered + np.column_stack((data["ql"].ravel(), data["qm"].ravel())))))
assert gradient_error < 1e-9

wake_nodes = data["x_w"][0]
te_nodes = np.unique(edges)
node_index = {int(node): i for i, node in enumerate(te_nodes)}
ns = len(te_nodes)
wake_panels = []
for a, b in edges:
    a, b = node_index[int(a)], node_index[int(b)]
    wake_panels.extend([wake_nodes[[a, a + ns, b + ns]], wake_nodes[[a, b + ns, b]]])
wake_panels = np.ascontiguousarray(wake_panels)
wake_normals = np.cross(wake_panels[:, 1] - wake_panels[:, 0], wake_panels[:, 2] - wake_panels[:, 0])
wake_normals /= np.linalg.norm(wake_normals, axis=1)[:, None]
wake_mu = np.repeat(data["mu_wake"].ravel(), 2)
assert np.all(wake_normals @ np.cross(flow, [0., 1., 0.]) > .99)
assert np.max(abs(data["mu_wake"].ravel() - (mu[upper] - mu[lower]))) < 1e-12

# VortexAD stores the negative physical surface-potential trace:
# phi' = -D_body mu - S sigma - D_wake mu_wake, sigma=-U.n.
trace = np.arange(len(faces), dtype=np.int64)
B, grad_B = influence(centers, triangles, normals, trace, 0)
D_ext, grad_D = influence(centers, triangles, normals, trace, 1)
W, grad_W = influence(centers, wake_panels, wake_normals, np.full(len(faces), -1, dtype=np.int64), 1)
field_velocity = flow - np.einsum("ijk,j->ik", grad_B, sigma)
field_velocity -= np.einsum("ijk,j->ik", grad_D, mu) + np.einsum("ijk,j->ik", grad_W, wake_mu)
normal_error = np.einsum("ij,ij->i", field_velocity, normals)
interior_potential = -(D_ext - np.eye(len(faces))) @ mu - B @ sigma - W @ wake_mu
field_cp = 1 - np.sum(field_velocity ** 2, axis=1)
field_report = dict(
    velocity_definition="U - grad(S)sigma - grad(D)mu - grad(Dwake)mu_wake; exterior self traces",
    wall_normal_rms=float(np.sqrt(np.average(normal_error ** 2, weights=areas))),
    wall_normal_max=float(np.max(abs(normal_error))),
    interior_dirichlet_rms=float(np.sqrt(np.average(interior_potential ** 2, weights=areas))),
    independent_field_cp_min=float(field_cp.min()), independent_field_cp_max=float(field_cp.max()),
    surface_reconstruction_cp_vs_field_rms=float(np.sqrt(np.average((data["Cp_raw"].ravel() - field_cp) ** 2, weights=areas))),
    core_strict_centroid_gate_passed=bool(np.sqrt(np.average(normal_error ** 2, weights=areas)) <= 1e-6
                                         and np.max(abs(normal_error)) <= 1e-5),
)
for name, selected in (("wing", tags == 30), ("body", tags == 20),
                       ("seam", np.isin(np.arange(len(faces)), fixture["seam_faces"]))):
    field_report[name] = dict(normal_rms=float(np.sqrt(np.average(normal_error[selected] ** 2, weights=areas[selected]))),
                              normal_max=float(np.max(abs(normal_error[selected]))))
approach = []
for offset in (1e-2, 1e-3, 1e-4):
    targets = centers + offset * np.sqrt(areas[:, None]) * normals
    plain = np.full(len(faces), -1, dtype=np.int64)
    _, sb = influence(targets, triangles, normals, plain, 0)
    _, db = influence(targets, triangles, normals, plain, 1)
    _, dw = influence(targets, wake_panels, wake_normals, plain, 1)
    velocity = flow - np.einsum("ijk,j->ik", sb, sigma)
    velocity -= np.einsum("ijk,j->ik", db, mu) + np.einsum("ijk,j->ik", dw, wake_mu)
    approach.append(dict(offset_sqrt_area=offset,
                         velocity_difference_vs_exterior_trace_rms=float(np.sqrt(np.average(
                             np.sum((velocity - field_velocity) ** 2, axis=1), weights=areas))),
                         wall_normal_rms=float(np.sqrt(np.average(
                             np.einsum("ij,ij->i", velocity, normals) ** 2, weights=areas)))))
assert all(np.isfinite(list(row.values())).all() for row in approach)
assert approach[-1]["velocity_difference_vs_exterior_trace_rms"] < approach[0]["velocity_difference_vs_exterior_trace_rms"]

force = -data["Cp_raw"].ravel()[:, None] * area_vector
reference = np.asarray(result["references"]["moment"])
shift = np.array([.17, -.11, .09]) * result["references"]["chord"]
shifted_moment = np.cross(centers - reference - shift, force).sum(axis=0)
expected_moment = np.cross(centers - reference, force).sum(axis=0) - np.cross(shift, force.sum(axis=0))
moment_shift_error = float(np.max(abs(shifted_moment - expected_moment)))
audit = dict(
    status="geometry_load_gradient_audited", stencil_reports=stencil_reports,
    recovered_gradient_max_abs_error=gradient_error,
    independent_field=field_report, exterior_approach=approach,
    moment_reference_shift_error=moment_shift_error,
    wing_body_AIC_crossblock_l2=float(np.linalg.norm(data["AIC_mu"][0][np.ix_(tags == 30, tags == 20)])),
    pressure_reconstruction_is_not_independent_wall_validation=True,
    physical_accuracy_accepted=False, elapsed_seconds=time.perf_counter() - started,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
)
(args.run / "independent_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
np.savez_compressed(args.run / "independent_field.npz", velocity=field_velocity,
                    normal_error=normal_error, cp=field_cp, interior_potential=interior_potential)
print(json.dumps(audit, indent=2, allow_nan=False), flush=True)
