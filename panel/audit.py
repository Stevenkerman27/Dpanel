"""Independent NumPy/Numba audit of the explicit-cut differentiable solver."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
parser.add_argument("--out", type=Path, help="New directory for a revised audit; keep previous audits intact")
parser.add_argument("--ncpu", type=int, default=8)
parser.add_argument("--off-surface", action="store_true")
parser.add_argument("--wake-samples", type=int, default=0, help="0 skips; -1 evaluates every wake face")
args = parser.parse_args()
args.run = args.run.resolve()
audit_out = args.run if args.out is None else args.out.resolve()
if args.out is not None:
    assert not audit_out.exists()
    audit_out.mkdir(parents=True)
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = str(args.ncpu)
os.environ["NUMBA_CACHE_DIR"] = str(audit_out / "numba_cache")
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
coordinates, faces = data["coordinates"], data["faces"]
triangles = np.ascontiguousarray(coordinates[faces])
centers = triangles.mean(axis=1)
area_vector = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]) / 2
area = np.linalg.norm(area_vector, axis=1)
normals = area_vector / area[:, None]
assert np.max(abs(normals - data["panel_normal"])) < 1e-12
assert np.max(abs(area - data["panel_area"])) < 1e-12
mu, sigma, gamma = data["mu"], data["sigma"], data["gamma"]
wake = np.ascontiguousarray(coordinates[data["wake_faces"]])
wake_cross = np.cross(wake[:, 1] - wake[:, 0], wake[:, 2] - wake[:, 0])
wake_area = np.linalg.norm(wake_cross, axis=1) / 2
wake_normals = wake_cross / (2 * wake_area[:, None])
direction = data["wake_linear_direction"] if "wake_linear_direction" in data else np.tile([0., 1., 0.], (len(wake), 1))
wc = wake.mean(axis=1)
wake_mu, wake_slope = data["wake_basis"] @ gamma, data["wake_slope"] @ gamma
alpha = np.deg2rad(result["parameters"]["alpha_deg"])
flow = np.array([np.cos(alpha), 0., np.sin(alpha)])

# This evaluator does not call JAX or the solver's kernel. Linear wake velocity
# follows by differentiating the analytic line-integral moment identity.
def evaluate(targets, trace):
    sb, gs = influence(targets, triangles, normals, trace, 0)
    db, gd = influence(targets, triangles, normals, trace, 1)
    sw, gsw = influence(targets, wake, wake_normals, np.full(len(targets), -1, dtype=np.int64), 0)
    dw, gdw = influence(targets, wake, wake_normals, np.full(len(targets), -1, dtype=np.int64), 1)
    relative_y = np.sum((targets[:, None] - wc[None]) * direction[None], axis=2)
    height = np.sum((targets[:, None] - wc[None]) * wake_normals[None], axis=2)
    source_gradient = np.sum(gsw * direction[None], axis=2)
    moment = relative_y * dw + height * source_gradient
    hessian_y = np.zeros(gsw.shape)
    for k, q in ((0, 1), (1, 2), (2, 0)):
        edge = wake[:, q] - wake[:, k]
        length = np.linalg.norm(edge, axis=1)
        outward = np.cross(edge / length[:, None], wake_normals)
        r0, r1 = wake[None, :, k] - targets[:, None], wake[None, :, q] - targets[:, None]
        rad0, rad1 = np.linalg.norm(r0, axis=2), np.linalg.norm(r1, axis=2)
        dot, product = np.sum(r0 * r1, axis=2), rad0 * rad1
        plus = product + dot
        opposite = dot < 0
        plus[opposite] = np.sum(np.cross(r0, r1)**2, axis=2)[opposite] / (product - dot)[opposite]
        gradient_log = length[None, :, None] / plus[:, :, None] * (r0 / rad0[:, :, None] + r1 / rad1[:, :, None])
        hessian_y -= np.sum(outward * direction, axis=1)[None, :, None] * gradient_log / (4 * np.pi)
    hessian_y -= np.sum(wake_normals * direction, axis=1)[None, :, None] * gdw
    moment_gradient = relative_y[:, :, None] * gdw + height[:, :, None] * hessian_y
    moment_gradient += direction[None] * dw[:, :, None] + wake_normals[None] * source_gradient[:, :, None]
    velocity = flow - np.einsum("ijk,j->ik", gs, sigma) + np.einsum("ijk,j->ik", gd, mu)
    velocity += np.einsum("ijk,j->ik", gdw, wake_mu) + np.einsum("ijk,j->ik", moment_gradient, wake_slope)
    perturbation = -sb @ sigma + db @ mu + dw @ wake_mu + moment @ wake_slope
    return velocity, perturbation, sb, db, dw, moment


velocity, potential, sb, db_ext, dw, moment = evaluate(centers, np.arange(len(faces), dtype=np.int64))
kernel_error = dict(
    single=float(np.max(abs(sb - data["single"]))),
    doublet=float(np.max(abs(db_ext - np.eye(len(faces)) - data["doublet"]))),
    linear_wake_moment=float(np.max(abs(moment - data["wake_linear_moment"]))),
    constant_wake=float(np.max(abs(dw - data["wake_doublet"]))),
)
assert max(kernel_error.values()) < 1e-10
material_jump_error = None
if "wake_material_y" in data:
    material_stations = (np.asarray(fixture["gamma_stations"]) - np.asarray(fixture["origin"])[1]) / fixture["reference_chord"]
    material_stations *= result["parameters"]["span_factor"]
    dof = np.asarray(fixture["gamma_station_dof"])
    values = np.zeros(len(dof))
    values[dof >= 0] = gamma[dof[dof >= 0]]
    expected = np.interp(data["wake_material_y"], material_stations, values)
    recovered = wake_mu[:, None] + wake_slope[:, None] * np.sum((wake - wc[:, None]) * direction[:, None], axis=2)
    material_jump_error = float(np.max(abs(expected - recovered)))
    assert material_jump_error < 1e-11, "Moved wake lost its material circulation label"
normal_error = np.sum(velocity * normals, axis=1)
interior_potential = potential - mu
field_cp = 1 - np.sum(velocity**2, axis=1)
gradient = np.zeros((len(faces), 2))
affine_error = 0.
for i, row in enumerate(data["neighbor_indices"]):
    delta = centers[row] - centers[i]
    matrix = np.column_stack((delta @ data["panel_x_dir"][i], delta @ data["panel_y_dir"][i]))
    q, r = np.linalg.qr(matrix, mode="reduced")
    gradient[i] = np.linalg.solve(r, q.T @ (mu[row] - mu[i]))
    affine_error = max(affine_error, float(np.max(abs(np.linalg.solve(r, q.T @ (matrix @ [.37, -.21])) - [.37, -.21]))))
gradient_error = float(np.max(abs(gradient - np.column_stack((data["ql"], data["qm"])))))
assert gradient_error < 1e-9 and affine_error < 1e-12
cut_edges, cut_pairs = np.asarray(fixture["cut_edges"]), np.asarray(fixture["cut_face_pairs"])
cut_y = coordinates[cut_edges].mean(axis=1)[:, 1]
origin, chord = np.asarray(fixture["origin"]), fixture["reference_chord"]
stations = (np.asarray(fixture["gamma_stations"]) - origin[1]) / chord * result["parameters"]["span_factor"]
station_dof = np.asarray(fixture["gamma_station_dof"])
knot_values = np.zeros(len(stations))
knot_values[station_dof >= 0] = gamma[station_dof[station_dof >= 0]]
gamma_at_cut = np.interp(cut_y, stations, knot_values)
cut_jump_error = mu[cut_pairs[:, 0]] - mu[cut_pairs[:, 1]] - gamma_at_cut
cut_length = np.linalg.norm(coordinates[cut_edges[:, 1]] - coordinates[cut_edges[:, 0]], axis=1)
contact = np.array(fixture["cut_kind"]) == "smooth_wall_contact"
field = dict(
    wall_normal_rms=float(np.sqrt(np.average(normal_error**2, weights=area))),
    wall_normal_max=float(np.max(abs(normal_error))),
    interior_dirichlet_rms=float(np.sqrt(np.average(interior_potential**2, weights=area))),
    independent_field_cp_min=float(field_cp.min()), independent_field_cp_max=float(field_cp.max()),
    reconstructed_vs_field_cp_rms=float(np.sqrt(np.average((data["Cp_raw"] - field_cp)**2, weights=area))),
    strict_centroid_gate_passed=bool(np.sqrt(np.average(normal_error**2, weights=area)) <= 1e-6 and np.max(abs(normal_error)) <= 1e-5),
    contact_jump_rms=float(np.sqrt(np.average(cut_jump_error[contact]**2, weights=cut_length[contact]))) if contact.any() else None,
    exposed_te_jump_rms=float(np.sqrt(np.average(cut_jump_error[~contact]**2, weights=cut_length[~contact]))),
)
for name, mask in (("wing", data["component_ids"] != 20), ("body", data["component_ids"] == 20),
                   ("seam", np.isin(np.arange(len(faces)), fixture["seam_faces"]))):
    if mask.any():
        field[name] = dict(normal_rms=float(np.sqrt(np.average(normal_error[mask]**2, weights=area[mask]))),
                           normal_max=float(np.max(abs(normal_error[mask]))))
if args.off_surface:
    residuals = []
    for bary in ([.6, .2, .2], [.2, .6, .2], [.2, .2, .6]):
        sample = np.einsum("ijk,j->ik", triangles, bary)
        v, *_ = evaluate(sample, np.arange(len(faces), dtype=np.int64))
        residuals.append(np.sum(v * normals, axis=1))
    field["off_surface_in_face_normal_rms"] = float(np.sqrt(np.average(np.mean(np.array(residuals)**2, axis=0), weights=area)))
    field["off_surface_in_face_normal_max"] = float(np.max(np.abs(residuals)))
offsets = []
for factor in (1e-2, 1e-3, 1e-4):
    targets = centers + factor * np.sqrt(area[:, None]) * normals
    v, *_ = evaluate(targets, np.full(len(faces), -1, dtype=np.int64))
    offsets.append(dict(offset_sqrt_area=factor, velocity_vs_exterior_trace_rms=float(np.sqrt(
        np.average(np.sum((v - velocity)**2, axis=1), weights=area)))))
assert offsets[-1]["velocity_vs_exterior_trace_rms"] < offsets[0]["velocity_vs_exterior_trace_rms"]
wake_audit, wake_velocity = None, np.empty((0, 3))
sample_ids = np.empty(0, dtype=int)
if args.wake_samples:
    assert args.wake_samples == -1 or args.wake_samples > 0
    sample_ids = np.arange(len(wake)) if args.wake_samples == -1 else np.unique(np.linspace(0, len(wake) - 1, min(args.wake_samples, len(wake))).astype(int))
    sampled, sampled_jump = [], []
    for factor in (1e-3, 5e-4):
        delta = factor * np.sqrt(wake_area[sample_ids, None]) * wake_normals[sample_ids]
        positive, *_ = evaluate(wc[sample_ids] + delta, np.full(len(sample_ids), -1, dtype=np.int64))
        negative, *_ = evaluate(wc[sample_ids] - delta, np.full(len(sample_ids), -1, dtype=np.int64))
        sampled.append((positive + negative) / 2)
        sampled_jump.append(positive - negative)
    # Offsets define a two-sided principal-value quadrature, not a vortex core.
    # Keep its approach error visible; never change wall collocation positions.
    wake_velocity = 2 * sampled[1] - sampled[0]
    velocity_jump = 2 * sampled_jump[1] - sampled_jump[0]
    expected_jump = wake_slope[sample_ids, None] * direction[sample_ids]
    transport = np.sum(wake_velocity * expected_jump, axis=1)
    pressure_jump = -2 * np.sum(wake_velocity * velocity_jump, axis=1)
    probe = wc[sample_ids[:3]] + .13 * np.sqrt(wake_area[sample_ids[:3], None]) * wake_normals[sample_ids[:3]]
    probe_trace = np.full(len(probe), -1, dtype=np.int64)
    probe_velocity, *_ = evaluate(probe, probe_trace)
    derivative_fd = np.empty_like(probe)
    for column in range(3):
        shift = np.zeros(3)
        shift[column] = 1e-5
        _, plus, *_ = evaluate(probe + shift, probe_trace)
        _, minus, *_ = evaluate(probe - shift, probe_trace)
        derivative_fd[:, column] = (plus - minus) / 2e-5 + flow[column]
    field_derivative_error = float(np.max(abs(probe_velocity - derivative_fd)))
    assert field_derivative_error < 1e-6, "Independent analytic velocity differs from its potential derivative"
    flux = np.sum(wake_velocity * wake_normals[sample_ids], axis=1)
    approach = np.linalg.norm(sampled[1] - sampled[0], axis=1)
    wake_audit = dict(samples=len(sample_ids), velocity_definition="Two-sided average, linear zero-offset extrapolation",
                      offsets_sqrt_area=[1e-3, 5e-4],
                      velocity_vs_potential_fd_max_error=field_derivative_error,
                      material_advection_rms=float(np.sqrt(np.average(transport**2, weights=wake_area[sample_ids]))),
                      steady_pressure_jump_rms=float(np.sqrt(np.average(pressure_jump**2, weights=wake_area[sample_ids]))),
                      velocity_jump_vs_material_gradient_rms=float(np.sqrt(np.average(
                          np.sum((velocity_jump - expected_jump)**2, axis=1), weights=wake_area[sample_ids]))),
                      total_normal_rms=float(np.sqrt(np.average(flux**2, weights=wake_area[sample_ids]))),
                      total_normal_max=float(np.max(abs(flux))),
                      extrapolation_change_rms=float(np.sqrt(np.average(approach**2, weights=wake_area[sample_ids]))),
                      freestream_normal_rms=float(np.sqrt(np.average((wake_normals[sample_ids] @ flow)**2, weights=wake_area[sample_ids]))))
    hinge = result.get("wake_model", {}).get("hinge_x_normalized")
    for name, mask in (("near", wc[sample_ids, 0] <= (hinge if hinge is not None else 5.)),
                       ("far", wc[sample_ids, 0] > (hinge if hinge is not None else 5.))):
        if mask.any():
            wake_audit[name] = dict(samples=int(mask.sum()), total_normal_rms=float(np.sqrt(
                np.average(flux[mask]**2, weights=wake_area[sample_ids[mask]]))),
                material_advection_rms=float(np.sqrt(np.average(transport[mask]**2, weights=wake_area[sample_ids[mask]]))),
                steady_pressure_jump_rms=float(np.sqrt(np.average(pressure_jump[mask]**2, weights=wake_area[sample_ids[mask]]))))
audit = dict(status="independently_audited", kernel_max_abs_errors=kernel_error,
             material_vertex_jump_max_error=material_jump_error,
             gradient_qr_max_abs_error=gradient_error, affine_reproduction_max_abs_error=affine_error,
             independent_field=field, exterior_approach=offsets, wake_tangency=wake_audit, physical_accuracy_accepted=False,
             source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             elapsed_seconds=time.perf_counter() - started)
(audit_out / "independent_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
(audit_out / "audit.py.snapshot").write_bytes(Path(__file__).read_bytes())
np.savez_compressed(audit_out / "independent_field.npz", velocity=velocity, normal_error=normal_error, cp=field_cp,
                    cut_jump_error=cut_jump_error, interior_potential=interior_potential,
                    wake_sample_ids=sample_ids, wake_mean_velocity=wake_velocity)
print(json.dumps(audit, indent=2), flush=True)
