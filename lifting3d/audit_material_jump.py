"""Audit a deformed wake jump in its material coordinates."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent / "volume"))
import p2

parser = argparse.ArgumentParser()
parser.add_argument("--mesh", type=Path,
                    default=root / "results" / "graph_wake_v1" / "01" / "mesh" / "mesh.npz")
parser.add_argument("--field", type=Path,
                    default=root / "results" / "graph_wake_v1" / "01" / "solve" / "field.npz")
parser.add_argument("--result", type=Path,
                    default=root / "results" / "graph_wake_v1" / "01" / "solve" / "result.json")
parser.add_argument("--out", type=Path, default=root / "results" / "material_jump_audit_v1")
parser.add_argument("--ncpu", type=int, default=8)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

mesh_bytes = args.mesh.read_bytes()
field_bytes = args.field.read_bytes()
result_bytes = args.result.read_bytes()
mesh_sha256 = hashlib.sha256(mesh_bytes).hexdigest()
field_sha256 = hashlib.sha256(field_bytes).hexdigest()
result_sha256 = hashlib.sha256(result_bytes).hexdigest()
script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

mesh = np.load(args.mesh)
field = np.load(args.field)
run_result = json.loads(result_bytes)
metadata = json.loads(args.mesh.with_name("mesh.json").read_text(encoding="utf-8"))
outer_summary_path = args.mesh.parents[2] / "summary.json"
outer_summary = json.loads(outer_summary_path.read_text(encoding="utf-8")) if outer_summary_path.exists() else {}
points = field["points"]
reference_points = mesh["reference_points"]
tets = field["tets"]
phi = field["phi"]
upper_faces = mesh["wake_upper"]
lower_faces = mesh["wake_lower"]
upper_tets = mesh["wake_upper_tet"]
lower_tets = mesh["wake_lower_tet"]
stations = mesh["gamma_stations"]
gamma = field["gamma"]
gamma_values = np.r_[0.0, gamma, 0.0]
assert args.ncpu == 8
assert run_result["mesh_sha256"] == metadata["mesh_sha256"]
assert len(points) == len(reference_points)
assert np.array_equal(points, mesh["points"])
assert np.array_equal(tets, mesh["tets"])
assert len(upper_faces) == len(lower_faces) == len(upper_tets) == len(lower_tets)
assert upper_faces.shape[1] == lower_faces.shape[1] == 6
assert len(stations) == len(gamma_values)
assert np.max(np.abs(points[upper_faces] - points[lower_faces])) == 0.0
assert np.array_equal(mesh["cut_upper_nodes"], np.unique(mesh["cut_upper_nodes"]))
assert np.array_equal(mesh["cut_lower_nodes"], np.unique(mesh["cut_lower_nodes"]))
assert np.array_equal(mesh["reference_points"][mesh["te_upper_nodes"], 1], stations[1:-1])

length = float(metadata["chord"])
origin = np.asarray(metadata.get("normalization_origin", [0.0, 0.0, 0.0]), dtype=float)
points = (points - origin) / length
reference_points = (reference_points - origin) / length
phi = phi / length
gamma_values = gamma_values / length
stations = (stations - origin[1]) / length
kink_tolerance = 1.0e-12 * max(1.0, abs(stations[0]), abs(stations[-1]))
triangle_bary, triangle_weights = p2.triangle_quadrature(3)
triangle_shape, triangle_gradient = p2.triangle_shape(triangle_bary)

cut_y = reference_points[mesh["cut_upper_nodes"], 1]
cut_interval = np.searchsorted(stations, cut_y, side="right") - 1
cut_interval = np.clip(cut_interval, 0, len(stations) - 2)
cut_fraction = ((cut_y - stations[cut_interval])
                / (stations[cut_interval + 1] - stations[cut_interval]))
cut_expected = ((1.0 - cut_fraction) * gamma_values[cut_interval]
                + cut_fraction * gamma_values[cut_interval + 1])
cut_delta = phi[mesh["cut_upper_nodes"]] - phi[mesh["cut_lower_nodes"]]
node_rows = []
for i in range(len(cut_y)):
    node_rows.append({
        "upper_node": int(mesh["cut_upper_nodes"][i]),
        "lower_node": int(mesh["cut_lower_nodes"][i]),
        "x_reference": float(reference_points[mesh["cut_upper_nodes"][i], 0]),
        "eta_reference": float(cut_y[i]),
        "left_station": int(cut_interval[i]),
        "right_station": int(cut_interval[i] + 1),
        "left_B_weight": float(1.0 - cut_fraction[i]),
        "right_B_weight": float(cut_fraction[i]),
        "B_gamma_eta_ref": float(cut_expected[i]),
        "phi_upper_minus_lower": float(cut_delta[i]),
        "B_gamma_error": float(cut_delta[i] - cut_expected[i]),
        "physical_y_current": float(points[mesh["cut_upper_nodes"][i], 1]),
    })

point_rows = []
face_rows = []
weights_all = []
delta_error_all = []
dx_material_all = []
deta_material_error_all = []
tangent_x_trace_error_all = []
tangent_eta_trace_error_all = []
tangent_x_gamma_error_all = []
tangent_eta_gamma_error_all = []
normal_gradient_jump_all = []
cartesian_x_gradient_jump_all = []
wrong_physical_y_error_all = []
physical_y_minus_eta_all = []
crossing_face_count = 0

for face_index in range(len(upper_faces)):
    upper_nodes = upper_faces[face_index]
    lower_nodes = lower_faces[face_index]
    upper_element = int(upper_tets[face_index])
    lower_element = int(lower_tets[face_index])
    xyz = points[upper_nodes]
    reference_xyz = reference_points[upper_nodes]
    delta_nodes = phi[upper_nodes] - phi[lower_nodes]

    face_crossings = stations[
        (stations > np.min(reference_xyz[:, 1]) + kink_tolerance)
        & (stations < np.max(reference_xyz[:, 1]) - kink_tolerance)]
    if len(face_crossings):
        crossing_face_count += 1

    param_jacobian = np.einsum("ia,qib->qab", reference_xyz[:, [0, 1]], triangle_gradient)
    param_determinant = np.linalg.det(param_jacobian)
    assert np.all(np.abs(param_determinant) > 1.0e-14)
    inverse_param_jacobian = np.linalg.inv(param_jacobian)
    physical_jacobian = np.einsum("ia,qib->qab", xyz, triangle_gradient)
    physical_area_metric = np.linalg.norm(
        np.cross(physical_jacobian[:, :, 0], physical_jacobian[:, :, 1]), axis=1)
    tangents = np.einsum("qia,qab->qib", physical_jacobian, inverse_param_jacobian)
    tangent_x = tangents[:, :, 0]
    tangent_eta = tangents[:, :, 1]
    surface_normal = np.cross(tangent_x, tangent_eta)
    surface_normal /= np.linalg.norm(surface_normal, axis=1)[:, None]

    delta_p2 = triangle_shape @ delta_nodes
    delta_reference_gradient = np.einsum("qia,i->qa", triangle_gradient, delta_nodes)
    delta_material_gradient = np.einsum(
        "qa,qab->qb", delta_reference_gradient, inverse_param_jacobian)

    x_reference_q = triangle_shape @ reference_xyz[:, 0]
    eta_reference_q = triangle_shape @ reference_xyz[:, 1]
    physical_y_q = triangle_shape @ xyz[:, 1]
    gamma_interval = np.searchsorted(stations, eta_reference_q, side="right") - 1
    gamma_interval = np.clip(gamma_interval, 0, len(stations) - 2)
    gamma_fraction = ((eta_reference_q - stations[gamma_interval])
                      / (stations[gamma_interval + 1] - stations[gamma_interval]))
    gamma_eta = ((1.0 - gamma_fraction) * gamma_values[gamma_interval]
                 + gamma_fraction * gamma_values[gamma_interval + 1])
    gamma_slope = ((gamma_values[gamma_interval + 1] - gamma_values[gamma_interval])
                   / (stations[gamma_interval + 1] - stations[gamma_interval]))
    delta_error = delta_p2 - gamma_eta
    dx_material = delta_material_gradient[:, 0]
    deta_error = delta_material_gradient[:, 1] - gamma_slope

    upper_tet_nodes = tets[upper_element]
    lower_tet_nodes = tets[lower_element]
    upper_local_vertices = [int(np.flatnonzero(upper_tet_nodes[:4] == node)[0])
                            for node in upper_nodes[:3]]
    lower_local_vertices = [int(np.flatnonzero(lower_tet_nodes[:4] == node)[0])
                            for node in lower_nodes[:3]]
    upper_bary = np.zeros((len(triangle_bary), 4))
    lower_bary = np.zeros((len(triangle_bary), 4))
    for local_face_vertex in range(3):
        upper_bary[:, upper_local_vertices[local_face_vertex]] = triangle_bary[:, local_face_vertex]
        lower_bary[:, lower_local_vertices[local_face_vertex]] = triangle_bary[:, local_face_vertex]

    for local_edge, (first_vertex, second_vertex) in enumerate(p2.TRI_EDGES):
        upper_tet_edge = next(
            edge for edge, pair in enumerate(p2.TET_EDGES)
            if set(pair) == {upper_local_vertices[first_vertex], upper_local_vertices[second_vertex]})
        lower_tet_edge = next(
            edge for edge, pair in enumerate(p2.TET_EDGES)
            if set(pair) == {lower_local_vertices[first_vertex], lower_local_vertices[second_vertex]})
        assert upper_nodes[3 + local_edge] == upper_tet_nodes[4 + upper_tet_edge]
        assert lower_nodes[3 + local_edge] == lower_tet_nodes[4 + lower_tet_edge]

    _, upper_reference_gradient = p2.p2_shape(upper_bary)
    _, lower_reference_gradient = p2.p2_shape(lower_bary)
    upper_map_jacobian = np.einsum(
        "ia,qib->qab", points[upper_tet_nodes], upper_reference_gradient)
    lower_map_jacobian = np.einsum(
        "ia,qib->qab", points[lower_tet_nodes], lower_reference_gradient)
    upper_physical_gradient = np.einsum(
        "qia,qab->qib", upper_reference_gradient, np.linalg.inv(upper_map_jacobian))
    lower_physical_gradient = np.einsum(
        "qia,qab->qib", lower_reference_gradient, np.linalg.inv(lower_map_jacobian))
    upper_velocity = np.einsum("qia,i->qa", upper_physical_gradient, phi[upper_tet_nodes])
    lower_velocity = np.einsum("qia,i->qa", lower_physical_gradient, phi[lower_tet_nodes])
    raw_gradient_jump = upper_velocity - lower_velocity
    raw_jump_tangent_x = np.einsum("qa,qa->q", raw_gradient_jump, tangent_x)
    raw_jump_tangent_eta = np.einsum("qa,qa->q", raw_gradient_jump, tangent_eta)
    raw_jump_normal = np.einsum("qa,qa->q", raw_gradient_jump, surface_normal)

    current_y_interval = np.searchsorted(stations, physical_y_q, side="right") - 1
    current_y_interval = np.clip(current_y_interval, 0, len(stations) - 2)
    current_y_fraction = ((physical_y_q - stations[current_y_interval])
                          / (stations[current_y_interval + 1] - stations[current_y_interval]))
    gamma_of_current_y = (
        (1.0 - current_y_fraction) * gamma_values[current_y_interval]
        + current_y_fraction * gamma_values[current_y_interval + 1])
    wrong_y_error = delta_p2 - gamma_of_current_y
    eta_shift = physical_y_q - eta_reference_q

    area_weights = triangle_weights * physical_area_metric
    weights_all.extend(area_weights.tolist())
    delta_error_all.extend(delta_error.tolist())
    dx_material_all.extend(dx_material.tolist())
    deta_material_error_all.extend(deta_error.tolist())
    tangent_x_trace_error_all.extend((raw_jump_tangent_x - delta_material_gradient[:, 0]).tolist())
    tangent_eta_trace_error_all.extend((raw_jump_tangent_eta - delta_material_gradient[:, 1]).tolist())
    tangent_x_gamma_error_all.extend(raw_jump_tangent_x.tolist())
    tangent_eta_gamma_error_all.extend((raw_jump_tangent_eta - gamma_slope).tolist())
    normal_gradient_jump_all.extend(raw_jump_normal.tolist())
    cartesian_x_gradient_jump_all.extend(raw_gradient_jump[:, 0].tolist())
    wrong_physical_y_error_all.extend(wrong_y_error.tolist())
    physical_y_minus_eta_all.extend(eta_shift.tolist())

    face_rows.append({
        "face": face_index,
        "crossed_station_count": int(len(face_crossings)),
        "area": float(np.sum(area_weights)),
        "delta_error_rms": float(np.sqrt(np.sum(area_weights * delta_error**2) / np.sum(area_weights))),
        "dx_material_rms": float(np.sqrt(np.sum(area_weights * dx_material**2) / np.sum(area_weights))),
        "deta_material_error_rms": float(np.sqrt(np.sum(area_weights * deta_error**2) / np.sum(area_weights))),
        "normal_gradient_jump_rms": float(np.sqrt(np.sum(area_weights * raw_jump_normal**2) / np.sum(area_weights))),
    })

    for q in range(len(triangle_weights)):
        point_rows.append({
            "face": face_index,
            "gauss_point": q,
            "x_reference": float(x_reference_q[q]),
            "eta_reference": float(eta_reference_q[q]),
            "physical_x": float(triangle_shape[q] @ xyz[:, 0]),
            "physical_y": float(physical_y_q[q]),
            "physical_z": float(triangle_shape[q] @ xyz[:, 2]),
            "area_weight": float(area_weights[q]),
            "delta_phi_p2": float(delta_p2[q]),
            "gamma_eta_reference": float(gamma_eta[q]),
            "delta_error": float(delta_error[q]),
            "d_delta_d_x_reference": float(delta_material_gradient[q, 0]),
            "d_delta_d_eta_reference": float(delta_material_gradient[q, 1]),
            "gamma_eta_slope": float(gamma_slope[q]),
            "deta_slope_error": float(deta_error[q]),
            "tangent_x": json.dumps(tangent_x[q].tolist()),
            "tangent_eta": json.dumps(tangent_eta[q].tolist()),
            "raw_gradient_jump_x": float(raw_gradient_jump[q, 0]),
            "raw_jump_dot_tangent_x": float(raw_jump_tangent_x[q]),
            "raw_jump_dot_tangent_eta": float(raw_jump_tangent_eta[q]),
            "raw_jump_normal_component": float(raw_jump_normal[q]),
            "raw_jump_dot_tx_error_vs_trace": float(
                raw_jump_tangent_x[q] - delta_material_gradient[q, 0]),
            "raw_jump_dot_teta_error_vs_trace": float(
                raw_jump_tangent_eta[q] - delta_material_gradient[q, 1]),
            "raw_jump_dot_tx_error_vs_material_zero": float(raw_jump_tangent_x[q]),
            "raw_jump_dot_teta_error_vs_gamma_slope": float(
                raw_jump_tangent_eta[q] - gamma_slope[q]),
            "gamma_current_physical_y_diagnostic_only": float(gamma_of_current_y[q]),
            "error_if_gamma_wrongly_tied_to_current_y": float(wrong_y_error[q]),
        })

def weighted_summary(weights, values):
    weights = np.asarray(weights, dtype=float)
    values = np.asarray(values, dtype=float)
    return {
        "area_weighted_rms": float(np.sqrt(np.sum(weights * values**2) / np.sum(weights))),
        "absolute_max": float(np.max(np.abs(values))),
        "signed_mean": float(np.sum(weights * values) / np.sum(weights)),
    }

def unweighted_summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "rms": float(np.sqrt(np.mean(values**2))),
        "absolute_max": float(np.max(np.abs(values))),
    }

with (args.out / "cut_nodes_B.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(node_rows[0]))
    writer.writeheader()
    writer.writerows(node_rows)
with (args.out / "wake_gauss_points.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(point_rows[0]))
    writer.writeheader()
    writer.writerows(point_rows)
with (args.out / "wake_faces.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(face_rows[0]))
    writer.writeheader()
    writer.writerows(face_rows)

weights_all = np.asarray(weights_all, dtype=float)
summary = {
    "status": "completed",
    "scope": "P2 material-coordinate wake-jump and one-sided gradient audit; iteration-01 field retained from failed outer geometry run",
    "ncpu_requested": args.ncpu,
    "mesh": str(args.mesh),
    "field": str(args.field),
    "result": str(args.result),
    "mesh_sha256": mesh_sha256,
    "field_sha256": field_sha256,
    "result_json_sha256": result_sha256,
    "audit_script_sha256": script_sha256,
    "solver_sha256": run_result.get("solver_sha256"),
    "mesh_metadata_sha256": metadata.get("mesh_sha256"),
    "solver_status": run_result.get("status"),
    "solver_converged": run_result.get("converged"),
    "solver_failure": run_result.get("message"),
    "outer_geometry_run": {
        "summary_path": str(outer_summary_path),
        "status": outer_summary.get("status"),
        "failure": outer_summary.get("failure"),
        "completed_field_iteration_count": len(outer_summary.get("history", [])),
        "note": "Outer relaxation status is distinct from the converged solve field audited here.",
    },
    "mesh_wake_material_coordinates": metadata.get("wake_material_coordinates"),
    "material_coordinate_used_by_solver": "reference_points[:,1] at cut lower nodes",
    "wake_faces": int(len(upper_faces)),
    "quadrature_order": 3,
    "gauss_points_per_face": int(len(triangle_weights)),
    "cut_node_pairs": int(len(cut_y)),
    "station_count_including_zero_tips": int(len(stations)),
    "crossing_face_count": int(crossing_face_count),
    "single_gamma_interval_face_count": int(len(upper_faces) - crossing_face_count),
    "kink_tolerance": kink_tolerance,
    "upper_lower_coordinate_pair_max": float(np.max(np.abs(points[upper_faces] - points[lower_faces]))),
    "x_coordinate_motion_max": float(np.max(np.abs(points[:, 0] - reference_points[:, 0]))),
    "B_gamma_eta_ref_at_cut_nodes": unweighted_summary(
        cut_delta - cut_expected),
    "face_quadrature": {
        "delta_phi_minus_gamma_eta_ref": weighted_summary(weights_all, delta_error_all),
        "material_d_delta_d_x_ref_expected_zero": weighted_summary(weights_all, dx_material_all),
        "material_d_delta_d_eta_ref_minus_gamma_slope": weighted_summary(
            weights_all, deta_material_error_all),
        "raw_gradient_jump_dot_tx_minus_trace_d_delta_dx_ref": weighted_summary(
            weights_all, tangent_x_trace_error_all),
        "raw_gradient_jump_dot_teta_minus_trace_d_delta_deta_ref": weighted_summary(
            weights_all, tangent_eta_trace_error_all),
        "raw_gradient_jump_dot_tx_expected_material_zero": weighted_summary(
            weights_all, tangent_x_gamma_error_all),
        "raw_gradient_jump_dot_teta_minus_gamma_slope": weighted_summary(
            weights_all, tangent_eta_gamma_error_all),
        "raw_gradient_jump_normal_component_unconstrained": weighted_summary(
            weights_all, normal_gradient_jump_all),
        "raw_gradient_jump_cartesian_x_diagnostic_not_acceptance": weighted_summary(
            weights_all, cartesian_x_gradient_jump_all),
        "delta_phi_minus_gamma_of_current_physical_y_wrong_coordinate_diagnostic": weighted_summary(
            weights_all, wrong_physical_y_error_all),
        "physical_y_minus_eta_ref_diagnostic_only": weighted_summary(
            weights_all, physical_y_minus_eta_all),
    },
    "solver_reported": {
        "potential_jump_max_error": run_result.get("potential_jump_max_error"),
        "wake_diagnostics": run_result.get("wake_diagnostics"),
        "cl_pressure": run_result.get("cl_pressure"),
        "cl_jump": run_result.get("cl_jump"),
    },
    "outputs": ["cut_nodes_B.csv", "wake_gauss_points.csv", "wake_faces.csv"],
}
(args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps(summary, indent=2))
