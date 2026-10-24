"""Audit the P2 wake trace against its piecewise-linear gamma basis."""
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
parser.add_argument("--mesh", type=Path, default=root / "meshes" / "strips_v1" / "wing015_s24" / "mesh.npz")
parser.add_argument("--result", type=Path,
                    default=root / "results" / "strips_v1" / "wing015_s24_a5")
parser.add_argument("--out", type=Path, default=root / "audit" / "wake_basis_s24_v1")
parser.add_argument("--ncpu", type=int, default=8)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

mesh_bytes = args.mesh.read_bytes()
field_path = args.result / "field.npz"
result_path = args.result / "result.json"
field_bytes = field_path.read_bytes()
result_bytes = result_path.read_bytes()
mesh_sha256 = hashlib.sha256(mesh_bytes).hexdigest()
field_sha256 = hashlib.sha256(field_bytes).hexdigest()
result_sha256 = hashlib.sha256(result_bytes).hexdigest()
script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

mesh = np.load(args.mesh)
field = np.load(field_path)
run_result = json.loads(result_bytes)
mesh_metadata = json.loads(args.mesh.with_name("mesh.json").read_text(encoding="utf-8"))
points = field["points"]
phi = field["phi"]
upper_faces = mesh["wake_upper"]
lower_faces = mesh["wake_lower"]
stations = mesh["gamma_stations"]
gamma = field["gamma"]
assert args.ncpu == 8
assert len(upper_faces) == len(lower_faces)
assert upper_faces.shape[1] == 6
assert len(stations) == len(gamma) + 2
assert np.max(np.abs(points[mesh["cut_upper_nodes"]] - points[mesh["cut_lower_nodes"]])) == 0.0
cut_map = np.arange(len(points), dtype=int)
cut_map[mesh["cut_upper_nodes"]] = mesh["cut_lower_nodes"]
assert np.array_equal(cut_map[upper_faces], lower_faces)
assert run_result["mesh_sha256"] == mesh_metadata["mesh_sha256"]
assert np.max(np.abs(points - mesh["points"])) <= 1.0e-12
assert len(run_result["potential_jump"]) == len(stations)

triangle_bary, triangle_weights = p2.triangle_quadrature(3)
shape, reference_gradient = p2.triangle_shape(triangle_bary)
gamma_values = np.r_[0.0, gamma, 0.0]
reported_jump = np.asarray(run_result["potential_jump"], dtype=float)
gamma_scale = max(float(np.max(np.abs(gamma_values))), 1.0e-14)
span = float(stations[-1] - stations[0])
kink_tolerance = 1.0e-12 * max(1.0, abs(stations[0]), abs(stations[-1]))

cut_y = points[mesh["cut_upper_nodes"], 1]
cut_interval = np.searchsorted(stations, cut_y, side="right") - 1
cut_interval = np.clip(cut_interval, 0, len(stations) - 2)
cut_fraction = ((cut_y - stations[cut_interval])
                / (stations[cut_interval + 1] - stations[cut_interval]))
cut_expected = ((1.0 - cut_fraction) * gamma_values[cut_interval]
                + cut_fraction * gamma_values[cut_interval + 1])
cut_delta = phi[mesh["cut_upper_nodes"]] - phi[mesh["cut_lower_nodes"]]
node_rows = []
for node_index in range(len(mesh["cut_upper_nodes"])):
    node_rows.append({
        "upper_node": int(mesh["cut_upper_nodes"][node_index]),
        "lower_node": int(mesh["cut_lower_nodes"][node_index]),
        "x": float(points[mesh["cut_upper_nodes"][node_index], 0]),
        "y": float(cut_y[node_index]),
        "left_station": int(cut_interval[node_index]),
        "right_station": int(cut_interval[node_index] + 1),
        "left_weight": float(1.0 - cut_fraction[node_index]),
        "right_weight": float(cut_fraction[node_index]),
        "B_gamma": float(cut_expected[node_index]),
        "phi_upper_minus_lower": float(cut_delta[node_index]),
        "B_gamma_error": float(cut_delta[node_index] - cut_expected[node_index]),
    })

point_rows = []
face_rows = []
all_area_weight = []
all_delta_error = []
all_dx = []
all_dy_error = []
group_values = {"kink_crossing": [], "single_interval": []}
pure_quadratic_values = []

for face_index in range(len(upper_faces)):
    upper_nodes = upper_faces[face_index]
    lower_nodes = lower_faces[face_index]
    coordinate_error = float(np.max(np.abs(points[upper_nodes] - points[lower_nodes])))
    assert coordinate_error <= 1.0e-12
    xyz = points[upper_nodes]
    delta_nodes = phi[upper_nodes] - phi[lower_nodes]
    x_min = float(np.min(xyz[:, 0]))
    x_max = float(np.max(xyz[:, 0]))
    y_min = float(np.min(xyz[:, 1]))
    y_max = float(np.max(xyz[:, 1]))
    crossed = stations[(stations > y_min + kink_tolerance) & (stations < y_max - kink_tolerance)]
    group = "kink_crossing" if len(crossed) else "single_interval"

    physical_jacobian = np.einsum("ia,qib->qab", xyz[:, :2], reference_gradient)
    determinant = np.linalg.det(physical_jacobian)
    assert np.all(determinant > 0.0) or np.all(determinant < 0.0)
    physical_gradient = np.einsum("qia,qab->qib", reference_gradient,
                                  np.linalg.inv(physical_jacobian))
    delta_p2 = shape @ delta_nodes
    delta_gradient = np.einsum("i,qia->qa", delta_nodes, physical_gradient)
    xq = shape @ xyz[:, 0]
    yq = shape @ xyz[:, 1]
    gamma_interval = np.searchsorted(stations, yq, side="right") - 1
    gamma_interval = np.clip(gamma_interval, 0, len(stations) - 2)
    gamma_fraction = (yq - stations[gamma_interval]) / (
        stations[gamma_interval + 1] - stations[gamma_interval])
    gamma_exact = ((1.0 - gamma_fraction) * gamma_values[gamma_interval]
                   + gamma_fraction * gamma_values[gamma_interval + 1])
    gamma_slope = ((gamma_values[gamma_interval + 1] - gamma_values[gamma_interval])
                   / (stations[gamma_interval + 1] - stations[gamma_interval]))
    delta_error = delta_p2 - gamma_exact
    dx_error = delta_gradient[:, 0]
    dy_error = delta_gradient[:, 1] - gamma_slope

    metric = np.abs(determinant)
    area_weights = triangle_weights * metric
    area = float(np.sum(area_weights))
    face_delta_rms = float(np.sqrt(np.sum(area_weights * delta_error**2) / area))
    face_dx_rms = float(np.sqrt(np.sum(area_weights * dx_error**2) / area))
    face_dy_rms = float(np.sqrt(np.sum(area_weights * dy_error**2) / area))
    face_rows.append({
        "face": face_index,
        "group": group,
        "crossed_station_count": int(len(crossed)),
        "crossed_station_y": ";".join(f"{v:.17g}" for v in crossed),
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "area": area,
        "delta_rms": face_delta_rms,
        "delta_max": float(np.max(np.abs(delta_error))),
        "dx_rms": face_dx_rms,
        "dx_max": float(np.max(np.abs(dx_error))),
        "dy_slope_error_rms": face_dy_rms,
        "dy_slope_error_max": float(np.max(np.abs(dy_error))),
    })
    group_values[group].append((area_weights, delta_error, dx_error, dy_error))
    all_area_weight.extend(area_weights.tolist())
    all_delta_error.extend(delta_error.tolist())
    all_dx.extend(dx_error.tolist())
    all_dy_error.extend(dy_error.tolist())

    for point_index in range(len(triangle_weights)):
        point_rows.append({
            "face": face_index,
            "gauss_point": point_index,
            "group": group,
            "x": float(xq[point_index]),
            "y": float(yq[point_index]),
            "area_weight": float(area_weights[point_index]),
            "delta_phi_p2": float(delta_p2[point_index]),
            "gamma_piecewise_linear": float(gamma_exact[point_index]),
            "delta_error": float(delta_error[point_index]),
            "d_delta_phi_dx": float(dx_error[point_index]),
            "d_delta_phi_dy": float(delta_gradient[point_index, 1]),
            "gamma_slope": float(gamma_slope[point_index]),
            "dy_slope_error": float(dy_error[point_index]),
        })

    quadratic_nodes = gamma_scale * (1.0 - (2.0 * (xyz[:, 1] - stations[0]) / span - 1.0) ** 2)
    quadratic_p2 = shape @ quadratic_nodes
    quadratic_gradient = np.einsum("i,qia->qa", quadratic_nodes, physical_gradient)
    quadratic_exact = gamma_scale * (1.0 - (2.0 * (yq - stations[0]) / span - 1.0) ** 2)
    quadratic_dy_exact = (-4.0 * gamma_scale / span
                          * (2.0 * (yq - stations[0]) / span - 1.0))
    pure_quadratic_values.extend(zip(
        area_weights.tolist(),
        (quadratic_p2 - quadratic_exact).tolist(),
        quadratic_gradient[:, 0].tolist(),
        (quadratic_gradient[:, 1] - quadratic_dy_exact).tolist()))

with (args.out / "wake_gauss_points.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(point_rows[0]))
    writer.writeheader()
    writer.writerows(point_rows)
with (args.out / "wake_faces.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(face_rows[0]))
    writer.writeheader()
    writer.writerows(face_rows)
with (args.out / "wake_nodes_B.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(node_rows[0]))
    writer.writeheader()
    writer.writerows(node_rows)

def summarize(weight_values, error_values):
    weights = np.asarray(weight_values, dtype=float)
    errors = np.asarray(error_values, dtype=float)
    return {
        "area_weighted_rms": float(np.sqrt(np.sum(weights * errors**2) / np.sum(weights))),
        "absolute_max": float(np.max(np.abs(errors))),
    }

group_summary = {}
for group, records in group_values.items():
    if not records:
        group_summary[group] = {"face_count": 0}
        continue
    weights = np.concatenate([record[0] for record in records])
    group_summary[group] = {"face_count": len(records)}
    for index, name in enumerate(["delta_error", "dx_error", "dy_slope_error"], start=1):
        values = np.concatenate([record[index] for record in records])
        group_summary[group][name] = summarize(weights, values)

all_area_weight = np.asarray(all_area_weight)
old_summary_path = root / "audit" / "wake_basis_v1" / "summary.json"
old_summary = json.loads(old_summary_path.read_text(encoding="utf-8"))
summary = {
    "status": "completed",
    "scope": "Independent P2 wake-trace representation audit; not a causal lift/Cp study",
    "mesh": str(args.mesh),
    "field": str(field_path),
    "mesh_sha256": mesh_sha256,
    "field_sha256": field_sha256,
    "result_json_sha256": result_sha256,
    "script_sha256": script_sha256,
    "ncpu_requested": args.ncpu,
    "p2_triangle_quadrature_order": 3,
    "gauss_points_per_face": int(len(triangle_weights)),
    "face_count": int(len(upper_faces)),
    "station_count_including_zero_tips": int(len(stations)),
    "gamma_unknown_count": int(len(gamma)),
    "station_kink_tolerance": kink_tolerance,
    "wake_strip_count_from_mesh_metadata": mesh_metadata.get("wake_strips"),
    "face_kink_crossing_count": int(sum(row["crossed_station_count"] > 0 for row in face_rows)),
    "face_within_one_gamma_interval_count": int(sum(row["crossed_station_count"] == 0 for row in face_rows)),
    "B_gamma_at_cut_nodes": {
        "rms": float(np.sqrt(np.mean((cut_delta - cut_expected)**2))),
        "absolute_max": float(np.max(np.abs(cut_delta - cut_expected))),
    },
    "reported_potential_jump_vs_gamma_station_values": {
        "rms": float(np.sqrt(np.mean((reported_jump - gamma_values)**2))),
        "absolute_max": float(np.max(np.abs(reported_jump - gamma_values))),
    },
    "gamma_amplitude_scale_for_quadratic_patch": gamma_scale,
    "quadratic_patch_definition": "A*(1-(2*(y-y_min)/span-1)^2), A=max(abs([0,gamma,0]))",
    "coordinate_pair_max_abs": float(np.max(np.abs(points[upper_faces] - points[lower_faces]))),
    "global": {
        "delta_error": summarize(all_area_weight, all_delta_error),
        "dx_error": summarize(all_area_weight, all_dx),
        "dy_slope_error": summarize(all_area_weight, all_dy_error),
    },
    "groups": group_summary,
    "pure_quadratic_patch": {
        "potential_error": summarize(
            [v[0] for v in pure_quadratic_values], [v[1] for v in pure_quadratic_values]),
        "dx_error": summarize(
            [v[0] for v in pure_quadratic_values], [v[2] for v in pure_quadratic_values]),
        "dy_error": summarize(
            [v[0] for v in pure_quadratic_values], [v[3] for v in pure_quadratic_values]),
    },
    "old_244_face_audit_comparison": {
        "source_summary": str(old_summary_path),
        "old_face_count": old_summary["face_count"],
        "old_kink_crossing_face_count": old_summary["groups"]["kink_crossing"]["face_count"],
        "old_within_one_interval_face_count": old_summary["groups"]["single_interval"]["face_count"],
        "old_global": old_summary["global"],
        "note": "Read-only comparison to the previously recorded 244-face audit.",
    },
    "reference_run_metrics_not_attributed": {
        "cl_pressure": run_result.get("cl_pressure"),
        "cl_jump": run_result.get("cl_jump"),
        "note": "CL values are copied from result.json for context only; no causal inference is made.",
    },
    "outputs": ["wake_nodes_B.csv", "wake_gauss_points.csv", "wake_faces.csv"],
}
(args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps(summary, indent=2))
