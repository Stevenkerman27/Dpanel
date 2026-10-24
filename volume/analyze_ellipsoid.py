"""Compare a converged M=0 P2 CAD field with the exact ellipsoid reference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "volume"))
import p2

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mesh", type=Path, required=True)
parser.add_argument("--field", type=Path, required=True)
parser.add_argument("--result", type=Path, required=True)
parser.add_argument("--native-axes", type=float, nargs=3, required=True)
parser.add_argument("--native-center", type=float, nargs=3, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--record-only", action="store_true")
args = parser.parse_args()

metadata_path = args.mesh.with_name("mesh.json")
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
result = json.loads(args.result.read_text(encoding="utf-8"))
if result["status"] != "converged" or result["input"]["mach"] != 0.0:
    raise SystemExit("Exact ellipsoid comparison requires a converged M=0 solve")
native_axes = np.asarray(args.native_axes, dtype=float)
native_center = np.asarray(args.native_center, dtype=float)
if not np.isfinite(native_axes).all() or np.any(native_axes <= 0.0) or not np.isfinite(native_center).all():
    raise SystemExit("Require finite center and positive finite semiaxes")

native_length = metadata.get("native_length")
native_origin = np.asarray(metadata.get("native_origin", [0.0, 0.0, 0.0]), dtype=float)
if native_length is None:
    normalized_axes = native_axes.copy()
    normalized_center = native_center.copy()
else:
    normalized_axes = native_axes / float(native_length)
    normalized_center = (native_center - native_origin) / float(native_length)
if not np.isfinite(normalized_axes).all() or np.any(normalized_axes <= 0.0):
    raise SystemExit("Mesh metadata gives invalid normalized ellipsoid axes")

depolarization = np.empty(3)
for axis in range(3):
    axis_squared = normalized_axes[axis] ** 2
    integrand = lambda s: np.prod(normalized_axes) / (
        2.0 * (axis_squared + s) * np.sqrt(np.prod(normalized_axes**2 + s))
    )
    depolarization[axis] = quad(integrand, 0.0, np.inf, epsabs=1e-12, epsrel=1e-12)[0]

sphere_depolarization = np.empty(3)
for axis in range(3):
    integrand = lambda s: 1.0 / (2.0 * (1.0 + s) ** 2.5)
    sphere_depolarization[axis] = quad(integrand, 0.0, np.inf, epsabs=1e-13, epsrel=1e-13)[0]
sphere_flow = np.array([0.0, 0.0, 1.0])
sphere_normal = np.array([0.3, -0.4, np.sqrt(0.75)])
sphere_normal /= np.linalg.norm(sphere_normal)
sphere_velocity = (np.eye(3) - np.outer(sphere_normal, sphere_normal)) @ (
    sphere_flow / (1.0 - sphere_depolarization)
)
sphere_patch_velocity = 1.5 * (sphere_flow - np.dot(sphere_flow, sphere_normal) * sphere_normal)
sphere_patch_error = float(np.linalg.norm(sphere_velocity - sphere_patch_velocity))
sphere_patch_cp_error = float(
    abs((1.0 - np.dot(sphere_velocity, sphere_velocity))
        - (1.0 - 2.25 * (1.0 - np.dot(sphere_flow, sphere_normal) ** 2)))
)
assert abs(float(np.sum(sphere_depolarization)) - 1.0) < 1e-12
assert np.max(abs(sphere_depolarization - 1.0 / 3.0)) < 1e-12
assert sphere_patch_error < 1e-12 and sphere_patch_cp_error < 1e-12
assert abs(float(np.sum(depolarization)) - 1.0) < 1e-10

with np.load(args.field) as field:
    points = np.asarray(field["points"])
    triangles = np.asarray(field["boundary_triangles"])
    tags = np.asarray(field["boundary_tags"])
    wall = tags == 10
    wall_triangles = triangles[wall]
    shape, _ = p2.triangle_shape(p2.triangle_quadrature(3)[0])
    wall_points = np.einsum("qi,bik->bqk", shape, points[wall_triangles], optimize=True)
    cp_computed = np.asarray(field["surface_cp"])[wall]
    cp_weight = np.asarray(field["surface_area_weight"])[wall]
    raw_wall_normal_velocity = np.asarray(field["surface_normal_velocity"])[wall]
    computed_normals = np.asarray(field["surface_normal"])[wall]

relative_points = (wall_points - normalized_center) / normalized_axes
ellipsoid_level_error = np.sum(relative_points**2, axis=2) - 1.0
reference_normals = relative_points / normalized_axes
reference_normals /= np.linalg.norm(reference_normals, axis=2)[:, :, None]

alpha_deg = float(result["input"]["alpha"])
alpha = np.deg2rad(alpha_deg)
freestream = np.array([np.cos(alpha), 0.0, np.sin(alpha)])
scaled_freestream = freestream / (1.0 - depolarization)
reference_velocity = scaled_freestream - (
    np.einsum("bqi,i->bq", reference_normals, scaled_freestream)[:, :, None] * reference_normals
)
reference_cp = 1.0 - np.sum(reference_velocity**2, axis=2)
cp_error = cp_computed - reference_cp
cp_rms = float(np.sqrt(np.sum(cp_weight * cp_error**2) / np.sum(cp_weight)))
raw_wall_vn_rms = float(
    np.sqrt(np.sum(cp_weight * raw_wall_normal_velocity**2) / np.sum(cp_weight))
)
normal_alignment = np.sum(computed_normals * reference_normals, axis=2)
normal_axis_error = np.minimum(
    np.sum((computed_normals - reference_normals) ** 2, axis=2),
    np.sum((computed_normals + reference_normals) ** 2, axis=2),
)
normal_axis_rms = float(np.sqrt(np.sum(cp_weight * normal_axis_error) / np.sum(cp_weight)))
normal_alignment_mean = float(np.sum(cp_weight * normal_alignment) / np.sum(cp_weight))
level_rms = float(np.sqrt(np.sum(cp_weight * ellipsoid_level_error**2) / np.sum(cp_weight)))
level_max = float(np.max(np.abs(ellipsoid_level_error)))
body_load = result["loads"]["by_tag"]["10"]["force_over_sref"]
pressure_cl = float(result["loads"]["lift_over_sref"])
pressure_cd = float(result["loads"]["drag_over_sref"])
gates = {
    "cp_area_weighted_rms_below_0p03": cp_rms < 0.03,
    "raw_wall_normal_velocity_rms_below_0p03": raw_wall_vn_rms < 0.03,
    "abs_pressure_CL_below_0p005": abs(pressure_cl) < 0.005,
    "abs_pressure_CD_below_0p005": abs(pressure_cd) < 0.005,
}
analysis = {
    "status": "passed" if all(gates.values()) else ("recorded_gate_failure" if args.record_only else "failed"),
    "scope": "Independent M=0 exact ellipsoid Cp comparison; no analytic velocity is imposed on the solver",
    "mesh": str(args.mesh.resolve()),
    "field": str(args.field.resolve()),
    "solve_result": str(args.result.resolve()),
    "mesh_sha256": hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
    "field_sha256": hashlib.sha256(args.field.read_bytes()).hexdigest(),
    "solve_result_sha256": hashlib.sha256(args.result.read_bytes()).hexdigest(),
    "alpha_deg": alpha_deg,
    "native_axes": native_axes.tolist(),
    "native_center": native_center.tolist(),
    "native_origin_from_mesh_metadata": native_origin.tolist(),
    "native_length_from_mesh_metadata": native_length,
    "normalized_axes": normalized_axes.tolist(),
    "normalized_center": normalized_center.tolist(),
    "depolarization_factors": depolarization.tolist(),
    "depolarization_sum_error": float(np.sum(depolarization) - 1.0),
    "sphere_patch": {
        "factors": sphere_depolarization.tolist(),
        "factor_sum": float(np.sum(sphere_depolarization)),
        "tangent_velocity_absolute_error": sphere_patch_error,
        "cp_absolute_error": sphere_patch_cp_error,
    },
    "wall_quadrature_samples": int(cp_computed.size),
    "cp_area_weighted_rms_error": cp_rms,
    "raw_wall_normal_velocity_area_weighted_rms": raw_wall_vn_rms,
    "computed_fluid_outward_vs_body_outward_normal_dot_mean": normal_alignment_mean,
    "computed_vs_reference_unoriented_normal_axis_rms": normal_axis_rms,
    "ellipsoid_level_error_area_weighted_rms": level_rms,
    "ellipsoid_level_error_max_abs": level_max,
    "pressure_load": {
        "CL": pressure_cl,
        "CD": pressure_cd,
        "CS": float(body_load[1]),
        "force_over_sref_xyz": body_load,
    },
    "gates": gates,
    "reference": "N_i=abc/2 integral_0_inf ds/((a_i^2+s)*sqrt(product_j(a_j^2+s))); q=(I-n*n^T)@(U/(1-N)); Cp=1-|q|^2",
}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(analysis, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps(analysis, indent=2))
if analysis["status"] == "failed":
    raise SystemExit(1)
