import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad


ROOT = Path(__file__).resolve().parent / "reference"
ROOT.mkdir(parents=True, exist_ok=True)

levels = (2, 3, 4)
meshes = {}
axes_by_case = {
    "sphere": np.array([1.0, 1.0, 1.0]),
    "prolate_ellipsoid": np.array([3.0, 1.0, 1.0]),
}
center = np.zeros(3)
scale = 1.0
rotation = np.eye(3)
wake_lines = np.empty((0,), dtype=np.float64)

base_xyz = np.array(
    [
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
    ],
    dtype=np.float64,
)
base_tri = np.array(
    [
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
        [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
    ],
    dtype=np.int64,
)

for face_index in range(len(base_tri)):
    face = base_tri[face_index]
    face_normal = np.cross(base_xyz[face[1]] - base_xyz[face[0]], base_xyz[face[2]] - base_xyz[face[0]])
    if np.dot(face_normal, base_xyz[face].mean(axis=0)) < 0.0:
        base_tri[face_index, 1], base_tri[face_index, 2] = base_tri[face_index, 2], base_tri[face_index, 1]

for level in levels:
    xyz = base_xyz.copy()
    tri = base_tri.copy()
    for subdivision in range(level):
        midpoint_cache = {}
        refined_tri = []
        for face in tri:
            midpoint_indices = []
            for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                edge_key = tuple(sorted(edge))
                if edge_key not in midpoint_cache:
                    midpoint = (xyz[edge[0]] + xyz[edge[1]]) * 0.5
                    midpoint = midpoint / np.linalg.norm(midpoint)
                    midpoint_cache[edge_key] = len(xyz)
                    xyz = np.vstack((xyz, midpoint))
                midpoint_indices.append(midpoint_cache[edge_key])
            ab, bc, ca = midpoint_indices
            refined_tri.extend(
                (
                    (face[0], ab, ca),
                    (ab, face[1], bc),
                    (ca, bc, face[2]),
                    (ab, bc, ca),
                )
            )
        tri = np.asarray(refined_tri, dtype=np.int64)

    meshes[level] = (xyz.copy(), tri.copy())
    for shape_name, axes in axes_by_case.items():
        shape_xyz = xyz * axes
        shape_center = center.copy()
        shape_scale = float(scale)
        shape_rotation = rotation.copy()
        metadata = {
            "case": shape_name,
            "subdivision_level": level,
            "face_count": int(len(tri)),
            "node_count": int(len(shape_xyz)),
            "axes_semiaxes": axes.tolist(),
            "center": shape_center.tolist(),
            "scale": shape_scale,
            "rotation": shape_rotation.tolist(),
            "geometry_transform": "xyz = center + scale * (base_xyz @ rotation.T)",
            "units": "dimensionless; sphere radius = 1",
            "flow_reference": "U_inf = (1, 0, 0); rho = 1",
        }
        np.savez_compressed(
            ROOT / f"{shape_name}_L{level}.npz",
            xyz=shape_xyz.astype(np.float64),
            tri=tri.astype(np.int64),
            tags=np.ones(len(tri), dtype=np.int64),
            wake_lines=wake_lines,
            axes=axes.astype(np.float64),
            center=shape_center.astype(np.float64),
            scale=np.asarray(shape_scale, dtype=np.float64),
            rotation=shape_rotation.astype(np.float64),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )

xyz, tri = meshes[3]

axis = np.array([1.0, 2.0, 3.0])
axis = axis / np.linalg.norm(axis)
theta = np.deg2rad(37.0)
rotation_37 = np.array(
    [
        [np.cos(theta) + axis[0] ** 2 * (1.0 - np.cos(theta)),
         axis[0] * axis[1] * (1.0 - np.cos(theta)) - axis[2] * np.sin(theta),
         axis[0] * axis[2] * (1.0 - np.cos(theta)) + axis[1] * np.sin(theta)],
        [axis[1] * axis[0] * (1.0 - np.cos(theta)) + axis[2] * np.sin(theta),
         np.cos(theta) + axis[1] ** 2 * (1.0 - np.cos(theta)),
         axis[1] * axis[2] * (1.0 - np.cos(theta)) - axis[0] * np.sin(theta)],
        [axis[2] * axis[0] * (1.0 - np.cos(theta)) - axis[1] * np.sin(theta),
         axis[2] * axis[1] * (1.0 - np.cos(theta)) + axis[0] * np.sin(theta),
         np.cos(theta) + axis[2] ** 2 * (1.0 - np.cos(theta))],
    ],
    dtype=np.float64,
)
variants = (
    ("sphere_L3_scale_0p001", np.zeros(3), 0.001, np.eye(3)),
    ("sphere_L3_scale_1000", np.zeros(3), 1000.0, np.eye(3)),
    ("sphere_L3_translate", np.array([100.0, -30.0, 20.0]), 1.0, np.eye(3)),
    ("sphere_L3_rotate_axis_1_2_3_deg37", np.zeros(3), 1.0, rotation_37),
)
for case_name, transform_center, transform_scale, transform_rotation in variants:
    transformed_xyz = transform_center + transform_scale * (xyz @ transform_rotation.T)
    metadata = {
        "case": case_name,
        "base_case": "sphere_L3",
        "subdivision_level": 3,
        "face_count": int(len(tri)),
        "node_count": int(len(transformed_xyz)),
        "axes_semiaxes_before_transform": [1.0, 1.0, 1.0],
        "center": transform_center.tolist(),
        "scale": transform_scale,
        "rotation": transform_rotation.tolist(),
        "rotation_axis": axis.tolist() if "rotate" in case_name else None,
        "rotation_angle_degrees": 37.0 if "rotate" in case_name else 0.0,
        "geometry_transform": "xyz = center + scale * (base_xyz @ rotation.T)",
        "units": "base sphere radius = 1; geometric transform only",
    }
    np.savez_compressed(
        ROOT / f"{case_name}.npz",
        xyz=transformed_xyz.astype(np.float64),
        tri=tri.astype(np.int64),
        tags=np.ones(len(tri), dtype=np.int64),
        wake_lines=wake_lines,
        axes=np.array([1.0, 1.0, 1.0], dtype=np.float64),
        center=transform_center.astype(np.float64),
        scale=np.asarray(transform_scale, dtype=np.float64),
        rotation=transform_rotation.astype(np.float64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )

ellipsoid_axes = np.array([3.0, 1.0, 1.0])
depolarization = np.empty(3, dtype=np.float64)
for axis_index in range(3):
    other_indices = [index for index in range(3) if index != axis_index]
    axis_squared = ellipsoid_axes[axis_index] ** 2
    other_squared_1 = ellipsoid_axes[other_indices[0]] ** 2
    other_squared_2 = ellipsoid_axes[other_indices[1]] ** 2
    integrand = lambda s: ellipsoid_axes.prod() / (
        2.0 * (axis_squared + s) * np.sqrt((ellipsoid_axes**2 + s).prod())
    )
    depolarization[axis_index] = quad(integrand, 0.0, np.inf, epsabs=1e-12, epsrel=1e-12)[0]

flow = np.array([1.0, 0.0, 0.0])
surface_directions = np.array(
    [
        [1.0, 0.3, -0.2],
        [-0.4, 0.7, 0.5],
        [0.2, -0.3, 0.9],
        [0.6, 0.2, 0.7],
    ],
    dtype=np.float64,
)
surface_directions = surface_directions / np.linalg.norm(surface_directions, axis=1)[:, None]
surface_points = ellipsoid_axes * surface_directions
surface_normals = surface_points / (ellipsoid_axes**2)
surface_normals = surface_normals / np.linalg.norm(surface_normals, axis=1)[:, None]
ellipsoid_samples = []
normal_derivative_max = 0.0
surface_velocity_error_max = 0.0
for point_index in range(len(surface_points)):
    point = surface_points[point_index]
    normal = surface_normals[point_index]
    lambda_value = 0.0
    depolarization_at_lambda = np.empty(3, dtype=np.float64)
    depolarization_derivative = np.empty(3, dtype=np.float64)
    for axis_index in range(3):
        axis_squared = ellipsoid_axes[axis_index] ** 2
        integrand = lambda s: ellipsoid_axes.prod() / (
            2.0 * (axis_squared + s) * np.sqrt((ellipsoid_axes**2 + s).prod())
        )
        depolarization_at_lambda[axis_index] = quad(
            integrand, lambda_value, np.inf, epsabs=1e-12, epsrel=1e-12
        )[0]
        depolarization_derivative[axis_index] = -ellipsoid_axes.prod() / (
            2.0 * (axis_squared + lambda_value) * np.sqrt((ellipsoid_axes**2 + lambda_value).prod())
        )

    lambda_gradient = 2.0 * point / (ellipsoid_axes**2 + lambda_value)
    lambda_gradient = lambda_gradient / np.sum(point**2 / (ellipsoid_axes**2 + lambda_value) ** 2)
    disturbance_coefficients = depolarization_at_lambda / (1.0 - depolarization)
    disturbance_potential = np.sum(flow * point * disturbance_coefficients)
    total_potential = np.dot(flow, point) + disturbance_potential
    total_velocity = flow * (1.0 + disturbance_coefficients)
    total_velocity += np.sum(
        flow * point * depolarization_derivative / (1.0 - depolarization)
    ) * lambda_gradient
    total_normal_velocity = np.dot(total_velocity, normal)
    tangent_velocity = total_velocity - total_normal_velocity * normal
    closed_form_velocity = (
        np.eye(3) - np.outer(normal, normal)
    ) @ (flow / (1.0 - depolarization))
    normal_derivative_max = max(normal_derivative_max, abs(total_normal_velocity))
    surface_velocity_error_max = max(
        surface_velocity_error_max,
        np.linalg.norm(tangent_velocity - closed_form_velocity),
    )
    ellipsoid_samples.append(
        {
            "xyz": point.tolist(),
            "normal": normal.tolist(),
            "total_potential": float(total_potential),
            "total_velocity": total_velocity.tolist(),
            "total_potential_normal_derivative": float(total_normal_velocity),
            "closed_form_tangent_velocity": closed_form_velocity.tolist(),
        }
    )
assert normal_derivative_max < 1e-10
assert surface_velocity_error_max < 1e-10

with (ROOT / "analytic_reference.json").open("w", encoding="utf-8") as reference_file:
    json.dump(
        {
            "flow_reference": {"U_inf": [1.0, 0.0, 0.0], "rho": 1.0},
            "sphere": {
                "radius": 1.0,
                "surface_velocity": "1.5 * (U - (U dot n) * n)",
                "Cp": "1 - dot(V_surface, V_surface) / dot(U, U)",
                "force": [0.0, 0.0, 0.0],
            },
            "prolate_ellipsoid_axes": ellipsoid_axes.tolist(),
            "depolarization_factors": depolarization.tolist(),
            "ellipsoid_exterior_total_potential": "Phi = U dot x + psi; psi = sum_i(U_i*x_i*L_i(lambda)/(1-L_i(0)))",
            "confocal_coordinate": "sum_i(x_i^2/(a_i^2+lambda)) = 1",
            "depolarization_at_lambda": "L_i(lambda) = abc/2 * integral_lambda_inf ds / ((a_i^2+s)*sqrt(product_j(a_j^2+s)))",
            "ellipsoid_surface_potential": "sum_i(U_i * x_i / (1 - L_i(0)))",
            "ellipsoid_surface_velocity": "P_tangent * diag(1/(1-L_i)) * U",
            "ellipsoid_Cp": "1 - dot(V_surface, V_surface) / dot(U, U)",
            "ellipsoid_surface_checks": {
                "points": ellipsoid_samples,
                "max_abs_total_potential_normal_derivative": normal_derivative_max,
                "max_surface_velocity_error_vs_closed_form": surface_velocity_error_max,
                "tolerance": 1e-10,
            },
            "reference_force": [0.0, 0.0, 0.0],
            "integral": "L_i = abc/2 * integral_0_inf ds / ((a_i^2+s)*sqrt(product_j(a_j^2+s)))",
            "quadrature": "scipy.integrate.quad, epsabs=epsrel=1e-12",
        },
        reference_file,
        indent=2,
        sort_keys=True,
    )
