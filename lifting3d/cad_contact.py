"""Discover wall/wake CAD contact curves and their spanwise projections."""
import numpy as np
from scipy.optimize import brentq


def validate_te_polyline(te_polyline):
    te_polyline = np.asarray(te_polyline, dtype=float)
    if te_polyline.ndim != 2 or te_polyline.shape[1] != 3 or len(te_polyline) < 2:
        raise ValueError("TE polyline must be an N by 3 array with at least two points")
    if not np.isfinite(te_polyline).all():
        raise ValueError("TE polyline coordinates must be finite")
    if np.any(np.diff(te_polyline[:, 1]) <= 0.0):
        raise ValueError("TE polyline y coordinates must be strictly increasing")
    if np.any(te_polyline[:, 2] != 0.0):
        raise ValueError("TE polyline must lie exactly in z=0")
    return te_polyline


def _curve_points(curve_tag, parameters):
    return np.asarray(gmsh.model.getValue(1, int(curve_tag), np.asarray(parameters).ravel().tolist())
                      ).reshape(-1, 3)


def _te_distance(points, te_polyline):
    clamped_y = np.clip(points[:, 1], te_polyline[0, 1], te_polyline[-1, 1])
    te_x = np.interp(clamped_y, te_polyline[:, 1], te_polyline[:, 0])
    te_points = np.column_stack([te_x, clamped_y, np.zeros(len(points))])
    return np.linalg.norm(points - te_points, axis=1)


def _curve_projection(curve_tag, endpoint_point_tags, te_polyline, scan_samples, te_tolerance):
    parameter_min, parameter_max = gmsh.model.getParametrizationBounds(1, curve_tag)
    parameter_min = float(parameter_min[0])
    parameter_max = float(parameter_max[0])
    parameters = np.linspace(parameter_min, parameter_max, scan_samples)
    sampled_points = _curve_points(curve_tag, parameters)
    sampled_dy = np.asarray([
        gmsh.model.getDerivative(1, int(curve_tag), [float(parameter)])[1]
        for parameter in parameters
    ])

    stationary = []
    for index in range(scan_samples - 1):
        left_parameter, right_parameter = parameters[index:index + 2]
        left_dy, right_dy = sampled_dy[index:index + 2]
        if left_dy == 0.0 and parameter_min < left_parameter < parameter_max:
            stationary.append(float(left_parameter))
        if left_dy * right_dy < 0.0:
            root = brentq(
                lambda parameter: gmsh.model.getDerivative(
                    1, int(curve_tag), [float(parameter)])[1],
                float(left_parameter), float(right_parameter), xtol=1e-13,
            )
            stationary.append(float(root))
    if sampled_dy[-1] == 0.0 and parameter_min < parameters[-1] < parameter_max:
        stationary.append(float(parameters[-1]))
    stationary.sort()
    unique_stationary = []
    for parameter in stationary:
        if not unique_stationary or abs(parameter - unique_stationary[-1]) > 1e-10 * max(
                1.0, abs(parameter_max - parameter_min)):
            unique_stationary.append(parameter)

    stationary_points = _curve_points(curve_tag, unique_stationary)
    endpoint_coordinates = np.asarray([
        gmsh.model.getValue(0, int(point_tag), []) for point_tag in endpoint_point_tags
    ]).reshape(-1, 3) if endpoint_point_tags else np.empty((0, 3))
    extreme_points = np.vstack([endpoint_coordinates, stationary_points])
    if len(extreme_points) == 0:
        extreme_points = sampled_points[[0, -1]]

    te_distances = _te_distance(sampled_points, te_polyline)
    stationary_records = []
    for parameter, point in zip(unique_stationary, stationary_points):
        sample_index = int(np.argmin(abs(parameters - parameter)))
        left_index = max(0, sample_index - 1)
        right_index = min(scan_samples - 1, sample_index + 1)
        stationary_records.append({
            "parameter": parameter,
            "y": float(point[1]),
            "abs_dy_dparameter": float(abs(gmsh.model.getDerivative(
                1, int(curve_tag), [parameter])[1])),
            "parameter_bracket": [
                float(parameters[left_index]), float(parameters[right_index]),
            ],
            "bracket_y_sample_range": float(
                np.ptp(sampled_points[left_index:right_index + 1, 1])),
        })
    return {
        "curve_tag": int(curve_tag),
        "endpoint_point_tags": [int(point_tag) for point_tag in endpoint_point_tags],
        "parameter_bounds": [parameter_min, parameter_max],
        "scan_sample_count": int(scan_samples),
        "maximum_sampled_y_step": float(np.max(abs(np.diff(sampled_points[:, 1])))),
        "sampled_y_min": float(sampled_points[:, 1].min()),
        "sampled_y_max": float(sampled_points[:, 1].max()),
        "projection_y_min": float(extreme_points[:, 1].min()),
        "projection_y_max": float(extreme_points[:, 1].max()),
        "stationary_points": stationary_records,
        "te_polyline_max_sample_distance": float(te_distances.max()),
        "is_user_te_polyline": bool(te_distances.max() <= te_tolerance),
        "te_polyline_tolerance": float(te_tolerance),
    }


def discover_contact_intervals(wall_surface_tags, wake_surface_tags, te_polyline,
                               scan_samples=129, te_tolerance=1e-7,
                               merge_tolerance=1e-9):
    """Return merged y intervals and per-curve/component CAD diagnostics.

    Gmsh must be initialized, the OCC model synchronized, and surface tags must
    refer to the final fluid wall and internal wake surfaces.
    """
    global gmsh
    try:
        import gmsh
    except ImportError as exception:
        raise RuntimeError("Import gmsh from the project's configured runtime before CAD discovery") from exception

    te_polyline = validate_te_polyline(te_polyline)
    if scan_samples < 3 or te_tolerance < 0.0 or merge_tolerance < 0.0:
        raise ValueError("Require at least 3 curve samples and nonnegative geometric tolerances")

    wall_curve_surfaces = {}
    wake_curve_surfaces = {}
    wall_embedded_curve_surfaces = {}
    wake_embedded_curve_surfaces = {}
    for surface_tag in wall_surface_tags:
        curves = gmsh.model.getBoundary(
            [(2, int(surface_tag))], combined=False, oriented=False)
        for dimension, curve_tag in curves:
            if dimension == 1:
                wall_curve_surfaces.setdefault(int(curve_tag), []).append(int(surface_tag))
        embedded = gmsh.model.mesh.getEmbedded(2, int(surface_tag))
        for dimension, curve_tag in embedded:
            if dimension == 1:
                wall_embedded_curve_surfaces.setdefault(
                    int(curve_tag), []).append(int(surface_tag))
    for surface_tag in wake_surface_tags:
        curves = gmsh.model.getBoundary(
            [(2, int(surface_tag))], combined=False, oriented=False)
        for dimension, curve_tag in curves:
            if dimension == 1:
                wake_curve_surfaces.setdefault(int(curve_tag), []).append(int(surface_tag))
        embedded = gmsh.model.mesh.getEmbedded(2, int(surface_tag))
        for dimension, curve_tag in embedded:
            if dimension == 1:
                wake_embedded_curve_surfaces.setdefault(
                    int(curve_tag), []).append(int(surface_tag))

    wall_all_curve_surfaces = {}
    wake_all_curve_surfaces = {}
    for curve_map in [wall_curve_surfaces, wall_embedded_curve_surfaces]:
        for curve_tag, surface_tags in curve_map.items():
            wall_all_curve_surfaces.setdefault(curve_tag, []).extend(surface_tags)
    for curve_map in [wake_curve_surfaces, wake_embedded_curve_surfaces]:
        for curve_tag, surface_tags in curve_map.items():
            wake_all_curve_surfaces.setdefault(curve_tag, []).extend(surface_tags)
    shared_curves = sorted(set(wall_all_curve_surfaces) & set(wake_all_curve_surfaces))
    curve_records = {}
    curve_endpoints = {}
    for curve_tag in shared_curves:
        endpoint_entities = gmsh.model.getBoundary(
            [(1, curve_tag)], combined=False, oriented=False)
        endpoint_tags = sorted(set(int(tag) for dimension, tag in endpoint_entities
                                   if dimension == 0))
        curve_endpoints[curve_tag] = endpoint_tags
        record = _curve_projection(
            curve_tag, endpoint_tags, te_polyline, scan_samples, te_tolerance)
        record["wall_surface_tags"] = sorted(set(wall_all_curve_surfaces[curve_tag]))
        record["wake_surface_tags"] = sorted(set(wake_all_curve_surfaces[curve_tag]))
        record["wall_curve_sources"] = [
            source for source, curve_map in [
                ("boundary", wall_curve_surfaces),
                ("embedded", wall_embedded_curve_surfaces),
            ] if curve_tag in curve_map]
        record["wake_curve_sources"] = [
            source for source, curve_map in [
                ("boundary", wake_curve_surfaces),
                ("embedded", wake_embedded_curve_surfaces),
            ] if curve_tag in curve_map]
        record["excluded_reason"] = "matches_user_te_polyline" if record["is_user_te_polyline"] else None
        curve_records[curve_tag] = record

    contact_curves = [
        curve_tag for curve_tag in shared_curves
        if not curve_records[curve_tag]["is_user_te_polyline"]
    ]
    parent = {curve_tag: curve_tag for curve_tag in contact_curves}

    def find(curve_tag):
        while parent[curve_tag] != curve_tag:
            parent[curve_tag] = parent[parent[curve_tag]]
            curve_tag = parent[curve_tag]
        return curve_tag

    endpoint_curves = {}
    for curve_tag in contact_curves:
        endpoint_tags = curve_endpoints[curve_tag]
        for point_tag in endpoint_tags:
            endpoint_curves.setdefault(point_tag, []).append(curve_tag)
    for touching_curves in endpoint_curves.values():
        for curve_tag in touching_curves[1:]:
            parent[find(curve_tag)] = find(touching_curves[0])

    components = {}
    for curve_tag in contact_curves:
        components.setdefault(find(curve_tag), []).append(curve_tag)

    component_records = []
    for component_index, component_curves in enumerate(components.values()):
        component_record = {
            "component_index": int(component_index),
            "all_curve_tags": sorted(component_curves),
            "included_curve_tags": sorted(component_curves),
            "endpoint_point_tags": sorted(set(
                point_tag for curve_tag in component_curves
                for point_tag in curve_endpoints[curve_tag])),
            "status": "contact_candidate",
        }
        component_record["y_interval"] = [
            min(curve_records[curve_tag]["projection_y_min"] for curve_tag in component_curves),
            max(curve_records[curve_tag]["projection_y_max"] for curve_tag in component_curves),
        ]
        component_records.append(component_record)
    for curve_tag in shared_curves:
        if curve_records[curve_tag]["is_user_te_polyline"]:
            component_records.append({
                "component_index": int(len(component_records)),
                "all_curve_tags": [int(curve_tag)],
                "included_curve_tags": [],
                "excluded_te_curve_tags": [int(curve_tag)],
                "endpoint_point_tags": curve_endpoints[curve_tag],
                "status": "excluded_te_only",
            })

    sorted_components = sorted(
        [record for record in component_records if record["status"] == "contact_candidate"],
        key=lambda record: record["y_interval"][0],
    )
    merged_intervals = []
    merged_component_indices = []
    for record in sorted_components:
        lower, upper = record["y_interval"]
        if not merged_intervals or lower > merged_intervals[-1][1] + merge_tolerance:
            merged_intervals.append([lower, upper])
            merged_component_indices.append([record["component_index"]])
        else:
            merged_intervals[-1][1] = max(merged_intervals[-1][1], upper)
            merged_component_indices[-1].append(record["component_index"])

    for component_index, record in enumerate(component_records):
        if record["status"] == "contact_candidate":
            record["merged_interval_index"] = next(
                index for index, indices in enumerate(merged_component_indices)
                if component_index in indices)

    diagnostics = {
        "status": "contacts_found" if merged_intervals else "no_wall_wake_contact_after_te_exclusion",
        "wall_surface_tags": [int(tag) for tag in wall_surface_tags],
        "wake_surface_tags": [int(tag) for tag in wake_surface_tags],
        "te_polyline_point_count": int(len(te_polyline)),
        "curve_scan_samples": int(scan_samples),
        "curve_endpoint_topology": "Only non-TE curves are connected into components when they share an OCC endpoint point tag; TE curves are excluded before union.",
        "projection_method": (
            "Curve endpoint y values plus sign-changing roots of OCC dy/du found by uniform parameter scan "
            "and scipy brentq; sampled extrema and brackets are retained per curve."
        ),
        "sampling_error_note": (
            "maximum_sampled_y_step and bracketing y sample range are resolution diagnostics, "
            "not certified global error bounds for arbitrary CAD curves."
        ),
        "shared_curve_count": int(len(shared_curves)),
        "shared_curve_source_counts": {
            "wall_boundary_wake_boundary": int(len(
                set(wall_curve_surfaces) & set(wake_curve_surfaces))),
            "wall_embedded_wake_boundary": int(len(
                set(wall_embedded_curve_surfaces) & set(wake_curve_surfaces))),
            "wall_boundary_wake_embedded": int(len(
                set(wall_curve_surfaces) & set(wake_embedded_curve_surfaces))),
            "wall_embedded_wake_embedded": int(len(
                set(wall_embedded_curve_surfaces) & set(wake_embedded_curve_surfaces))),
        },
        "surface_curve_discovery": (
            "Union of OCC surface boundary curves and Gmsh mesh embedded curves "
            "from getEmbedded(2, surface_tag); a shared curve must occur on both "
            "a selected wall surface and a selected wake surface."
        ),
        "excluded_te_curve_count": int(sum(
            record["is_user_te_polyline"] for record in curve_records.values())),
        "contact_component_count": int(sum(
            record["status"] == "contact_candidate" for record in component_records)),
        "merged_intervals": merged_intervals,
        "merged_component_indices": merged_component_indices,
        "components": component_records,
        "curves": [curve_records[curve_tag] for curve_tag in shared_curves],
    }
    return merged_intervals, diagnostics
