"""Validate a declared trailing-edge polyline against trimmed wing CAD edges."""
import numpy as np


def validate_declared_te(gmsh, wing_solids, te, tolerance):
    te = np.asarray(te, dtype=float)
    if te.ndim != 2 or te.shape[1] != 3 or len(te) < 2:
        raise ValueError("Declared TE must be an N by 3 array with at least two points")
    if not np.isfinite(te).all():
        raise ValueError("Declared TE coordinates must be finite")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("TE tolerance must be finite and positive")
    if len(wing_solids) == 0:
        raise ValueError("At least one wing CAD solid is required")

    wing_solids = [(int(dimension), int(tag)) for dimension, tag in wing_solids]
    if any(dimension != 3 for dimension, _ in wing_solids):
        raise ValueError("wing_solids entries must be dimension-3 CAD entities")

    boundary_surfaces = gmsh.model.getBoundary(
        wing_solids, combined=False, oriented=False)
    boundary_curves = gmsh.model.getBoundary(
        [(int(dimension), int(tag)) for dimension, tag in boundary_surfaces
         if int(dimension) == 2],
        combined=False, oriented=False)
    curve_tags = sorted(set(
        int(tag) for dimension, tag in boundary_curves if int(dimension) == 1))
    if not curve_tags:
        raise ValueError("Wing CAD solids have no boundary curves")

    curve_bounds = {}
    for curve_tag in curve_tags:
        lower, upper = gmsh.model.getParametrizationBounds(1, curve_tag)
        curve_bounds[curve_tag] = (float(lower[0]), float(upper[0]))

    sample_points = []
    sample_segments = []
    sample_positions = []
    for segment_index in range(len(te) - 1):
        for position, fraction in enumerate([0.0, 0.5, 1.0]):
            sample_points.append((1.0 - fraction) * te[segment_index]
                                 + fraction * te[segment_index + 1])
            sample_segments.append(segment_index)
            sample_positions.append(["start", "midpoint", "end"][position])
    sample_points = np.asarray(sample_points)

    records = []
    for sample_index, point in enumerate(sample_points):
        candidate_records = []
        for curve_tag in curve_tags:
            closest, parameters = gmsh.model.getClosestPoint(
                1, curve_tag, point.tolist())
            closest = np.asarray(closest, dtype=float).reshape(-1, 3)[0]
            parameter = float(np.asarray(parameters, dtype=float).ravel()[0])
            lower, upper = curve_bounds[curve_tag]
            parameter_tolerance = 1e-10 * max(1.0, abs(upper - lower))
            parameter_in_bounds = (
                lower - parameter_tolerance <= parameter
                <= upper + parameter_tolerance)
            distance = float(np.linalg.norm(closest - point))
            candidate_records.append({
                "curve_tag": curve_tag,
                "parameter": parameter,
                "parameter_bounds": [lower, upper],
                "parameter_in_bounds": bool(parameter_in_bounds),
                "distance": distance,
            })

        in_bounds = [candidate for candidate in candidate_records
                     if candidate["parameter_in_bounds"]]
        if in_bounds:
            closest_record = min(in_bounds, key=lambda item: item["distance"])
        else:
            closest_record = min(candidate_records, key=lambda item: item["distance"])
        matched = (
            closest_record["parameter_in_bounds"]
            and closest_record["distance"] <= tolerance)
        records.append({
            "segment_index": int(sample_segments[sample_index]),
            "sample_position": sample_positions[sample_index],
            "point": point.tolist(),
            "matched": bool(matched),
            "nearest_boundary_curve": closest_record,
        })

    failed = [record for record in records if not record["matched"]]
    diagnostics = {
        "status": "passed" if not failed else "failed",
        "wing_solid_tags": [tag for _, tag in wing_solids],
        "boundary_curve_tags": curve_tags,
        "te_segment_count": int(len(te) - 1),
        "sample_count": int(len(records)),
        "tolerance": float(tolerance),
        "maximum_trimmed_boundary_distance": float(max(
            record["nearest_boundary_curve"]["distance"] for record in records)),
        "all_samples_on_trimmed_wing_boundary": not failed,
        "samples": records,
    }
    if failed:
        first = failed[0]
        exception = ValueError(
            "Declared TE does not lie on a trimmed wing boundary curve: "
            f"segment {first['segment_index']} {first['sample_position']} "
            f"has distance {first['nearest_boundary_curve']['distance']:.9g}, "
            f"parameter_in_bounds={first['nearest_boundary_curve']['parameter_in_bounds']}, "
            f"tolerance={tolerance:.9g}")
        exception.diagnostics = diagnostics
        raise exception
    return diagnostics
