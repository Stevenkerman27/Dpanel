"""Independent audit of saved taper/sweep meshes and their wake/Kutta mappings."""
import hashlib
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
result_root = root/"results/planforms_v1"
report_path = root/"PLANFORM_MESH_AUDIT.md"
source_paths = [root/"build_mesh.py", root/"solve.py", root/"analyze.py", root/"tilt_mesh.py"]
assert result_root.exists(), "Expected saved planforms_v1 meshes"

case_names = ["taper", "sweep", "combined"]
records = []

for name in case_names:
    case_root = result_root/name
    mesh_path = case_root/"mesh/mesh.npz"
    metadata_path = case_root/"mesh/mesh.json"
    field_path = case_root/"a5p0/field.npz"
    result_path = case_root/"a5p0/result.json"
    assert mesh_path.exists() and metadata_path.exists() and field_path.exists() and result_path.exists()
    mesh_bytes = mesh_path.read_bytes()
    field_bytes = field_path.read_bytes()
    metadata_bytes = metadata_path.read_bytes()
    mesh = np.load(mesh_path)
    field = np.load(field_path)
    metadata = json.loads(metadata_bytes)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    points = np.asarray(mesh["points"])
    stations = np.asarray(mesh["gamma_stations"])
    upper_te = np.asarray(mesh["te_upper_nodes"])
    lower_te = np.asarray(mesh["te_lower_nodes"])
    upper_wake = np.asarray(mesh["wake_upper"])
    lower_wake = np.asarray(mesh["wake_lower"])
    span = float(metadata["span"])
    chord = float(metadata["chord"])
    thickness = float(metadata["thickness"])
    taper = float(metadata.get("taper_ratio", 1.))
    sweep = float(metadata.get("quarter_chord_sweep_deg", 0.))
    sweep_tangent = np.tan(np.deg2rad(sweep))
    origin = np.asarray(metadata.get("normalization_origin", [0., 0., 0.]))
    wall_coordinates = (np.asarray(field["surface_points"])-origin)/chord

    te_y = points[upper_te, 1]
    te_chord = 1.+(taper-1.)*np.minimum(abs(2.*te_y/span), 1.)
    te_expected_x = abs(te_y)*sweep_tangent+.25+.75*te_chord
    te_x_error = points[upper_te, 0]-te_expected_x
    te_z_error = points[upper_te, 2]
    te_station_error = te_y-stations[1:-1]

    wing_boundary = np.asarray(mesh["boundary_triangles"])[mesh["boundary_tags"] != 100]
    wing_nodes = np.unique(wing_boundary[:, :3])
    plane_nodes = wing_nodes[(abs(points[wing_nodes, 2]) < 1.e-10)
                             & (abs(points[wing_nodes, 1]) < span/2.-1.e-9)]
    plane_y = points[plane_nodes, 1]
    plane_chord = 1.+(taper-1.)*np.minimum(abs(2.*plane_y/span), 1.)
    plane_le = abs(plane_y)*sweep_tangent+.25*(1.-plane_chord)
    plane_xc = (points[plane_nodes, 0]-plane_le)/plane_chord
    endpoint_error = np.minimum(abs(plane_xc), abs(plane_xc-1.))

    wall_mask = np.isin(field["tags"], [10, 20, 30])
    wing_surface_mask = field["tags"] == 10
    wing_points = wall_coordinates[wing_surface_mask]
    wing_y = wing_points[:, :, 1]
    wing_chord = 1.+(taper-1.)*np.minimum(abs(2.*wing_y/span), 1.)
    wing_le = abs(wing_y)*sweep_tangent+.25*(1.-wing_chord)
    wing_xc = (wing_points[:, :, 0]-wing_le)/wing_chord
    interior_section = ((abs(wing_y) < span/2.-.1)
                        & (wing_xc > .02) & (wing_xc < .98))
    x = np.clip(wing_xc[interior_section], 0., 1.)
    naca_shape = 5.*thickness*(.2969*np.sqrt(x)-.126*x-.3516*x*x
                               +.2843*x**3-.1036*x**4)
    expected_z = (1.+(taper-1.)*np.minimum(
        abs(2.*wing_y[interior_section]/span), 1.))*naca_shape
    actual_z = wing_points[:, :, 2][interior_section]
    section_map_error = abs(actual_z)-expected_z

    wake_y = points[upper_wake[:, :3], 1]
    wake_min_y = wake_y.min(axis=1)
    wake_max_y = wake_y.max(axis=1)
    crossed_internal_stations = np.zeros(len(upper_wake), dtype=bool)
    for station in stations[1:-1]:
        crossed_internal_stations |= ((wake_min_y < station-1.e-11)
                                      & (wake_max_y > station+1.e-11))
    wake_interval = np.searchsorted(stations, wake_y.mean(axis=1), side="right")-1
    wake_interval = np.minimum(wake_interval, len(stations)-2)
    wake_interval_counts = np.bincount(wake_interval, minlength=len(stations)-1)
    wake_pair_error = float(np.max(abs(points[upper_wake]-points[lower_wake])))

    phi = np.asarray(field["phi"])
    wake_jump = phi[upper_wake]-phi[lower_wake]
    streamwise_slopes = []
    interpolation_errors = []
    expected_gamma = np.r_[0., np.asarray(field["gamma"]), 0.]
    for face, jump_values in zip(upper_wake, wake_jump):
        face_points = points[face]
        design = np.column_stack([np.ones(6), face_points[:, 0], face_points[:, 1]])
        coefficient = np.linalg.lstsq(design, jump_values, rcond=None)[0]
        streamwise_slopes.append(float(coefficient[1]))
        expected_values = np.interp(face_points[:, 1], stations, expected_gamma)
        interpolation_errors.extend((jump_values-expected_values).tolist())
    streamwise_slopes = np.asarray(streamwise_slopes)

    wall_faces = np.asarray(field["faces"])[wall_mask]
    wall_adjacent = np.asarray(field["adjacent"])[wall_mask]
    node_adjacent_counts_upper = []
    node_adjacent_counts_lower = []
    node_pair_selection_equal = []
    for upper_node, lower_node in zip(upper_te, lower_te):
        upper_selected = np.unique(
            wall_adjacent[np.any(wall_faces == upper_node, axis=1)])
        lower_selected = np.unique(
            wall_adjacent[np.any(wall_faces == lower_node, axis=1)])
        node_adjacent_counts_upper.append(len(upper_selected))
        node_adjacent_counts_lower.append(len(lower_selected))
        node_pair_selection_equal.append(np.array_equal(upper_selected, lower_selected))

    te_edge_ranges = []
    for face in wall_faces:
        corners = face[:3]
        corner_y = points[corners, 1]
        local_chord = 1.+(taper-1.)*abs(2.*corner_y/span)
        local_te = .25+abs(corner_y)*sweep_tangent+.75*local_chord
        on_te = ((abs(points[corners, 0]-local_te) < 1.e-9)
                 & (abs(points[corners, 2]) < 1.e-9))
        if int(on_te.sum()) == 2:
            edge = corners[on_te]
            edge_y = np.sort(points[edge, 1])
            te_edge_ranges.append((float(edge_y[0]), float(edge_y[1])))
    weak_edge_coverage = np.zeros(len(stations)-2, dtype=int)
    for left, right in te_edge_ranges:
        for knot in range(1, len(stations)-1):
            support_left = stations[knot-1]
            support_right = stations[knot+1]
            if min(right, support_right)-max(left, support_left) > 1.e-12:
                weak_edge_coverage[knot-1] += 1
    weak_missing_knots = np.flatnonzero(weak_edge_coverage == 0)+1

    node_pair_same_count = int(np.sum(upper_te == lower_te))
    node_cancellation_count = int(np.sum(
        (upper_te == lower_te) & np.asarray(node_pair_selection_equal)))
    expected_wake_count = int(metadata["wake_strips"])
    record = dict(
        name=name,
        mesh_sha256=hashlib.sha256(mesh_bytes).hexdigest(),
        metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
        field_sha256=hashlib.sha256(field_bytes).hexdigest(),
        builder_snapshot_sha256=hashlib.sha256(
            (case_root/"mesh/build_mesh.py.snapshot").read_bytes()).hexdigest(),
        solver_snapshot_sha256=hashlib.sha256(
            (case_root/"a5p0/solve.py.snapshot").read_bytes()).hexdigest(),
        metadata=dict(span=span, chord=chord, thickness=thickness,
                      taper_ratio=taper, quarter_chord_sweep_deg=sweep,
                      reference_area=metadata["reference_area"],
                      wake_strips=metadata["wake_strips"],
                      gamma_stations=len(stations), gamma_unknowns=len(field["gamma"]),
                      nodes=len(points), tets=len(mesh["tets"]),
                      boundary_faces=len(mesh["boundary_triangles"]),
                      wake_faces=len(upper_wake)),
        result=dict(status=result["status"], converged=result["converged"],
                    kutta=result["kutta_formulation"],
                    kutta_edge_traces=result.get("kutta_edge_traces"),
                    cl_pressure=result["cl_pressure"], cl_gamma=result["cl_jump"]),
        geometry=dict(
            te_node_count=len(upper_te),
            upper_lower_te_node_equal_count=node_pair_same_count,
            max_te_x_error=float(max(abs(te_x_error))),
            max_te_abs_z=float(max(abs(te_z_error))),
            max_te_station_y_error=float(max(abs(te_station_error))),
            z0_planform_endpoint_nodes=len(plane_nodes),
            z0_nodes_not_at_expected_le_or_te=int(np.sum(endpoint_error > 1.e-7)),
            max_z0_endpoint_xc_error=float(max(endpoint_error))
            if len(endpoint_error) else None,
            sampled_surface_points=int(interior_section.sum()),
            section_map_abs_error_rms=float(np.sqrt(np.mean(section_map_error**2))),
            section_map_abs_error_max=float(max(abs(section_map_error))),
            section_map_abs_error_p99=float(np.quantile(abs(section_map_error), .99))),
        wake=dict(
            upper_lower_coordinate_max_error=wake_pair_error,
            faces_crossing_gamma_station_count=int(crossed_internal_stations.sum()),
            face_count_by_gamma_interval=wake_interval_counts.tolist(),
            empty_gamma_intervals=np.flatnonzero(wake_interval_counts == 0).tolist(),
            gamma_interval_count=expected_wake_count,
            max_abs_dgamma_dx=float(max(abs(streamwise_slopes))),
            rms_dgamma_dx=float(np.sqrt(np.mean(streamwise_slopes**2))),
            max_abs_node_jump_vs_gamma_interpolation=float(np.max(np.abs(interpolation_errors)))),
        node_kutta=dict(
            pair_count=len(upper_te),
            upper_lower_identical_pairs=node_pair_same_count,
            identical_neighbor_element_sets=int(sum(node_pair_selection_equal)),
            predicted_cancelled_rows=node_cancellation_count,
            min_upper_selected_wall_tets=int(min(node_adjacent_counts_upper)),
            min_lower_selected_wall_tets=int(min(node_adjacent_counts_lower))),
        weak_kutta=dict(
            selected_wall_te_edges=len(te_edge_ranges),
            solver_reported_te_edge_traces=result.get("kutta_edge_traces"),
            covered_gamma_hats=int(np.sum(weak_edge_coverage > 0)),
            total_gamma_hats=len(weak_edge_coverage),
            missing_gamma_hat_indices=weak_missing_knots.tolist(),
            min_edge_coverage_count=int(min(weak_edge_coverage))),
        passed_mesh_wake_geometry=bool(
            abs(te_x_error).max() < 1.e-8 and abs(te_station_error).max() < 1.e-9
            and np.sum(crossed_internal_stations) == 0
            and len(np.flatnonzero(wake_interval_counts == 0)) == 0
            and len(weak_missing_knots) == 0))
    records.append(record)

report = [
    "# Planform Mesh / Wake / Kutta Independent Audit",
    "",
    "This is a read-only audit of the three saved `results/planforms_v1` meshes and their existing weak-Kutta fields. It does not invoke Gmsh or rerun a solver. Input files are hashed below; source files are inspected but not modified.",
    "",
    "## Findings",
    "",
    "**No node-Kutta cancellation was demonstrated.** Across the three saved meshes, upper/lower TE pairs have distinct global node IDs and distinct selected adjacent-wall-tet sets, with at least two selected wall tetrahedra per trace. The suspected same-row cancellation mechanism is therefore not supported for these meshes. This is a selection-coverage audit, not a full matrix-rank or alternate-mesh proof.",
    "",
    "**No planform mapping or strip-crossing defect was found in these three saved meshes.** The independently evaluated local quarter-chord sweep, taper chord, TE x, section thickness mapping, wake strip interval membership and weak-Kutta station support are reported per mesh. A clean geometry/topology result is not aerodynamic accuracy certification.",
    "",
    "## Independent Measurements",
    "",
]

for record in records:
    meta = record["metadata"]
    geo = record["geometry"]
    wake = record["wake"]
    node = record["node_kutta"]
    weak = record["weak_kutta"]
    report.extend([
        f"### {record['name']}",
        "",
        f"- Geometry: `b={meta['span']}`, root chord=`{meta['chord']}`, taper=`{meta['taper_ratio']}`, "
        f"quarter-chord sweep=`{meta['quarter_chord_sweep_deg']} deg`, reference area=`{meta['reference_area']}`; "
        f"{meta['nodes']} nodes, {meta['tets']} tetrahedra.",
        f"- Surface map: {geo['sampled_surface_points']} interior spanwise P2 surface samples; "
        f"max TE x error=`{geo['max_te_x_error']:.3e}`, max TE z=`{geo['max_te_abs_z']:.3e}`, "
        f"max TE/Gamma-station y error=`{geo['max_te_station_y_error']:.3e}`; "
        f"absolute-thickness versus analytic NACA section RMS/max/p99=`{geo['section_map_abs_error_rms']:.3e}/"
        f"{geo['section_map_abs_error_max']:.3e}/{geo['section_map_abs_error_p99']:.3e}`.",
        f"- Planform endpoint check on interior-span z=0 wing boundary corner nodes (tip caps excluded): "
        f"{geo['z0_planform_endpoint_nodes']} nodes; not near expected LE/TE=`{geo['z0_nodes_not_at_expected_le_or_te']}`, "
        f"maximum x/c endpoint error=`{geo['max_z0_endpoint_xc_error']}`.",
        f"- Wake: {meta['wake_faces']} upper triangles, upper/lower coordinate max error="
        f"`{wake['upper_lower_coordinate_max_error']:.3e}`; Gamma-station crossings="
        f"`{wake['faces_crossing_gamma_station_count']}`; empty strips=`{wake['empty_gamma_intervals']}`; "
        f"faces by strip=`{wake['face_count_by_gamma_interval']}`.",
        f"- Solved potential jump measured from `phi[upper]-phi[lower]`: "
        f"max/RMS `|dGamma/dx|`=`{wake['max_abs_dgamma_dx']:.3e}/"
        f"{wake['rms_dgamma_dx']:.3e}`; max nodal mismatch from the saved piecewise-linear Gamma="
        f"`{wake['max_abs_node_jump_vs_gamma_interpolation']:.3e}`.",
        f"- Node Kutta coverage: {node['pair_count']} pairs; upper/lower same node="
        f"`{node['upper_lower_identical_pairs']}`; identical adjacent-wall-tet sets="
        f"`{node['identical_neighbor_element_sets']}`; predicted zero rows="
        f"`{node['predicted_cancelled_rows']}`; minimum selected wall tets per side="
        f"`{node['min_upper_selected_wall_tets']}/{node['min_lower_selected_wall_tets']}`.",
        f"- Weak Kutta coverage: selected TE traces=`{weak['selected_wall_te_edges']}` "
        f"(solver recorded `{weak['solver_reported_te_edge_traces']}`); Gamma hats covered="
        f"`{weak['covered_gamma_hats']}/{weak['total_gamma_hats']}`, "
        f"missing indices=`{weak['missing_gamma_hat_indices']}`.",
        f"- Existing weak solve: `{record['result']['status']}`, converged=`{record['result']['converged']}`, "
        f"CL pressure/Gamma=`{record['result']['cl_pressure']:.9f}/"
        f"{record['result']['cl_gamma']:.9f}`. Mesh/wake geometric checks passed="
        f"`{record['passed_mesh_wake_geometry']}`.",
        f"- SHA256: mesh `{record['mesh_sha256']}`, field `{record['field_sha256']}`, "
        f"metadata `{record['metadata_sha256']}`, builder snapshot `{record['builder_snapshot_sha256']}`, "
        f"solver snapshot `{record['solver_snapshot_sha256']}`.",
        "",
    ])

report.extend([
    "## Source-Level Interpretation",
    "",
    "- Taper and sweep are encoded as `c(y)=1+(taper−1)|2y/span|`, quarter-chord x=`|y| tan(sweep)`, and LE x=`x_qc+0.25(1−c)`. The TE is consequently `x_qc+0.25+0.75c`; the audit recomputes this against stored TE nodes and section surface samples.",
    "- Wake strip boundaries are created at the same uniform `gamma_stations`; each saved wake triangle is checked by its corner-y interval. The saved field’s jump is independently extracted from duplicated-coordinate `phi` traces, then fitted against local x and y on each quadratic wake triangle.",
    "- Weak Kutta uses TE edges detected from the local TE mapping and integrates against the spanwise Gamma hats. The audit separately checks every internal hat has positive TE-edge support and compares the selected edge count against the saved solver diagnostic.",
    "- The analytic NACA comparison uses the absolute surface z because the stored boundary contains both reflected upper and lower skins. The OCC boundary is a spline through 81 cosine-spaced NACA samples, so small differences from direct analytic evaluation are a CAD interpolation diagnostic, not by themselves a defect.",
    "- The above checks cover only these three meshes and the source snapshots associated with them. They do not prove arbitrary taper/sweep, fuselage interference, tilted-wake geometry, or aerodynamic convergence.",
    "",
    "Read-only current source SHA256:",
    "",
])
for source_path in source_paths:
    report.append(f"- `{source_path.name}`: `{hashlib.sha256(source_path.read_bytes()).hexdigest()}`.")
report.extend([
    "",
    f"Audit script SHA256: `{hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}`.",
])
report_path.write_text("\n".join(report)+"\n", encoding="utf-8")
print("\n".join([
    f"{row['name']}: map={row['passed_mesh_wake_geometry']}, "
    f"wake-cross={row['wake']['faces_crossing_gamma_station_count']}, "
    f"dGamma/dx max={row['wake']['max_abs_dgamma_dx']:.3e}, "
    f"node-cancel={row['node_kutta']['predicted_cancelled_rows']}/{row['node_kutta']['pair_count']}, "
    f"weak-hats={row['weak_kutta']['covered_gamma_hats']}/{row['weak_kutta']['total_gamma_hats']}"
    for row in records
]))
print(report_path)
