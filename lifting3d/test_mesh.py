"""Independent dense geometry and wake/topology checks."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import root_wake
import cad_basis

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent/"volume"))
import p2

parser = argparse.ArgumentParser()
parser.add_argument("--mesh", type=Path, required=True)
parser.add_argument("--report", type=Path, help="Optional JSON report path; default is mesh-parent/tests.json")
args = parser.parse_args()
mesh = np.load(args.mesh)
metadata = json.loads(args.mesh.with_name("mesh.json").read_text())
points, tets = mesh["points"], mesh["tets"]
records = []
barycentric = np.array([[i, j, k, 8-i-j-k] for i in range(9)
                        for j in range(9-i) for k in range(9-i-j)])/8.
_, derivative = p2.p2_shape(barycentric)
minimum = np.inf
for first in range(0, len(tets), 256):
    jacobian = np.einsum("eia,qib->eqab", points[tets[first:first+256]], derivative)
    minimum = min(minimum, float(np.linalg.det(jacobian).min()))
records.append(dict(test="dense_P2_Jacobian", samples_per_element=len(barycentric),
                    minimum=minimum, passed=minimum > 0.))
records.append(dict(test="all_nodes_used", passed=len(np.unique(tets)) == len(points)))
records.append(dict(test="cut_coordinate_pairing",
                    passed=bool(np.array_equal(points[mesh["cut_upper_nodes"]], points[mesh["cut_lower_nodes"]]))))
records.append(dict(test="wake_face_coordinate_pairing",
                    passed=bool(np.array_equal(points[mesh["wake_upper"]], points[mesh["wake_lower"]]))))
stations = mesh["gamma_stations"]
length = float(metadata.get("chord", 1.))
origin = np.array(metadata.get("normalization_origin", [0., 0., 0.]))
cad_mesh = metadata.get("geometry_family") == "imported_cad_planar_te"
cad_tolerance = float(metadata.get("cad_coordinate_tolerance", 1.e-10))
if cad_mesh:
    te_polyline = (np.asarray(mesh["trailing_edge_points"], dtype=float)-origin)/length
    te_station_indices = np.asarray(mesh["te_station_indices"])
    assert te_polyline.ndim == 2 and te_polyline.shape[1] == 3 and len(te_polyline) >= 2
    assert np.isfinite(te_polyline).all() and np.all(np.diff(te_polyline[:, 1]) > 0.)
    assert np.issubdtype(te_station_indices.dtype, np.integer) and te_station_indices.ndim == 1
    assert len(te_station_indices) > 0 and np.all(np.diff(te_station_indices) > 0)
    assert np.all(te_station_indices > 0) and np.all(te_station_indices < len(stations)-1)
    normalized_stations = (stations-origin[1])/length
    te_coordinates = (points[mesh["te_upper_nodes"]]-origin)/length
    expected_stations = normalized_stations[te_station_indices]
    expected_te_x = np.interp(expected_stations, te_polyline[:, 1], te_polyline[:, 0])
    te_x_error = te_coordinates[:, 0]-expected_te_x
    te_station_error = te_coordinates[:, 1]-expected_stations
    te_x_tolerance = cad_tolerance
    te_station_tolerance = cad_tolerance
else:
    te_coordinates = (points[mesh["te_upper_nodes"]]-origin)/length
    te_chord = 1.+(metadata.get("taper_ratio", 1.)-1.)*abs(
        2.*te_coordinates[:, 1]/(metadata["span"]/length))
    expected_te_x = (.25+abs(te_coordinates[:, 1])
                     *np.tan(np.deg2rad(metadata.get("quarter_chord_sweep_deg", 0.)))
                     +.75*te_chord)
    expected_stations = stations[1:-1]
    te_x_error = te_coordinates[:, 0]-expected_te_x
    te_station_error = None
    te_x_tolerance = 1.e-10
    te_station_tolerance = 1.e-12
root_support = metadata.get("root_wake_support")
if cad_mesh:
    contacts = np.asarray(metadata.get("wall_contact_intervals", []), dtype=float).reshape(-1, 2)
    normalized_contacts = (contacts-origin[1])/length
    expected_map = cad_basis.contact_station_map(normalized_stations, normalized_contacts)
    records.append(dict(test="cad_contact_basis_map",
                        contact_intervals=len(normalized_contacts),
                        passed=bool(np.array_equal(mesh["gamma_station_dof"], expected_map))))
    assert abs(normalized_stations[0]-te_polyline[0, 1]) < cad_tolerance
    assert abs(normalized_stations[-1]-te_polyline[-1, 1]) < cad_tolerance
elif root_support:
    limits = np.array(root_support["root_interval_y"])
    expected_stations = expected_stations[(expected_stations <= limits[0]+1e-12)
                                         | (expected_stations >= limits[1]-1e-12)]
    expected_map = root_wake.merge_root_basis((stations-origin[1])/length, (limits-origin[1])/length,
                                             metadata["span"]/length/2.)
    exposed = (np.array(root_support["exposed_te_intervals_y"])-origin[1])/length
    support = root_wake.mapped_weak_test_support((stations-origin[1])/length, expected_map, exposed)
    records.append(dict(test="root_wake_row_coverage", minimum_support=float(support.sum(axis=0).min()),
                        passed=bool(np.array_equal(mesh["gamma_station_dof"], expected_map)
                                    and np.all(support.sum(axis=0) > 0.)
                                    and np.all(support[:, 0] > 0.))))
if not cad_mesh:
    te_station_error = points[mesh["te_upper_nodes"], 1]-expected_stations
records.append(dict(test="TE_station_pairing", passed=bool(np.all(np.diff(stations) > 0.)
                    and len(mesh["te_upper_nodes"]) == len(expected_stations)
                    and np.max(abs(te_station_error)) < te_station_tolerance
                    and np.max(abs(te_x_error)) < te_x_tolerance)))
if cad_mesh:
    wake_nodes_upper = np.asarray(mesh["wake_upper"]).reshape(-1)
    wake_nodes_lower = np.asarray(mesh["wake_lower"]).reshape(-1)
    normalized_stations = (stations-origin[1])/length
    wake_strip_checks = []
    for wake_faces in [mesh["wake_upper"], mesh["wake_lower"]]:
        for face in wake_faces:
            face_y = (points[face, 1]-origin[1])/length
            face_min_y, face_max_y = float(face_y.min()), float(face_y.max())
            inside_one_strip = np.any(
                (face_min_y >= normalized_stations[:-1]-cad_tolerance)
                & (face_max_y <= normalized_stations[1:]+cad_tolerance))
            wake_strip_checks.append(bool(inside_one_strip))
    records.append(dict(
        test="cad_wake_tri6_within_gamma_strip",
        faces_checked=len(wake_strip_checks),
        offending_faces=int(np.count_nonzero(~np.asarray(wake_strip_checks))),
        normalized_tolerance=cad_tolerance,
        passed=bool(len(wake_strip_checks) > 0 and np.all(wake_strip_checks))))

    # A wall/wake edge may be a smooth-body contact. Only the declared TE
    # contributes to Kutta tests; check both endpoints and the Tri6 midpoint.
    edge_local = [(0, 1, 3), (1, 2, 4), (2, 0, 5)]
    wake_edge_keys = set()
    for wake_faces in [mesh["wake_upper"], mesh["wake_lower"]]:
        for face in wake_faces:
            for first, second, midpoint in edge_local:
                wake_edge_keys.add(tuple(sorted((int(face[first]), int(face[second])))))
    wall_te_edges = {}
    contact_edge_count = 0
    for face, tag in zip(mesh["boundary_triangles"], mesh["boundary_tags"]):
        if int(tag) == 100:
            continue
        for first, second, midpoint in edge_local:
            key = tuple(sorted((int(face[first]), int(face[second]))))
            if key in wake_edge_keys:
                xyz = (points[face[[first, second, midpoint]]]-origin)/length
                edge_te_x = np.interp(xyz[:, 1], te_polyline[:, 1], te_polyline[:, 0])
                on_te = (np.max(abs(xyz[:, 0]-edge_te_x)) <= cad_tolerance
                         and np.max(abs(xyz[:, 2])) <= cad_tolerance
                         and xyz[:, 1].min() >= normalized_stations[0]-cad_tolerance
                         and xyz[:, 1].max() <= normalized_stations[-1]+cad_tolerance)
                if on_te:
                    wall_te_edges[key] = int(face[midpoint])
                else:
                    contact_edge_count += 1

    gamma_dof = np.asarray(mesh["gamma_station_dof"], dtype=int)
    gamma_dof_count = int(gamma_dof.max()+1)
    te_support = np.zeros(gamma_dof_count)
    te_edge_length = 0.
    gauss_x, gauss_w = np.polynomial.legendre.leggauss(8)
    gauss_t = .5*(gauss_x+1.)
    gauss_w = .5*gauss_w
    for (first, second), midpoint in wall_te_edges.items():
        edge_points = points[[first, second, midpoint]]
        edge_integral = np.zeros(gamma_dof_count)
        for t, weight in zip(gauss_t, gauss_w):
            shape = np.array([(1.-t)*(1.-2.*t), t*(2.*t-1.), 4.*t*(1.-t)])
            derivative = np.array([4.*t-3., 4.*t-1., 4.-8.*t])
            position = shape @ edge_points
            tangent = derivative @ edge_points
            ds = float(np.linalg.norm(tangent))/length
            y = (position[1]-origin[1])/length
            interval = int(np.clip(np.searchsorted(normalized_stations, y, side="right")-1,
                                   0, len(normalized_stations)-2))
            fraction = ((y-normalized_stations[interval])
                        /(normalized_stations[interval+1]-normalized_stations[interval]))
            station_basis = np.zeros(len(normalized_stations))
            station_basis[interval] = 1.-fraction
            station_basis[interval+1] = fraction
            reduced_basis = np.zeros(gamma_dof_count)
            for station_index, dof in enumerate(gamma_dof):
                if dof >= 0:
                    reduced_basis[dof] += station_basis[station_index]
            edge_integral += weight*ds*reduced_basis
            te_edge_length += weight*ds
        te_support += edge_integral
    records.append(dict(
        test="cad_wall_te_hat_integral_support",
        physical_wall_te_edges=len(wall_te_edges),
        excluded_wall_wake_contact_edge_occurrences=contact_edge_count,
        normalized_total_edge_length=te_edge_length,
        reduced_dof_support=te_support.tolist(),
        unsupported_reduced_dofs=np.flatnonzero(te_support <= 0.).tolist(),
        passed=bool(len(wall_te_edges) > 0 and np.all(te_support > 0.))))

    shared_wake_nodes = np.intersect1d(wake_nodes_upper, wake_nodes_lower)
    tip_y = normalized_stations[[0, -1]]
    shared_y = (points[shared_wake_nodes, 1]-origin[1])/length
    shared_is_tip = (np.isclose(shared_y, tip_y[0], rtol=0., atol=cad_tolerance)
                     | np.isclose(shared_y, tip_y[1], rtol=0., atol=cad_tolerance))
    cut_upper = np.asarray(mesh["cut_upper_nodes"], dtype=np.int64)
    cut_lower = np.asarray(mesh["cut_lower_nodes"], dtype=np.int64)
    cut_pairs_valid = (len(cut_upper) == len(cut_lower)
                       and len(np.unique(cut_upper)) == len(cut_upper)
                       and len(np.unique(cut_lower)) == len(cut_lower)
                       and len(np.intersect1d(cut_upper, cut_lower)) == 0)
    cut_pair_coordinate_error = float(np.max(
        abs((points[cut_upper]-points[cut_lower])/length))) if cut_pairs_valid and len(cut_upper) else np.inf
    records.append(dict(
        test="cad_tip_and_cut_node_topology",
        shared_wake_node_count=len(shared_wake_nodes),
        shared_wake_nodes_at_station_endpoints=bool(len(shared_wake_nodes) > 0 and np.all(shared_is_tip)),
        cut_upper_lower_disjoint=cut_pairs_valid,
        max_cut_pair_coordinate_error_normalized=cut_pair_coordinate_error,
        passed=bool(len(shared_wake_nodes) > 0 and np.all(shared_is_tip)
                    and cut_pairs_valid and cut_pair_coordinate_error <= cad_tolerance)))
else:
    reflection_map = np.arange(len(points))
    half = len(tets)//2
    upper = tets[:half].ravel()
    lower = tets[half:][:, [0, 2, 1, 3, 6, 5, 4, 7, 9, 8]].ravel()
    reflection_map[upper] = lower
    reference_points = mesh["reference_points"] if "reference_points" in mesh else points
    error = float(np.max(abs((reference_points[reflection_map[np.unique(upper)]]-origin)
                             -(reference_points[np.unique(upper)]-origin)*[1., 1., -1.])))
    records.append(dict(test="reference_element_reflection_pairing", error=error, passed=error < 1e-12))
all_faces = np.vstack([mesh["boundary_triangles"], mesh["wake_upper"], mesh["wake_lower"]])
all_adjacent = np.concatenate([mesh["boundary_tet"], mesh["wake_upper_tet"], mesh["wake_lower_tet"]])
face_map = {}
for element, tet in enumerate(tets):
    for omitted in range(4):
        face_map.setdefault(tuple(sorted(np.delete(tet[:4], omitted))), []).append(element)
valid = all(len(pair) <= 2 for pair in face_map.values())
valid &= sum(len(pair) == 1 for pair in face_map.values()) == len(all_faces)
for face, element in zip(all_faces, all_adjacent):
    valid &= face_map.get(tuple(sorted(face[:3]))) == [element]
    for i, j, mid in [(0, 1, 3), (1, 2, 4), (2, 0, 5)]:
        local = [list(tets[element, :4]).index(int(face[node])) for node in [i, j]]
        edge = next(k for k, pair in enumerate(p2.TET_EDGES) if set(pair) == set(local))
        valid &= face[mid] == tets[element, 4+edge]
records.append(dict(test="all_exterior_faces_adjacency_and_midnodes", passed=bool(valid)))
target = args.report if args.report else args.mesh.parent/"tests.json"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(records, indent=2), encoding="utf-8")
print(json.dumps(records, indent=2))
assert all(row["passed"] for row in records)
