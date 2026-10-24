"""Conforming P2 full potential with spanwise wake jump and explicit Kutta equations."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
from scipy.sparse import bmat, coo_matrix, csr_matrix, save_npz
from scipy.sparse.linalg import MatrixRankWarning, splu
from threadpoolctl import threadpool_limits
import schur
import root_wake
import cad_basis

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent/"volume"))
import p2

parser = argparse.ArgumentParser()
parser.add_argument("--mesh", type=Path, required=True)
parser.add_argument("--alpha", type=float, default=5.)
parser.add_argument("--mach", type=float, default=0.)
parser.add_argument("--quad-order", type=int, default=3, choices=[3, 4])
parser.add_argument("--ncpu", type=int, default=8)
parser.add_argument("--flux-audit", action="store_true", help="Measure raw flux jumps on every internal tetrahedron face")
parser.add_argument("--kutta", choices=["node", "weak"], default="node",
                    help="TE derivative equality: nodal adjacent-cell average or spanwise hat-weighted edge integral")
parser.add_argument("--linear-solver", choices=["augmented", "schur"], default="augmented",
                    help="Factor the full Kutta system or use exact potential/circulation block elimination")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
started = time.perf_counter()
args.out.mkdir(parents=True, exist_ok=True)
source_bytes = {path.name: path.read_bytes() for path in
                [Path(__file__), Path(p2.__file__), root/"PROTOCOL.md", root/"schur.py", root/"SCHUR_PROTOCOL.md",
                 root/"root_wake.py", root/"ROOT_WAKE_PROTOCOL.md", root/"cad_basis.py", root/"CAD_PROTOCOL.md"]}
result = dict(status="failed", converged=False, alpha_deg=args.alpha, mach=args.mach, history=[],
              method="P2 Galerkin, conforming wake cut, continuous piecewise-linear material-span jump",
              scope="Symmetric finite-thickness straight or trapezoidal wing, optional ellipsoid fuselage; validation in progress",
              solver_sha256=hashlib.sha256(source_bytes["solve.py"]).hexdigest(),
              linear_solver=args.linear_solver,
              schur_sha256=hashlib.sha256(source_bytes["schur.py"]).hexdigest(),
              p2_sha256=hashlib.sha256(source_bytes["p2.py"]).hexdigest(),
              protocol_sha256=hashlib.sha256(source_bytes["PROTOCOL.md"]).hexdigest())
try:
    assert np.isfinite([args.mach, args.alpha]).all() and 0. <= args.mach < 1. and args.ncpu > 0, "Require finite alpha, 0<=Mach<1 and positive NCPU"
    threadpool_limits(args.ncpu)
    warnings.simplefilter("error", MatrixRankWarning)
    result["mesh_sha256"] = hashlib.sha256(args.mesh.read_bytes()).hexdigest()
    raw = np.load(args.mesh)
    metadata = json.loads(args.mesh.with_name("mesh.json").read_text())
    cad_mesh = metadata.get("geometry_family") == "imported_cad_planar_te"
    assert np.issubdtype(raw["boundary_tags"].dtype, np.integer) and np.isin(
        raw["boundary_tags"], [10, 20, 30, 100]).all(), "Physical boundary tags must be integers in {10,20,30,100}"
    for key in ["cut_upper_nodes", "cut_lower_nodes", "te_upper_nodes", "te_lower_nodes"]:
        indices = raw[key]
        assert indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer), f"{key} must contain integer node indices"
        assert np.all(indices >= 0) and np.all(indices < len(raw["points"])), f"{key} contains an out-of-range node index"
    # Include both traces in the topological audit, with separate tags for the wake.
    all_faces = np.vstack([raw["boundary_triangles"], raw["wake_upper"], raw["wake_lower"]])
    all_tags = np.concatenate([np.where(raw["boundary_tags"] == 100, 100, 10), np.full(len(raw["wake_upper"]), 20),
                               np.full(len(raw["wake_lower"]), 30)])
    all_adjacent = np.concatenate([raw["boundary_tet"], raw["wake_upper_tet"], raw["wake_lower_tet"]])
    np.savez_compressed(args.out/"topology_input.npz", points=raw["points"], tets=raw["tets"],
                        boundary_triangles=all_faces, boundary_tags=all_tags, boundary_tet=all_adjacent)
    points, tets, faces, tags, adjacent, _, reoriented, mismatch = p2.load_mesh(args.out/"topology_input.npz")
    tags = np.concatenate([raw["boundary_tags"], np.full(len(raw["wake_upper"]), 201),
                            np.full(len(raw["wake_lower"]), 202)])
    assert mismatch == 0, "Exported face-to-tet adjacency disagrees with independent topology"
    physical_points = points.copy()
    physical_length = float(metadata["chord"])
    normalization_origin = np.array(metadata.get("normalization_origin", [0., 0., 0.]))
    assert physical_length > 0. and np.isfinite(physical_length)
    points = (points-normalization_origin)/physical_length
    assert np.isfinite(points).all(), "Normalized coordinates must be finite"
    te_polyline = None
    if cad_mesh:
        te_polyline = (raw["trailing_edge_points"]-normalization_origin)/physical_length
        assert te_polyline.ndim == 2 and te_polyline.shape[1] == 3 and len(te_polyline) >= 2
        assert np.isfinite(te_polyline).all() and np.all(np.diff(te_polyline[:, 1]) > 0.)
        assert np.max(abs(te_polyline[:, 2])) < 1e-12, "Imported CAD currently requires a planar TE"
        assert args.kutta == "weak", "Imported CAD requires the topological-bank weak Kutta formulation"
        result["scope"] = "Imported closed CAD, declared planar piecewise-linear TE, full-volume topological wake cut; validation in progress"
    reference_area = metadata["reference_area"]/physical_length**2
    assert np.isfinite(reference_area) and reference_area > 0., "Reference area must be finite and positive"
    assert np.isfinite(metadata["span"]) and metadata["span"] > 0., "Span must be finite and positive"
    assert len(raw["wake_upper"]) == len(raw["wake_lower"]) > 0, "Wake trace face counts must agree and be positive"
    pairing_distance = np.linalg.norm(points[raw["wake_upper"]][:, :, None]
                                      -points[raw["wake_lower"]][:, None, :], axis=3)
    assert np.max(np.min(pairing_distance, axis=2)) < 1e-12, "Wake face rows do not form physical pairs"
    upper, lower = raw["cut_upper_nodes"], raw["cut_lower_nodes"]
    assert len(upper) == len(lower) > 0 and len(np.unique(upper)) == len(upper) and len(np.unique(lower)) == len(lower), "Cut-node pairing must be one-to-one"
    assert not np.intersect1d(upper, lower).size, "Upper and lower cut nodes must be distinct"
    assert np.max(abs(points[upper]-points[lower])) == 0., "Wake cut coordinate pairing disagrees"
    canonical = np.arange(len(points))
    canonical[lower] = upper
    independent, column = np.unique(canonical, return_inverse=True)
    transform = coo_matrix((np.ones(len(points)), (np.arange(len(points)), column)),
                           shape=(len(points), len(independent))).tocsr()
    physical_stations = raw["gamma_stations"]
    stations = (physical_stations-normalization_origin[1])/physical_length
    assert np.isfinite(stations).all() and np.all(np.diff(stations) > 0.), "Gamma stations must be finite and strictly increasing"
    station_dof = raw["gamma_station_dof"] if "gamma_station_dof" in raw else np.r_[-1, np.arange(len(stations)-2), -1]
    assert station_dof.shape == stations.shape and np.issubdtype(station_dof.dtype, np.integer)
    gamma_count = int(station_dof.max()+1)
    assert gamma_count > 0 and station_dof[0] == station_dof[-1] == -1
    assert np.array_equal(np.unique(station_dof[station_dof >= 0]), np.arange(gamma_count)), "Gamma mapping must use every reduced DOF"
    root_support = metadata.get("root_wake_support")
    expected_te_stations = stations[1:-1]
    if cad_mesh:
        contacts = np.asarray(metadata.get("wall_contact_intervals", []), dtype=float).reshape(-1, 2)
        contacts = (contacts-normalization_origin[1])/physical_length
        expected_map = cad_basis.contact_station_map(stations, contacts)
        assert np.array_equal(station_dof, expected_map), "CAD contact equivalence map disagrees with recorded intervals"
        te_station_indices = raw["te_station_indices"]
        assert np.issubdtype(te_station_indices.dtype, np.integer) and te_station_indices.ndim == 1
        assert np.all(np.diff(te_station_indices) > 0.) and np.all(te_station_indices > 0) and np.all(te_station_indices < len(stations)-1)
        expected_te_stations = stations[te_station_indices]
        assert abs(stations[0]-te_polyline[0, 1]) < 1e-12 and abs(stations[-1]-te_polyline[-1, 1]) < 1e-12
    elif root_support:
        assert args.kutta == "weak", "Merged wall-contact root requires weak Kutta equations"
        root_limits = (np.array(root_support["root_interval_y"])-normalization_origin[1])/physical_length
        expected_map = root_wake.merge_root_basis(stations, root_limits, metadata["span"]/physical_length/2.)
        assert np.array_equal(station_dof, expected_map), "Root circulation map disagrees with declared wall-contact constraint"
        expected_te_stations = expected_te_stations[(expected_te_stations <= root_limits[0]+1e-12)
                                                   | (expected_te_stations >= root_limits[1]-1e-12)]
    else:
        assert np.array_equal(station_dof, np.r_[-1, np.arange(len(stations)-2), -1]), "Reduced root mapping requires explicit geometry metadata"
    assert len(raw["te_upper_nodes"]) == len(raw["te_lower_nodes"]) == len(expected_te_stations), "TE pairs must cover exposed station knots"
    te_match_tolerance = float(metadata.get("cad_coordinate_tolerance", 1e-10)) if cad_mesh else 1e-10
    assert len(expected_te_stations) > 0
    assert np.max(abs(points[raw["te_upper_nodes"], 1]-expected_te_stations)) < te_match_tolerance, "TE nodes must match exposed Gamma station coordinates"
    assert np.max(abs(points[raw["te_upper_nodes"]]-points[raw["te_lower_nodes"]])) < 1e-12, "TE trace coordinates must match"
    material_coordinates = bool(metadata.get("wake_material_coordinates", False))
    jump_coordinate = points[:, 1]
    if material_coordinates:
        assert "reference_points" in raw, "Material wake requires undeformed reference coordinates"
        reference_coordinates = (raw["reference_points"]-normalization_origin)/physical_length
        assert reference_coordinates.shape == points.shape and np.isfinite(reference_coordinates).all()
        jump_coordinate = reference_coordinates[:, 1]
    assert np.all(jump_coordinate[lower] > stations[0]) and np.all(jump_coordinate[lower] < stations[-1]), "Wake material coordinates must lie inside Gamma endpoints"
    interval = np.searchsorted(stations, jump_coordinate[lower], side="right")-1
    interval = np.minimum(interval, len(stations)-2)
    fraction = (jump_coordinate[lower]-stations[interval])/(stations[interval+1]-stations[interval])
    knot_indices = np.column_stack([interval, interval+1]).ravel()
    jump_values = -np.column_stack([1.-fraction, fraction]).ravel()
    jump_rows = np.repeat(lower, 2)
    mapped_knots = station_dof[knot_indices]
    nonzero_knots = mapped_knots >= 0
    jump = coo_matrix((jump_values[nonzero_knots],
                       (jump_rows[nonzero_knots], mapped_knots[nonzero_knots])),
                      shape=(len(points), gamma_count)).tocsr()
    reference_nodes = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.],
                                [.5, .5, 0., 0.], [0., .5, .5, 0.], [.5, 0., .5, 0.],
                                [.5, 0., 0., .5], [0., .5, 0., .5], [0., 0., .5, .5]])
    kutta = np.zeros((gamma_count, len(points)))
    bank_node_sign = np.zeros(len(points), dtype=int)
    bank_node_sign[upper], bank_node_sign[lower] = 1, -1
    physical_wall = np.isin(tags, [10, 20, 30])
    wall_faces = faces[physical_wall]
    wall_adjacent = adjacent[physical_wall]
    if args.kutta == "node":
        for row, pair in enumerate(zip(raw["te_upper_nodes"], raw["te_lower_nodes"])):
            for sign, node in zip([1., -1.], pair):
                selected = np.unique(wall_adjacent[np.any(wall_faces == node, axis=1)])
                assert len(selected) > 0
                for element in selected:
                    local = int(np.flatnonzero(tets[element] == node)[0])
                    _, reference_gradient = p2.p2_shape(reference_nodes[local][None])
                    jacobian = points[tets[element]].T@reference_gradient[0]
                    gradient = reference_gradient[0]@np.linalg.inv(jacobian)
                    kutta[row, tets[element]] += sign*gradient[:, 0]/len(selected)
    else:
        # Integrate the same raw derivative traces against the Gamma hats on each TE edge.
        line_points, line_weights = np.polynomial.legendre.leggauss(args.quad_order)
        line_points, line_weights = .5*(line_points+1.), .5*line_weights
        kutta_measure = np.zeros(gamma_count)
        te_edge_count = 0
        for face, element in zip(wall_faces, wall_adjacent):
            corners = face[:3]
            span = metadata["span"]/physical_length
            if cad_mesh:
                local_te = np.interp(points[corners, 1], te_polyline[:, 1], te_polyline[:, 0])
                on_te = ((abs(points[corners, 0]-local_te) < te_match_tolerance)
                         & (abs(points[corners, 2]) < te_match_tolerance)
                         & (points[corners, 1] >= stations[0]-1e-12)
                         & (points[corners, 1] <= stations[-1]+1e-12))
            else:
                local_chord = 1.+(metadata.get("taper_ratio", 1.)-1.)*abs(2.*points[corners, 1]/span)
                local_te = .25+abs(points[corners, 1])*np.tan(np.deg2rad(metadata.get("quarter_chord_sweep_deg", 0.)))+.75*local_chord
                on_te = (abs(points[corners, 0]-local_te) < 1e-9) & (abs(points[corners, 2]) < 1e-9)
            if on_te.sum() != 2:
                continue
            edge = corners[on_te]
            edge = edge[np.argsort(points[edge, 1])]
            left, right = points[edge, 1]
            assert right > left, "TE edge must have positive spanwise length"
            breaks = np.r_[left, stations[(stations > left+1e-12) & (stations < right-1e-12)], right]
            local = [int(np.flatnonzero(tets[element, :4] == node)[0]) for node in edge]
            if cad_mesh:
                edge_signs = np.unique(bank_node_sign[edge])
                edge_signs = edge_signs[edge_signs != 0]
                assert len(edge_signs) == 1, "TE edge must belong to exactly one topological wake bank"
                sign = float(edge_signs[0])
                edge_measure = float(np.linalg.norm(points[edge[1]]-points[edge[0]])/(right-left))
            else:
                sign = float(np.sign(points[tets[element, :4], 2].mean()))
                assert sign != 0., "TE wall element must lie on one side of the wing"
                edge_measure = 1.
            te_edge_count += 1
            for first_y, last_y in zip(breaks[:-1], breaks[1:]):
                y = first_y+(last_y-first_y)*line_points
                edge_fraction = (y-left)/(right-left)
                barycentric = np.zeros((len(y), 4))
                barycentric[:, local[0]], barycentric[:, local[1]] = 1.-edge_fraction, edge_fraction
                _, reference_gradient = p2.p2_shape(barycentric)
                jacobian = np.einsum("ia,qib->qab", points[tets[element]], reference_gradient)
                gradient = np.einsum("qia,qab->qib", reference_gradient, np.linalg.inv(jacobian))
                interval_y = np.minimum(np.searchsorted(stations, y, side="right")-1, len(stations)-2)
                fraction_y = (y-stations[interval_y])/(stations[interval_y+1]-stations[interval_y])
                for knots, hat in [(interval_y, 1.-fraction_y), (interval_y+1, fraction_y)]:
                    for knot in np.unique(knots):
                        dof = station_dof[knot]
                        if dof < 0:
                            continue
                        weight = line_weights*(last_y-first_y)*hat*(knots == knot)*edge_measure
                        kutta[dof, tets[element]] += sign*np.einsum("q,qi->i", weight, gradient[:, :, 0])
                        kutta_measure[dof] += .5*weight.sum()
        assert np.all(kutta_measure > 0.), "Every weak Kutta test must intersect the TE"
        kutta /= kutta_measure[:, None]
        result["kutta_edge_traces"] = te_edge_count
        result["kutta_test_measure"] = kutta_measure.tolist()
        result["kutta_measure_definition"] = "Physical TE edge arclength, bank labels from local cut topology" if cad_mesh else "Spanwise edge measure, bank labels from reflected half mesh"
    kutta = csr_matrix(kutta)
    kutta_psi, kutta_gamma = kutta@transform, kutta@jump
    alpha = np.deg2rad(args.alpha)
    flow = np.array([np.cos(alpha), 0., np.sin(alpha)])
    initial_phi = points@flow
    geometry = p2.evaluate_surface(points, tets, faces, tags, adjacent, initial_phi, 0., args.quad_order)
    triangle_bary, triangle_weights = p2.triangle_quadrature(args.quad_order)
    face_shape, _ = p2.triangle_shape(triangle_bary)
    far = tags == 100
    far_natural = np.einsum("bqa,a->bq", geometry["normal"][far], flow)
    local_rhs = np.einsum("qi,bq,bq->bi", face_shape, geometry["area_weight"][far], far_natural)
    rhs = np.zeros(len(points))
    np.add.at(rhs, faces[far].ravel(), local_rhs.ravel())
    gauge_node = int(faces[far][0, 0])
    assert gauge_node not in lower
    gauge_column = int(column[gauge_node])
    free = np.setdiff1d(np.arange(len(independent)), [gauge_column])
    psi = points[independent]@flow
    gamma = np.zeros(gamma_count)
    for iteration in range(30):
        phi = transform@psi+jump@gamma
        volume_residual, matrix, local_mach2, beta, energy = p2.evaluate(
            points, tets, phi, args.mach, args.quad_order, True)
        residual = np.concatenate([(transform.T@(volume_residual-rhs))[free], kutta@phi])
        norm = float(np.linalg.norm(residual))
        result["history"].append(dict(iteration=iteration, residual_norm=norm,
                                      kutta_max=float(np.max(abs(kutta@phi))),
                                      max_volume_mach=float(np.sqrt(local_mach2.max()))))
        print(iteration, norm, flush=True)
        if norm < 1e-9:
            break
        a_pp = transform.T@matrix@transform
        a_pg = transform.T@matrix@jump
        augmented = bmat([[a_pp[free][:, free], a_pg[free]],
                          [kutta_psi[:, free], kutta_gamma]], format="csc")
        linear_started = time.perf_counter()
        if args.linear_solver == "schur":
            update, linear_diagnostics = schur.solve_step(
                a_pp[free][:, free], a_pg[free], kutta_psi[:, free], kutta_gamma, residual)
            result["history"][-1].update(linear_diagnostics)
        else:
            factor = splu(augmented)
            update = factor.solve(-residual)
            pivots = abs(factor.U.diagonal())
            result["history"][-1].update(
                linear_solver="augmented", augmented_lu_nonzeros=int(factor.L.nnz+factor.U.nnz),
                lu_min_abs_pivot=float(pivots.min()),
                lu_pivot_ratio=float(pivots.min()/pivots.max()),
                pivot_note="Augmented LU pivot magnitude ratio; not a spectral condition number")
            del factor
        relative_linear_residual = float(np.linalg.norm(augmented@update+residual)/norm)
        result["history"][-1].update(linear_relative_residual=relative_linear_residual,
                                    linear_solve_seconds=time.perf_counter()-linear_started)
        assert relative_linear_residual < 1e-9, "Linear update does not satisfy the original augmented equations"
        assert np.isfinite(update).all(), "Nonfinite augmented Kutta solution"
        step = 1.
        accepted = False
        for backtrack in range(24):
            trial_psi, trial_gamma = psi.copy(), gamma+step*update[len(free):]
            trial_psi[free] += step*update[:len(free)]
            trial_phi = transform@trial_psi+jump@trial_gamma
            try:
                trial_r, _, _, _, _ = p2.evaluate(points, tets, trial_phi, args.mach, args.quad_order, False)
                trial_residual = np.concatenate([(transform.T@(trial_r-rhs))[free], kutta@trial_phi])
                trial_norm = np.linalg.norm(trial_residual)
            except ValueError:
                trial_norm = np.inf
            if trial_norm < norm:
                psi, gamma = trial_psi, trial_gamma
                accepted = True
                result["history"][-1]["accepted_step"] = step
                break
            step *= .5
        assert accepted, "Newton backtracking failed; no admissible residual-reducing step"
    assert norm < 1e-9, "Nonlinear iteration limit"
    surface = p2.evaluate_surface(points, tets, faces, tags, adjacent, phi, args.mach, args.quad_order)
    wall = physical_wall
    force = np.sum(surface["cp"][wall, :, None]*surface["normal"][wall]
                   *surface["area_weight"][wall, :, None], axis=(0, 1))/reference_area
    lift_axis = np.array([-np.sin(alpha), 0., np.cos(alpha)])
    jump_error = phi[upper]-phi[lower]+jump[lower]@gamma
    wall_weight = surface["area_weight"][wall]
    vn = surface["normal_velocity"][wall]
    station_gamma = root_wake.expand_station_values(station_dof, gamma)
    result.update(status="converged_subsonic_unvalidated_lifting", converged=True,
                  nodes=len(points), elements=len(tets), gamma_unknowns=gamma_count,
                  cl_pressure=float(force@lift_axis), cd_pressure=float(force@flow),
                  cl_jump=float(2.*np.trapezoid(station_gamma, stations)/reference_area),
                  pressure_force_coefficient=force.tolist(),
                  gamma_stations=physical_stations.tolist(),
                  potential_jump=(station_gamma*physical_length).tolist(),
                  gamma_station_dof=station_dof.tolist(), root_wake_support=root_support,
                  geometry_family=metadata.get("geometry_family", "parametric_wing"),
                  wall_contact_intervals=metadata.get("wall_contact_intervals", []),
                  normalization_length=physical_length, normalization_origin=normalization_origin.tolist(),
                  quadrature_order=args.quad_order, ncpu_requested=args.ncpu,
                  kutta_formulation=args.kutta,
                  wake_tilt_deg=metadata.get("wake_tilt_deg", 0.),
                  wake_strips=metadata.get("wake_strips", 0),
                  wake_material_coordinates=material_coordinates,
                  wake_model=metadata.get("wake_model", "fixed_planar"),
                  jump_representation=metadata.get("jump_representation", "Legacy nodal interpolation; streamwise constancy approximate"),
                  volume_quadrature_points=len(p2.tetra_quadrature(args.quad_order)[0]),
                  face_quadrature_points=len(triangle_bary),
                  kutta_max=float(np.max(abs(kutta@phi))),
                  potential_jump_max_error=float(np.max(abs(jump_error))*physical_length),
                  wall_normal_rms=float(np.sqrt(np.average(vn**2, weights=wall_weight))),
                  wall_normal_max=float(np.max(abs(vn))),
                  raw_mass_by_tag={str(tag): float(surface["integrated_mass_flux"][tags == tag].sum()*physical_length**2)
                                   for tag in [10, 20, 30, 100, 201, 202]},
                  weak_mass_residual_norm=float(np.linalg.norm((transform.T@(volume_residual-rhs))[free])*physical_length**2),
                  weak_mass_residual_per_chord2=float(np.linalg.norm((transform.T@(volume_residual-rhs))[free])),
                  weak_gauge_reaction=float((transform.T@(volume_residual-rhs))[gauge_column]*physical_length**2),
                  prescribed_outer_mass=float(rhs.sum()*physical_length**2), min_beta=float(beta.min()),
                  max_local_mach=float(max(np.sqrt(local_mach2.max()), surface["max_local_mach"].max())),
                  cp_min=float(surface["cp"][wall].min()), cp_max=float(surface["cp"][wall].max()),
                  faces_reoriented=reoriented, elapsed_seconds=time.perf_counter()-started)
    result["component_loads"] = {}
    for tag in [10, 20, 30]:
        selected = tags == tag
        if not np.any(selected):
            continue
        component_force = np.sum(surface["cp"][selected, :, None]*surface["normal"][selected]
                                 *surface["area_weight"][selected, :, None], axis=(0, 1))/reference_area
        result["component_loads"][str(tag)] = dict(force_coefficient=component_force.tolist(),
            cl=float(component_force@lift_axis), cd=float(component_force@flow),
            area=float(surface["area_weight"][selected].sum()*physical_length**2),
            normal_rms=float(np.sqrt(np.average(surface["normal_velocity"][selected]**2,
                                                 weights=surface["area_weight"][selected]))))
    surface_points = np.einsum("qi,bia->bqa", face_shape, points[faces])
    wake_up, wake_low = tags == 201, tags == 202
    up_points, low_points = surface_points[wake_up], surface_points[wake_low]
    distance = np.linalg.norm(up_points[:, :, None]-low_points[:, None, :], axis=3)
    paired = np.argmin(distance, axis=2)
    assert np.max(np.min(distance, axis=2)) < 1e-10, "Wake quadrature points do not form matching pairs"
    lower_flux = np.take_along_axis(surface["normal_mass_flux"][wake_low], paired, axis=1)
    wake_flux_jump = surface["normal_mass_flux"][wake_up]+lower_flux
    lower_cp = np.take_along_axis(surface["cp"][wake_low], paired, axis=1)
    wake_cp_jump = surface["cp"][wake_up]-lower_cp
    result["wake_diagnostics"] = dict(
        raw_paired_normal_mass_flux_jump_rms=float(np.sqrt(np.average(wake_flux_jump**2,
            weights=surface["area_weight"][wake_up]))),
        raw_paired_normal_mass_flux_jump_max=float(np.max(abs(wake_flux_jump))),
        raw_pressure_jump_rms=float(np.sqrt(np.average(wake_cp_jump**2,
            weights=surface["area_weight"][wake_up]))),
        wake_upper_normal_velocity_rms=float(np.sqrt(np.average(surface["normal_velocity"][wake_up]**2,
            weights=surface["area_weight"][wake_up]))),
        model=metadata.get("wake_model", "Fixed planar wake with recorded tilt; no free-wake tangency or pressure-continuity iteration"),
        conservation="Paired raw gradients, separate from weak nodal continuity")
    wall_points, wall_cp = surface_points[wall], surface["cp"][wall]
    if cad_mesh:
        region = np.full(wall_cp.shape, "surface", dtype="<U12")
        tip_distance = np.minimum(abs(wall_points[:, :, 1]-stations[0]), abs(wall_points[:, :, 1]-stations[-1]))
        region[tip_distance < .1] = "tip"
        region_names = ["surface", "tip"]
    else:
        local_fraction = np.minimum(abs(2.*wall_points[:, :, 1]/(metadata["span"]/physical_length)), 1.)
        local_chord = 1.+(metadata.get("taper_ratio", 1.)-1.)*local_fraction
        local_le = (.5*metadata["span"]/physical_length*local_fraction
                    *np.tan(np.deg2rad(metadata.get("quarter_chord_sweep_deg", 0.)))+.25*(1.-local_chord))
        wall_chord_coordinate = (wall_points[:, :, 0]-local_le)/local_chord
        region = np.full(wall_cp.shape, "interior", dtype="<U12")
        region[wall_chord_coordinate < .1] = "leading"
        region[wall_chord_coordinate > .95] = "trailing"
        region[abs(wall_points[:, :, 1]) > metadata["span"]/physical_length/2.-.1] = "tip"
        region_names = ["interior", "leading", "trailing", "tip"]
    result["wall_regions"] = {}
    for name in region_names:
        selected = region == name
        if not np.any(selected):
            continue
        result["wall_regions"][name] = dict(
            area=float(wall_weight[selected].sum()*physical_length**2),
            normal_rms=float(np.sqrt(np.average(vn[selected]**2, weights=wall_weight[selected]))),
            normal_max=float(np.max(abs(vn[selected]))),
            cp_min=float(wall_cp[selected].min()), cp_max=float(wall_cp[selected].max()))
    result["wall_region_definition"] = ("Complete disjoint cover; within .1 reference chord of declared TE tip span coordinates, otherwise surface; no inferred CAD leading edge"
        if cad_mesh else "Complete disjoint cover; tip |y|>b/2-.1 root chord, otherwise local TE x/c>.95, LE x/c<.1, remaining interior")
    result["internal_face_flux_jump"] = dict(status="not_requested", flag="--flux-audit",
        note="No claim of raw internal-face conservation without this measurement")
    if args.flux_audit:
        result["internal_face_flux_jump"] = p2.internal_face_flux_jump(points, tets, phi, args.mach, args.quad_order)
        result["internal_face_flux_jump"]["units"] = "Coordinates/chord; velocity/Uinf; mass flux/rhoinf/Uinf"
    np.savez_compressed(args.out/"field.npz", points=physical_points, tets=tets,
                        phi=phi*physical_length, psi=psi*physical_length, gamma=gamma*physical_length,
                        gamma_station_dof=station_dof,
                        faces=faces, tags=tags, adjacent=adjacent,
                        surface_points=surface_points*physical_length+normalization_origin,
                        surface_velocity=surface["velocity"], cp=surface["cp"],
                        surface_normal=surface["normal"], surface_weight=surface["area_weight"]*physical_length**2,
                        surface_normal_velocity=surface["normal_velocity"])
except Exception as exception:
    result.update(status="failed_exception", converged=False, exception_type=type(exception).__name__, message=str(exception),
                  elapsed_seconds=time.perf_counter()-started)
    if "augmented" in locals() and ("singular" in str(exception).lower() or "Nonfinite augmented" in str(exception)):
        save_npz(args.out/"failed_matrix.npz", augmented)
        np.savez_compressed(args.out/"failed_linear_system.npz", residual=residual, free=free,
                            independent=independent, points=physical_points)
result["elapsed_seconds"] = time.perf_counter()-started
(args.out/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
for name, contents in source_bytes.items():
    (args.out/(name+".snapshot")).write_bytes(contents)
print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2))
if not result["converged"]:
    raise SystemExit(2)
