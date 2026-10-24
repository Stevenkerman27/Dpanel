"""Read-only audit of merged-root wake Tri6 geometry and Gamma traces."""
import hashlib
import argparse
import json
from pathlib import Path
import shutil

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
parser = argparse.ArgumentParser()
parser.add_argument("--out", default="audit/root_cut_v1")
args = parser.parse_args()
OUT = ROOT / args.out
if OUT.exists():
    assert not any((OUT/name).exists()
                   for name in ["audit.json", "REPORT.md", "audit_root_cut.py.snapshot"]), \
        f"Refusing to overwrite audit evidence: {OUT}"
else:
    OUT.mkdir(parents=True)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tri6_shape(barycentric):
    l0 = barycentric[:, 0]
    l1 = barycentric[:, 1]
    l2 = barycentric[:, 2]
    dl = np.array([[-1., -1.], [1., 0.], [0., 1.]])
    values = np.column_stack([
        l0*(2.*l0-1.), l1*(2.*l1-1.), l2*(2.*l2-1.),
        4.*l0*l1, 4.*l1*l2, 4.*l2*l0,
    ])
    derivatives = np.empty((len(barycentric), 6, 2))
    for i, li in enumerate([l0, l1, l2]):
        derivatives[:, i, :] = (4.*li[:, None]-1.)*dl[i]
    derivatives[:, 3, :] = 4.*(l1[:, None]*dl[0]+l0[:, None]*dl[1])
    derivatives[:, 4, :] = 4.*(l2[:, None]*dl[1]+l1[:, None]*dl[2])
    derivatives[:, 5, :] = 4.*(l0[:, None]*dl[2]+l2[:, None]*dl[0])
    return values, derivatives


def mapped_basis(y, stations, station_dof, dof_count):
    y = np.asarray(y, dtype=float).reshape(-1)
    interval = np.searchsorted(stations, y, side="right")-1
    interval = np.clip(interval, 0, len(stations)-2)
    fraction = (y-stations[interval])/(stations[interval+1]-stations[interval])
    result = np.zeros((len(y), dof_count))
    left_dof = station_dof[interval]
    right_dof = station_dof[interval+1]
    rows = np.arange(len(y))
    left_valid = left_dof >= 0
    right_valid = right_dof >= 0
    result[rows[left_valid], left_dof[left_valid]] += 1.-fraction[left_valid]
    result[rows[right_valid], right_dof[right_valid]] += fraction[right_valid]
    return result


def trace_metrics(points, faces, stations, station_values, shape, derivatives):
    face_points = points[faces]
    nodal_values = np.interp(face_points[:, :, 1], stations, station_values)
    trace = np.einsum("qi,fi->fq", shape, nodal_values)
    dtrace_ref = np.einsum("qia,fi->fqa", derivatives, nodal_values)
    sample_points = np.einsum("qi,fij->fqj", shape, face_points)
    dpoints_ref = np.einsum("qia,fij->fqaj", derivatives, face_points)
    jacobian_xy = dpoints_ref[:, :, :, :2]
    gradient_xy = np.linalg.solve(jacobian_xy, dtrace_ref[..., None])[..., 0]
    expected = np.interp(sample_points[:, :, 1], stations, station_values)
    error = trace-expected
    return dict(
        max_trace_error=float(np.max(np.abs(error))),
        rms_trace_error=float(np.sqrt(np.mean(error**2))),
        max_abs_streamwise_derivative=float(np.max(np.abs(gradient_xy[:, :, 0]))),
    )


# Degree-four barycentric lattice is overdetermined for a quadratic trace.
barycentric = np.array([[i, j, 4-i-j] for i in range(5)
                        for j in range(5-i)])/4.
shape, derivatives = tri6_shape(barycentric)

batch_path = RESULTS / "geometry_robustness_v2/progress.json"
batch = json.loads(batch_path.read_text(encoding="utf-8"))
planform_path = RESULTS / "planform_root_v1/progress.json"
planform_batch = json.loads(planform_path.read_text(encoding="utf-8"))
cases = [
    dict(name="root_wake_pilot_v2",
         mesh=RESULTS/"root_wake_pilot_v2/mesh/mesh.npz",
         metadata=RESULTS/"root_wake_pilot_v2/mesh/mesh.json",
         result=RESULTS/"root_wake_pilot_v2/solve/result.json"),
    dict(name="root_near_repair_v1",
         mesh=RESULTS/"root_near_repair_v1/mesh/mesh.npz",
         metadata=RESULTS/"root_near_repair_v1/mesh/mesh.json",
         result=RESULTS/"root_near_repair_v1/solve/result.json"),
]
for row in batch["cases"]:
    case_root = RESULTS/"geometry_robustness_v2"/row["case"]["name"]
    cases.append(dict(name="geometry_robustness_v2/"+row["case"]["name"],
                      mesh=case_root/"mesh/mesh.npz",
                      metadata=case_root/"mesh/mesh.json",
                      result=None))
for row in planform_batch["cases"]:
    assert row["status"] == "passed" and row["solve"]["converged"]
    case_root = RESULTS/"planform_root_v1"/row["case"]["name"]
    cases.append(dict(name="planform_root_v1/"+row["case"]["name"],
                      mesh=case_root/"mesh/mesh.npz",
                      metadata=case_root/"mesh/mesh.json",
                      result=case_root/"solve/result.json"))

records = []
source_hashes = {
    "lifting3d/audit_root_cut.py": sha256(Path(__file__)),
    "lifting3d/ROOT_WAKE_PROTOCOL.md": sha256(ROOT/"ROOT_WAKE_PROTOCOL.md"),
    "lifting3d/root_wake.py": sha256(ROOT/"root_wake.py"),
    "lifting3d/results/geometry_robustness_v2/progress.json": sha256(batch_path),
    "lifting3d/results/planform_root_v1/progress.json": sha256(planform_path),
    "lifting3d/results/planform_root_v1/attempt_001_progress.json": sha256(
        RESULTS/"planform_root_v1/attempt_001_progress.json"),
    "lifting3d/planform_root.py": sha256(ROOT/"planform_root.py"),
}

for case in cases:
    mesh_path = case["mesh"]
    metadata_path = case["metadata"]
    mesh = np.load(mesh_path, allow_pickle=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    points = np.asarray(mesh["points"], dtype=float)
    stations = np.asarray(mesh["gamma_stations"], dtype=float)
    station_dof = np.asarray(mesh["gamma_station_dof"], dtype=int)
    dof_count = int(station_dof.max()+1)
    root_contract = metadata["root_wake_support"]
    root_interval = np.asarray(root_contract["root_interval_y"], dtype=float)
    upper_faces = np.asarray(mesh["wake_upper"], dtype=int)
    lower_faces = np.asarray(mesh["wake_lower"], dtype=int)
    assert upper_faces.shape == lower_faces.shape and upper_faces.shape[1] == 6
    assert np.array_equal(station_dof, np.asarray(root_contract["station_to_dof"], dtype=int))
    assert root_contract["status"] == "constructed"

    all_faces = np.vstack([upper_faces, lower_faces])
    all_face_points = points[all_faces]
    upper_lower_coordinate_error = float(np.max(np.abs(
        points[upper_faces]-points[lower_faces])))
    face_sample_points = np.einsum("qi,fij->fqj", shape, all_face_points)
    midpoint_error = []
    for edge_node, corner_a, corner_b in [(3, 0, 1), (4, 1, 2), (5, 2, 0)]:
        midpoint_error.append(np.max(np.abs(
            all_face_points[:, edge_node, 1]
            -.5*(all_face_points[:, corner_a, 1]+all_face_points[:, corner_b, 1]))))

    station_basis = np.zeros((len(stations), dof_count))
    valid_station = station_dof >= 0
    station_basis[np.flatnonzero(valid_station), station_dof[valid_station]] = 1.
    plateau_knots = []
    for knot in range(1, len(stations)-1):
        if np.all(station_dof[knot-1:knot+2] == 0):
            plateau_knots.append(knot)

    crossing_counts = np.zeros(len(stations)-2, dtype=int)
    crossing_face_count = 0
    nonplateau_crossing_count = 0
    all_face_y = np.concatenate([all_face_points[:, :, 1],
                                 face_sample_points[:, :, 1]], axis=1)
    face_ranges = np.column_stack([all_face_y.min(axis=1), all_face_y.max(axis=1)])
    for knot in range(1, len(stations)-1):
        crosses = ((face_ranges[:, 0] < stations[knot]-1.e-11)
                   & (face_ranges[:, 1] > stations[knot]+1.e-11))
        crossing_counts[knot-1] = int(crosses.sum())
        crossing_face_count += int(crosses.sum())
        if knot not in plateau_knots:
            nonplateau_crossing_count += int(crosses.sum())

    body_faces = np.asarray(mesh["boundary_triangles"], dtype=int)[
        np.asarray(mesh["boundary_tags"], dtype=int) == 20]
    body_nodes = np.unique(body_faces)
    cut_upper = np.asarray(mesh["cut_upper_nodes"], dtype=int)
    cut_lower = np.asarray(mesh["cut_lower_nodes"], dtype=int)
    shared_wake_nodes = np.intersect1d(upper_faces.reshape(-1), lower_faces.reshape(-1))
    tip_node_mask = (
        np.isclose(points[shared_wake_nodes, 1], stations[0], rtol=0., atol=1.e-10)
        | np.isclose(points[shared_wake_nodes, 1], stations[-1], rtol=0., atol=1.e-10)
    )
    shared_wake_basis = mapped_basis(
        points[shared_wake_nodes, 1], stations, station_dof, dof_count)
    cut_upper_lower_shared = np.intersect1d(cut_upper, cut_lower)
    body_shared_upper = np.intersect1d(body_nodes, cut_upper)
    body_shared_lower = np.intersect1d(body_nodes, cut_lower)
    root_basis_upper = mapped_basis(points[body_shared_upper, 1],
                                    stations, station_dof, dof_count)
    root_basis_lower = mapped_basis(points[body_shared_lower, 1],
                                    stations, station_dof, dof_count)
    root_unit = np.zeros(dof_count)
    root_unit[0] = 1.
    shared_wake_unit_gamma = shared_wake_basis @ root_unit
    shared_wake_tip_only = bool(len(shared_wake_nodes) > 0 and np.all(tip_node_mask))
    shared_wake_tip_unit_gamma_max = float(
        np.max(np.abs(shared_wake_unit_gamma))) if len(shared_wake_nodes) else float("inf")
    shared_basis_error = max(
        float(np.max(np.abs(root_basis_upper-root_unit))),
        float(np.max(np.abs(root_basis_lower-root_unit))))
    shared_y = np.r_[points[body_shared_upper, 1], points[body_shared_lower, 1]]
    shared_inside_root = bool(np.all(shared_y >= root_interval[0]-1.e-10)
                              and np.all(shared_y <= root_interval[1]+1.e-10))

    arbitrary_coefficients = np.sin(.713*np.arange(dof_count)+.29)
    arbitrary_coefficients[0] = .731
    arbitrary_station_values = station_basis@arbitrary_coefficients
    basis_trace_error = 0.
    basis_streamwise_derivative = 0.
    for dof in range(dof_count):
        unit = np.zeros(dof_count)
        unit[dof] = 1.
        metrics = trace_metrics(points, all_faces, stations,
                                station_basis@unit, shape, derivatives)
        basis_trace_error = max(basis_trace_error, metrics["max_trace_error"])
        basis_streamwise_derivative = max(
            basis_streamwise_derivative, metrics["max_abs_streamwise_derivative"])
    arbitrary_trace = trace_metrics(
        points, all_faces, stations,
        arbitrary_station_values, shape, derivatives)
    arbitrary_shared_gamma = np.interp(shared_y, stations, arbitrary_station_values)
    arbitrary_root_value = float(arbitrary_coefficients[0])
    arbitrary_shared_error = float(np.max(np.abs(arbitrary_shared_gamma-arbitrary_root_value)))

    actual_solution = None
    if case["result"] is not None:
        result_path = case["result"]
        result = json.loads(result_path.read_text(encoding="utf-8"))
        field_path = result_path.with_name("field.npz")
        field = np.load(field_path, allow_pickle=False)
        actual_station_values = np.asarray(result["potential_jump"], dtype=float)
        actual_dof_values = np.asarray(field["gamma"], dtype=float)
        actual_station_dof = np.asarray(result["gamma_station_dof"], dtype=int)
        assert np.array_equal(actual_station_dof, station_dof)
        expanded_actual = np.zeros(len(stations))
        expanded_actual[valid_station] = actual_dof_values[station_dof[valid_station]]
        dof_expansion_error = float(np.max(np.abs(expanded_actual-actual_station_values)))
        root_station_mask = station_dof == 0
        root_station_values = actual_station_values[root_station_mask]
        actual_root_value = float(actual_dof_values[0])
        actual_shared_gamma = np.interp(shared_y, stations, actual_station_values)
        actual_trace = trace_metrics(
            points, all_faces, stations,
            actual_station_values, shape, derivatives)
        actual_solution = dict(
            status=result.get("status"),
            converged=bool(result.get("converged")),
            reduced_gamma_dof0=actual_root_value,
            root_station_value_min=float(root_station_values.min()),
            root_station_value_max=float(root_station_values.max()),
            root_station_nonzero=bool(abs(actual_root_value) > 1.e-10),
            root_station_constant_error=float(np.max(np.abs(root_station_values-actual_root_value))),
            reduced_to_station_expansion_error=dof_expansion_error,
            body_cut_shared_gamma_error=float(np.max(np.abs(actual_shared_gamma-actual_root_value))),
            p2_trace=actual_trace,
            field_sha256=sha256(field_path),
            result_sha256=sha256(result_path),
        )
        source_hashes[str(result_path.relative_to(ROOT.parent))] = sha256(result_path)
        source_hashes[str(field_path.relative_to(ROOT.parent))] = sha256(field_path)

    body_shared_pair_count = min(len(body_shared_upper), len(body_shared_lower))
    record = dict(
        name=case["name"],
        mesh_sha256=sha256(mesh_path),
        metadata_sha256=sha256(metadata_path),
        mesh=dict(points=len(points), tets=len(mesh["tets"]),
                  upper_wake_tri6=len(upper_faces), lower_wake_tri6=len(lower_faces),
                  requested_wake_strips=metadata.get("wake_strips"),
                  actual_wake_strips=metadata.get("actual_wake_strips")),
        gamma_map=dict(stations=stations.tolist(), station_to_dof=station_dof.tolist(),
                       reduced_dofs=dof_count, root_interval_y=root_interval.tolist(),
                       constant_plateau_knots=plateau_knots,
                       arbitrary_dof0_value=arbitrary_root_value),
        wake_topology=dict(
            upper_lower_coordinate_max_error=upper_lower_coordinate_error,
            shared_upper_lower_node_id_count=int(np.sum(upper_faces == lower_faces)),
            shared_upper_lower_unique_node_id_count=len(shared_wake_nodes),
            shared_upper_lower_nodes_tip_only=shared_wake_tip_only,
            shared_upper_lower_tip_unit_gamma_max=shared_wake_tip_unit_gamma_max,
            max_edge_midpoint_y_affinity_error=float(max(midpoint_error)),
            face_crossing_count_by_station=crossing_counts.tolist(),
            total_face_station_crossings=crossing_face_count,
            nonplateau_face_station_crossings=nonplateau_crossing_count,
            all_crossings_confined_to_constant_root_plateau=bool(
                nonplateau_crossing_count == 0)),
        tri6_trace=dict(
            basis_vector_max_interpolation_error=basis_trace_error,
            basis_vector_max_abs_dgamma_dx=basis_streamwise_derivative,
            arbitrary_vector_max_interpolation_error=arbitrary_trace["max_trace_error"],
            arbitrary_vector_rms_interpolation_error=arbitrary_trace["rms_trace_error"],
            arbitrary_vector_max_abs_dgamma_dx=arbitrary_trace["max_abs_streamwise_derivative"]),
        body_cut_topology=dict(
            cut_upper_lower_free_node_intersection_count=len(cut_upper_lower_shared),
            cut_upper_lower_free_nodes_disjoint=len(cut_upper_lower_shared) == 0,
            tag20_body_node_count=len(body_nodes),
            body_nodes_shared_with_upper_cut=len(body_shared_upper),
            body_nodes_shared_with_lower_cut=len(body_shared_lower),
            paired_shared_node_count=body_shared_pair_count,
            shared_nodes_inside_root_interval=shared_inside_root,
            mapped_gamma_basis_max_error_from_dof0=shared_basis_error,
            arbitrary_mapped_gamma_max_error_from_nonzero_dof0=arbitrary_shared_error),
        solved_gamma=actual_solution,
    )
    record["passed_topology"] = bool(
        upper_lower_coordinate_error < 1.e-12
        and shared_wake_tip_only and shared_wake_tip_unit_gamma_max < 1.e-12
        and len(cut_upper_lower_shared) == 0
        and nonplateau_crossing_count == 0
        and len(body_shared_upper) > 0 and len(body_shared_lower) > 0
        and shared_inside_root and shared_basis_error < 1.e-12
        and arbitrary_shared_error < 1.e-12)
    record["passed_arbitrary_trace"] = bool(
        basis_trace_error < 1.e-11
        and basis_streamwise_derivative < 1.e-10
        and arbitrary_trace["max_trace_error"] < 1.e-11
        and arbitrary_trace["max_abs_streamwise_derivative"] < 1.e-10)
    if actual_solution is not None:
        record["passed_solved_root_gamma"] = bool(
            actual_solution["status"].startswith("converged")
            and actual_solution["root_station_nonzero"]
            and actual_solution["root_station_constant_error"] < 1.e-10
            and actual_solution["reduced_to_station_expansion_error"] < 1.e-10
            and actual_solution["body_cut_shared_gamma_error"] < 1.e-10
            and actual_solution["p2_trace"]["max_trace_error"] < 1.e-10
            and actual_solution["p2_trace"]["max_abs_streamwise_derivative"] < 1.e-9)
    record["passed"] = bool(record["passed_topology"] and record["passed_arbitrary_trace"]
                            and record.get("passed_solved_root_gamma", True))
    records.append(record)
    source_hashes[str(mesh_path.relative_to(ROOT.parent))] = record["mesh_sha256"]
    source_hashes[str(metadata_path.relative_to(ROOT.parent))] = record["metadata_sha256"]

audit = dict(
    status="completed",
    scope="independent root-cut geometry/topology and mapped Gamma trace audit, including swept/tapered solved cases; no Cp accuracy claim",
    method=dict(
        tri6_basis="independently coded barycentric quadratic Lagrange trace",
        arbitrary_coefficients="all reduced unit vectors plus one deterministic nonzero vector with DOF0=.731",
        streamwise_derivative="surface coordinate Jacobian in x-y; direct 2x2 solve, no SVD or pseudoinverse",
        matrix_or_solver_assembly_called=False,
        geometry_robustness_cases="mesh-only; no solver fields read",
        solved_gamma_cases=[row["name"] for row in cases if row["result"] is not None]),
    source_sha256=source_hashes,
    cases=records,
)
(OUT/"audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
shutil.copyfile(Path(__file__), OUT/"audit_root_cut.py.snapshot")

report = [
    "# Independent Root-Cut / Gamma Trace Audit",
    "",
    f"This audit reads saved meshes and saved reduced Gamma results for the pilot, near-repair, and {len(planform_batch['cases'])} swept/tapered planform cases. It does not call solver assembly, rerun a solver, or assess Cp accuracy. The 11 geometry-robustness matrix cases are mesh-only.",
    "",
    "The independently coded Tri6 barycentric basis was sampled on an overdetermined degree-four barycentric lattice. Every reduced unit vector was tested, plus a deterministic arbitrary vector with nonzero root coefficient `DOF 0 = .731`. For each face the P2 trace was compared to piecewise-linear `Gamma(y)` and its physical streamwise derivative was calculated by solving the local 2x2 x-y surface Jacobian.",
    "",
    "| Mesh | Wake Tri6 (upper/lower) | Requested/actual strips | Station crossings | Non-plateau crossings | Body-cut shared (upper/lower) | Basis trace max error | Basis max `|dGamma/dx|` | Topology | Trace |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
]
for row in records:
    topo = row["wake_topology"]
    trace = row["tri6_trace"]
    body = row["body_cut_topology"]
    mesh = row["mesh"]
    report.append(
        f"| `{row['name']}` | {mesh['upper_wake_tri6']}/{mesh['lower_wake_tri6']} "
        f"| {mesh['requested_wake_strips']}/{mesh['actual_wake_strips']} "
        f"| {topo['total_face_station_crossings']} "
        f"| {topo['nonplateau_face_station_crossings']} "
        f"| {body['body_nodes_shared_with_upper_cut']}/{body['body_nodes_shared_with_lower_cut']} "
        f"| {trace['basis_vector_max_interpolation_error']:.3e} "
        f"| {trace['basis_vector_max_abs_dgamma_dx']:.3e} "
        f"| {row['passed_topology']} | {row['passed_arbitrary_trace']} |")
report.extend([
    "",
    "## Interpretation",
    "",
    "A wake face crossing an internal Gamma knot is acceptable only when the knot lies inside the mapped constant root plateau (`station_to_dof == 0` on the adjacent plateau stations). Every other crossed knot is reported as a non-plateau crossing. Face ranges use all six Tri6 node y-coordinates and a degree-four barycentric geometry sample. Edge-midpoint y-affinity is reported as a curvature diagnostic, not a pass/fail gate.",
    "",
    "The necessary wake-face sharing condition is that any node ID shared by upper and lower wake faces lies only on the two spanwise tip boundaries, and the mapped unit-Gamma basis is zero there. Such sharing is valid: tip Gamma is prescribed zero. The cut jump-node sets themselves are separately required to be disjoint, since those are the free upper/lower trace DOFs.",
    "",
    "V1's retained `attempt_1` added the condition that every Tri6 edge midpoint y equal the arithmetic mean of its corner y values (`max(midpoint_error) < 1e-10`). This is not a topology or trace requirement: on a curved isoparametric P2 edge, the quadratic coordinate interpolation generally places its midpoint away from the straight endpoint chord. The P2 field is composed with that same coordinate map, so a field linear in physical y is represented exactly without midpoint affinity. V2 retains the measured deviation as a curvature diagnostic only.",
    "",
    f"The `shared_upper_lower_node_id_count` field records matching upper/lower face-node positions (range {min(row['wake_topology']['shared_upper_lower_node_id_count'] for row in records)}-{max(row['wake_topology']['shared_upper_lower_node_id_count'] for row in records)} in this audit); these correspond to unique shared node IDs (range {min(row['wake_topology']['shared_upper_lower_unique_node_id_count'] for row in records)}-{max(row['wake_topology']['shared_upper_lower_unique_node_id_count'] for row in records)}). All such IDs lie on spanwise tips, where the endpoint station map has no free Gamma DOF and the unit-DOF0 mapped basis is exactly zero. The required disjointness is instead between `cut_upper_nodes` and `cut_lower_nodes`, the two free jump traces; their intersection is empty in every audited case.",
    "",
    "At tag-20/body nodes shared with upper and lower cut-node sets, the mapped Gamma basis must equal the unit basis for reduced DOF 0. The arbitrary vector assigns that DOF a nonzero coefficient and verifies that all such shared nodes receive that same value. The pilot, near-repair, and five planform records additionally check saved converged reduced Gamma, its station expansion, and its Tri6 trace; the planform records also exercise swept/tapered body contact. The 11-case geometry-robustness batch remains mesh-only.",
    "",
    "The successful `planform_root_v1/progress.json` records the resumed five-case run. The separate `attempt_001_progress.json` records the earlier metadata-rejection attempt and is retained as provenance. No solver or mesh was rerun for this audit.",
    "",
    "No conclusions about Cp accuracy, wall impermeability, Kutta physical validity, or overall lifting-flow accuracy are made.",
    "",
    "All input SHA-256 values and per-mesh results are in `audit.json`. The exact audit source is preserved as `audit_root_cut.py.snapshot`.",
])
(OUT/"REPORT.md").write_text("\n".join(report)+"\n", encoding="utf-8")

print(json.dumps({
    "status": audit["status"],
    "case_count": len(records),
    "passed": sum(row["passed"] for row in records),
    "cases": [{
        "name": row["name"],
        "passed": row["passed"],
        "crossings": row["wake_topology"]["total_face_station_crossings"],
        "nonplateau_crossings": row["wake_topology"]["nonplateau_face_station_crossings"],
        "shared_wake_tip_only": row["wake_topology"]["shared_upper_lower_nodes_tip_only"],
        "shared_wake_tip_unit_gamma": row["wake_topology"]["shared_upper_lower_tip_unit_gamma_max"],
        "cut_jump_nodes_disjoint": row["body_cut_topology"]["cut_upper_lower_free_nodes_disjoint"],
        "tri6_trace_error": row["tri6_trace"]["basis_vector_max_interpolation_error"],
        "dx_gamma": row["tri6_trace"]["basis_vector_max_abs_dgamma_dx"],
        "shared_root_gamma_error": row["body_cut_topology"]["mapped_gamma_basis_max_error_from_dof0"],
        "solved_gamma_nonzero": (row["solved_gamma"] or {}).get("root_station_nonzero"),
    } for row in records],
    "output": str(OUT),
}, indent=2))
