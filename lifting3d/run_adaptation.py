"""One frozen lifting3d indicator-guided mesh refinement comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
VOLUME = ROOT.parent / "volume"
sys.path.insert(0, str(VOLUME))
import error_indicator


def fixed_baseline_seam_audit(output_root):
    baseline_mesh = ROOT / "results/wingbody_matrix_v1/t024_r035_a090/mesh/mesh.npz"
    baseline_metadata_path = baseline_mesh.with_name("mesh.json")
    baseline_run = output_root / "baseline_current_solver"
    adapted_run = output_root / "solve"
    audit_path = output_root / "audits/fixed_baseline_seam.json"
    if audit_path.exists():
        raise FileExistsError(f"refusing to overwrite fixed-seam audit: {audit_path}")
    with np.load(baseline_mesh, allow_pickle=False) as archive:
        mesh = {key: archive[key] for key in archive.files}
    metadata = json.loads(baseline_metadata_path.read_text(encoding="utf-8"))
    wall = np.isin(mesh["boundary_tags"], [10, 20, 30])
    edge_components = {}
    for face, tag in zip(mesh["boundary_triangles"][wall], mesh["boundary_tags"][wall]):
        for first, second, midpoint in ((0, 1, 3), (1, 2, 4), (2, 0, 5)):
            edge = tuple(sorted((int(face[first]), int(face[second]))))
            edge_components.setdefault(edge, []).append((int(tag), int(face[midpoint])))
    seam_nodes = []
    for edge, sides in edge_components.items():
        if {side[0] for side in sides} == {20, 30}:
            seam_nodes.extend([edge[0], edge[1], sides[0][1]])
    if not seam_nodes:
        raise ValueError("baseline wall topology contains no shared tag-20/tag-30 seam")
    seam_nodes = np.unique(seam_nodes)
    seam_tree = cKDTree(mesh["points"][seam_nodes])
    threshold = 0.1 * float(metadata["chord"])

    records = {}
    for name, run in (("baseline_current_solver", baseline_run), ("adapted", adapted_run)):
        field_path = run / "field.npz"
        result_path = run / "result.json"
        if not field_path.is_file() or not result_path.is_file():
            records[name] = {"status": "unavailable", "reason": "solver field/result was not produced"}
            continue
        with np.load(field_path, allow_pickle=False) as archive:
            field = {key: archive[key] for key in archive.files}
        result = json.loads(result_path.read_text(encoding="utf-8"))
        coordinates = np.asarray(field["surface_points"])
        velocity = np.asarray(field["surface_velocity"])
        normal = np.asarray(field["surface_normal"])
        weight = np.asarray(field["surface_weight"])
        cp = np.asarray(field["cp"])
        tags = np.asarray(field["tags"])
        wall_mask = np.isin(tags, [10, 20, 30])
        distance, _ = seam_tree.query(coordinates.reshape(-1, 3), k=1)
        fixed_junction = wall_mask[:, None] & (distance.reshape(coordinates.shape[:2]) < threshold)
        normal_velocity = np.sum(velocity * normal, axis=2)
        selected_weight = weight[fixed_junction]
        selected_vn = normal_velocity[fixed_junction]
        selected_cp = cp[fixed_junction]
        if not len(selected_weight) or float(selected_weight.sum()) <= 0.0:
            raise ValueError(f"{name} has no raw quadrature samples in the fixed baseline junction region")
        records[name] = {
            "field_sha256": hashlib.sha256(field_path.read_bytes()).hexdigest(),
            "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            "sample_count": int(len(selected_weight)),
            "area": float(selected_weight.sum()),
            "raw_normal_velocity_rms": float(np.sqrt(np.average(selected_vn**2, weights=selected_weight))),
            "raw_normal_velocity_max_abs": float(np.max(np.abs(selected_vn))),
            "raw_cp_finite": bool(np.isfinite(selected_cp).all()),
            "raw_cp_min": float(np.nanmin(selected_cp)),
            "raw_cp_max": float(np.nanmax(selected_cp)),
        }
    audit = {
        "status": "measured" if all(row.get("status") != "unavailable" for row in records.values())
        else "partial_resource_rejected",
        "method": "fixed baseline body-wing seam P2-node KDTree; distance < 0.1 chord",
        "baseline_mesh_sha256": hashlib.sha256(baseline_mesh.read_bytes()).hexdigest(),
        "baseline_seam_nodes": int(len(seam_nodes)),
        "threshold": threshold,
        "regions_are_recomputed_from_raw_surface_quadrature": True,
        "original_analysis_files_modified": False,
        "runs": records,
        "delta_adapted_minus_baseline": {
            "area": records["adapted"]["area"] - records["baseline_current_solver"]["area"],
            "raw_normal_velocity_rms": (
                records["adapted"]["raw_normal_velocity_rms"]
                - records["baseline_current_solver"]["raw_normal_velocity_rms"]),
        } if records["adapted"].get("status") != "unavailable" else None,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(audit, indent=2))


parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, default=ROOT / "results/lifting_adaptation_v1")
parser.add_argument("--ncpu", type=int, default=8)
parser.add_argument("--fixed-baseline-seam-audit", action="store_true")
parser.add_argument("--mark-resource-rejected", action="store_true")
parser.add_argument("--resource-peak-gb", type=float)
parser.add_argument("--resource-free-ram-gb", type=float)
args = parser.parse_args()
args.out = args.out.resolve()
args.out.mkdir(parents=True, exist_ok=True)
if args.fixed_baseline_seam_audit:
    fixed_baseline_seam_audit(args.out)
    raise SystemExit(0)
if args.mark_resource_rejected:
    result_path = args.out / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") == "completed_with_gate_results":
        raise ValueError("refusing to relabel a completed experiment as resource rejected")
    solve_log = args.out / "solve.log"
    resource_record = {
        "status": "resource_rejected",
        "reason": "manually terminated during sparse LU to prevent system paging",
        "solver_pid": 39668,
        "peak_sampled_working_set_gb": args.resource_peak_gb,
        "free_physical_ram_at_termination_gb": args.resource_free_ram_gb,
        "solver_log_preserved": solve_log.is_file(),
        "solver_log_sha256": hashlib.sha256(solve_log.read_bytes()).hexdigest()
        if solve_log.is_file() else None,
        "solver_result_and_field_written": (args.out / "solve/result.json").is_file()
        and (args.out / "solve/field.npz").is_file(),
        "partial_topology_input_preserved": (args.out / "solve/topology_input.npz").is_file(),
        "partial_matrix_written": (args.out / "solve/failed_matrix.npz").is_file(),
        "annotation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    result["status"] = "resource_rejected"
    result["error"] = "Resource stop during sparse LU; not classified as numerical non-convergence."
    result["resource_rejection"] = resource_record
    if result.get("history"):
        result["history"][-1]["status"] = "resource_rejected"
        result["history"][-1]["termination_reason"] = resource_record["reason"]
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    (args.out / "resource_audit.json").write_text(
        json.dumps(resource_record, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(resource_record, indent=2))
    raise SystemExit(0)
started = time.perf_counter()
baseline = ROOT / "results/wingbody_matrix_v1/t024_r035_a090"
baseline_mesh_dir = baseline / "mesh"
baseline_mesh = baseline_mesh_dir / "mesh.npz"
baseline_mesh_metadata_path = baseline_mesh_dir / "mesh.json"
baseline_run = baseline / "m0_a5"
baseline_field = baseline_run / "field.npz"
baseline_result_path = baseline_run / "result.json"
baseline_analysis_path = baseline_run / "analysis.json"
build_out = args.out / "mesh"
solve_out = args.out / "solve"
matched_baseline_out = args.out / "baseline_current_solver"
baseline_indicator_out = args.out / "baseline_indicator"
adapted_indicator_out = args.out / "adapted_indicator"
background_path = args.out / "background_view.npz"

source_paths = [
    Path(__file__),
    ROOT / "build_mesh.py",
    ROOT / "solve.py",
    ROOT / "analyze.py",
    VOLUME / "p2.py",
    VOLUME / "error_indicator.py",
    ROOT / "LIFTING_ADAPTATION_PROTOCOL.md",
]
result = {
    "status": "failed",
    "scope": "single pre-registered adaptive-mesh comparison; experimental lifting reference",
    "geometry": {"thickness": 0.24, "body_length": 3.0, "body_radius": 0.35,
                 "body_aft": 0.9, "span": 4.0, "chord": 1.0, "reference_area": 4.0},
    "flow": {"mach": 0.0, "alpha_deg": 5.0, "kutta": "weak", "ncpu": args.ncpu},
    "mesh_target": {"h": 0.2, "far_size": 5.0 / 3.0, "curvature": 32,
                    "wake_strips": 24, "marked_and_wall_junction_size": 0.1},
    "history": [],
}

try:
    if args.ncpu != 8:
        raise ValueError("this pre-registered comparison requires --ncpu 8")
    if (args.out / "result.json").exists():
        raise FileExistsError(f"refusing to overwrite prior experiment evidence: {args.out / 'result.json'}")
    for path in [baseline_mesh, baseline_mesh_metadata_path, baseline_field,
                 baseline_result_path, baseline_analysis_path]:
        if not path.is_file():
            raise FileNotFoundError(f"required baseline evidence is missing: {path}")
    source_hashes = {str(path.relative_to(ROOT.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source_paths}
    result["source_sha256"] = source_hashes
    mesh_bytes = baseline_mesh.read_bytes()
    field_bytes = baseline_field.read_bytes()
    base_result_bytes = baseline_result_path.read_bytes()
    base_analysis_bytes = baseline_analysis_path.read_bytes()
    mesh_sha = hashlib.sha256(mesh_bytes).hexdigest()
    field_sha = hashlib.sha256(field_bytes).hexdigest()
    mesh_metadata = json.loads(baseline_mesh_metadata_path.read_text(encoding="utf-8"))
    base_solver = json.loads(base_result_bytes)
    base_analysis = json.loads(base_analysis_bytes)
    baseline_solver_snapshot = baseline_run / "solve.py.snapshot"
    baseline_solver_snapshot_sha = hashlib.sha256(baseline_solver_snapshot.read_bytes()).hexdigest()
    current_solver_sha = hashlib.sha256((ROOT / "solve.py").read_bytes()).hexdigest()
    if base_solver.get("mesh_sha256") != mesh_sha or base_analysis.get("mesh_sha256") != mesh_sha:
        raise ValueError("baseline mesh hash does not match solver/analysis provenance")
    if base_analysis.get("field_sha256") != field_sha:
        raise ValueError("baseline analysis field hash does not match supplied field")
    if (mesh_metadata.get("thickness"), mesh_metadata.get("body_radius"),
            mesh_metadata.get("body_aft"), mesh_metadata.get("body_length"),
            mesh_metadata.get("size"), mesh_metadata.get("curvature_elements_per_circle"),
            mesh_metadata.get("wake_strips")) != (0.24, 0.35, 0.9, 3.0, 0.2, 32, 24):
        raise ValueError("baseline mesh metadata does not match the frozen geometry")
    if (base_solver.get("mach"), base_solver.get("alpha_deg"),
            base_solver.get("kutta_formulation"), base_solver.get("converged")) != (0.0, 5.0, "weak", True):
        raise ValueError("baseline solver record does not match frozen flow/Kutta case")
    result["baseline_provenance"] = {
        "mesh_sha256": mesh_sha,
        "field_sha256": field_sha,
        "result_sha256": hashlib.sha256(base_result_bytes).hexdigest(),
        "analysis_sha256": hashlib.sha256(base_analysis_bytes).hexdigest(),
        "mesh_metadata_sha256": hashlib.sha256(baseline_mesh_metadata_path.read_bytes()).hexdigest(),
        "historical_solver_source_sha256": baseline_solver_snapshot_sha,
        "current_solver_source_sha256": current_solver_sha,
        "historical_solver_source_matches_current": baseline_solver_snapshot_sha == current_solver_sha,
        "mesh": str(baseline_mesh.relative_to(ROOT)),
        "field": str(baseline_field.relative_to(ROOT)),
        "source": "wingbody_matrix_v1/t024_r035_a090/m0_a5",
    }

    matched_baseline_out.mkdir(parents=True, exist_ok=True)
    matched_solve_command = [
        sys.executable, str(ROOT / "solve.py"), "--mesh", str(baseline_mesh),
        "--alpha", "5", "--mach", "0", "--ncpu", "8", "--kutta", "weak",
        "--flux-audit", "--out", str(matched_baseline_out),
    ]
    tick = time.perf_counter()
    matched_solve = subprocess.run(matched_solve_command, capture_output=True, text=True)
    matched_solve_seconds = time.perf_counter() - tick
    (args.out / "baseline_current_solver.log").write_text(
        matched_solve.stdout + matched_solve.stderr, encoding="utf-8")
    result["history"].append({"stage": "baseline_resolve_current_source",
                              "returncode": matched_solve.returncode,
                              "elapsed_seconds": matched_solve_seconds,
                              "command": matched_solve_command})
    if matched_solve.returncode or not (matched_baseline_out / "result.json").is_file():
        raise RuntimeError("same-mesh baseline re-solve with current solver failed")
    matched_baseline_result = json.loads(
        (matched_baseline_out / "result.json").read_text(encoding="utf-8"))
    matched_analysis_command = [
        sys.executable, str(ROOT / "analyze.py"),
        "--run", str(matched_baseline_out), "--mesh", str(baseline_mesh),
    ]
    matched_analysis = subprocess.run(matched_analysis_command, capture_output=True, text=True)
    (args.out / "baseline_current_solver_analyze.log").write_text(
        matched_analysis.stdout + matched_analysis.stderr, encoding="utf-8")
    result["history"].append({"stage": "baseline_current_source_analysis",
                              "returncode": matched_analysis.returncode})
    if matched_analysis.returncode or not (matched_baseline_out / "analysis.json").is_file():
        raise RuntimeError("same-mesh baseline raw-field analysis failed")
    matched_baseline_analysis = json.loads(
        (matched_baseline_out / "analysis.json").read_text(encoding="utf-8"))
    matched_baseline_field = matched_baseline_out / "field.npz"
    baseline_indicator_out.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    with threadpool_limits(limits=args.ncpu):
        mesh_data, field_data, points, tets, phi, faces, tags, potential_key, _, _ = \
            error_indicator.load_inputs(baseline_mesh, matched_baseline_field)
        if potential_key != "phi":
            raise ValueError("lifting baseline must use nodal potential key 'phi'")
        baseline_indicator = error_indicator.compute_indicator(
            mesh_data, points, tets, phi, faces, tags, 0.0, 3)
    elapsed = time.perf_counter() - tick
    matched_field_sha = hashlib.sha256(matched_baseline_field.read_bytes()).hexdigest()
    baseline_indicator_summary = {
        key: value for key, value in baseline_indicator.items()
        if not isinstance(value, np.ndarray)
    }
    baseline_indicator_summary.update(
        status="measured",
        mesh_sha256=mesh_sha,
        field_sha256=matched_field_sha,
        quadrature_order=3,
        elapsed_seconds=elapsed,
        indicator_source_sha256=source_hashes[str((VOLUME / "error_indicator.py").relative_to(ROOT.parent))],
    )
    np.savez_compressed(
        baseline_indicator_out / "indicator.npz",
        cell_bulk=baseline_indicator["cell_bulk"],
        cell_internal=baseline_indicator["cell_internal"],
        cell_wake_cut=baseline_indicator["cell_wake"],
        cell_wall=baseline_indicator["cell_wall"],
        cell_total=baseline_indicator["cell_total"],
        marked=baseline_indicator["marked"],
        dorfler_order=baseline_indicator["dorfler_order"],
        dorfler_cumulative_fraction=baseline_indicator["dorfler_cumulative_fraction"],
    )
    (baseline_indicator_out / "result.json").write_text(
        json.dumps(baseline_indicator_summary, indent=2, allow_nan=False), encoding="utf-8")
    result["history"].append({"stage": "baseline_indicator", "status": "measured",
                              "elapsed_seconds": elapsed})

    marked = baseline_indicator["marked"]
    marked_nodes = np.unique(tets[marked].ravel())
    wall_mask = np.isin(tags, [10, 20, 30])
    wall_tets = np.unique(mesh_data["boundary_tet"][wall_mask])
    wall_tets = wall_tets[wall_tets >= 0]

    seam_edges = {}
    for face, tag in zip(faces[wall_mask], tags[wall_mask]):
        for first, second, midpoint in ((0, 1, 3), (1, 2, 4), (2, 0, 5)):
            edge = tuple(sorted((int(face[first]), int(face[second]))))
            seam_edges.setdefault(edge, []).append((int(tag), int(face[midpoint])))
    seam_nodes = []
    for edge, sides in seam_edges.items():
        if {side[0] for side in sides} == {20, 30}:
            seam_nodes.extend([edge[0], edge[1], sides[0][1]])
    if not seam_nodes:
        raise ValueError("baseline wall topology contains no shared tag-20/tag-30 seam")
    seam_nodes = np.unique(seam_nodes)
    seam_tree = cKDTree(points[seam_nodes])
    seam_distance, _ = seam_tree.query(points[tets].reshape(-1, 3), k=1)
    junction_tets = np.flatnonzero(np.min(seam_distance.reshape(len(tets), 10), axis=1) < 0.1)
    selected_tets = np.unique(np.r_[wall_tets, junction_tets])
    selected_wall_junction_nodes = np.unique(tets[selected_tets].ravel())
    fine_nodes = np.unique(np.r_[marked_nodes, selected_wall_junction_nodes])

    corner_z = points[tets[:, :4], 2]
    z_tolerance = 1.0e-12 * max(1.0, float(np.max(np.abs(points[:, 2]))))
    straddling = (corner_z.min(axis=1) < -z_tolerance) & (corner_z.max(axis=1) > z_tolerance)
    if np.any(straddling):
        raise ValueError(f"baseline has {int(np.count_nonzero(straddling))} tetrahedra crossing z=0")
    upper_element_mask = corner_z.min(axis=1) >= -z_tolerance
    lower_element_mask = corner_z.max(axis=1) <= z_tolerance
    if int(np.count_nonzero(upper_element_mask)) != int(np.count_nonzero(lower_element_mask)):
        raise ValueError("baseline reflected upper/lower element counts differ")
    upper_tets = tets[upper_element_mask]
    vertex_size = np.full(len(points), 5.0 / 3.0)
    vertex_size[fine_nodes] = 0.1
    np.savez_compressed(background_path, points=points, tets=upper_tets, vertex_size=vertex_size)
    background_sha = hashlib.sha256(background_path.read_bytes()).hexdigest()
    result["background_field"] = {
        "path": str(background_path.relative_to(args.out)),
        "sha256": background_sha,
        "source_mesh_sha256": mesh_sha,
        "source_field_sha256": field_sha,
        "source_indicator_field_sha256": baseline_indicator_summary["field_sha256"],
        "indicator_marked_cells": int(marked.sum()),
        "indicator_dorfler_fraction": baseline_indicator_summary["marked_contribution_fraction"],
        "indicator_marked_vertices": int(len(marked_nodes)),
        "wall_adjacent_cells": int(len(wall_tets)),
        "junction_adjacent_cells": int(len(junction_tets)),
        "wall_junction_union_cells": int(len(selected_tets)),
        "junction_seam_nodes": int(len(seam_nodes)),
        "fine_vertices": int(len(fine_nodes)),
        "far_vertices": int(len(points) - len(fine_nodes)),
        "upper_half_tets": int(len(upper_tets)),
        "full_mesh_tets": int(len(tets)),
        "upper_lower_count_equal": True,
        "mesh_field": "points,tets,vertex_size; upper-half P2 volume elements only",
    }

    build_command = [
        sys.executable, str(ROOT / "build_mesh.py"),
        "--size", "0.2", "--curvature", "32", "--wake-strips", "24",
        "--body-length", "3", "--body-radius", "0.35", "--body-aft", "0.9",
        "--thickness", "0.24", "--radius", "5", "--far-size", str(5.0 / 3.0),
        "--background-view", str(background_path), "--out", str(build_out),
    ]
    tick = time.perf_counter()
    build = subprocess.run(build_command, capture_output=True, text=True)
    build_seconds = time.perf_counter() - tick
    (args.out / "mesh_build.log").write_text(build.stdout + build.stderr, encoding="utf-8")
    result["history"].append({"stage": "mesh_build", "returncode": build.returncode,
                              "elapsed_seconds": build_seconds, "command": build_command})
    if build.returncode:
        raise RuntimeError(f"CAD mesh build failed with return code {build.returncode}")
    built_mesh = build_out / "mesh.npz"
    built_metadata_path = build_out / "mesh.json"
    if not built_mesh.is_file() or not built_metadata_path.is_file():
        raise FileNotFoundError("mesh builder returned success without mesh.npz and mesh.json")
    built_metadata = json.loads(built_metadata_path.read_text(encoding="utf-8"))
    built_background = built_metadata.get("background_size_field") or {}
    if built_background.get("source_sha256") != background_sha or built_metadata.get("gmsh_threads") != 1:
        raise ValueError("builder metadata does not confirm the expected background hash and one Gmsh thread")
    result["adapted_mesh"] = {
        "mesh_sha256": hashlib.sha256(built_mesh.read_bytes()).hexdigest(),
        "mesh_metadata_sha256": hashlib.sha256(built_metadata_path.read_bytes()).hexdigest(),
        "points": built_metadata.get("points"),
        "tets": built_metadata.get("tets"),
        "gmsh_threads": built_metadata.get("gmsh_threads"),
        "background_source_sha256": built_background.get("source_sha256"),
        "minimum_dense_jacobian": built_metadata.get("min_dense_jacobian"),
    }

    mesh_test_command = [sys.executable, str(ROOT / "test_mesh.py"), "--mesh", str(built_mesh)]
    tick = time.perf_counter()
    mesh_test = subprocess.run(mesh_test_command, capture_output=True, text=True)
    mesh_test_seconds = time.perf_counter() - tick
    (args.out / "mesh_test.log").write_text(mesh_test.stdout + mesh_test.stderr, encoding="utf-8")
    result["history"].append({"stage": "mesh_test", "returncode": mesh_test.returncode,
                              "elapsed_seconds": mesh_test_seconds, "command": mesh_test_command})
    if mesh_test.returncode:
        raise RuntimeError(f"adapted mesh tests failed with return code {mesh_test.returncode}")

    solve_command = [
        sys.executable, str(ROOT / "solve.py"), "--mesh", str(built_mesh),
        "--alpha", "5", "--mach", "0", "--ncpu", "8", "--kutta", "weak",
        "--flux-audit", "--out", str(solve_out),
    ]
    tick = time.perf_counter()
    solve = subprocess.run(solve_command, capture_output=True, text=True)
    solve_seconds = time.perf_counter() - tick
    (args.out / "solve.log").write_text(solve.stdout + solve.stderr, encoding="utf-8")
    result["history"].append({"stage": "solve", "returncode": solve.returncode,
                              "elapsed_seconds": solve_seconds, "command": solve_command})
    if not (solve_out / "result.json").is_file() or not (solve_out / "field.npz").is_file():
        raise RuntimeError("solver did not preserve result.json and field.npz")
    adapted_result = json.loads((solve_out / "result.json").read_text(encoding="utf-8"))
    if solve.returncode or not adapted_result.get("converged"):
        result["adapted_solver_result"] = adapted_result
        raise RuntimeError(f"adapted solver did not converge (return code {solve.returncode})")

    analyze_command = [sys.executable, str(ROOT / "analyze.py"),
                       "--run", str(solve_out), "--mesh", str(built_mesh)]
    tick = time.perf_counter()
    analysis_run = subprocess.run(analyze_command, capture_output=True, text=True)
    analysis_seconds = time.perf_counter() - tick
    (args.out / "analyze.log").write_text(
        analysis_run.stdout + analysis_run.stderr, encoding="utf-8")
    result["history"].append({"stage": "analyze", "returncode": analysis_run.returncode,
                              "elapsed_seconds": analysis_seconds, "command": analyze_command})
    if analysis_run.returncode or not (solve_out / "analysis.json").is_file():
        raise RuntimeError("raw surface analysis failed")
    adapted_analysis = json.loads((solve_out / "analysis.json").read_text(encoding="utf-8"))

    adapted_indicator_out.mkdir(parents=True, exist_ok=True)
    with np.load(built_mesh, allow_pickle=False) as archive:
        adapted_mesh_data = {key: archive[key] for key in archive.files}
    with np.load(solve_out / "field.npz", allow_pickle=False) as archive:
        adapted_field_data = {key: archive[key] for key in archive.files}
    if not np.array_equal(adapted_mesh_data["points"], adapted_field_data["points"]):
        raise ValueError("adapted solver field points differ from the input mesh ordering")
    if not np.array_equal(adapted_mesh_data["tets"], adapted_field_data["tets"]):
        raise ValueError("adapted solver field tets differ from the input mesh ordering")
    adapted_mesh_bytes = built_mesh.read_bytes()
    adapted_field_bytes = (solve_out / "field.npz").read_bytes()
    adapted_phi = np.asarray(adapted_field_data["phi"])
    adapted_faces = np.asarray(adapted_mesh_data["boundary_triangles"], dtype=np.int64)
    adapted_tags = np.asarray(adapted_mesh_data["boundary_tags"], dtype=np.int64)
    with threadpool_limits(limits=args.ncpu):
        adapted_indicator = error_indicator.compute_indicator(
            adapted_mesh_data, adapted_mesh_data["points"], adapted_mesh_data["tets"],
            adapted_phi, adapted_faces, adapted_tags, 0.0, 3)
    np.savez_compressed(
        adapted_indicator_out / "indicator.npz",
        cell_bulk=adapted_indicator["cell_bulk"],
        cell_internal=adapted_indicator["cell_internal"],
        cell_wake_cut=adapted_indicator["cell_wake"],
        cell_wall=adapted_indicator["cell_wall"],
        cell_total=adapted_indicator["cell_total"],
        marked=adapted_indicator["marked"],
        dorfler_order=adapted_indicator["dorfler_order"],
        dorfler_cumulative_fraction=adapted_indicator["dorfler_cumulative_fraction"],
    )
    adapted_indicator_summary = {
        key: value for key, value in adapted_indicator.items()
        if not isinstance(value, np.ndarray)
    }
    adapted_indicator_summary.update(
        status="measured",
        mesh_sha256=hashlib.sha256(adapted_mesh_bytes).hexdigest(),
        field_sha256=hashlib.sha256(adapted_field_bytes).hexdigest(),
        quadrature_order=3,
        indicator_source_sha256=source_hashes[str((VOLUME / "error_indicator.py").relative_to(ROOT.parent))],
    )
    (adapted_indicator_out / "result.json").write_text(
        json.dumps(adapted_indicator_summary, indent=2, allow_nan=False), encoding="utf-8")

    with np.load(matched_baseline_field, allow_pickle=False) as old_field:
        old_cp = np.asarray(old_field["cp"])
        old_gamma = np.asarray(old_field["gamma"])
        old_tags = np.asarray(old_field["tags"])
        old_weight = np.asarray(old_field["surface_weight"])
    with np.load(solve_out / "field.npz", allow_pickle=False) as new_field:
        new_cp = np.asarray(new_field["cp"])
        new_gamma = np.asarray(new_field["gamma"])
        new_tags = np.asarray(new_field["tags"])
        new_weight = np.asarray(new_field["surface_weight"])
    old_finite = bool(np.isfinite(old_cp).all())
    new_finite = bool(np.isfinite(new_cp).all())
    cp_by_tag = {}
    for tag in [10, 20, 30]:
        old_selected = old_tags == tag
        new_selected = new_tags == tag
        if np.any(old_selected) and np.any(new_selected):
            old_w = old_weight[old_selected]
            new_w = new_weight[new_selected]
            old_values = old_cp[old_selected]
            new_values = new_cp[new_selected]
            old_mean = float(np.average(old_values, weights=old_w))
            new_mean = float(np.average(new_values, weights=new_w))
            cp_by_tag[str(tag)] = {
                "baseline_area_weighted_mean": old_mean,
                "adapted_area_weighted_mean": new_mean,
                "mean_delta": new_mean - old_mean,
                "baseline_area_weighted_rms": float(np.sqrt(np.average(old_values**2, weights=old_w))),
                "adapted_area_weighted_rms": float(np.sqrt(np.average(new_values**2, weights=new_w))),
            }
    baseline_summary = {
        "historical_matrix_result": base_solver,
        "historical_matrix_analysis": base_analysis,
        "result": matched_baseline_result,
        "analysis": matched_baseline_analysis,
        "raw_cp": {"finite": old_finite, "count": int(old_cp.size),
                   "finite_count": int(np.isfinite(old_cp).sum()),
                   "min": float(np.nanmin(old_cp)), "max": float(np.nanmax(old_cp))},
        "indicator": {"bulk": baseline_indicator_summary["bulk_total"],
                      "internal": baseline_indicator_summary["internal_total"],
                      "wake": baseline_indicator_summary["wake_total"],
                      "wall": baseline_indicator_summary["wall_total"],
                      "total": baseline_indicator_summary["dorfler_total"]},
    }
    adapted_summary = {
        "result": adapted_result,
        "analysis": adapted_analysis,
        "raw_cp": {"finite": new_finite, "count": int(new_cp.size),
                   "finite_count": int(np.isfinite(new_cp).sum()),
                   "min": float(np.nanmin(new_cp)), "max": float(np.nanmax(new_cp))},
        "indicator": {"bulk": adapted_indicator_summary["bulk_total"],
                      "internal": adapted_indicator_summary["internal_total"],
                      "wake": adapted_indicator_summary["wake_total"],
                      "wall": adapted_indicator_summary["wall_total"],
                      "total": adapted_indicator_summary["dorfler_total"]},
    }
    result["comparison"] = {
        "baseline": baseline_summary,
        "adapted": adapted_summary,
        "delta": {
            "cl_pressure": adapted_result["cl_pressure"] - matched_baseline_result["cl_pressure"],
            "cl_gamma": adapted_result["cl_jump"] - matched_baseline_result["cl_jump"],
            "cd_pressure": adapted_result["cd_pressure"] - matched_baseline_result["cd_pressure"],
            "wall_normal_rms": adapted_result["wall_normal_rms"] - matched_baseline_result["wall_normal_rms"],
            "junction_normal_rms": (
                adapted_analysis.get("regions", {}).get("junction", {}).get("normal_rms", float("nan"))
                - matched_baseline_analysis.get("regions", {}).get("junction", {}).get("normal_rms", float("nan"))),
            "weak_mass_residual_norm": (
                adapted_result["weak_mass_residual_norm"] - matched_baseline_result["weak_mass_residual_norm"]),
            "pressure_circulation_lift_difference": (
                adapted_analysis["pressure_circulation_lift_difference"]
                - matched_baseline_analysis["pressure_circulation_lift_difference"]),
        },
        "gamma_station_comparison": {
            "stations_equal": bool(np.array_equal(
                np.asarray(matched_baseline_result["gamma_stations"]),
                np.asarray(adapted_result["gamma_stations"]))),
            "baseline_stations": matched_baseline_result["gamma_stations"],
            "adapted_stations": adapted_result["gamma_stations"],
            "baseline": old_gamma.tolist(),
            "adapted": new_gamma.tolist(),
            "pointwise_delta": (new_gamma - old_gamma).tolist()
            if (len(new_gamma) == len(old_gamma)
                and np.array_equal(np.asarray(matched_baseline_result["gamma_stations"]),
                                   np.asarray(adapted_result["gamma_stations"]))) else None,
            "note": "Gamma vector comparison is station-wise only when vector lengths and station arrays agree.",
        },
        "raw_cp_area_weighted_by_tag": cp_by_tag,
        "indicator_reduction_claim": False,
        "physical_gate_values_are_not_rethresholded": True,
    }
    result["gate_record"] = {
        "adapted_solver_converged": bool(adapted_result.get("converged")),
        "component_force_closure_error": adapted_analysis.get("component_sum_error"),
        "component_force_closure_passed_at_existing_1e-10": bool(
            adapted_analysis.get("component_sum_error", float("inf")) < 1.0e-10),
        "raw_cp_all_finite": new_finite,
        "raw_wall_normal_rms": adapted_result.get("wall_normal_rms"),
        "raw_junction_normal_rms": adapted_analysis.get("regions", {}).get("junction", {}).get("normal_rms"),
        "raw_mass_by_tag": adapted_result.get("raw_mass_by_tag"),
        "weak_mass_residual_norm": adapted_result.get("weak_mass_residual_norm"),
        "weak_gauge_reaction": adapted_result.get("weak_gauge_reaction"),
        "wake_diagnostics": adapted_result.get("wake_diagnostics"),
        "internal_face_flux_jump": adapted_result.get("internal_face_flux_jump"),
        "existing_thresholds_changed": False,
    }
    result.update(status="completed_with_gate_results",
                  baseline_indicator_seconds=baseline_indicator_summary["elapsed_seconds"],
                  adapted_indicator_elements=len(adapted_mesh_data["tets"]),
                  adapted_indicator_total=adapted_indicator_summary["dorfler_total"],
                  elapsed_seconds=time.perf_counter() - started)
except Exception as error:
    result.update(status="failed", error=f"{type(error).__name__}: {error}",
                  elapsed_seconds=time.perf_counter() - started)
finally:
    (args.out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")

print(json.dumps({key: result.get(key) for key in
                  ("status", "error", "background_field", "adapted_mesh",
                   "gate_record", "elapsed_seconds")}, indent=2))
if result["status"] != "completed_with_gate_results":
    raise SystemExit(1)
