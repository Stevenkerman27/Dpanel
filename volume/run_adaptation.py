"""Run one serial Gmsh PostView sphere adaptation and compare physical gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GMsh_RUNTIME = ROOT / "audit" / "runtime" / "python"
sys.path.insert(0, str(GMsh_RUNTIME))
import gmsh


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smoke_test_postview():
    gmsh.initialize()
    try:
        gmsh.model.add("postview_api_smoke")
        view = gmsh.view.add("scalar size on tetrahedron")
        coordinates = np.array(
            [[0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
        )
        sizes = np.full(4, 0.2)
        data = np.r_[coordinates[0], coordinates[1], coordinates[2], sizes]
        gmsh.view.addListData(view, "SS", 1, data.tolist())
        field = gmsh.model.mesh.field.add("PostView")
        gmsh.model.mesh.field.setNumber(field, "ViewTag", view)
        gmsh.model.mesh.field.setAsBackgroundMesh(field)
        volume = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        tet_type = gmsh.model.mesh.getElementType("Tetrahedron", 1)
        _, tet_nodes = gmsh.model.mesh.getElementsByType(tet_type)
        if len(node_tags) < 8 or len(tet_nodes) == 0:
            raise RuntimeError("PostView smoke CAD mesh generation returned no volume mesh")
        return {
            "gmsh_version": gmsh.__version__,
            "data_type": "SS",
            "input_elements": 1,
            "generated_nodes": len(node_tags),
            "generated_linear_tets": len(tet_nodes) // 4,
            "postview_field": field,
            "status": "passed",
        }
    finally:
        gmsh.finalize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh", type=Path, default=ROOT / "volume/meshes/p2/sphere_coarse/mesh.npz"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "audit/adaptation_sphere_v3")
    parser.add_argument("--ncpu", type=int, default=8)
    parser.add_argument(
        "--wall-refine",
        action="store_true",
        help="Use 0.5*base size on Dörfler-marked or wall-adjacent corner nodes",
    )
    args = parser.parse_args()
    args.mesh = args.mesh.resolve()
    args.out = args.out.resolve()
    if args.ncpu < 1:
        parser.error("--ncpu must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"Refusing to overwrite non-empty output: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    status = {"status": "failed", "history": []}
    try:
        smoke = smoke_test_postview()
        with np.load(args.mesh) as archive:
            mesh = {key: archive[key] for key in archive.files}
        with open(args.mesh.with_name("mesh.json"), encoding="utf-8") as source:
            metadata = json.load(source)
        if metadata.get("shape") != "sphere":
            raise ValueError("adaptation driver currently accepts only shape='sphere'")
        tets = np.asarray(mesh["tets"], dtype=np.int64)
        base_size = float(metadata["mesh_size"])
        baseline_solve_out = args.out / "baseline_solve_m0_sphere_exact_q3"
        baseline_command = [
            sys.executable,
            str(ROOT / "volume/solve_p2.py"),
            "--mesh",
            str(args.mesh),
            "--mach",
            "0",
            "--boundary",
            "sphere-exact",
            "--quad-order",
            "3",
            "--ncpu",
            str(args.ncpu),
            "--out",
            str(baseline_solve_out),
        ]
        baseline_run = subprocess.run(
            baseline_command, cwd=ROOT, env=os.environ.copy(),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        (args.out / "baseline_solve.log").write_text(
            baseline_run.stdout, encoding="utf-8"
        )
        if baseline_run.returncode:
            raise RuntimeError(
                f"source-mesh P2 baseline solve failed: {baseline_run.returncode}"
            )
        baseline_field_path = baseline_solve_out / "field.npz"
        baseline_result_path = baseline_field_path.with_name("result.json")
        baseline = json.loads(baseline_result_path.read_text(encoding="utf-8"))
        if baseline.get("status") != "converged":
            raise ValueError("current source-mesh baseline is not converged")
        baseline_field_hash = sha256(baseline_field_path)
        indicator_out = args.out / "baseline_indicator_m0_q3"
        indicator_command = [
            sys.executable,
            str(ROOT / "volume/error_indicator.py"),
            "--mesh",
            str(args.mesh),
            "--field",
            str(baseline_field_path),
            "--mach",
            "0",
            "--quad-order",
            "3",
            "--ncpu",
            str(args.ncpu),
            "--out",
            str(indicator_out),
        ]
        indicator_run = subprocess.run(
            indicator_command, cwd=ROOT, env=os.environ.copy(),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        (args.out / "baseline_indicator.log").write_text(
            indicator_run.stdout, encoding="utf-8"
        )
        indicator_result_path = indicator_out / "result.json"
        indicator_result = json.loads(indicator_result_path.read_text(encoding="utf-8"))
        if indicator_run.returncode or indicator_result.get("status") != "measured":
            raise RuntimeError(
                f"source-mesh indicator failed: {indicator_result.get('error')}"
            )
        if indicator_result.get("field_sha256") != baseline_field_hash:
            raise ValueError(
                "baseline indicator field_sha256 does not match the exact baseline field file"
            )
        baseline_indicator_summary = {
            key: indicator_result[key]
            for key in ("dorfler_total", "bulk_total", "internal_total", "wall_total", "marked_count")
        }
        with np.load(indicator_out / "indicator.npz") as archive:
            indicator = {key: archive[key] for key in archive.files}
        marks = np.asarray(indicator["marked"], dtype=bool)
        if marks.shape != (len(tets),):
            raise ValueError("Dorfler cell marks do not match source tetrahedra")
        target_size = np.full(len(mesh["points"]), base_size, dtype=np.float64)
        marked_cells = np.flatnonzero(marks)
        marked_vertex_nodes = np.unique(tets[marked_cells, :4])
        wall_adjacent_elements = np.empty(0, dtype=np.int64)
        wall_vertex_nodes = np.empty(0, dtype=np.int64)
        farfield_vertex_nodes = np.empty(0, dtype=np.int64)
        if args.wall_refine:
            corner_face_to_tet = {}
            for element, tet in enumerate(tets):
                for omitted in range(4):
                    face_corners = np.delete(tet[:4], omitted)
                    corner_face_to_tet[tuple(sorted(face_corners.tolist()))] = element
            wall_elements = []
            wall_triangles = np.asarray(mesh["boundary_triangles"], dtype=np.int64)
            boundary_tags = np.asarray(mesh["boundary_tags"], dtype=np.int64)
            for face in wall_triangles[boundary_tags != 100]:
                face_key = tuple(sorted(face[:3].tolist()))
                if face_key not in corner_face_to_tet:
                    raise ValueError("could not map a physical wall face to its source tetrahedron")
                wall_elements.append(corner_face_to_tet[face_key])
            wall_adjacent_elements = np.unique(wall_elements)
            wall_vertex_nodes = np.unique(tets[wall_adjacent_elements, :4])
            farfield_vertex_nodes = np.unique(wall_triangles[boundary_tags == 100, :3])
            refine_vertex_nodes = np.union1d(marked_vertex_nodes, wall_vertex_nodes)
            target_size[refine_vertex_nodes] = 0.5 * base_size
            target_size[farfield_vertex_nodes] = base_size
            adaptation_method = "dorfler50_union_wall_adjacent_corner_nodes_h50"
            target_factor = 0.5
        else:
            target_size[marked_vertex_nodes] = 0.65 * base_size
            adaptation_method = "dorfler50_marked_corner_nodes_h65"
            target_factor = 0.65
            refine_vertex_nodes = marked_vertex_nodes
        background_path = args.out / "background_size.npz"
        np.savez_compressed(
            background_path,
            points=mesh["points"],
            tets=tets[:, :4],
            vertex_size=target_size,
        )
        adaptation_config = {
            "method": adaptation_method,
            "dorfler_fraction": 0.5,
            "base_size": base_size,
            "refinement_factor": target_factor,
            "marked_cells": int(np.count_nonzero(marks)),
            "marked_corner_nodes": int(len(marked_vertex_nodes)),
            "wall_adjacent_tets": int(len(wall_adjacent_elements)),
            "wall_adjacent_corner_nodes": int(len(wall_vertex_nodes)),
            "refined_corner_nodes_before_farfield_override": int(
                len(np.union1d(marked_vertex_nodes, wall_vertex_nodes))
            ),
            "farfield_corner_nodes_held_at_base": int(len(farfield_vertex_nodes)),
            "final_refined_corner_nodes": int(
                np.count_nonzero(target_size < base_size)
            ),
            "farfield_target_size": base_size,
            "reason": (
                "The wall residual and curved wall geometry are separate refinement targets "
                "from the volume-energy Dörfler indicator."
                if args.wall_refine
                else "Original v7 default indicator-only mapping."
            ),
        }
        preflight = {
            "status": "registered_before_remesh",
            "source_mesh": str(args.mesh),
            "source_mesh_sha256": sha256(args.mesh),
            "source_field": str(baseline_field_path),
            "source_field_sha256": baseline_field_hash,
            "source_indicator": str(indicator_out / "indicator.npz"),
            "source_indicator_sha256": sha256(indicator_out / "indicator.npz"),
            "indicator_field_sha256": indicator_result["field_sha256"],
            "indicator_matches_baseline_field": True,
            "baseline_result": str(baseline_result_path),
            "baseline_result_sha256": sha256(baseline_result_path),
            "baseline_indicator_result": str(indicator_result_path),
            "baseline_indicator_result_sha256": sha256(indicator_result_path),
            "postview_smoke_test": smoke,
            "mesh_points": len(mesh["points"]),
            "mesh_tets": len(tets),
            "marked_cells": int(np.count_nonzero(marks)),
            "adaptation": adaptation_config,
            "dorfler_fraction": 0.5,
            "base_size": base_size,
            "marked_vertex_target_size": target_factor * base_size,
            "unmarked_vertex_target_size": base_size,
            "gates": {
                "cp_area_rms_and_max": "strictly below baseline",
                "absolute_lift_and_drag": "no larger than baseline",
                "wall_velocity_rms_mass_flux_rms_and_boundary_closure": "no larger than baseline",
                "weak_free_residual": "at most 1e-10",
                "positive_sampled_curved_jacobian_and_body_farfield_tags": "required",
                "indicator_total": "diagnostic only; not a physical acceptance gate",
            },
            "baseline": {
                "sphere_reference": baseline["sphere_reference"],
                "loads": baseline["loads"],
                "diagnostics": baseline["diagnostics"],
            },
        }
        (args.out / "preflight.json").write_text(
            json.dumps(preflight, indent=2, allow_nan=False), encoding="utf-8"
        )
        (args.out / "adaptation_config.json").write_text(
            json.dumps(adaptation_config, indent=2, allow_nan=False), encoding="utf-8"
        )
        status.update(preflight=preflight)
        mesh_out = args.out / "mesh"
        builder = ROOT / "volume/build_curved_mesh.py"
        command = [
            sys.executable,
            str(builder),
            "--shape",
            "sphere",
            "--size",
            str(base_size),
            "--radius",
            str(metadata["reference_radius"]),
            "--farfield-factor",
            str(metadata["farfield_radius"] / metadata["reference_radius"]),
            "--out",
            str(mesh_out),
            "--background-view",
            str(background_path),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(GMsh_RUNTIME) + os.pathsep + env.get("PYTHONPATH", "")
        (args.out / "remesh_command.json").write_text(
            json.dumps({"command": command}, indent=2), encoding="utf-8"
        )
        try:
            with (args.out / "remesh.log").open("w", encoding="utf-8") as log:
                run = subprocess.run(
                    command, cwd=ROOT, env=env, text=True, stdout=log,
                    stderr=subprocess.STDOUT, timeout=300,
                )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("CAD remesh timed out after 300 seconds") from error
        if run.returncode:
            raise RuntimeError(f"CAD remesh failed with exit code {run.returncode}")
        status["history"].append({"step": "cad_remesh", "status": "passed"})

        solve_out = args.out / "solve_m0_sphere_exact_q3"
        solve_command = [
            sys.executable,
            str(ROOT / "volume/solve_p2.py"),
            "--mesh",
            str(mesh_out / "mesh.npz"),
            "--mach",
            "0",
            "--boundary",
            "sphere-exact",
            "--quad-order",
            "3",
            "--ncpu",
            str(args.ncpu),
            "--out",
            str(solve_out),
        ]
        solve = subprocess.run(solve_command, cwd=ROOT, env=env, text=True, capture_output=True)
        (args.out / "solve_stdout.txt").write_text(solve.stdout, encoding="utf-8")
        (args.out / "solve_stderr.txt").write_text(solve.stderr, encoding="utf-8")
        result_path = solve_out / "result.json"
        solved = json.loads(result_path.read_text(encoding="utf-8"))
        if solve.returncode or solved.get("status") != "converged":
            raise RuntimeError(f"P2 solver failed: {solved.get('error', solve.returncode)}")
        status["history"].append({"step": "solve_p2", "status": "passed"})

        indicator_out = args.out / "indicator_m0_q3"
        indicator_command = [
            sys.executable,
            str(ROOT / "volume/error_indicator.py"),
            "--mesh",
            str(mesh_out / "mesh.npz"),
            "--field",
            str(solve_out / "field.npz"),
            "--mach",
            "0",
            "--quad-order",
            "3",
            "--ncpu",
            str(args.ncpu),
            "--out",
            str(indicator_out),
        ]
        measured = subprocess.run(
            indicator_command, cwd=ROOT, env=env, text=True, capture_output=True
        )
        (args.out / "indicator_stdout.txt").write_text(measured.stdout, encoding="utf-8")
        (args.out / "indicator_stderr.txt").write_text(measured.stderr, encoding="utf-8")
        indicator_result = json.loads((indicator_out / "result.json").read_text(encoding="utf-8"))
        if measured.returncode or indicator_result.get("status") != "measured":
            raise RuntimeError(f"error indicator failed: {indicator_result.get('error')}")
        status["history"].append({"step": "indicator", "status": "passed"})

        baseline_ref = baseline["sphere_reference"]
        adapted_ref = solved["sphere_reference"]
        baseline_flux = baseline["diagnostics"]["raw_boundary_tet10_gradient_normal_metrics"]["10"]
        adapted_flux = solved["diagnostics"]["raw_boundary_tet10_gradient_normal_metrics"]["10"]
        adapted_closure = abs(solved["diagnostics"]["raw_total_boundary_integrated_mass_flux"])
        baseline_closure = abs(baseline["diagnostics"]["raw_total_boundary_integrated_mass_flux"])
        cp_rms_pass = adapted_ref["area_weighted_cp_rms_error"] < baseline_ref["area_weighted_cp_rms_error"]
        cp_max_pass = adapted_ref["max_abs_cp_error"] < baseline_ref["max_abs_cp_error"]
        lift_pass = abs(adapted_ref["lift_error_over_sref"]) <= abs(baseline_ref["lift_error_over_sref"])
        drag_pass = abs(adapted_ref["drag_error_over_sref"]) <= abs(baseline_ref["drag_error_over_sref"])
        wall_velocity_pass = (
            adapted_flux["area_weighted_rms_normal_velocity"]
            <= baseline_flux["area_weighted_rms_normal_velocity"]
        )
        wall_mass_pass = (
            adapted_flux["area_weighted_rms_normal_mass_flux"]
            <= baseline_flux["area_weighted_rms_normal_mass_flux"]
        )
        closure_pass = adapted_closure <= baseline_closure
        remesh_metadata = json.loads((mesh_out / "mesh.json").read_text(encoding="utf-8"))
        tags = remesh_metadata["physical_tags"]
        tag_pass = "farfield" in tags and (
            "sphere_or_ellipsoid" in tags or "body" in tags
        )
        quality_pass = remesh_metadata["curved_jacobian_check"]["nonpositive_jacobian_samples"] == 0
        residual_pass = solved["convergence"]["final_free_residual_norm"] <= 1.0e-10
        gates = {
            "cp_rms_strict_improvement": bool(cp_rms_pass),
            "cp_max_strict_improvement": bool(cp_max_pass),
            "lift_error_nonincrease": bool(lift_pass),
            "drag_error_nonincrease": bool(drag_pass),
            "raw_wall_velocity_rms_nonincrease": bool(wall_velocity_pass),
            "raw_wall_mass_flux_rms_nonincrease": bool(wall_mass_pass),
            "raw_boundary_mass_closure_nonincrease": bool(closure_pass),
            "weak_free_residual": bool(residual_pass),
            "positive_curved_jacobian_samples": bool(quality_pass),
            "body_and_farfield_tags": bool(tag_pass),
        }
        status.update(
            status="completed",
            adaptation=adaptation_config,
            physical_gates=gates,
            all_physical_gates_passed=all(gates.values()),
            adapted={
                "mesh_sha256": sha256(mesh_out / "mesh.npz"),
                "counts": remesh_metadata["counts"],
                "mesh_quality": remesh_metadata["quality"],
                "curved_jacobian_check": remesh_metadata["curved_jacobian_check"],
                "sphere_reference": adapted_ref,
                "loads": solved["loads"],
                "diagnostics": solved["diagnostics"],
                "indicator": {
                    key: indicator_result[key]
                    for key in ("dorfler_total", "bulk_total", "internal_total", "wall_total", "marked_count")
                },
            },
            baseline_comparison={
                "cp_rms_ratio": adapted_ref["area_weighted_cp_rms_error"] / baseline_ref["area_weighted_cp_rms_error"],
                "cp_max_ratio": adapted_ref["max_abs_cp_error"] / baseline_ref["max_abs_cp_error"],
                "baseline_indicator_total": baseline_indicator_summary["dorfler_total"],
                "adapted_indicator_total": indicator_result["dorfler_total"],
                "baseline_indicator_components": baseline_indicator_summary,
                "adapted_indicator_components": {
                    key: indicator_result[key]
                    for key in ("dorfler_total", "bulk_total", "internal_total", "wall_total", "marked_count")
                },
            },
        )
    except Exception as error:
        status.update(error=f"{type(error).__name__}: {error}")
    finally:
        status["wall_seconds"] = time.perf_counter() - started
        (args.out / "result.json").write_text(
            json.dumps(status, indent=2, allow_nan=False), encoding="utf-8"
        )
    print(json.dumps(status, indent=2, allow_nan=False))
    return 0 if status.get("status") == "completed" and status.get("all_physical_gates_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
