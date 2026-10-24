"""Build and solve one translated STEP ellipsoid, then check independent analytic Cp."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VOLUME = ROOT / "volume"
RESULTS = VOLUME / "results" / "cad_ellipsoid_v1"
OUT = RESULTS / "split_step_replay_v2"
sys.path.insert(0, str(ROOT / "audit" / "runtime" / "python"))
import gmsh

assert not OUT.exists(), "Preserve existing CAD ellipsoid evidence"
OUT.mkdir(parents=True)
native_axes = np.array([3.0, 1.0, 1.0])
native_center = np.array([2.0, -3.0, 1.0])
ellipsoid = OUT / "ellipsoid"
ellipsoid.mkdir()
step_path = ellipsoid / "input.step"

gmsh.initialize()
gmsh.model.add("translated_prolate_ellipsoid")
occ = gmsh.model.occ
solid = occ.addSphere(*native_center, 1.0)
occ.dilate([(3, solid)], *native_center, *native_axes)
solid_copy = occ.copy([(3, solid)])[0][1]
left_box = occ.addBox(-10.0, -10.0, -10.0, native_center[0] + 10.0, 20.0, 20.0)
right_box = occ.addBox(native_center[0], -10.0, -10.0, 10.0, 20.0, 20.0)
left_halves, _ = occ.intersect([(3, solid)], [(3, left_box)], removeObject=True, removeTool=True)
right_halves, _ = occ.intersect([(3, solid_copy)], [(3, right_box)], removeObject=True, removeTool=True)
gmsh.model.occ.synchronize()
gmsh.write(str(step_path))
gmsh.finalize()

mesh_out = ellipsoid / "mesh"
mesh_command = [
    sys.executable, str(VOLUME / "build_cad_mesh.py"),
    "--cad", str(step_path),
    "--size", "0.06",
    "--curvature", "32",
    "--radius", "5",
    "--far-size", "1",
    "--ncpu", "8",
    "--out", str(mesh_out),
]
started = time.perf_counter()
mesh_run = subprocess.run(mesh_command, cwd=ROOT, capture_output=True, text=True)
mesh_seconds = time.perf_counter() - started
(ellipsoid / "mesh.log").write_text(mesh_run.stdout + mesh_run.stderr, encoding="utf-8")
if mesh_run.returncode != 0:
    raise SystemExit(f"CAD mesh build failed with exit {mesh_run.returncode}; evidence retained in {ellipsoid}")

mesh_metadata = json.loads((mesh_out / "mesh.json").read_text(encoding="utf-8"))
expected_origin = native_center
expected_length = 6.0
origin_error = float(np.max(np.abs(np.asarray(mesh_metadata["native_origin"]) - expected_origin)))
length_error = abs(float(mesh_metadata["native_length"]) - expected_length)
normalized_axes = native_axes / float(mesh_metadata["native_length"])
normalized_center = (native_center - np.asarray(mesh_metadata["native_origin"])) / float(mesh_metadata["native_length"])
mapping_record = {
    "expected_native_center": native_center.tolist(),
    "recorded_native_origin": mesh_metadata["native_origin"],
    "native_origin_max_abs_error": origin_error,
    "expected_native_length": expected_length,
    "recorded_native_length": mesh_metadata["native_length"],
    "native_length_abs_error": length_error,
    "normalized_axes": normalized_axes.tolist(),
    "normalized_geometric_center": normalized_center.tolist(),
    "note": "OCC STEP B-spline bounding box center differs from ellipsoid geometric center; use mapped center in analytic reference and report deviation.",
    "bbox_mapping_matches_requested_center": origin_error < 1e-5 and length_error < 1e-5,
}
(ellipsoid / "native_mapping.json").write_text(json.dumps(mapping_record, indent=2), encoding="utf-8")

solve_results = []
for alpha in [0, 5]:
    case_name = f"m0_alpha_{alpha}"
    solve_out = ellipsoid / case_name / "solve"
    solve_command = [
        sys.executable, str(VOLUME / "solve_p2.py"),
        "--mesh", str(mesh_out / "mesh.npz"),
        "--mach", "0",
        "--alpha", str(alpha),
        "--boundary", "uniform",
        "--sref", "1",
        "--ncpu", "8",
        "--quad-order", "3",
        "--out", str(solve_out),
    ]
    started = time.perf_counter()
    solve_run = subprocess.run(solve_command, cwd=ROOT, capture_output=True, text=True)
    solve_seconds = time.perf_counter() - started
    solve_out.mkdir(parents=True, exist_ok=True)
    (solve_out / "solve.log").write_text(solve_run.stdout + solve_run.stderr, encoding="utf-8")
    result_path = solve_out / "result.json"
    if not result_path.is_file():
        raise SystemExit(f"Missing solve result for alpha={alpha}; preserve logs in {solve_out}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    solve_results.append({
        "case": case_name,
        "command": solve_command,
        "exit_code": solve_run.returncode,
        "wall_seconds": solve_seconds,
        "status": result["status"],
        "mesh_sha256": result.get("mesh_sha256"),
        "result": str(result_path.relative_to(ROOT)),
    })
    if solve_run.returncode != 0 or result["status"] != "converged":
        raise SystemExit(f"P2 solve failed for alpha={alpha}; evidence retained in {solve_out}")
    analysis_path = ellipsoid / case_name / "analysis.json"
    analysis_command = [
        sys.executable, str(VOLUME / "analyze_ellipsoid.py"),
        "--mesh", str(mesh_out / "mesh.npz"),
        "--field", str(solve_out / "field.npz"),
        "--result", str(result_path),
        "--native-axes", *[str(value) for value in native_axes],
        "--native-center", *[str(value) for value in native_center],
        "--out", str(analysis_path),
    ]
    analysis_run = subprocess.run(analysis_command, cwd=ROOT, capture_output=True, text=True)
    (ellipsoid / case_name / "analysis.log").write_text(
        analysis_run.stdout + analysis_run.stderr, encoding="utf-8"
    )
    solve_results[-1]["analysis_command"] = analysis_command
    solve_results[-1]["analysis_exit_code"] = analysis_run.returncode
    solve_results[-1]["analysis"] = str(analysis_path.relative_to(ROOT))
    if analysis_run.returncode != 0:
        raise SystemExit(f"Analytic ellipsoid gate failed for alpha={alpha}; evidence retained in {analysis_path}")

compact = VOLUME / "meshes" / "curved" / "ellipsoid_s05_v4_compact"
compact_solve = VOLUME / "results" / "mesh_rejection_v1" / "p2_meaningful" / "ellipsoid_s05_v4_compact_m0_q3"
if (compact / "mesh.npz").is_file() and (compact_solve / "field.npz").is_file():
    compact_case = OUT / "legacy_compact_m0"
    compact_case.mkdir()
    compact_result = compact_solve / "result.json"
    compact_analysis = compact_case / "analysis.json"
    compact_command = [
        sys.executable, str(VOLUME / "analyze_ellipsoid.py"),
        "--mesh", str(compact / "mesh.npz"),
        "--field", str(compact_solve / "field.npz"),
        "--result", str(compact_result),
        "--native-axes", "3", "1", "1",
        "--native-center", "0", "0", "0",
        "--record-only",
        "--out", str(compact_analysis),
    ]
    compact_run = subprocess.run(compact_command, cwd=ROOT, capture_output=True, text=True)
    (compact_case / "analysis.log").write_text(
        compact_run.stdout + compact_run.stderr, encoding="utf-8"
    )
    if compact_run.returncode != 0:
        raise SystemExit("Legacy compact ellipsoid M=0 analysis could not be evaluated; evidence retained")
    legacy_analysis = str(compact_analysis.relative_to(ROOT))
else:
    legacy_analysis = None

summary = {
    "status": "physical_gates_passed_normalization_mismatch",
    "scope": "One sphere+dilate STEP prolate ellipsoid split into two closed solids for import; builder fuses them, M=0 alpha 0/5",
    "step": str(step_path.relative_to(ROOT)),
    "step_sha256": hashlib.sha256(step_path.read_bytes()).hexdigest(),
    "builder_command": mesh_command,
    "builder_exit_code": mesh_run.returncode,
    "builder_wall_seconds": mesh_seconds,
    "mesh_sha256": mesh_metadata["mesh_sha256"],
    "builder_source_sha256": mesh_metadata["source_sha256"],
    "native_mapping": mapping_record,
    "STEP_closed_solids_before_builder_fuse": len(left_halves) + len(right_halves),
    "solve_cases": solve_results,
    "legacy_compact_m0_analysis": legacy_analysis,
    "test_driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "analyzer_sha256": hashlib.sha256((VOLUME / "analyze_ellipsoid.py").read_bytes()).hexdigest(),
    "prespecified_gates": {
        "area_weighted_Cp_RMS": "< 0.03",
        "raw_wall_normal_velocity_RMS": "< 0.03",
        "absolute_pressure_CL": "< 0.005",
        "absolute_pressure_CD": "< 0.005",
        "depolarization_factor_sum_error": "< 1e-10",
        "native_origin_and_length_absolute_error": "reported; current STEP B-spline bbox mismatch is not silently accepted as centered",
    },
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(OUT / "test_cad_ellipsoid.py.snapshot").write_bytes(Path(__file__).read_bytes())
print(json.dumps(summary, indent=2))
