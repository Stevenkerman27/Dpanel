"""Dry-run-first geometry and topology robustness matrix for aft-body wake roots."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_DEFAULT = ROOT / "results/geometry_robustness_v1"
BODY_LENGTH = 4.0
SPAN = 4.0
CHORD = 1.0
FAR_RADIUS = 5.0
ALPHA = 5.0
MACH = 0.0
NCPU = 8
SOLVER_TIMEOUT = 180
SIZE = 0.25
CURVATURE = 16
WAKE_STRIPS = 16

cases = []
for aft in [1.2, 1.5, 2.0]:
    for thickness in [0.12, 0.24, 0.36]:
        cases.append(dict(name=f"t{int(round(100*thickness)):03d}_r035_a{int(round(100*aft)):03d}",
                          thickness=thickness, radius=0.35, aft=aft))
for radius in [0.15, 0.50]:
    cases.append(dict(name=f"t024_r{int(round(100*radius)):03d}_a150",
                      thickness=0.24, radius=radius, aft=1.5))

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
parser.add_argument("--execute", action="store_true",
                    help="Build and solve only after root-wake integration is present")
parser.add_argument("--linear-solver", choices=["augmented", "schur"], default="schur")
args = parser.parse_args()
args.out = args.out.resolve()

if not args.execute:
    print(json.dumps({
        "status": "prepared_dry_run",
        "execution_started": False,
        "required_next_gate": "user authorization to execute after root-merge pilot review",
        "case_count": len(cases),
        "solver_timeout_seconds": SOLVER_TIMEOUT,
        "ncpu": NCPU,
        "linear_solver": args.linear_solver,
        "root_wake": "merged",
        "cases": cases,
    }, indent=2))
    raise SystemExit(0)

args.out.mkdir(parents=True, exist_ok=True)
if (args.out / "progress.json").exists() or any(args.out.iterdir()):
    raise SystemExit(f"refusing to overwrite existing geometry evidence: {args.out}")

started = time.perf_counter()
build_script = ROOT / "build_mesh.py"
solve_script = ROOT / "solve.py"
mesh_test_script = ROOT / "test_mesh.py"
analysis_script = ROOT / "analyze.py"
protocol_path = ROOT / "GEOMETRY_ROBUSTNESS_PROTOCOL.md"
root_protocol_path = ROOT / "ROOT_WAKE_PROTOCOL.md"
source_paths = [Path(__file__), build_script, solve_script, mesh_test_script,
                analysis_script, ROOT / "root_wake.py", root_protocol_path,
                ROOT / "schur.py", ROOT.parent / "volume/p2.py", protocol_path]
source_hashes = {str(path.relative_to(ROOT.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in source_paths}
results = []
commands = []
run_record = {
    "status": "running",
    "execution_started": True,
    "scope": "geometry/topology robustness and solver sanity; no absolute Cp/lift accuracy claim",
    "ncpu": NCPU,
    "linear_solver": args.linear_solver,
    "solver_timeout_seconds": SOLVER_TIMEOUT,
    "geometry_defaults": {
        "body_length": BODY_LENGTH, "span": SPAN, "chord": CHORD,
        "far_radius": FAR_RADIUS, "far_size": 5.0 / 3.0,
        "size": SIZE, "curvature": CURVATURE, "wake_strips": WAKE_STRIPS,
        "mach": MACH, "alpha_deg": ALPHA, "kutta": "weak",
    },
    "root_wake": "merged",
    "source_sha256": source_hashes,
    "cases": results,
    "commands": commands,
}
(args.out / "protocol.snapshot.md").write_bytes(protocol_path.read_bytes())
(args.out / "progress.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")

help_command = [sys.executable, str(build_script), "--help"]
help_run = subprocess.run(help_command, capture_output=True, text=True)
builder_help = help_run.stdout + help_run.stderr
root_wake_interface_supported = "--root-wake" in builder_help and "merged" in builder_help
commands.append({
    "stage": "root_wake_interface_preflight",
    "command": help_command,
    "returncode": help_run.returncode,
    "root_wake_merged_supported": root_wake_interface_supported,
})
solver_help_command = [sys.executable, str(solve_script), "--help"]
solver_help_run = subprocess.run(solver_help_command, capture_output=True, text=True)
solver_help = solver_help_run.stdout + solver_help_run.stderr
solver_backend_supported = (
    "--linear-solver" in solver_help and args.linear_solver in solver_help)
commands.append({
    "stage": "solver_backend_interface_preflight",
    "command": solver_help_command,
    "returncode": solver_help_run.returncode,
    "linear_solver_requested": args.linear_solver,
    "linear_solver_supported": solver_backend_supported,
})
if (help_run.returncode or not root_wake_interface_supported
        or solver_help_run.returncode or not solver_backend_supported):
    run_record.update(
        status="interface_pending",
        error="root-wake builder options or selected solver backend are not exposed; no CAD or solver launched",
        elapsed_seconds=time.perf_counter() - started)
    (args.out / "progress.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    (args.out / "result.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    print(json.dumps({"status": run_record["status"], "error": run_record["error"],
                      "root_wake_merged_supported": root_wake_interface_supported,
                      "solver_backend_supported": solver_backend_supported}, indent=2))
    raise SystemExit(2)

for case in cases:
    thickness = float(case["thickness"])
    radius = float(case["radius"])
    aft = float(case["aft"])
    body_center_x = aft - BODY_LENGTH / 2.0
    te_x = CHORD
    section = (te_x - body_center_x) / (BODY_LENGTH / 2.0)
    rejection = None
    if not np.isfinite([thickness, radius, aft, body_center_x, section]).all():
        rejection = "nonfinite geometry parameter"
    elif thickness <= 0.0 or radius <= 0.0:
        rejection = "thickness and body radius must be positive"
    elif abs(section) >= 1.0:
        rejection = "wing trailing-edge plane misses or is tangent to ellipsoid"
    else:
        root_half_width = radius * math.sqrt(1.0 - section**2)
        if root_half_width <= 0.0:
            rejection = "ellipsoid/TE contact is tangent"
        elif root_half_width >= SPAN / 2.0:
            rejection = "body covers the entire wing TE; no exposed TE remains"
        elif body_center_x - BODY_LENGTH / 2.0 <= 0.5 - FAR_RADIUS:
            rejection = "body nose is not strictly inside the farfield x boundary"
        elif body_center_x + BODY_LENGTH / 2.0 >= 0.5 + FAR_RADIUS:
            rejection = "body tail is not strictly inside the farfield x boundary"
        elif radius >= FAR_RADIUS:
            rejection = "body radius reaches/exceeds the farfield side boundary"
    if rejection:
        case_record = dict(case=case, status="geometry_rejected", reason=rejection)
        results.append(case_record)
        run_record["cases"] = results
        (args.out / "progress.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
        continue

    root_interval = [-root_half_width, root_half_width]
    case_root = args.out / case["name"]
    mesh_root = case_root / "mesh"
    solve_root = case_root / "solve"
    case_root.mkdir(parents=True)
    case_record = {
        "case": case,
        "geometry_preflight": {
            "status": "passed",
            "body_center_x": body_center_x,
            "root_interval_y": root_interval,
            "exposed_te_intervals_y": [[-SPAN / 2.0, -root_half_width],
                                       [root_half_width, SPAN / 2.0]],
        },
        "status": "mesh_pending",
    }
    results.append(case_record)
    mesh_command = [
        sys.executable, str(build_script),
        "--size", str(SIZE), "--span", str(SPAN), "--thickness", str(thickness),
        "--curvature", str(CURVATURE), "--wake-strips", str(WAKE_STRIPS),
        "--body-length", str(BODY_LENGTH), "--body-radius", str(radius),
        "--body-aft", str(aft), "--radius", str(FAR_RADIUS), "--far-size", str(5.0 / 3.0),
        "--root-wake", "merged", "--out", str(mesh_root),
    ]
    tick = time.perf_counter()
    try:
        mesh_run = subprocess.run(mesh_command, capture_output=True, text=True, timeout=300)
        mesh_elapsed = time.perf_counter() - tick
        mesh_stdout, mesh_stderr, mesh_returncode, mesh_timeout = (
            mesh_run.stdout, mesh_run.stderr, mesh_run.returncode, False)
    except subprocess.TimeoutExpired as error:
        mesh_elapsed = time.perf_counter() - tick
        mesh_stdout = error.stdout or ""
        mesh_stderr = error.stderr or ""
        if isinstance(mesh_stdout, bytes):
            mesh_stdout = mesh_stdout.decode(errors="replace")
        if isinstance(mesh_stderr, bytes):
            mesh_stderr = mesh_stderr.decode(errors="replace")
        mesh_returncode, mesh_timeout = None, True
    (case_root / "mesh.log").write_text(mesh_stdout + mesh_stderr, encoding="utf-8")
    commands.append({"case": case["name"], "stage": "mesh", "command": mesh_command,
                     "returncode": mesh_returncode, "timeout": mesh_timeout,
                     "elapsed_seconds": mesh_elapsed})
    case_record["mesh"] = {"returncode": mesh_returncode, "timeout": mesh_timeout,
                           "elapsed_seconds": mesh_elapsed}
    mesh_path = mesh_root / "mesh.npz"
    metadata_path = mesh_root / "mesh.json"
    if mesh_timeout or mesh_returncode != 0 or not mesh_path.is_file() or not metadata_path.is_file():
        case_record["status"] = "mesh_failed"
        run_record["cases"] = results
        (args.out / "progress.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
        continue

    mesh_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    case_record["mesh"].update(
        mesh_sha256=hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        points=mesh_metadata.get("points"),
        tets=mesh_metadata.get("tets"),
        requested_wake_strips=mesh_metadata.get("wake_strips"),
        actual_wake_strips=mesh_metadata.get("actual_wake_strips"),
        cad_root_reconciliation=mesh_metadata.get("cad_root_reconciliation"),
        minimum_dense_jacobian=mesh_metadata.get("min_dense_jacobian"),
        body_tags=mesh_metadata.get("cad_component_surfaces",
                                    mesh_metadata.get("cad_surface_components")),
        root_wake_support=mesh_metadata.get("root_wake_support"))
    expected_geometry = {
        "thickness": thickness,
        "body_length": BODY_LENGTH,
        "body_radius": radius,
        "body_aft": aft,
        "span": SPAN,
        "size": SIZE,
        "curvature_elements_per_circle": CURVATURE,
        "wake_strips": WAKE_STRIPS,
    }
    geometry_metadata_match = True
    for key, expected in expected_geometry.items():
        actual = mesh_metadata.get(key)
        if actual is None or not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-12):
            geometry_metadata_match = False
    case_record["mesh"]["geometry_metadata_match"] = geometry_metadata_match
    root_contract = mesh_metadata.get("root_wake_support", {})
    if not geometry_metadata_match:
        case_record["status"] = "geometry_metadata_rejected"
        case_record["error"] = "built mesh metadata differs from frozen case inputs"
        run_record["cases"] = results
        (args.out / "progress.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
        continue
    expected_exposed = [[-SPAN / 2.0, -root_half_width],
                        [root_half_width, SPAN / 2.0]]
    try:
        metadata_root_interval = np.asarray(root_contract["root_interval_y"], dtype=float)
        metadata_exposed_intervals = np.asarray(root_contract["exposed_te_intervals_y"], dtype=float)
        root_intervals_match = (
            metadata_root_interval.shape == (2,)
            and np.allclose(metadata_root_interval, root_interval, rtol=0.0, atol=1.0e-10)
            and metadata_exposed_intervals.shape == (2, 2)
            and np.allclose(metadata_exposed_intervals, expected_exposed, rtol=0.0, atol=1.0e-10))
    except (KeyError, TypeError, ValueError):
        root_intervals_match = False
    station_to_dof = root_contract.get("station_to_dof") if isinstance(root_contract, dict) else None
    weak_test_support = root_contract.get("weak_test_support_integrals") if isinstance(root_contract, dict) else None
    try:
        station_to_dof_array = np.asarray(station_to_dof, dtype=int)
        support_array = np.asarray(weak_test_support, dtype=float)
        root_arrays_valid = (
            station_to_dof_array.ndim == 1
            and np.array_equal(station_to_dof_array, np.load(mesh_path)["gamma_station_dof"])
            and support_array.ndim == 2
            and support_array.shape[1] == len(np.unique(station_to_dof_array[station_to_dof_array >= 0]))
            and np.isfinite(support_array).all()
            and np.all(support_array.sum(axis=0) > 0.0))
    except (TypeError, ValueError):
        root_arrays_valid = False
    if (not isinstance(root_contract, dict)
            or root_contract.get("status") != "constructed"
            or not root_arrays_valid
            or not root_intervals_match):
        case_record["status"] = "root_wake_metadata_rejected"
        case_record["error"] = (
            "mesh metadata lacks constructed root support arrays or its root/exposed TE intervals disagree")
        run_record["cases"] = results
        (args.out / "progress.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
        continue

    mesh_test_command = [sys.executable, str(mesh_test_script), "--mesh", str(mesh_path)]
    tick = time.perf_counter()
    mesh_test_run = subprocess.run(mesh_test_command, capture_output=True, text=True)
    mesh_test_elapsed = time.perf_counter() - tick
    (case_root / "mesh_test.log").write_text(
        mesh_test_run.stdout + mesh_test_run.stderr, encoding="utf-8")
    tests_path = mesh_root / "tests.json"
    mesh_tests = json.loads(tests_path.read_text(encoding="utf-8")) if tests_path.is_file() else []
    commands.append({"case": case["name"], "stage": "mesh_test",
                     "command": mesh_test_command, "returncode": mesh_test_run.returncode,
                     "elapsed_seconds": mesh_test_elapsed, "tests": mesh_tests})
    row_coverage = [row for row in mesh_tests
                    if row.get("test") == "root_wake_row_coverage" and row.get("passed") is True]
    row_coverage_test_present = any(
        row.get("test") == "root_wake_row_coverage" for row in mesh_tests)
    jacobian = [row for row in mesh_tests
                if row.get("test") == "dense_P2_Jacobian" and row.get("passed") is True]
    adjacency = [row for row in mesh_tests
                 if row.get("test") == "all_exterior_faces_adjacency_and_midnodes"
                 and row.get("passed") is True]
    body_tags_present = all(tag in np.unique(np.load(mesh_path)["boundary_tags"])
                            for tag in [20, 30])
    case_record["mesh_test"] = {"returncode": mesh_test_run.returncode,
                                "elapsed_seconds": mesh_test_elapsed,
                                "root_wake_row_coverage": bool(row_coverage),
                                "root_wake_row_coverage_test_present": row_coverage_test_present,
                                "positive_dense_jacobian": bool(jacobian),
                                "exterior_adjacency": bool(adjacency),
                                "body_and_wing_tags_present": bool(body_tags_present)}
    if (mesh_test_run.returncode or not row_coverage or not row_coverage_test_present
            or not jacobian or not adjacency or not body_tags_present):
        case_record["status"] = "mesh_topology_rejected"
        run_record["cases"] = results
        (args.out / "progress.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
        continue

    solve_command = [
        sys.executable, str(solve_script), "--mesh", str(mesh_path),
        "--alpha", str(ALPHA), "--mach", str(MACH), "--ncpu", str(NCPU),
        "--kutta", "weak", "--linear-solver", args.linear_solver, "--out", str(solve_root),
    ]
    tick = time.perf_counter()
    try:
        solve_run = subprocess.run(solve_command, capture_output=True, text=True,
                                   timeout=SOLVER_TIMEOUT)
        solve_elapsed = time.perf_counter() - tick
        solve_stdout, solve_stderr, solve_returncode, solve_timeout = (
            solve_run.stdout, solve_run.stderr, solve_run.returncode, False)
    except subprocess.TimeoutExpired as error:
        solve_elapsed = time.perf_counter() - tick
        solve_stdout = error.stdout or ""
        solve_stderr = error.stderr or ""
        if isinstance(solve_stdout, bytes):
            solve_stdout = solve_stdout.decode(errors="replace")
        if isinstance(solve_stderr, bytes):
            solve_stderr = solve_stderr.decode(errors="replace")
        solve_returncode, solve_timeout = None, True
    solve_root.mkdir(parents=True, exist_ok=True)
    (solve_root / "run.log").write_text(solve_stdout + solve_stderr, encoding="utf-8")
    commands.append({"case": case["name"], "stage": "solve", "command": solve_command,
                     "returncode": solve_returncode, "timeout": solve_timeout,
                     "elapsed_seconds": solve_elapsed})
    case_record["solve"] = {"returncode": solve_returncode, "timeout": solve_timeout,
                            "elapsed_seconds": solve_elapsed}
    result_path = solve_root / "result.json"
    field_path = solve_root / "field.npz"
    if solve_timeout:
        case_record["status"] = "solver_timeout"
    elif not result_path.is_file() or not field_path.is_file():
        case_record["status"] = "solver_no_output"
    else:
        solver_result = json.loads(result_path.read_text(encoding="utf-8"))
        with np.load(field_path, allow_pickle=False) as field:
            cp = np.asarray(field["cp"])
            cp_finite = bool(np.isfinite(cp).all())
        case_record["solve"].update(
            result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
            field_sha256=hashlib.sha256(field_path.read_bytes()).hexdigest(),
            solver_status=solver_result.get("status"),
            converged=bool(solver_result.get("converged")),
            raw_cp_finite=cp_finite,
            cp_count=int(cp.size),
            cp_min=float(np.nanmin(cp)) if cp.size else None,
            cp_max=float(np.nanmax(cp)) if cp.size else None,
            weak_residual=solver_result.get("weak_mass_residual_norm"),
            min_beta=solver_result.get("min_beta"),
            max_local_mach=solver_result.get("max_local_mach"),
            raw_mass_by_tag=solver_result.get("raw_mass_by_tag"),
            pressure_force_coefficient=solver_result.get("pressure_force_coefficient"),
            cl_pressure=solver_result.get("cl_pressure"),
            cl_gamma=solver_result.get("cl_jump"),
            cd_pressure=solver_result.get("cd_pressure"),
            kutta_max=solver_result.get("kutta_max"),
            history=solver_result.get("history"),
        )
        analyze_command = [sys.executable, str(analysis_script),
                           "--run", str(solve_root), "--mesh", str(mesh_path)]
        analyze_run = subprocess.run(analyze_command, capture_output=True, text=True)
        (solve_root / "analyze.log").write_text(
            analyze_run.stdout + analyze_run.stderr, encoding="utf-8")
        analysis_path = solve_root / "analysis.json"
        if analyze_run.returncode == 0 and analysis_path.is_file():
            analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
            case_record["solve"]["analysis"] = analysis
            closure = analysis.get("component_sum_error")
            case_record["solve"]["component_force_closure_error"] = closure
            case_record["solve"]["component_force_closure_passed"] = (
                closure is not None and closure < 1.0e-10)
        solver_sanity = (
            solve_returncode == 0 and solver_result.get("converged") and cp_finite
            and np.isfinite(solver_result.get("pressure_force_coefficient", [np.nan])).all()
            and case_record["solve"].get("component_force_closure_passed", False))
        case_record["status"] = "solver_sanity_passed" if solver_sanity else "solver_sanity_failed"

    run_record["cases"] = results
    (args.out / "progress.json").write_text(
        json.dumps(run_record, indent=2, allow_nan=False), encoding="utf-8")

run_record["status"] = "completed_with_case_results"
run_record["elapsed_seconds"] = time.perf_counter() - started
(args.out / "progress.json").write_text(json.dumps(run_record, indent=2, allow_nan=False),
                                        encoding="utf-8")
(args.out / "result.json").write_text(json.dumps(run_record, indent=2, allow_nan=False),
                                       encoding="utf-8")
print(json.dumps({
    "status": run_record["status"],
    "cases": [{"name": row["case"]["name"], "status": row["status"]} for row in results],
    "elapsed_seconds": run_record["elapsed_seconds"],
}, indent=2))
