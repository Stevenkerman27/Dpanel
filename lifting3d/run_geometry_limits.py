"""Serialized root-wake geometry-limit matrix with frozen-source checks."""
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
OUT_DEFAULT = ROOT / "results" / "geometry_limits_v2"
BODY_LENGTH = 4.0
SPAN = 4.0
CHORD = 1.0
FAR_RADIUS = 5.0
FAR_SIZE = 5.0 / 3.0
SIZE = 0.25
CURVATURE = 16
WAKE_STRIPS = 16
ALPHA = 5.0
MACH = 0.0
NCPU = 8
MESH_TIMEOUT = 90
SOLVE_TIMEOUT = 180
FORCE_TOLERANCE = 1.0e-10

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
parser.add_argument("--execute", action="store_true")
parser.add_argument("--reuse-v1", action="store_true",
                    help="Reuse geometry_limits_v1 meshes and rejection evidence; never invoke the mesh builder")
args = parser.parse_args()
args.out = args.out.resolve()
if not args.execute:
    print(json.dumps({
        "status": "prepared_dry_run",
        "execution_started": False,
        "case_count": 14,
        "mesh_timeout_seconds": MESH_TIMEOUT,
        "solve_timeout_seconds": SOLVE_TIMEOUT,
        "ncpu": NCPU,
        "linear_solver": "schur",
        "kutta": "weak",
        "reuse_v1": args.reuse_v1,
        "output": str(args.out),
    }, indent=2))
    raise SystemExit(0)

if args.out.exists():
    raise SystemExit(f"refusing to overwrite previous geometry evidence: {args.out}")
args.out.mkdir(parents=True)

build_script = ROOT / "build_mesh.py"
solve_script = ROOT / "solve.py"
mesh_test_script = ROOT / "test_mesh.py"
protocol_path = ROOT / "GEOMETRY_LIMITS_PROTOCOL.md"
source_paths = [
    Path(__file__), protocol_path, build_script, solve_script,
    ROOT / "root_wake.py", mesh_test_script, ROOT.parent / "volume" / "p2.py",
    ROOT / "schur.py", ROOT / "PROTOCOL.md", ROOT / "ROOT_WAKE_PROTOCOL.md",
    ROOT / "SCHUR_PROTOCOL.md",
]
source_hashes = {
    str(path.relative_to(ROOT.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in source_paths
}
frozen_source_paths = [path for path in source_paths if path != build_script]
frozen_source_hashes = {
    str(path.relative_to(ROOT.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in frozen_source_paths
}
v1_result_path = ROOT / "results" / "geometry_limits_v1" / "result.json"
v1_result = json.loads(v1_result_path.read_text(encoding="utf-8"))
v1_cases = {row["input"]["name"]: row for row in v1_result["cases"]}
v1_builder_hash = v1_result["source_sha256_before"][str(build_script.relative_to(ROOT.parent))]
current_builder_hash = hashlib.sha256(build_script.read_bytes()).hexdigest()
pilot_mesh_path = ROOT / "results" / "root_wake_pilot_v2" / "mesh" / "mesh.npz"
pilot_mesh_hash = hashlib.sha256(pilot_mesh_path.read_bytes()).hexdigest()
source_hashes[str(pilot_mesh_path.relative_to(ROOT.parent))] = pilot_mesh_hash
(args.out / "run_geometry_limits.py.snapshot").write_bytes(Path(__file__).read_bytes())
(args.out / "GEOMETRY_LIMITS_PROTOCOL.md.snapshot").write_bytes(protocol_path.read_bytes())

knot_radius = 0.25 / math.sqrt(1.0 - (1.5 / 2.0)**2)
cases = [
    {"name": "legal_aft_0p999", "expected": "legal", "aft": 0.999, "radius": 0.35, "thickness": 0.24},
    {"name": "legal_aft_1p0001", "expected": "legal", "aft": 1.0001, "radius": 0.35, "thickness": 0.24},
    {"name": "legal_aft_1p001", "expected": "legal", "aft": 1.001, "radius": 0.35, "thickness": 0.24},
    {"name": "legal_aft_1p01", "expected": "legal", "aft": 1.01, "radius": 0.35, "thickness": 0.24},
    {"name": "legal_aft_3p0", "expected": "legal", "aft": 3.0, "radius": 0.35, "thickness": 0.24},
    {"name": "legal_root_knot_exact", "expected": "legal", "aft": 1.5, "radius": knot_radius, "thickness": 0.24},
    {"name": "legal_root_knot_plus_1e-9", "expected": "legal", "aft": 1.5, "radius": knot_radius + 1.0e-9, "thickness": 0.24},
    {"name": "legal_thickness_0p6", "expected": "legal", "aft": 1.5, "radius": 0.35, "thickness": 0.6},
    {"name": "reject_aft_tangent_1p0", "expected": "builder_reject", "reason_token": "tangent", "aft": 1.0, "radius": 0.35, "thickness": 0.24},
    {"name": "reject_aft_3p0001", "expected": "builder_reject", "reason_token": "maximum girth", "aft": 3.0001, "radius": 0.35, "thickness": 0.24},
    {"name": "reject_fullspan_radius_3p05", "expected": "builder_reject", "reason_token": "full wing span", "aft": 1.5, "radius": 3.05, "thickness": 0.24},
    {"name": "reject_sweep_20", "expected": "builder_reject", "reason_token": "rectangular unswept", "aft": 1.5, "radius": 0.35, "thickness": 0.24, "sweep": 20.0},
    {"name": "reject_wake_tilt_2", "expected": "builder_reject", "reason_token": "planar cut", "aft": 1.5, "radius": 0.35, "thickness": 0.24, "wake_tilt": 2.0},
    {"name": "reject_node_kutta_pilot_mesh", "expected": "solver_reject"},
]

run_record = {
    "status": "running",
    "execution_started": True,
    "scope": "Geometry/topology limits and solver sanity only; no absolute Cp or aerodynamic accuracy claim",
    "frozen_settings": {
        "span": SPAN, "chord": CHORD, "body_length": BODY_LENGTH,
        "size": SIZE, "curvature": CURVATURE, "wake_strips": WAKE_STRIPS,
        "far_radius": FAR_RADIUS, "far_size": FAR_SIZE, "alpha_deg": ALPHA,
        "mach": MACH, "kutta": "weak", "linear_solver": "schur", "ncpu": NCPU,
        "mesh_timeout_seconds": MESH_TIMEOUT, "solve_timeout_seconds": SOLVE_TIMEOUT,
    },
    "root_knot_radius": knot_radius,
    "source_sha256_before": source_hashes,
    "frozen_source_sha256_before": frozen_source_hashes,
    "builder_provenance": {
        "v1_mesh_builder_sha256": v1_builder_hash,
        "current_builder_sha256": current_builder_hash,
        "current_builder_matches_v1": current_builder_hash == v1_builder_hash,
        "comparison_required": False,
        "note": "Meshes are reused byte-for-byte; current builder is recorded only to disclose source difference.",
    },
    "mesh_reuse": {
        "enabled": args.reuse_v1,
        "source_result": str(v1_result_path.relative_to(ROOT.parent)),
        "mesh_rebuilt": False if args.reuse_v1 else None,
    },
    "cases": [],
}
(args.out / "progress.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
failures = []

if args.reuse_v1:
    for case in cases:
        case_root = args.out / case["name"]
        case_root.mkdir()
        record = {"input": case.copy(), "status": "running"}
        run_record["cases"].append(record)
        old_record = v1_cases[case["name"]]

        if case["expected"] == "builder_reject" or case["expected"] == "solver_reject":
            old_pass = bool(old_record.get("passed"))
            record.update(
                status="reused_expected_rejection" if old_pass else "reused_rejection_failed",
                passed=old_pass,
                evidence_source="geometry_limits_v1",
                original_status=old_record.get("status"),
                original_failure_message=old_record.get("failure_message"),
                original_builder_sha256=old_record.get("failure_record", {}).get("source_sha256"),
                original_solver_result=old_record.get("result"),
                note="Original rejection evidence is carried forward; no builder or solver was run for this reject case.",
            )
        elif not old_record.get("mesh_sha256"):
            record.update(
                status="mesh_unavailable_for_reuse",
                passed=False,
                evidence_source="geometry_limits_v1",
                original_status=old_record.get("status"),
                original_failure_message=old_record.get("failure_message"),
                original_builder_sha256=old_record.get("failure_record", {}).get("source_sha256"),
                note="v1 did not save mesh.npz for this legal case; rebuild intentionally skipped.",
            )
        else:
            old_case_root = ROOT / "results" / "geometry_limits_v1" / case["name"]
            mesh_path = old_case_root / "mesh" / "mesh.npz"
            metadata_path = old_case_root / "mesh" / "mesh.json"
            tests_path = old_case_root / "mesh" / "tests.json"
            if not (mesh_path.is_file() and metadata_path.is_file() and tests_path.is_file()):
                record.update(status="reuse_artifact_missing", passed=False)
            else:
                mesh_sha = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
                metadata_sha = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                mesh = np.load(mesh_path, allow_pickle=False)
                tests = json.loads(tests_path.read_text(encoding="utf-8"))
                tags = sorted(int(tag) for tag in np.unique(mesh["boundary_tags"]))
                geometry_match = all(
                    math.isclose(float(metadata.get(key, math.nan)), expected,
                                 rel_tol=0.0, abs_tol=1.0e-12)
                    for key, expected in [
                        ("size", SIZE), ("span", SPAN), ("thickness", case["thickness"]),
                        ("body_length", BODY_LENGTH), ("body_radius", case["radius"]),
                        ("body_aft", case["aft"]), ("curvature_elements_per_circle", CURVATURE),
                        ("wake_strips", WAKE_STRIPS),
                    ])
                tag_pass = all(tag in tags for tag in [20, 30, 100])
                tests_pass = bool(tests and all(row.get("passed") is True for row in tests))
                v1_mesh_match = mesh_sha == old_record["mesh_sha256"]
                root_expected = case["aft"] >= CHORD
                root_record = metadata.get("root_wake_support")
                root_pass = (not root_expected and not root_record) or (
                    isinstance(root_record, dict) and root_record.get("status") == "constructed")
                map_pass = True
                root_half_width = None
                exact_knot_error = None
                near_knot_gap = None
                if root_expected and isinstance(root_record, dict):
                    body_center_x = case["aft"] - BODY_LENGTH / 2.0
                    section = (CHORD - body_center_x) / (BODY_LENGTH / 2.0)
                    root_half_width = case["radius"] * math.sqrt(1.0 - section**2)
                    expected_interval = np.array([-root_half_width, root_half_width])
                    saved_interval = np.asarray(root_record.get("root_interval_y"), dtype=float)
                    station_map = np.asarray(mesh["gamma_station_dof"], dtype=int)
                    stations = np.asarray(mesh["gamma_stations"], dtype=float)
                    metadata_map = np.asarray(root_record.get("station_to_dof"), dtype=int)
                    map_pass = bool(
                        saved_interval.shape == (2,)
                        and np.allclose(saved_interval, expected_interval, rtol=0.0, atol=1.0e-10)
                        and np.array_equal(station_map, metadata_map)
                        and station_map.shape == stations.shape
                        and np.isfinite(stations).all()
                        and np.all(np.diff(stations) > 0.0)
                        and np.all(station_map[[0, -1]] == -1))
                    endpoint_gaps = np.abs(stations - root_half_width)
                    exact_knot_error = float(endpoint_gaps.min())
                    if case["name"].startswith("legal_root_knot"):
                        map_pass = map_pass and exact_knot_error < 1.0e-12
                        if "plus_1e-9" in case["name"]:
                            near_knot_gap = float(abs(root_half_width - 0.25))
                            map_pass = map_pass and near_knot_gap > 1.0e-10
                reuse_checks_pass = bool(
                    v1_mesh_match and geometry_match and tag_pass and tests_pass
                    and root_pass and map_pass)
                record.update(
                    reused_mesh_path=str(mesh_path.relative_to(ROOT.parent)),
                    mesh_sha256=mesh_sha, expected_v1_mesh_sha256=old_record["mesh_sha256"],
                    mesh_sha256_matches_v1=v1_mesh_match,
                    mesh_metadata_sha256=metadata_sha, mesh_points=metadata.get("points"),
                    mesh_tets=metadata.get("tets"), boundary_tags=tags,
                    required_boundary_tags=[20, 30, 100],
                    geometry_metadata_match=geometry_match,
                    saved_mesh_tests=tests, saved_mesh_tests_pass=tests_pass,
                    root_wake_support=root_record, root_interval_half_width=root_half_width,
                    exact_0p25_station_distance=exact_knot_error,
                    root_plus_1e9_perturbation=near_knot_gap,
                    reuse_checks_pass=reuse_checks_pass,
                )
                if not reuse_checks_pass:
                    record.update(status="reuse_validation_failed", passed=False)
                else:
                    solve_out = case_root / "solve"
                    solve_command = [
                        sys.executable, str(solve_script), "--mesh", str(mesh_path),
                        "--alpha", str(ALPHA), "--mach", str(MACH), "--ncpu", str(NCPU),
                        "--kutta", "weak", "--linear-solver", "schur", "--out", str(solve_out),
                    ]
                    tick = time.perf_counter()
                    try:
                        solve_run = subprocess.run(
                            solve_command, capture_output=True, text=True,
                            timeout=SOLVE_TIMEOUT, cwd=ROOT.parent)
                        stdout, stderr = solve_run.stdout, solve_run.stderr
                        returncode, timed_out = solve_run.returncode, False
                    except subprocess.TimeoutExpired as error:
                        stdout, stderr = error.stdout or "", error.stderr or ""
                        returncode, timed_out = None, True
                    if isinstance(stdout, bytes):
                        stdout = stdout.decode(errors="replace")
                    if isinstance(stderr, bytes):
                        stderr = stderr.decode(errors="replace")
                    (case_root / "solve.log").write_text(stdout + stderr, encoding="utf-8")
                    solve_elapsed = time.perf_counter() - tick
                    result_path = solve_out / "result.json"
                    field_path = solve_out / "field.npz"
                    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
                    if result.get("converged") and field_path.is_file():
                        field = np.load(field_path, allow_pickle=False)
                        finite = all(np.isfinite(field[key]).all() for key in
                                     ["cp", "surface_velocity", "surface_normal", "surface_weight", "gamma"])
                        reduced_gamma = np.asarray(field["gamma"])
                        saved_map = np.asarray(field["gamma_station_dof"], dtype=int)
                        mesh_map = np.asarray(mesh["gamma_station_dof"], dtype=int)
                        stations = np.asarray(mesh["gamma_stations"], dtype=float)
                        expanded_gamma = np.zeros(len(stations))
                        free = saved_map >= 0
                        expanded_gamma[free] = reduced_gamma[saved_map[free]]
                        jump = np.asarray(result["potential_jump"], dtype=float)
                        jump_error = float(np.max(abs(
                            expanded_gamma - jump / float(result["normalization_length"]))))
                        wall = np.isin(field["tags"], [10, 20, 30])
                        raw_force = np.sum(
                            field["cp"][wall, :, None] * field["surface_normal"][wall]
                            * field["surface_weight"][wall, :, None], axis=(0, 1)) / float(metadata["reference_area"])
                        component_force = np.sum(
                            [np.asarray(row["force_coefficient"])
                             for row in result["component_loads"].values()], axis=0)
                        force_error = float(np.max(abs(
                            raw_force - np.asarray(result["pressure_force_coefficient"]))))
                        component_error = float(np.max(abs(
                            component_force - np.asarray(result["pressure_force_coefficient"]))))
                        surface_ok = bool(
                            finite and np.all(field["surface_weight"] > 0.0)
                            and result["min_beta"] > 0.0
                            and len(reduced_gamma) == int(mesh_map.max() + 1)
                            and np.array_equal(saved_map, mesh_map)
                            and len(jump) == len(stations)
                            and len(result["gamma_stations"]) == len(stations)
                            and jump_error < 1.0e-10
                            and force_error < FORCE_TOLERANCE
                            and component_error < FORCE_TOLERANCE)
                        record.update(
                            status="passed" if surface_ok else "surface_sanity_failed",
                            passed=surface_ok, solve_command=solve_command,
                            solve_returncode=returncode, solve_timeout=timed_out,
                            solve_elapsed_seconds=solve_elapsed, converged=result.get("converged"),
                            gamma_station_count=len(stations), gamma_reduced_dof_count=len(reduced_gamma),
                            gamma_expansion_max_abs_error=jump_error,
                            raw_force_wall_face_count=int(wall.sum()),
                            raw_force_vs_result_max_abs_error=force_error,
                            component_force_closure_max_abs_error=component_error,
                            pressure_force_coefficient=result.get("pressure_force_coefficient"),
                            cl_pressure=result.get("cl_pressure"), cl_jump=result.get("cl_jump"),
                            min_beta=result.get("min_beta"), surface_sanity_pass=surface_ok,
                            result=result)
                    else:
                        record.update(
                            status="solver_timeout" if timed_out else "solver_failed",
                            passed=False, solve_command=solve_command,
                            solve_returncode=returncode, solve_timeout=timed_out,
                            solve_elapsed_seconds=solve_elapsed, result=result)

        if not record.get("passed", False):
            failures.append(case["name"])
        current_frozen_hashes = {
            str(path.relative_to(ROOT.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in frozen_source_paths
        }
        if current_frozen_hashes != frozen_source_hashes:
            run_record["status"] = "frozen_source_hash_changed"
            run_record["frozen_source_sha256_after_change"] = current_frozen_hashes
            run_record["cases_completed_before_source_change"] = len(run_record["cases"])
            (args.out / "progress.json").write_text(
                json.dumps(run_record, indent=2, allow_nan=False), encoding="utf-8")
            raise SystemExit("Frozen solver/test/root-wake source hash changed; stopped v2")
        run_record["cases_completed"] = len(run_record["cases"])
        run_record["failed_cases_so_far"] = failures.copy()
        (args.out / "progress.json").write_text(
            json.dumps(run_record, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({
            "case": case["name"], "status": record["status"],
            "passed": record.get("passed", False),
            "cases_completed": len(run_record["cases"]),
        }, indent=2), flush=True)

    run_record["frozen_source_sha256_after"] = {
        str(path.relative_to(ROOT.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in frozen_source_paths
    }
    run_record["frozen_source_hashes_unchanged"] = (
        run_record["frozen_source_sha256_after"] == frozen_source_hashes)
    run_record["source_sha256_after"] = {
        str(path.relative_to(ROOT.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    run_record["builder_provenance"]["current_builder_sha256_at_completion"] = (
        run_record["source_sha256_after"][str(build_script.relative_to(ROOT.parent))])
    run_record["failed_cases"] = failures
    run_record["passed"] = len(cases) - len(failures)
    run_record["total"] = len(cases)
    run_record["status"] = (
        "passed" if not failures and run_record["frozen_source_hashes_unchanged"]
        else "completed_with_failures")
    run_record["elapsed_seconds"] = sum(
        row.get("solve_elapsed_seconds", 0.0) for row in run_record["cases"])
    (args.out / "progress.json").write_text(
        json.dumps(run_record, indent=2, allow_nan=False), encoding="utf-8")
    (args.out / "result.json").write_text(
        json.dumps(run_record, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "status": run_record["status"], "passed": run_record["passed"],
        "total": run_record["total"], "failed_cases": failures,
        "frozen_source_hashes_unchanged": run_record["frozen_source_hashes_unchanged"],
        "output": str(args.out),
    }, indent=2))
    if run_record["status"] != "passed":
        raise SystemExit(2)
    raise SystemExit(0)

for case in cases:
    case_root = args.out / case["name"]
    case_root.mkdir()
    record = {"input": case.copy(), "status": "running"}
    run_record["cases"].append(record)

    if case["expected"] == "solver_reject":
        mesh_path = pilot_mesh_path
        solve_out = case_root / "solve"
        command = [
            sys.executable, str(solve_script), "--mesh", str(mesh_path),
            "--alpha", str(ALPHA), "--mach", str(MACH), "--ncpu", str(NCPU),
            "--kutta", "node", "--linear-solver", "schur", "--out", str(solve_out),
        ]
        tick = time.perf_counter()
        try:
            run = subprocess.run(command, capture_output=True, text=True,
                                 timeout=SOLVE_TIMEOUT, cwd=ROOT.parent)
            stdout, stderr, returncode, timed_out = run.stdout, run.stderr, run.returncode, False
        except subprocess.TimeoutExpired as error:
            stdout, stderr, returncode, timed_out = error.stdout or "", error.stderr or "", None, True
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        (case_root / "solve.log").write_text(stdout + stderr, encoding="utf-8")
        result_path = solve_out / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        message = result.get("message", "")
        passed = bool(not timed_out and returncode != 0
                      and "requires weak Kutta" in message)
        record.update(
            status="expected_solver_rejection" if passed else "solver_reject_failed",
            command=command, returncode=returncode, timeout=timed_out,
            elapsed_seconds=time.perf_counter() - tick,
            failure_message=message, result=result,
            reused_pilot_mesh_sha256=hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
            passed=passed,
        )
    else:
        mesh_out = case_root / "mesh"
        mesh_command = [
            sys.executable, str(build_script),
            "--size", str(SIZE), "--span", str(SPAN),
            "--thickness", str(case["thickness"]), "--curvature", str(CURVATURE),
            "--wake-strips", str(WAKE_STRIPS), "--body-length", str(BODY_LENGTH),
            "--body-radius", str(case["radius"]), "--body-aft", str(case["aft"]),
            "--radius", str(FAR_RADIUS), "--far-size", str(FAR_SIZE),
            "--root-wake", "auto", "--out", str(mesh_out),
        ]
        if case.get("sweep", 0.0):
            mesh_command.extend(["--sweep", str(case["sweep"])])
        if case.get("wake_tilt", 0.0):
            mesh_command.extend(["--wake-tilt", str(case["wake_tilt"])])
        tick = time.perf_counter()
        try:
            run = subprocess.run(mesh_command, capture_output=True, text=True,
                                 timeout=MESH_TIMEOUT, cwd=ROOT.parent)
            stdout, stderr, returncode, timed_out = run.stdout, run.stderr, run.returncode, False
        except subprocess.TimeoutExpired as error:
            stdout, stderr, returncode, timed_out = error.stdout or "", error.stderr or "", None, True
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        mesh_elapsed = time.perf_counter() - tick
        (case_root / "mesh.log").write_text(stdout + stderr, encoding="utf-8")
        mesh_path = mesh_out / "mesh.npz"
        mesh_metadata_path = mesh_out / "mesh.json"
        failure_path = mesh_out / "failure.json"
        build_failure = json.loads(failure_path.read_text(encoding="utf-8")) if failure_path.is_file() else {}
        build_message = build_failure.get("message", "")

        if case["expected"] == "builder_reject":
            passed = bool(not timed_out and returncode != 0
                          and case["reason_token"].lower() in build_message.lower())
            record.update(
                status="expected_builder_rejection" if passed else "builder_reject_failed",
                command=mesh_command, returncode=returncode, timeout=timed_out,
                elapsed_seconds=mesh_elapsed, failure_message=build_message,
                failure_record=build_failure, passed=passed,
            )
        elif timed_out or returncode != 0 or not mesh_path.is_file() or not mesh_metadata_path.is_file():
            record.update(status="mesh_failed", command=mesh_command, returncode=returncode,
                          timeout=timed_out, elapsed_seconds=mesh_elapsed,
                          failure_message=build_message, failure_record=build_failure, passed=False)
        else:
            mesh = np.load(mesh_path, allow_pickle=False)
            mesh_metadata = json.loads(mesh_metadata_path.read_text(encoding="utf-8"))
            mesh_sha256 = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
            tests_command = [sys.executable, str(mesh_test_script), "--mesh", str(mesh_path)]
            tick = time.perf_counter()
            tests_run = subprocess.run(tests_command, capture_output=True, text=True,
                                       timeout=MESH_TIMEOUT, cwd=ROOT.parent)
            tests_elapsed = time.perf_counter() - tick
            tests_stdout = tests_run.stdout + tests_run.stderr
            (case_root / "mesh_test.log").write_text(tests_stdout, encoding="utf-8")
            tests_path = mesh_out / "tests.json"
            mesh_tests = json.loads(tests_path.read_text(encoding="utf-8")) if tests_path.is_file() else []
            mesh_tests_pass = bool(tests_run.returncode == 0 and mesh_tests
                                   and all(row.get("passed") is True for row in mesh_tests))
            boundary_tags_pass = all(tag in np.unique(mesh["boundary_tags"]) for tag in [20, 30, 100])
            geometry_match = all(
                math.isclose(float(mesh_metadata.get(key, math.nan)), expected,
                             rel_tol=0.0, abs_tol=1.0e-12)
                for key, expected in [
                    ("size", SIZE), ("span", SPAN), ("thickness", case["thickness"]),
                    ("body_length", BODY_LENGTH), ("body_radius", case["radius"]),
                    ("body_aft", case["aft"]), ("curvature_elements_per_circle", CURVATURE),
                    ("wake_strips", WAKE_STRIPS),
                ])
            root_expected = case["aft"] >= CHORD
            root_record = mesh_metadata.get("root_wake_support")
            root_pass = (not root_expected and not root_record) or (
                isinstance(root_record, dict) and root_record.get("status") == "constructed")
            map_pass = False
            root_half_width = None
            exact_knot_error = None
            near_knot_gap = None
            if root_expected and isinstance(root_record, dict):
                body_center_x = case["aft"] - BODY_LENGTH / 2.0
                section = (CHORD - body_center_x) / (BODY_LENGTH / 2.0)
                root_half_width = case["radius"] * math.sqrt(1.0 - section**2)
                expected_interval = np.array([-root_half_width, root_half_width])
                saved_interval = np.asarray(root_record.get("root_interval_y"), dtype=float)
                station_map = np.asarray(mesh["gamma_station_dof"], dtype=int)
                stations = np.asarray(mesh["gamma_stations"], dtype=float)
                metadata_map = np.asarray(root_record.get("station_to_dof"), dtype=int)
                expected_knot_interval = case["name"].startswith("legal_root_knot")
                map_pass = bool(
                    saved_interval.shape == (2,)
                    and np.allclose(saved_interval, expected_interval, rtol=0.0, atol=1.0e-10)
                    and np.array_equal(station_map, metadata_map)
                    and station_map.shape == stations.shape
                    and np.isfinite(stations).all()
                    and np.all(np.diff(stations) > 0.0)
                    and np.all(station_map[[0, -1]] == -1))
                endpoint_gaps = np.abs(stations - root_half_width)
                exact_knot_error = float(endpoint_gaps.min())
                if expected_knot_interval:
                    map_pass = map_pass and exact_knot_error < 1.0e-12
                    if "plus_1e-9" in case["name"]:
                        near_knot_gap = float(abs(root_half_width - 0.25))
                        map_pass = map_pass and near_knot_gap > 1.0e-10
            topology_pass = bool(mesh_tests_pass and boundary_tags_pass and geometry_match
                                 and root_pass and (not root_expected or map_pass))
            record.update(
                status="mesh_topology_passed" if topology_pass else "mesh_topology_failed",
                command=mesh_command, returncode=returncode, timeout=False,
                mesh_elapsed_seconds=mesh_elapsed,
                mesh_sha256=mesh_sha256,
                mesh_metadata_sha256=hashlib.sha256(mesh_metadata_path.read_bytes()).hexdigest(),
                mesh_points=mesh_metadata.get("points"), mesh_tets=mesh_metadata.get("tets"),
                requested_wake_strips=mesh_metadata.get("wake_strips"),
                actual_wake_strips=mesh_metadata.get("actual_wake_strips"),
                minimum_dense_jacobian=mesh_metadata.get("min_dense_jacobian"),
                mesh_test_command=tests_command,
                mesh_test_elapsed_seconds=tests_elapsed,
                body_boundary_tags=sorted(int(tag) for tag in np.unique(mesh["boundary_tags"])
                                          if tag in [10, 20, 30, 100]),
                geometry_metadata_match=geometry_match,
                independent_mesh_test_returncode=tests_run.returncode,
                independent_mesh_tests=mesh_tests,
                root_wake_support=root_record,
                root_interval_half_width=root_half_width,
                exact_0p25_station_distance=exact_knot_error,
                root_plus_1e9_perturbation=near_knot_gap,
                topology_pass=topology_pass,
            )
            if topology_pass:
                solve_out = case_root / "solve"
                solve_command = [
                    sys.executable, str(solve_script), "--mesh", str(mesh_path),
                    "--alpha", str(ALPHA), "--mach", str(MACH), "--ncpu", str(NCPU),
                    "--kutta", "weak", "--linear-solver", "schur", "--out", str(solve_out),
                ]
                tick = time.perf_counter()
                try:
                    solve_run = subprocess.run(solve_command, capture_output=True, text=True,
                                               timeout=SOLVE_TIMEOUT, cwd=ROOT.parent)
                    solve_stdout, solve_stderr = solve_run.stdout, solve_run.stderr
                    solve_returncode, solve_timeout = solve_run.returncode, False
                except subprocess.TimeoutExpired as error:
                    solve_stdout, solve_stderr = error.stdout or "", error.stderr or ""
                    solve_returncode, solve_timeout = None, True
                if isinstance(solve_stdout, bytes):
                    solve_stdout = solve_stdout.decode(errors="replace")
                if isinstance(solve_stderr, bytes):
                    solve_stderr = solve_stderr.decode(errors="replace")
                (case_root / "solve.log").write_text(solve_stdout + solve_stderr, encoding="utf-8")
                solve_elapsed = time.perf_counter() - tick
                result_path = solve_out / "result.json"
                field_path = solve_out / "field.npz"
                solve_result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
                if solve_result.get("converged") and field_path.is_file():
                    field = np.load(field_path, allow_pickle=False)
                    required_finite = all(np.isfinite(field[key]).all()
                                          for key in ["cp", "surface_velocity", "surface_normal",
                                                      "surface_weight", "gamma"])
                    reduced_gamma = np.asarray(field["gamma"])
                    saved_map = np.asarray(field["gamma_station_dof"], dtype=int)
                    mesh_map = np.asarray(mesh["gamma_station_dof"], dtype=int)
                    stations = np.asarray(mesh["gamma_stations"], dtype=float)
                    expanded_gamma = np.zeros(len(stations))
                    free_station = saved_map >= 0
                    expanded_gamma[free_station] = reduced_gamma[saved_map[free_station]]
                    result_jump = np.asarray(solve_result["potential_jump"], dtype=float)
                    jump_error = float(np.max(abs(
                        expanded_gamma - result_jump / float(solve_result["normalization_length"]))))
                    tag_values = np.asarray(field["tags"])
                    reference_area = float(mesh_metadata["reference_area"])
                    wall = np.isin(field["tags"], [10, 20, 30])
                    raw_force = np.sum(
                        field["cp"][wall, :, None] * field["surface_normal"][wall]
                        * field["surface_weight"][wall, :, None], axis=(0, 1)) / reference_area
                    component_force = np.sum(
                        [np.asarray(entry["force_coefficient"])
                         for entry in solve_result["component_loads"].values()], axis=0)
                    force_result_error = float(np.max(abs(
                        raw_force - np.asarray(solve_result["pressure_force_coefficient"]))))
                    force_component_error = float(np.max(abs(
                        component_force - np.asarray(solve_result["pressure_force_coefficient"]))))
                    surface_ok = bool(
                        required_finite and np.all(field["surface_weight"] > 0.0)
                        and solve_result["min_beta"] > 0.0
                        and len(reduced_gamma) == int(mesh_map.max() + 1)
                        and np.array_equal(saved_map, mesh_map)
                        and len(result_jump) == len(stations)
                        and len(solve_result["gamma_stations"]) == len(stations)
                        and jump_error < 1.0e-10
                        and force_result_error < FORCE_TOLERANCE
                        and force_component_error < FORCE_TOLERANCE)
                    record.update(
                        status="passed" if surface_ok else "surface_sanity_failed",
                        solve_command=solve_command, solve_returncode=solve_returncode,
                        solve_timeout=solve_timeout, solve_elapsed_seconds=solve_elapsed,
                        converged=solve_result.get("converged", False),
                        gamma_station_count=int(len(stations)),
                        gamma_reduced_dof_count=int(len(reduced_gamma)),
                        gamma_expansion_max_abs_error=jump_error,
                        raw_force_wall_face_count=int(wall.sum()),
                        cp_finite_count=int(np.isfinite(field["cp"]).sum()),
                        cp_sample_count=int(field["cp"].size),
                        raw_force_vs_result_max_abs_error=force_result_error,
                        component_force_closure_max_abs_error=force_component_error,
                        pressure_force_coefficient=solve_result.get("pressure_force_coefficient"),
                        cl_pressure=solve_result.get("cl_pressure"),
                        cl_jump=solve_result.get("cl_jump"),
                        min_beta=solve_result.get("min_beta"),
                        surface_sanity_pass=surface_ok,
                        result=solve_result,
                    )
                else:
                    record.update(status="solver_timeout" if solve_timeout else "solver_failed",
                                  solve_command=solve_command, solve_returncode=solve_returncode,
                                  solve_timeout=solve_timeout, solve_elapsed_seconds=solve_elapsed,
                                  result=solve_result, passed=False)
            record["passed"] = record["status"] == "passed"

    if not record.get("passed", False):
        failures.append(case["name"])
    current_hashes = {
        str(path.relative_to(ROOT.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    current_hashes[str(pilot_mesh_path.relative_to(ROOT.parent))] = hashlib.sha256(
        pilot_mesh_path.read_bytes()).hexdigest()
    if current_hashes != source_hashes:
        run_record["status"] = "source_hash_changed"
        run_record["source_sha256_after_change"] = current_hashes
        run_record["cases_completed_before_source_change"] = len(run_record["cases"])
        (args.out / "progress.json").write_text(
            json.dumps(run_record, indent=2, allow_nan=False), encoding="utf-8")
        raise SystemExit("Frozen source hash changed; stopped matrix immediately")
    run_record["cases_completed"] = len(run_record["cases"])
    run_record["failed_cases_so_far"] = failures.copy()
    (args.out / "progress.json").write_text(
        json.dumps(run_record, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "case": case["name"], "status": record["status"],
        "passed": record.get("passed", False),
        "cases_completed": len(run_record["cases"]),
    }, indent=2), flush=True)

run_record["source_sha256_after"] = {
    str(path.relative_to(ROOT.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in source_paths
}
run_record["source_sha256_after"][str(pilot_mesh_path.relative_to(ROOT.parent))] = hashlib.sha256(
    pilot_mesh_path.read_bytes()).hexdigest()
run_record["source_hashes_unchanged"] = (
    run_record["source_sha256_after"] == source_hashes)
run_record["failed_cases"] = failures
run_record["passed"] = len(cases) - len(failures)
run_record["total"] = len(cases)
run_record["status"] = "passed" if not failures and run_record["source_hashes_unchanged"] else "completed_with_failures"
run_record["elapsed_seconds"] = sum(
    row.get("mesh_elapsed_seconds", row.get("elapsed_seconds", 0.0))
    + row.get("solve_elapsed_seconds", 0.0) for row in run_record["cases"])
(args.out / "progress.json").write_text(
    json.dumps(run_record, indent=2, allow_nan=False), encoding="utf-8")
(args.out / "result.json").write_text(
    json.dumps(run_record, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({
    "status": run_record["status"],
    "passed": run_record["passed"],
    "total": run_record["total"],
    "failed_cases": failures,
    "source_hashes_unchanged": run_record["source_hashes_unchanged"],
    "output": str(args.out),
}, indent=2))
if run_record["status"] != "passed":
    raise SystemExit(2)
