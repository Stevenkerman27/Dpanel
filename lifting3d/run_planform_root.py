"""Run the frozen five-case merged-root planform matrix in serial."""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/planform_root_v1"
parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()
CASES = [
    dict(name="taper1_sweep20", taper=1.0, sweep=20.0),
    dict(name="taper0p5_sweep0", taper=0.5, sweep=0.0),
    dict(name="taper0p5_sweep20", taper=0.5, sweep=20.0),
    dict(name="taper1_sweepm10", taper=1.0, sweep=-10.0),
    dict(name="taper1p5_sweep0", taper=1.5, sweep=0.0),
]
SOURCE_PATHS = [
    ROOT / "build_mesh.py",
    ROOT / "solve.py",
    ROOT / "planform_root.py",
    ROOT / "root_wake.py",
    ROOT / "schur.py",
    ROOT / "test_mesh.py",
    ROOT / "PROTOCOL.md",
    ROOT / "SCHUR_PROTOCOL.md",
    ROOT / "ROOT_WAKE_PROTOCOL.md",
    ROOT / "ROOT_GEOMETRY_TOLERANCE.md",
]
PROTOCOL_PATH = ROOT / "PLANFORM_ROOT_RESULTS.md"

if OUT.exists() and any(OUT.iterdir()):
    assert args.resume, f"Refusing to overwrite evidence: {OUT}"
    prior_progress = OUT / "progress.json"
    if prior_progress.is_file() and not (OUT / "attempt_001_progress.json").exists():
        (OUT / "attempt_001_progress.json").write_bytes(prior_progress.read_bytes())
OUT.mkdir(parents=True, exist_ok=True)
started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
started_clock = time.perf_counter()
source_hashes_start = {
    str(path.relative_to(ROOT.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in SOURCE_PATHS
}
(OUT / "protocol.snapshot.md").write_bytes(PROTOCOL_PATH.read_bytes())

run_record = dict(
    status="running",
    started_at=started_at,
    ncpu=8,
    mesh_timeout_seconds=90,
    solve_timeout_seconds=180,
    frozen_parameters=dict(
        thickness=0.24, body_length=4.0, body_radius=0.35, body_aft=1.5,
        size=0.25, curvature=16, wake_strips=16, root_wake="merged",
        kutta="weak", linear_solver="schur", mach=0.0, alpha_deg=5.0,
    ),
    source_sha256_start=source_hashes_start,
    source_sha256_end=None,
    cases=[],
    commands=[],
    scope="Geometry/root-map and solver algebra robustness only; no Cp accuracy acceptance.",
)
(OUT / "progress.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")

for case in CASES:
    case_root = OUT / case["name"]
    mesh_root = case_root / "mesh"
    solve_root = case_root / "solve"
    case_root.mkdir(parents=True, exist_ok=True)
    if not mesh_root.exists():
        mesh_root.mkdir()
    record = dict(case=case, status="mesh_pending")
    run_record["cases"].append(record)
    (OUT / "progress.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")

    mesh_command = [
        sys.executable, str(ROOT / "build_mesh.py"),
        "--size", "0.25", "--span", "4", "--thickness", "0.24",
        "--curvature", "16", "--wake-strips", "16",
        "--body-length", "4", "--body-radius", "0.35", "--body-aft", "1.5",
        "--radius", "5", "--far-size", str(5.0 / 3.0), "--root-wake", "merged",
        "--taper", str(case["taper"]), "--sweep", str(case["sweep"]),
        "--out", str(mesh_root),
    ]
    existing_mesh = (mesh_root / "mesh.npz").is_file() and (mesh_root / "mesh.json").is_file()
    if args.resume and existing_mesh:
        mesh_record = dict(command=mesh_command, returncode=0, timeout=False,
                           elapsed_seconds=0.0, reused_existing_mesh=True)
    else:
        mesh_started = time.perf_counter()
        mesh_process = subprocess.Popen(mesh_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        mesh_timeout = False
        try:
            mesh_stdout, mesh_stderr = mesh_process.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            mesh_timeout = True
            kill = subprocess.run(["taskkill", "/PID", str(mesh_process.pid), "/T", "/F"],
                                  capture_output=True, text=True)
            mesh_stdout, mesh_stderr = mesh_process.communicate()
            mesh_stderr += "\nTimeout process-tree termination:\n" + kill.stdout + kill.stderr
        mesh_elapsed = time.perf_counter() - mesh_started
        (case_root / "mesh.log").write_text(mesh_stdout + mesh_stderr, encoding="utf-8")
        mesh_returncode = mesh_process.returncode
        mesh_record = dict(command=mesh_command, returncode=mesh_returncode,
                           timeout=mesh_timeout, elapsed_seconds=mesh_elapsed,
                           process_tree_terminated_on_timeout=mesh_timeout)
    mesh_returncode = mesh_record["returncode"]
    mesh_timeout = mesh_record["timeout"]
    run_record["commands"].append(dict(case=case["name"], stage="mesh", **mesh_record))
    record["mesh"] = mesh_record
    mesh_path = mesh_root / "mesh.npz"
    metadata_path = mesh_root / "mesh.json"
    if mesh_timeout or mesh_returncode != 0 or not mesh_path.is_file() or not metadata_path.is_file():
        record["status"] = "mesh_failed"
        run_record["status"] = "stopped_at_mesh_failure"
        (OUT / "progress.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
        break

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(mesh_path, allow_pickle=False) as mesh:
        mesh_points = mesh["points"]
        mesh_tets = mesh["tets"]
        mesh_boundary_tags = mesh["boundary_tags"]
        mesh_map = mesh["gamma_station_dof"]
    record["mesh"].update(
        sha256=hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        points=int(len(mesh_points)), tets=int(len(mesh_tets)),
        gamma_unknowns=metadata.get("gamma_unknowns"),
        requested_wake_strips=metadata.get("wake_strips"),
        actual_wake_strips=metadata.get("actual_wake_strips"),
        geometry_metadata=dict(
            taper_ratio=metadata.get("taper_ratio"),
            quarter_chord_sweep_deg=metadata.get("quarter_chord_sweep_deg"),
            body_aft=metadata.get("body_aft"),
            root_wake_support=metadata.get("root_wake_support"),
        ),
    )
    geometry_matches = (
        metadata.get("taper_ratio") == case["taper"]
        and metadata.get("quarter_chord_sweep_deg") == case["sweep"]
        and metadata.get("thickness") == 0.24
        and metadata.get("body_length") == 4.0
        and metadata.get("body_radius") == 0.35
        and metadata.get("body_aft") == 1.5
        and metadata.get("size") == 0.25
        and metadata.get("curvature_elements_per_circle") == 16
        and metadata.get("wake_strips") == 16
        and metadata.get("root_wake_support", {}).get("status") == "constructed"
        and set(np.unique(mesh_boundary_tags).tolist()).issubset({10, 20, 30, 100})
        and {20, 30}.issubset(set(np.unique(mesh_boundary_tags).tolist()))
    )
    record["mesh"]["metadata_match"] = bool(geometry_matches)
    if not geometry_matches:
        record["status"] = "mesh_metadata_rejected"
        run_record["status"] = "stopped_at_mesh_metadata_rejection"
        (OUT / "progress.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
        break

    mesh_test_command = [sys.executable, str(ROOT / "test_mesh.py"), "--mesh", str(mesh_path)]
    test_run = subprocess.run(mesh_test_command, capture_output=True, text=True)
    (case_root / "mesh_test.log").write_text(test_run.stdout + test_run.stderr, encoding="utf-8")
    tests_path = mesh_root / "tests.json"
    tests = json.loads(tests_path.read_text(encoding="utf-8")) if tests_path.is_file() else []
    record["mesh_test"] = dict(
        command=mesh_test_command,
        returncode=test_run.returncode,
        tests=tests,
        passed_count=sum(row.get("passed") is True for row in tests),
        total_count=len(tests),
        eight_of_eight=(test_run.returncode == 0 and len(tests) == 8
                        and all(row.get("passed") is True for row in tests)),
    )
    run_record["commands"].append(dict(
        case=case["name"], stage="mesh_test", command=mesh_test_command,
        returncode=test_run.returncode, tests=tests,
    ))
    if not record["mesh_test"]["eight_of_eight"]:
        record["status"] = "mesh_test_failed"
        run_record["status"] = "stopped_at_mesh_test_failure"
        (OUT / "progress.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
        break

    solve_root.mkdir()
    solve_command = [
        sys.executable, str(ROOT / "solve.py"), "--mesh", str(mesh_path),
        "--alpha", "5", "--mach", "0", "--ncpu", "8", "--kutta", "weak",
        "--linear-solver", "schur", "--out", str(solve_root),
    ]
    solve_started = time.perf_counter()
    solve_process = subprocess.Popen(solve_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    solve_timeout = False
    try:
        solve_stdout, solve_stderr = solve_process.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        solve_timeout = True
        kill = subprocess.run(["taskkill", "/PID", str(solve_process.pid), "/T", "/F"],
                              capture_output=True, text=True)
        solve_stdout, solve_stderr = solve_process.communicate()
        solve_stderr += "\nTimeout process-tree termination:\n" + kill.stdout + kill.stderr
    solve_elapsed = time.perf_counter() - solve_started
    (case_root / "solve.log").write_text(solve_stdout + solve_stderr, encoding="utf-8")
    solve_returncode = solve_process.returncode
    record["solve"] = dict(command=solve_command, returncode=solve_returncode,
                           timeout=solve_timeout, elapsed_seconds=solve_elapsed,
                           process_tree_terminated_on_timeout=solve_timeout)
    run_record["commands"].append(dict(case=case["name"], stage="solve",
                                       **record["solve"]))
    result_path = solve_root / "result.json"
    field_path = solve_root / "field.npz"
    if solve_timeout or solve_returncode != 0 or not result_path.is_file() or not field_path.is_file():
        record["status"] = "solver_failed"
        run_record["status"] = "stopped_at_solver_failure"
        (OUT / "progress.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
        break

    result = json.loads(result_path.read_text(encoding="utf-8"))
    with np.load(field_path, allow_pickle=False) as field:
        numeric_fields_finite = all(
            np.isfinite(field[key]).all()
            for key in field.files if np.issubdtype(field[key].dtype, np.number)
        )
        field_tags = field["tags"]
        surface_cp = field["cp"]
        surface_normals = field["surface_normal"]
        surface_weights = field["surface_weight"]
        surface_velocity = field["surface_velocity"]
        output_map = field["gamma_station_dof"]
        reduced_gamma = field["gamma"]
    station_map = np.asarray(result["gamma_station_dof"], dtype=int)
    potential_jump = np.asarray(result["potential_jump"], dtype=float)
    station_values = np.zeros(len(station_map))
    free_station = station_map >= 0
    station_values[free_station] = reduced_gamma[station_map[free_station]] * result["normalization_length"]
    map_complete = (
        station_map.shape == mesh_map.shape
        and np.array_equal(station_map, mesh_map)
        and np.array_equal(output_map, mesh_map)
        and np.array_equal(np.unique(station_map[station_map >= 0]),
                           np.arange(int(station_map.max()) + 1))
        and station_map[0] == -1 and station_map[-1] == -1
        and len(reduced_gamma) == int(station_map.max()) + 1
        and np.max(abs(station_values - potential_jump)) < 1e-12
    )
    kutta_measure = np.asarray(result.get("kutta_test_measure", []), dtype=float)
    positive_kutta_measure = (
        kutta_measure.size == int(station_map.max()) + 1
        and np.isfinite(kutta_measure).all()
        and np.all(kutta_measure > 0.0)
    )
    physical = np.isin(field_tags, [10, 20, 30])
    raw_force = np.sum(surface_cp[physical, :, None] * surface_normals[physical]
                       * surface_weights[physical, :, None], axis=(0, 1))
    raw_force /= metadata["reference_area"]
    component_force = np.sum([
        np.asarray(entry["force_coefficient"], dtype=float)
        for tag, entry in result["component_loads"].items() if int(tag) in [10, 20, 30]
    ], axis=0)
    raw_force_matches_components = (
        set(np.unique(field_tags).tolist()).issubset({10, 20, 30, 100, 201, 202})
        and np.isfinite(raw_force).all()
        and np.max(abs(raw_force - component_force)) < 1e-10
        and np.max(abs(raw_force - np.asarray(result["pressure_force_coefficient"]))) < 1e-10
    )
    record["solve"].update(
        status=result.get("status"),
        converged=result.get("converged"),
        nodes=result.get("nodes"),
        tets=result.get("elements"),
        gamma_unknowns=result.get("gamma_unknowns"),
        cl_pressure=result.get("cl_pressure"),
        cl_jump=result.get("cl_jump"),
        cd_pressure=result.get("cd_pressure"),
        weak_mass_residual_norm=result.get("weak_mass_residual_norm"),
        kutta_max=result.get("kutta_max"),
        kutta_test_measure=result.get("kutta_test_measure"),
        raw_fields_finite=bool(numeric_fields_finite),
        positive_kutta_measure=bool(positive_kutta_measure),
        station_map_complete_expansion=bool(map_complete),
        raw_wall_force_coefficient=raw_force.tolist(),
        component_force_coefficient=component_force.tolist(),
        raw_force_component_closure_error=float(np.max(abs(raw_force-component_force))),
        raw_wall_force_matches_components=bool(raw_force_matches_components),
        result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
        field_sha256=hashlib.sha256(field_path.read_bytes()).hexdigest(),
    )
    solver_gates = (
        result.get("converged") is True
        and numeric_fields_finite
        and positive_kutta_measure
        and map_complete
        and raw_force_matches_components
    )
    record["status"] = "passed" if solver_gates else "solver_gate_failed"
    run_record["status"] = "running" if solver_gates else "stopped_at_solver_gate_failure"
    (OUT / "progress.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    if not solver_gates:
        break

source_hashes_end = {
    str(path.relative_to(ROOT.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in SOURCE_PATHS
}
run_record["source_sha256_end"] = source_hashes_end
run_record["core_unchanged"] = source_hashes_start == source_hashes_end
run_record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
run_record["elapsed_seconds"] = time.perf_counter() - started_clock
if run_record["status"] == "running":
    run_record["status"] = "completed" if len(run_record["cases"]) == len(CASES) else "incomplete"
if not run_record["core_unchanged"]:
    run_record["status"] = "core_hash_changed"
(OUT / "progress.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
(OUT / "result.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
(OUT / "driver.py.snapshot").write_bytes(Path(__file__).read_bytes())
print(json.dumps({
    "status": run_record["status"],
    "cases": [{"name": row["case"]["name"], "status": row["status"]}
              for row in run_record["cases"]],
    "core_unchanged": run_record["core_unchanged"],
    "elapsed_seconds": run_record["elapsed_seconds"],
}, indent=2))
assert run_record["status"] == "completed" and run_record["core_unchanged"]
