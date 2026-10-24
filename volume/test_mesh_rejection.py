import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VOLUME = ROOT / "volume"
RESULTS = VOLUME / "results" / "mesh_rejection_v1"
sys.path.insert(0, str(VOLUME))
import p2


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_patch(path, points, tets, boundary, tags=None, boundary_tet=None):
    arrays = {
        "points": points,
        "tets": tets,
        "boundary_triangles": boundary,
        "boundary_tags": np.full(len(boundary), 100, dtype=np.int64) if tags is None else tags,
    }
    if boundary_tet is not None:
        arrays["boundary_tet"] = boundary_tet
    np.savez_compressed(path, **arrays)


RESULTS.mkdir(parents=True, exist_ok=True)
source_hashes = {
    "p2.py": sha256(VOLUME / "p2.py"),
    "solve.py": sha256(ROOT / "lifting3d" / "solve.py"),
    "test_p2.py": sha256(VOLUME / "test_p2.py"),
}

corners = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ]
)
corner_tets = [[0, 1, 2, 3], [0, 2, 1, 4]]
points = list(corners)
edge_node = {}
tets = []
for corner_tet in corner_tets:
    tet = list(corner_tet)
    for i, j in p2.TET_EDGES:
        key = tuple(sorted((corner_tet[i], corner_tet[j])))
        if key not in edge_node:
            edge_node[key] = len(points)
            points.append(0.5 * (corners[key[0]] + corners[key[1]]))
        tet.append(edge_node[key])
    tets.append(tet)
points = np.asarray(points)
tets = np.asarray(tets, dtype=np.int64)
face_map = {}
for tet_index, tet in enumerate(tets):
    for omitted in range(4):
        face_corners = tet[[i for i in range(4) if i != omitted]]
        key = tuple(sorted(face_corners.tolist()))
        face_edges = [
            edge_node[tuple(sorted((int(face_corners[0]), int(face_corners[1]))))],
            edge_node[tuple(sorted((int(face_corners[1]), int(face_corners[2]))))],
            edge_node[tuple(sorted((int(face_corners[2]), int(face_corners[0]))))],
        ]
        face_map.setdefault(key, []).append(np.r_[face_corners, face_edges])
boundary = np.asarray([items[0] for items in face_map.values() if len(items) == 1], dtype=np.int64)
boundary_tet = np.zeros(len(boundary), dtype=np.int64)
valid_path = RESULTS / "valid_two_tet_patch.npz"
write_patch(valid_path, points, tets, boundary, boundary_tet=boundary_tet)
p2.load_mesh(valid_path)

bad_cases = []

tags = np.full(len(boundary), 100.0)
tags[0] = 100.5
bad_cases.append(("float_boundary_tags", points, tets, boundary, tags, boundary_tet, "boundary_tags must be integer"))

bad_tets = tets.copy()
duplicate_midnode = len(points)
bad_points = np.vstack([points, points[edge_node[(0, 1)]]])
bad_tets[1, 4 + p2.TET_EDGES.index((0, 1))] = duplicate_midnode
bad_cases.append(("interior_shared_edge_mids_mismatch", bad_points, bad_tets, boundary, None, boundary_tet,
                  "sharing a corner edge must share the same midside node"))

bad_tets = tets.copy()
bad_tets[0, 9] = bad_tets[0, 0]
bad_cases.append(("repeated_tet10_node", points, bad_tets, boundary, None, boundary_tet,
                  "ten distinct node indices"))

orphan_points = np.vstack([points, [[3.0, 4.0, 5.0]]])
bad_cases.append(("orphan_point", orphan_points, tets, boundary, None, boundary_tet,
                  "point nodes unused"))

nan_points = points.copy()
nan_points[-1, 2] = np.nan
bad_cases.append(("nan_point", nan_points, tets, boundary, None, boundary_tet,
                  "finite coordinates"))

bad_cases.append(("float_boundary_tet_adjacency", points, tets, boundary, None,
                  boundary_tet.astype(float), "boundary_tet adjacency must be integer"))

mesh_negative_results = []
for case_name, case_points, case_tets, case_boundary, case_tags, case_adjacent, expected_error in bad_cases:
    case_dir = RESULTS / "p2_negative" / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = case_dir / "mesh.npz"
    write_patch(mesh_path, case_points, case_tets, case_boundary, case_tags, case_adjacent)
    started = time.perf_counter()
    try:
        p2.load_mesh(mesh_path)
        actual_error = None
    except Exception as error:
        actual_error = f"{type(error).__name__}: {error}"
    elapsed = time.perf_counter() - started
    passed = actual_error is not None and expected_error in actual_error
    record = {
        "case": case_name,
        "path": str(mesh_path.relative_to(ROOT)),
        "input_sha256": sha256(mesh_path),
        "expected_error_class": expected_error,
        "actual_error": actual_error,
        "exit_code": 0 if passed else 1,
        "elapsed_seconds": elapsed,
    }
    (case_dir / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    mesh_negative_results.append(record)

wing_mesh = ROOT / "lifting3d" / "meshes" / "wing03" / "mesh.npz"
wing_metadata = ROOT / "lifting3d" / "meshes" / "wing03" / "mesh.json"
with np.load(wing_mesh) as archive:
    wing_arrays = {key: np.array(archive[key]) for key in archive.files}

entry_cases = [
    ("physical_tag_float", "boundary_tags", lambda value: value.astype(float), "Physical boundary tags must be integers"),
    ("cut_index_float", "cut_upper_nodes", lambda value: value.astype(float), "cut_upper_nodes must contain integer"),
    ("cut_index_out_of_range", "cut_upper_nodes", lambda value: np.r_[value[:-1], len(wing_arrays["points"])], "cut_upper_nodes contains an out-of-range"),
    ("te_index_float", "te_upper_nodes", lambda value: value.astype(float), "te_upper_nodes must contain integer"),
    ("te_index_out_of_range", "te_upper_nodes", lambda value: np.r_[value[:-1], len(wing_arrays["points"])], "te_upper_nodes contains an out-of-range"),
]
entry_results = []
for case_name, key, mutation, expected_error in entry_cases:
    case_dir = RESULTS / "lifting3d_early_reject" / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    bad_arrays = {name: value.copy() for name, value in wing_arrays.items()}
    bad_arrays[key] = mutation(bad_arrays[key])
    bad_mesh = case_dir / "mesh.npz"
    np.savez_compressed(bad_mesh, **bad_arrays)
    shutil.copy2(wing_metadata, case_dir / "mesh.json")
    out_dir = case_dir / "solver_out"
    command = [
        sys.executable,
        str(ROOT / "lifting3d" / "solve.py"),
        "--mesh", str(bad_mesh),
        "--alpha", "5",
        "--mach", "0",
        "--ncpu", "8",
        "--out", str(out_dir),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    combined_output = completed.stdout + completed.stderr
    passed = completed.returncode != 0 and expected_error in combined_output and not (out_dir / "topology_input.npz").exists()
    record = {
        "case": case_name,
        "command": command,
        "exit_code": completed.returncode,
        "passed": passed,
        "expected_error_class": expected_error,
        "output": combined_output,
        "topology_input_written": (out_dir / "topology_input.npz").exists(),
        "input_sha256": sha256(bad_mesh),
        "mesh_metadata_sha256": sha256(case_dir / "mesh.json"),
        "elapsed_seconds": elapsed,
    }
    (case_dir / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    entry_results.append(record)

meaningful_cases = [
    ("sphere_coarse_m0_q3", "volume/meshes/p2/sphere_coarse/mesh.npz", 0.0, "sphere-exact", 3.14159265358979),
    ("sphere_coarse_m02_q3", "volume/meshes/p2/sphere_coarse/mesh.npz", 0.2, "uniform", 3.14159265358979),
    ("sphere_fine_m0_q3", "volume/meshes/p2/sphere_fine/mesh.npz", 0.0, "sphere-exact", 3.14159265358979),
    ("sphere_fine_m02_q3", "volume/meshes/p2/sphere_fine/mesh.npz", 0.2, "uniform", 3.14159265358979),
    ("ellipsoid_s05_v4_compact_m0_q3", "volume/meshes/curved/ellipsoid_s05_v4_compact/mesh.npz", 0.0, "uniform", 1.0),
    ("ellipsoid_s05_v4_compact_m02_q3", "volume/meshes/curved/ellipsoid_s05_v4_compact/mesh.npz", 0.2, "uniform", 1.0),
    ("ellipsoid_s05_v4_m0_q3", "volume/meshes/curved/ellipsoid_s05_v4/mesh.npz", 0.0, "uniform", 1.0),
    ("ellipsoid_s05_v4_m02_q3", "volume/meshes/curved/ellipsoid_s05_v4/mesh.npz", 0.2, "uniform", 1.0),
    ("wingbody_s08_v2_m0_q3", "volume/meshes/curved/wingbody_s08_v2/mesh.npz", 0.0, "uniform", 1.0),
    ("wingbody_s08_v2_m02_q3", "volume/meshes/curved/wingbody_s08_v2/mesh.npz", 0.2, "uniform", 1.0),
]
meaningful_results = []
for case_name, relative_mesh, mach, boundary_mode, sref in meaningful_cases:
    mesh_path = ROOT / relative_mesh
    output_dir = RESULTS / "p2_meaningful" / case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(VOLUME / "solve_p2.py"),
        "--mesh", str(mesh_path),
        "--mach", str(mach),
        "--alpha", "0",
        "--boundary", boundary_mode,
        "--sref", str(sref),
        "--ncpu", "8",
        "--quad-order", "3",
        "--out", str(output_dir),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    (output_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    result_path = output_dir / "result.json"
    solver_result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
    expected_status = "failed" if "ellipsoid_s05_v4_m" in case_name else "converged"
    record = {
        "case": case_name,
        "command": command,
        "exit_code": completed.returncode,
        "expected_status": expected_status,
        "actual_status": solver_result.get("status") if solver_result else None,
        "mesh_path": str(mesh_path.relative_to(ROOT)),
        "mesh_sha256": sha256(mesh_path),
        "elapsed_seconds": elapsed,
        "result_json": str(result_path.relative_to(ROOT)) if result_path.exists() else None,
        "result_error": solver_result.get("error") if solver_result else None,
    }
    (output_dir / "regression_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    meaningful_results.append(record)

unittest_command = [sys.executable, "-m", "unittest", "volume.test_p2", "-v"]
unit_started = time.perf_counter()
unit_test = subprocess.run(unittest_command, cwd=ROOT, capture_output=True, text=True)
unit_elapsed = time.perf_counter() - unit_started
(RESULTS / "test_p2_stdout.txt").write_text(unit_test.stdout, encoding="utf-8")
(RESULTS / "test_p2_stderr.txt").write_text(unit_test.stderr, encoding="utf-8")

summary = {
    "source_hashes": source_hashes,
    "valid_two_tet_patch": {
        "path": str(valid_path.relative_to(ROOT)),
        "input_sha256": sha256(valid_path),
        "status": "loaded",
    },
    "p2_negative_cases": mesh_negative_results,
    "lifting3d_entry_negative_cases": entry_results,
    "p2_meaningful_cases": meaningful_results,
    "unittest": {
        "command": unittest_command,
        "exit_code": unit_test.returncode,
        "elapsed_seconds": unit_elapsed,
        "stdout": unit_test.stdout,
        "stderr": unit_test.stderr,
    },
}
(RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

checks_passed = (
    all(item["exit_code"] == 0 for item in mesh_negative_results)
    and all(item["passed"] for item in entry_results)
    and all(item["actual_status"] == item["expected_status"] for item in meaningful_results)
    and unit_test.returncode == 0
)
print(json.dumps({
    "status": "passed" if checks_passed else "failed",
    "p2_negative_passed": sum(item["exit_code"] == 0 for item in mesh_negative_results),
    "p2_negative_total": len(mesh_negative_results),
    "lifting3d_entry_passed": sum(item["passed"] for item in entry_results),
    "lifting3d_entry_total": len(entry_results),
    "meaningful_statuses": {item["case"]: item["actual_status"] for item in meaningful_results},
    "unittest_exit_code": unit_test.returncode,
    "summary": str(RESULTS / "summary.json"),
}, indent=2))
if not checks_passed:
    raise SystemExit(1)
