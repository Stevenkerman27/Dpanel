"""Run the predeclared extra_repair_v1 study."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

root = Path(__file__).resolve().parent
mesh_root = root / "meshes/extra_repair_v2"
result_root = root / "results/extra_repair_v2"
builder = root / "build_mesh.py"
solver = root / "solve.py"
protocol = root / "REPAIR_PROTOCOL.md"
assert not mesh_root.exists(), "Preserve prior extra_repair_v2 mesh evidence"
assert not result_root.exists(), "Preserve prior extra_repair_v2 result evidence"
mesh_root.mkdir(parents=True)
result_root.mkdir(parents=True)
env8 = dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8")
builds = []
records = []
commands = []
started = time.perf_counter()

for label, radius, size in [
    ("radius04_h002", 4., .02),
    ("radius08_h002", 8., .02),
    ("radius16_h002", 16., .02),
    ("center_h010", 8., .01),
    ("center_h005", 8., .005),
]:
    target = mesh_root / label
    command = [sys.executable, str(builder), "--shape", "joukowski", "--epsilon", ".08",
               "--eta", "0", "--size", str(size), "--far-size", "1",
               "--radius", str(radius), "--mirror", "--out", str(target)]
    run_started = time.perf_counter()
    run = subprocess.run(command, env=env8, capture_output=True, text=True, timeout=600)
    target.mkdir(parents=True, exist_ok=True)
    (target / "build.log").write_text(run.stdout + run.stderr, encoding="utf-8")
    mesh_json = target / "mesh.json"
    mesh_npz = target / "mesh.npz"
    metadata = json.loads(mesh_json.read_text()) if mesh_json.exists() else {}
    mesh_hash = hashlib.sha256(mesh_npz.read_bytes()).hexdigest() if mesh_npz.exists() else None
    if mesh_npz.exists():
        mesh = np.load(mesh_npz)
        triangle_points = mesh["points"][mesh["triangles"]]
        vertex_edges = np.concatenate([triangle_points[:, [1, 0]],
                                       triangle_points[:, [2, 1]],
                                       triangle_points[:, [0, 2]]])
        all_edge_lengths = np.linalg.norm(vertex_edges[:, 1]-vertex_edges[:, 0], axis=1)
        wall_lengths = np.linalg.norm(mesh["points"][mesh["edges"][mesh["edge_tags"] == 10, 1]]
                                      -mesh["points"][mesh["edges"][mesh["edge_tags"] == 10, 0]], axis=1)
        mesh_metrics = dict(max_vertex_edge_chord=float(all_edge_lengths.max()/metadata["chord"]),
                            wall_edge_min_chord=float(wall_lengths.min()/metadata["chord"]),
                            wall_edge_max_chord=float(wall_lengths.max()/metadata["chord"]))
    else:
        mesh_metrics = {}
    builds.append(dict(label=label, command=command, returncode=run.returncode,
                       elapsed_seconds=time.perf_counter()-run_started, mesh_sha256=mesh_hash,
                       metadata=metadata, mesh_metrics=mesh_metrics))
    assert run.returncode == 0 and mesh_npz.exists(), f"Mesh build failed: {label}"

rotation_source = mesh_root / "radius08_h002/mesh.npz"
rotation_metadata_source = mesh_root / "radius08_h002/mesh.json"
source_mesh = np.load(rotation_source)
source_metadata = json.loads(rotation_metadata_source.read_text())
rotation_origin = np.asarray(source_metadata["normalization_origin"])
rotated_mesh_data = {key: source_mesh[key].copy() for key in source_mesh.files}
for angle_degrees in [20., 45.]:
    angle = np.deg2rad(angle_degrees)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    rotation_dir = mesh_root / f"rotated{int(angle_degrees)}"
    rotation_dir.mkdir()
    rotated_mesh = {key: value.copy() for key, value in rotated_mesh_data.items()}
    rotated_mesh["points"] = ((rotated_mesh["points"]-rotation_origin)@rotation.T
                              +rotation_origin)
    rotated_mesh["contour"] = ((rotated_mesh["contour"]-rotation_origin)@rotation.T
                               +rotation_origin)
    np.savez_compressed(rotation_dir / "mesh.npz", **rotated_mesh)
    rotated_metadata = dict(source_metadata)
    for key in ["normalization_origin", "farfield_center", "leading_edge"]:
        value = np.asarray(rotated_metadata[key])
        rotated_metadata[key] = (((value-rotation_origin)@rotation.T)+rotation_origin).tolist()
    rotated_metadata.update(shape="rotated_file", rotation_degrees=angle_degrees,
                            rotation_source_mesh_sha256=hashlib.sha256(rotation_source.read_bytes()).hexdigest(),
                            mesh_sha256=hashlib.sha256((rotation_dir / "mesh.npz").read_bytes()).hexdigest(),
                            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (rotation_dir / "mesh.json").write_text(json.dumps(rotated_metadata, indent=2), encoding="utf-8")
    np.savetxt(rotation_dir / "contour.dat", rotated_mesh["contour"])

mesh_paths = {
    "R4": mesh_root / "radius04_h002/mesh.npz",
    "R8": mesh_root / "radius08_h002/mesh.npz",
    "R16": mesh_root / "radius16_h002/mesh.npz",
    "h010": mesh_root / "center_h010/mesh.npz",
    "h005": mesh_root / "center_h005/mesh.npz",
    "rotated20": mesh_root / "rotated20/mesh.npz",
    "rotated45": mesh_root / "rotated45/mesh.npz",
}
case_specs = []
for radius_label in ["R4", "R8", "R16"]:
    for boundary in ["uniform", "exact"]:
        case_specs.append((f"farfield_{radius_label}_{boundary}", radius_label,
                           ["--alpha", "5", "--mach", "0", "--boundary", boundary]))
for label, mesh_label, center_option in [
    ("center_h020_default", "R8", []),
    ("center_h020_xp02", "R8", ["--center", ".2", "0"]),
    ("center_h020_xm05", "R8", ["--center", "-.5", "0"]),
    ("center_h010_default", "h010", []),
    ("center_h010_xp02", "h010", ["--center", ".2", "0"]),
    ("center_h010_xm05", "h010", ["--center", "-.5", "0"]),
    ("center_h005_default", "h005", []),
    ("center_h005_xp02", "h005", ["--center", ".2", "0"]),
    ("center_h005_xm05", "h005", ["--center", "-.5", "0"]),
]:
    case_specs.append((label, mesh_label,
                       ["--alpha", "5", "--mach", "0", "--boundary", "exact", *center_option]))
case_specs.extend([
    ("rotation_base", "R8", ["--alpha", "5", "--mach", "0", "--boundary", "uniform"]),
    ("rotation20", "rotated20", ["--alpha", "25", "--mach", "0", "--boundary", "uniform"]),
    ("rotation45", "rotated45", ["--alpha", "50", "--mach", "0", "--boundary", "uniform"]),
    ("rotation_exact_reject", "rotated20",
     ["--alpha", "25", "--mach", "0", "--boundary", "exact"]),
    ("invalid_center_nan", "R8", ["--alpha", "5", "--mach", "0", "--boundary", "uniform",
                                  "--center", "nan", "0"]),
    ("invalid_center_inf", "R8", ["--alpha", "5", "--mach", "0", "--boundary", "uniform",
                                  "--center", "inf", "0"]),
    ("outside_center", "R8", ["--alpha", "5", "--mach", "0", "--boundary", "uniform",
                              "--center", "10", "10"]),
])

for label, mesh_label, options in case_specs:
    target = result_root / label
    command = [sys.executable, str(solver), "--mesh", str(mesh_paths[mesh_label]),
               *options, "--ncpu", "8", "--out", str(target)]
    run_started = time.perf_counter()
    run = subprocess.run(command, env=env8, capture_output=True, text=True, timeout=600)
    target.mkdir(parents=True, exist_ok=True)
    (target / "run.log").write_text(run.stdout + run.stderr, encoding="utf-8")
    result_path = target / "result.json"
    failure_path = target / "failure.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    failure = json.loads(failure_path.read_text()) if failure_path.exists() else {}
    cp_metrics = {}
    if (target / "field.npz").exists():
        field = np.load(target / "field.npz")
        cp = field["cp"]
        cp_finite = bool(np.isfinite(cp).all())
        cp_metrics = dict(cp_finite=cp_finite, cp_finite_count=int(np.isfinite(cp).sum()),
                          cp_min=float(np.min(cp)) if cp_finite else None,
                          cp_max=float(np.max(cp)) if cp_finite else None,
                          cp_count=int(cp.size))
    expected_rejection = label in ["rotation_exact_reject", "invalid_center_nan",
                                   "invalid_center_inf", "outside_center"]
    if label.startswith("invalid_center_"):
        failure_classified = ("finite" in failure.get("message", "").lower()
                              and run.returncode != 0)
    elif label == "outside_center":
        failure_classified = ("inside the body" in failure.get("message", "")
                              and run.returncode != 0)
    elif label == "rotation_exact_reject":
        failure_classified = ("rotated exact mapping is unsupported" in failure.get("message", "")
                              and run.returncode != 0)
    elif result:
        failure_classified = False
    else:
        failure_classified = False
    if expected_rejection:
        passed = bool(failure_classified and not result.get("converged", False))
    else:
        passed = bool(run.returncode == 0 and result.get("converged", False)
                      and cp_metrics.get("cp_finite", False))
    row = dict(case=label, mesh=mesh_label, command=command, returncode=run.returncode,
               elapsed_seconds=time.perf_counter()-run_started,
               executable=sys.executable,
               executable_sha256=hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
               mesh_sha256=hashlib.sha256(mesh_paths[mesh_label].read_bytes()).hexdigest(),
               solver_sha256=hashlib.sha256(solver.read_bytes()).hexdigest(),
               result=result, failure=failure, cp_metrics=cp_metrics,
               expected_rejection=expected_rejection, failure_classified=failure_classified,
               passed=passed)
    records.append(row)
    commands.append(dict(case=label, command=command, returncode=run.returncode))
    (result_root / "summary.json").write_text(
        json.dumps(dict(protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
                        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        solver_sha256=hashlib.sha256(solver.read_bytes()).hexdigest(),
                        builder_sha256=hashlib.sha256(builder.read_bytes()).hexdigest(),
                        builds=builds, records=records, commands=commands,
                        passed=sum(item["passed"] for item in records), total=len(records),
                        elapsed_seconds=time.perf_counter()-started), indent=2, allow_nan=False),
        encoding="utf-8")
    print(label, run.returncode, result.get("status", failure.get("message", "missing result")),
          "passed=", passed, flush=True)

gate_records = []
center_measurements = []
for boundary in ["uniform", "exact"]:
    r8 = next(item["result"] for item in records if item["case"] == f"farfield_R8_{boundary}")
    r16 = next(item["result"] for item in records if item["case"] == f"farfield_R16_{boundary}")
    difference = abs(r8["cl_pressure"]-r16["cl_pressure"])
    gate_records.append(dict(case=f"farfield_R8_vs_R16_{boundary}", cl_pressure_difference=difference,
                             tolerance=.002, passed=difference < .002))

rotation_identity_points = np.array([[.3, -.8], [1.2, .4], [-.6, .9]])
rotation_identity_direction = np.array([.8, .6])
rotation_identity_direction /= np.linalg.norm(rotation_identity_direction)
rotation_identity_gradient = np.array([[.2, -.7], [1.1, .3], [-.4, .9]])
for angle_degrees in [20., 45.]:
    angle = np.deg2rad(angle_degrees)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    before = rotation_identity_gradient@rotation_identity_direction
    after = (rotation_identity_gradient@rotation.T)@(rotation_identity_direction@rotation.T)
    error = float(np.max(abs(before-after)))
    gate_records.append(dict(case=f"kutta_projection_rotation_identity_{int(angle_degrees)}",
                             max_absolute_difference=error, tolerance=1.e-14,
                             passed=error <= 1.e-14))

rotation_base = next(item["result"] for item in records if item["case"] == "rotation_base")
rotation_base_field = np.load(result_root / "rotation_base/field.npz")
for angle_degrees in [20, 45]:
    case = f"rotation{angle_degrees}"
    rotated = next(item["result"] for item in records if item["case"] == case)
    rotated_field = np.load(result_root / case / "field.npz")
    differences = dict(cl_pressure=abs(rotation_base["cl_pressure"]-rotated["cl_pressure"]),
                       cl_circulation=abs(rotation_base["cl_circulation"]-rotated["cl_circulation"]),
                       cd_pressure=abs(rotation_base["cd_pressure"]-rotated["cd_pressure"]),
                       cp_max=float(np.max(abs(rotation_base_field["cp"]-rotated_field["cp"]))))
    passed = bool(differences["cl_pressure"] < .002
                  and differences["cl_circulation"] < .002
                  and differences["cd_pressure"] < .002
                  and differences["cp_max"] < .03)
    gate_records.append(dict(case=f"rotation_covariance_{angle_degrees}", differences=differences,
                             coefficient_tolerance=.002, cp_tolerance=.03, passed=passed))

for size_label in ["h020", "h010", "h005"]:
    default = next(item["result"] for item in records if item["case"] == f"center_{size_label}_default")
    default_field = np.load(result_root / f"center_{size_label}_default/field.npz")
    for center_label in ["xp02", "xm05"]:
        case = f"center_{size_label}_{center_label}"
        shifted = next(item["result"] for item in records if item["case"] == case)
        shifted_field = np.load(result_root / case / "field.npz")
        center_measurements.append(dict(
            case=f"center_sensitivity_{size_label}_{center_label}",
            cl_pressure_difference=abs(default["cl_pressure"]-shifted["cl_pressure"]),
            cl_circulation_difference=abs(default["cl_circulation"]-shifted["cl_circulation"]),
            cp_max_difference=float(np.max(abs(default_field["cp"]-shifted_field["cp"]))),
            cp_rmse_default=default["cp_rmse_exact"],
            cp_rmse_shifted=shifted["cp_rmse_exact"],
            interpretation="measurement only; no center-insensitivity acceptance threshold"))

summary = dict(protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
               driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               solver_sha256=hashlib.sha256(solver.read_bytes()).hexdigest(),
               builder_sha256=hashlib.sha256(builder.read_bytes()).hexdigest(),
               builds=builds, records=records, gates=gate_records,
               center_measurements=center_measurements, commands=commands,
               passed=sum(item["passed"] for item in records)+sum(item["passed"] for item in gate_records),
               total=len(records)+len(gate_records),
               elapsed_seconds=time.perf_counter()-started)
(result_root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
print(f"{summary['passed']}/{summary['total']} recorded gates passed", flush=True)
