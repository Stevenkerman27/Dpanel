"""Run the predeclared airfoil2d extra_v1 robustness matrix."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

root = Path(__file__).resolve().parent
mesh_root = root / "meshes/extra_v1"
result_root = root / "results/extra_v1"
builder = root / "build_mesh.py"
solver = root / "solve.py"
protocol = root / "EXTRA_PROTOCOL.md"
assert not mesh_root.exists(), "Preserve prior extra_v1 mesh evidence"
assert not result_root.exists(), "Preserve prior extra_v1 result evidence"
mesh_root.mkdir(parents=True)
result_root.mkdir(parents=True)
env8 = dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8")
builds = []
records = []
commands = []
started = time.perf_counter()

for label, epsilon, eta, radius, mirror in [
    ("eta04", .08, .04, 8., False),
    ("epsilon030", .30, 0., 8., True),
    ("radius04", .08, 0., 4., True),
    ("radius08", .08, 0., 8., True),
    ("radius16", .08, 0., 16., True),
]:
    target = mesh_root / label
    command = [sys.executable, str(builder), "--shape", "joukowski",
               "--size", ".02", "--radius", str(radius),
               "--epsilon", str(epsilon), "--eta", str(eta), "--out", str(target)]
    if mirror:
        command.append("--mirror")
    run_started = time.perf_counter()
    run = subprocess.run(command, env=env8, capture_output=True, text=True, timeout=300)
    target.mkdir(parents=True, exist_ok=True)
    (target / "build.log").write_text(run.stdout + run.stderr, encoding="utf-8")
    mesh_json = target / "mesh.json"
    mesh_npz = target / "mesh.npz"
    metadata = json.loads(mesh_json.read_text()) if mesh_json.exists() else {}
    builds.append(dict(label=label, command=command, returncode=run.returncode,
                       elapsed_seconds=time.perf_counter()-run_started,
                       mesh_sha256=hashlib.sha256(mesh_npz.read_bytes()).hexdigest() if mesh_npz.exists() else None,
                       metadata=metadata))
    (result_root / "summary.json").write_text(
        json.dumps(dict(protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
                        builds=builds, records=records, commands=commands), indent=2), encoding="utf-8")

naca_mesh = mesh_root / "naca_invalid_exact"
command = [sys.executable, str(builder), "--shape", "naca", "--size", ".02",
           "--radius", "8", "--thickness", ".12", "--mirror", "--out", str(naca_mesh)]
run_started = time.perf_counter()
run = subprocess.run(command, env=env8, capture_output=True, text=True, timeout=300)
naca_mesh.mkdir(parents=True, exist_ok=True)
(naca_mesh / "build.log").write_text(run.stdout + run.stderr, encoding="utf-8")
metadata_path = naca_mesh / "mesh.json"
mesh_path = naca_mesh / "mesh.npz"
builds.append(dict(label="naca_invalid_exact", command=command, returncode=run.returncode,
                   elapsed_seconds=time.perf_counter()-run_started,
                   mesh_sha256=hashlib.sha256(mesh_path.read_bytes()).hexdigest() if mesh_path.exists() else None,
                   metadata=json.loads(metadata_path.read_text()) if metadata_path.exists() else {}))

rotation_source = root / "meshes/general_v1/j02/mesh.npz"
rotation_metadata_source = root / "meshes/general_v1/j02/mesh.json"
rotation_dir = mesh_root / "rotated20"
rotation_dir.mkdir()
source_mesh = np.load(rotation_source)
rotated_mesh = {key: source_mesh[key].copy() for key in source_mesh.files}
source_metadata = json.loads(rotation_metadata_source.read_text())
rotation_origin = np.asarray(source_metadata["normalization_origin"])
angle = np.deg2rad(20.)
rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
rotated_mesh["points"] = (rotated_mesh["points"]-rotation_origin) @ rotation.T + rotation_origin
rotated_mesh["contour"] = (rotated_mesh["contour"]-rotation_origin) @ rotation.T + rotation_origin
np.savez_compressed(rotation_dir / "mesh.npz", **rotated_mesh)
rotated_metadata = dict(source_metadata)
rotated_metadata.update(shape="rotated_file", rotation_degrees=20.,
                        rotation_source_mesh_sha256=hashlib.sha256(rotation_source.read_bytes()).hexdigest(),
                        mesh_sha256=hashlib.sha256((rotation_dir / "mesh.npz").read_bytes()).hexdigest(),
                        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(rotation_dir / "mesh.json").write_text(json.dumps(rotated_metadata, indent=2), encoding="utf-8")
np.savetxt(rotation_dir / "contour.dat", rotated_mesh["contour"])

mesh_paths = {
    "eta04": mesh_root / "eta04/mesh.npz",
    "epsilon030": mesh_root / "epsilon030/mesh.npz",
    "radius04": mesh_root / "radius04/mesh.npz",
    "radius08": mesh_root / "radius08/mesh.npz",
    "radius16": mesh_root / "radius16/mesh.npz",
    "naca_invalid_exact": naca_mesh / "mesh.npz",
    "rotated20": rotation_dir / "mesh.npz",
}
case_specs = [
    ("cambered_exact", "eta04", ["--alpha", "5", "--mach", "0", "--boundary", "exact"]),
    ("thick_exact", "epsilon030", ["--alpha", "5", "--mach", "0", "--boundary", "exact"]),
    ("farfield_R4", "radius04", ["--alpha", "5", "--mach", "0", "--boundary", "uniform"]),
    ("farfield_R8", "radius08", ["--alpha", "5", "--mach", "0", "--boundary", "uniform"]),
    ("farfield_R16", "radius16", ["--alpha", "5", "--mach", "0", "--boundary", "uniform"]),
    ("cpu1_cambered", "eta04", ["--alpha", "5", "--mach", "0", "--boundary", "exact", "--ncpu", "1"]),
    ("cpu8_cambered", "eta04", ["--alpha", "5", "--mach", "0", "--boundary", "exact", "--ncpu", "8"]),
    ("basis_default", "radius08", ["--alpha", "5", "--mach", "0", "--boundary", "exact"]),
    ("basis_xp02", "radius08", ["--alpha", "5", "--mach", "0", "--boundary", "exact",
                                "--center", ".2", "0"]),
    ("basis_xm05", "radius08", ["--alpha", "5", "--mach", "0", "--boundary", "exact",
                                "--center", "-.5", "0"]),
    ("rotation_base", "radius08", ["--alpha", "5", "--mach", "0", "--boundary", "uniform"]),
    ("rotation_20deg", "rotated20", ["--alpha", "25", "--mach", "0", "--boundary", "uniform"]),
    ("invalid_mach", "radius08", ["--alpha", "5", "--mach", "1", "--boundary", "uniform"]),
    ("invalid_quadrature", "radius08", ["--alpha", "5", "--mach", "0", "--boundary", "uniform",
                                        "--quadrature", "2"]),
    ("outside_center", "radius08", ["--alpha", "5", "--mach", "0", "--boundary", "exact",
                                    "--center", "10", "10"]),
    ("nonfinite_center", "radius08", ["--alpha", "5", "--mach", "0", "--boundary", "exact",
                                      "--center", "nan", "0"]),
    ("unsupported_exact_geometry", "naca_invalid_exact",
     ["--alpha", "5", "--mach", "0", "--boundary", "exact"]),
]
for label, mesh_label, options in case_specs:
    target = result_root / label
    command = [sys.executable, str(solver), "--mesh", str(mesh_paths[mesh_label]),
               *options, "--out", str(target)]
    ncpu = 1 if "--ncpu" in options and options[options.index("--ncpu")+1] == "1" else 8
    env = dict(os.environ, OMP_NUM_THREADS=str(ncpu),
               OPENBLAS_NUM_THREADS=str(ncpu), MKL_NUM_THREADS=str(ncpu))
    run_started = time.perf_counter()
    run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=300)
    target.mkdir(parents=True, exist_ok=True)
    (target / "run.log").write_text(run.stdout + run.stderr, encoding="utf-8")
    result_path = target / "result.json"
    failure_path = target / "failure.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    failure = json.loads(failure_path.read_text()) if failure_path.exists() else {}
    row = dict(case=label, mesh=mesh_label, command=command, returncode=run.returncode,
               elapsed_seconds=time.perf_counter()-run_started,
               mesh_sha256=hashlib.sha256(mesh_paths[mesh_label].read_bytes()).hexdigest(),
               solver_sha256=hashlib.sha256(solver.read_bytes()).hexdigest(),
               result=result, failure=failure, passed=False)
    if label in ["cambered_exact", "thick_exact"]:
        expected_rejection = False
        exact_lift_error_pressure = abs(result.get("cl_pressure", np.nan)-result.get("cl_exact", np.nan))
        exact_lift_error_circulation = abs(result.get("cl_circulation", np.nan)-result.get("cl_exact", np.nan))
        row.update(expected_rejection=expected_rejection,
                   cl_pressure_relative_error=exact_lift_error_pressure/abs(result.get("cl_exact", np.nan)),
                   cl_circulation_relative_error=exact_lift_error_circulation/abs(result.get("cl_exact", np.nan)))
        row["passed"] = bool(run.returncode == 0 and result.get("converged", False)
                             and row["cl_pressure_relative_error"] < .01
                             and row["cl_circulation_relative_error"] < .01
                             and result.get("cp_rmse_exact", np.inf) < .03
                             and abs(result.get("cd_pressure", np.inf)) < .005)
    elif label.startswith("farfield_"):
        row["passed"] = bool(run.returncode == 0 and result.get("converged", False)
                             and np.isfinite(result.get("cl_pressure", np.nan))
                             and np.isfinite(result.get("cd_pressure", np.nan))
                             and np.isfinite(result.get("cp_rmse_exact", np.nan)))
    elif label in ["cpu1_cambered", "cpu8_cambered", "basis_default", "basis_xp02", "basis_xm05",
                   "rotation_base", "rotation_20deg"]:
        row["passed"] = bool(run.returncode == 0 and result.get("converged", False))
    else:
        if label == "outside_center":
            failure_classified = "inside the body" in failure.get("message", "")
        elif label == "nonfinite_center":
            failure_classified = "finite" in failure.get("message", "").lower()
        elif label in ["invalid_mach", "invalid_quadrature"]:
            failure_classified = (failure.get("exception_type") == "AssertionError"
                                  and "line 44" in run.stdout + run.stderr)
        elif label == "unsupported_exact_geometry":
            failure_classified = (failure.get("exception_type") == "AssertionError"
                                  and "line 91" in run.stdout + run.stderr)
        else:
            failure_classified = False
        row.update(expected_rejection=True,
                   failure_classified=failure_classified,
                   rejected=bool(run.returncode != 0 and not result.get("converged", False)
                                 and not failure.get("converged", False)
                                 and failure_classified))
        row["passed"] = row["rejected"]
    records.append(row)
    commands.append(dict(case=label, command=command, returncode=run.returncode))
    (result_root / "summary.json").write_text(
        json.dumps(dict(protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
                        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        solver_sha256=hashlib.sha256(solver.read_bytes()).hexdigest(),
                        builder_sha256=hashlib.sha256(builder.read_bytes()).hexdigest(),
                        builds=builds, records=records, commands=commands,
                        passed=sum(record["passed"] for record in records), total=len(records),
                        elapsed_seconds=time.perf_counter()-started), indent=2, allow_nan=False),
        encoding="utf-8")
    print(label, run.returncode, result.get("status", failure.get("message", "missing result")),
          "passed=", row["passed"], flush=True)

far4 = next(row["result"] for row in records if row["case"] == "farfield_R4")
far8 = next(row["result"] for row in records if row["case"] == "farfield_R8")
far16 = next(row["result"] for row in records if row["case"] == "farfield_R16")
farfield_cl_difference = abs(far8["cl_pressure"]-far16["cl_pressure"])
records.append(dict(case="farfield_R8_vs_R16_gate", cl_absolute_difference=farfield_cl_difference,
                    tolerance=.002, passed=farfield_cl_difference < .002))

cpu1 = next(row["result"] for row in records if row["case"] == "cpu1_cambered")
cpu8 = next(row["result"] for row in records if row["case"] == "cpu8_cambered")
cpu1_field = np.load(result_root / "cpu1_cambered/field.npz")
cpu8_field = np.load(result_root / "cpu8_cambered/field.npz")
cpu_differences = dict(cl_pressure=abs(cpu1["cl_pressure"]-cpu8["cl_pressure"]),
                       cl_circulation=abs(cpu1["cl_circulation"]-cpu8["cl_circulation"]),
                       cd_pressure=abs(cpu1["cd_pressure"]-cpu8["cd_pressure"]),
                       cp_max=float(np.max(abs(cpu1_field["cp"]-cpu8_field["cp"]))))
records.append(dict(case="ncpu1_vs_8_gate", differences=cpu_differences, tolerance=1.e-10,
                    same_mesh_sha256=next(row["mesh_sha256"] for row in records
                                          if row["case"] == "cpu1_cambered")
                    == next(row["mesh_sha256"] for row in records if row["case"] == "cpu8_cambered"),
                    passed=all(value <= 1.e-10 for value in cpu_differences.values())))

basis_default = next(row["result"] for row in records if row["case"] == "basis_default")
default_field = np.load(result_root / "basis_default/field.npz")
basis_differences = []
for label in ["basis_xp02", "basis_xm05"]:
    row = next(item for item in records if item["case"] == label)
    basis_result = row["result"]
    basis_field = np.load(result_root / label / "field.npz")
    difference = dict(case=label,
                      cl_pressure=abs(basis_result["cl_pressure"]-basis_default["cl_pressure"]),
                      cp_max=float(np.max(abs(basis_field["cp"]-default_field["cp"]))))
    difference["passed"] = bool(difference["cl_pressure"] < .005 and difference["cp_max"] < .05)
    basis_differences.append(difference)
records.append(dict(case="basis_center_sensitivity_gate", comparisons=basis_differences,
                    passed=all(row["passed"] for row in basis_differences)))

rotation_base = next(row["result"] for row in records if row["case"] == "rotation_base")
rotation_result = next(row["result"] for row in records if row["case"] == "rotation_20deg")
rotation_base_field = np.load(result_root / "rotation_base/field.npz")
rotation_field = np.load(result_root / "rotation_20deg/field.npz")
rotation_differences = dict(cl_pressure=abs(rotation_base["cl_pressure"]-rotation_result["cl_pressure"]),
                            cl_circulation=abs(rotation_base["cl_circulation"]-rotation_result["cl_circulation"]),
                            cd_pressure=abs(rotation_base["cd_pressure"]-rotation_result["cd_pressure"]),
                            cp_max=float(np.max(abs(rotation_base_field["cp"]-rotation_field["cp"]))))
records.append(dict(case="rotation_covariance_gate", differences=rotation_differences,
                    coefficient_tolerance=.002, cp_tolerance=.03,
                    passed=bool(rotation_differences["cl_pressure"] < .002
                                and rotation_differences["cl_circulation"] < .002
                                and rotation_differences["cd_pressure"] < .002
                                and rotation_differences["cp_max"] < .03)))

summary = dict(protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
               driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               solver_sha256=hashlib.sha256(solver.read_bytes()).hexdigest(),
               builder_sha256=hashlib.sha256(builder.read_bytes()).hexdigest(),
               builds=builds, records=records, commands=commands,
               passed=sum(row["passed"] for row in records), total=len(records),
               elapsed_seconds=time.perf_counter()-started)
(result_root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
print(f"{summary['passed']}/{summary['total']} extra_v1 gates passed", flush=True)
