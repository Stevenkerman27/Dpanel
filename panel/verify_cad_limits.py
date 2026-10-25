"""Probe CAD contact transitions as an ellipsoid aft tip crosses the TE."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

root = Path(__file__).resolve().parent
project = root.parent
runtime = project / "audit" / "runtime" / "python"
sys.path.insert(0, str(runtime))
import gmsh

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="runs/panel_cad_limits_v1")
parser.add_argument("--radius", type=float, default=5.)
parser.add_argument("--size", default=".3")
parser.add_argument("--far-size", default="1.5")
parser.add_argument("--ncpu", type=int, default=8)
args = parser.parse_args()

out = project / args.out
assert not out.exists(), "Preserve previous CAD-limit attempts"
out.mkdir(parents=True)
source_case = project / "lifting3d" / "results" / "cad_inputs_v1" / "naca0012_wing"
wing_source = source_case / "wing.step"
te_source = source_case / "trailing_edge.json"
assert wing_source.is_file() and te_source.is_file()

case_specs = [
    {"name": "aft_1_minus_1e-4", "aft_x": 1. - 1e-4},
    {"name": "aft_1_plus_1e-4", "aft_x": 1. + 1e-4},
    {"name": "aft_1_plus_1e-8", "aft_x": 1. + 1e-8},
    {"name": "aft_exact_1", "aft_x": 1.},
]
semiaxes = [2., .35, .35]
fixture_started = time.perf_counter()
gmsh.initialize()
gmsh.logger.start()
gmsh.option.setNumber("General.Verbosity", 0)
gmsh.option.setNumber("General.NumThreads", 8)

for case in case_specs:
    case["center"] = [case["aft_x"] - semiaxes[0], 0., 0.]
    case_dir = out / case["name"]
    input_dir = case_dir / "input"
    input_dir.mkdir(parents=True)
    shutil.copy2(wing_source, input_dir / "wing.step")
    shutil.copy2(te_source, input_dir / "trailing_edge.json")
    case["input_dir"] = input_dir

    gmsh.clear()
    gmsh.model.add(case["name"])
    occ = gmsh.model.occ
    center_x, center_y, center_z = case["center"]
    body = occ.addSphere(center_x, center_y, center_z, 1.)
    occ.dilate([(3, body)], center_x, center_y, center_z, *semiaxes)
    body_copy = occ.copy([(3, body)])[0][1]
    left_box = occ.addBox(
        center_x - 10., center_y - 10., center_z - 10.,
        10., 20., 20.)
    right_box = occ.addBox(
        center_x, center_y - 10., center_z - 10.,
        10., 20., 20.)
    left_halves, _ = occ.intersect(
        [(3, body)], [(3, left_box)], removeObject=True, removeTool=True)
    right_halves, _ = occ.intersect(
        [(3, body_copy)], [(3, right_box)], removeObject=True, removeTool=True)
    body_solids = [
        entity for entity in left_halves + right_halves if entity[0] == 3
    ]
    occ.synchronize()
    assert len(body_solids) == 2
    body_path = input_dir / "body.step"
    gmsh.write(str(body_path))
    case["body_path"] = body_path
    case["fixture"] = {
        "aft_x": case["aft_x"],
        "center": case["center"],
        "semiaxes": semiaxes,
        "closed_half_solid_count": len(body_solids),
        "body_sha256": hashlib.sha256(body_path.read_bytes()).hexdigest(),
    }

fixture_gmsh_log = gmsh.logger.get()
gmsh.logger.stop()
gmsh.finalize()
(out / "fixture_generation.gmsh.log").write_text(
    "\n".join(fixture_gmsh_log), encoding="utf-8")

results = []
for case in case_specs:
    case_dir = out / case["name"]
    cad_out = case_dir / "cad"
    command = [
        sys.executable, str(root / "cad.py"),
        "--wing-cad", str(case["input_dir"] / "wing.step"),
        "--body-cad", str(case["body_path"]),
        "--trailing-edge", str(case["input_dir"] / "trailing_edge.json"),
        "--size", args.size,
        "--far-size", args.far_size,
        "--radius", repr(args.radius),
        "--wake-strips", "8",
        "--ncpu", str(args.ncpu),
        "--out", str(cad_out),
    ]
    case_started = time.perf_counter()
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    (case_dir / "cad.stdout.log").write_text(process.stdout, encoding="utf-8")
    (case_dir / "cad.stderr.log").write_text(process.stderr, encoding="utf-8")
    failure_path = cad_out / "failure.json"
    metadata_path = cad_out / "mesh.json"
    surface_path = cad_out / "surface.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8")) if failure_path.exists() else None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
    surface = json.loads(surface_path.read_text(encoding="utf-8")) if surface_path.exists() else None
    intervals = metadata.get("wall_contact_intervals", []) if metadata else []
    contact_length = float(sum(upper - lower for lower, upper in intervals))
    result = {
        "name": case["name"],
        "aft_x": case["aft_x"],
        "delta_from_te": case["aft_x"] - 1.,
        "center": case["center"],
        "semiaxes": semiaxes,
        "body_sha256": case["fixture"]["body_sha256"],
        "command": command,
        "returncode": process.returncode,
        "status": "passed" if process.returncode == 0 and surface else "failed",
        "failure_stage": failure["stage"] if failure else None,
        "failure_type": failure["exception_type"] if failure else None,
        "failure_message": failure["message"] if failure else None,
        "wall_contact_intervals": intervals,
        "wall_contact_total_y_length": contact_length,
        "wall_panels": metadata["wall_panels"] if metadata else None,
        "wake_panels": metadata["wake_panels"] if metadata else None,
        "minimum_triangle_quality": (
            surface["audit"]["minimum_triangle_quality"] if surface else None
        ),
        "effective_surface_json": str(surface_path.resolve()) if surface else None,
        "cad_elapsed_seconds": metadata["elapsed_seconds"] if metadata else None,
        "case_elapsed_seconds": time.perf_counter() - case_started,
        "failure_json": str(failure_path.resolve()) if failure else None,
        "stdout_log": str((case_dir / "cad.stdout.log").resolve()),
        "stderr_log": str((case_dir / "cad.stderr.log").resolve()),
        "gmsh_log": str((cad_out / "gmsh.log").resolve()) if (cad_out / "gmsh.log").exists() else None,
        "fixture": case["fixture"],
    }
    (case_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    results.append(result)

manifest = {
    "status": "completed",
    "wing_source": str(wing_source.resolve()),
    "trailing_edge_source": str(te_source.resolve()),
    "source_hashes": {
        "wing_step": hashlib.sha256(wing_source.read_bytes()).hexdigest(),
        "trailing_edge": hashlib.sha256(te_source.read_bytes()).hexdigest(),
        "cad_py": hashlib.sha256((root / "cad.py").read_bytes()).hexdigest(),
        "verify_cad_limits_py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    },
    "body_semiaxes": semiaxes,
    "farfield_radius": args.radius,
    "cad_tolerance": 1e-7,
    "geometric_offsets_are_not_welded": True,
    "threads": 8,
    "fixture_generation_elapsed_seconds": time.perf_counter() - fixture_started,
    "results": results,
}
(out / "summary.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps({
    "output": str(out.resolve()),
    "results": [
        {
            "name": result["name"],
            "aft_x": result["aft_x"],
            "status": result["status"],
            "failure_stage": result["failure_stage"],
            "wall_contact_total_y_length": result["wall_contact_total_y_length"],
            "wall_panels": result["wall_panels"],
            "minimum_triangle_quality": result["minimum_triangle_quality"],
            "effective_surface_json": result["effective_surface_json"],
        }
        for result in results
    ],
}, indent=2), flush=True)
