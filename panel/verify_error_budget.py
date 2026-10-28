"""Separate geometry, discretization, potential-reconstruction and TE error budgets."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
project = root.parent
runtime = project / "audit" / "runtime" / "python"
sys.path.insert(0, str(runtime))
import gmsh

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="runs/panel_error_budget_v1")
parser.add_argument("--radius", type=float, default=8.)
parser.add_argument("--ncpu", type=int, default=8)
parser.add_argument("--audit-python", type=Path)
parser.add_argument("--aft", type=float, nargs="+", default=[0.9999, 1.0, 1.0001])
parser.add_argument("--sizes", type=float, nargs="+", default=[.3, .2, .15])
parser.add_argument("--summary-only", action="store_true")
args = parser.parse_args()

out = project / args.out
if args.summary_only:
    assert out.exists(), "--summary-only needs an existing output directory"
else:
    assert not out.exists(), "Preserve previous error-budget attempts"
    out.mkdir(parents=True)
source_case = project / "lifting3d" / "results" / "cad_inputs_v1" / "naca0012_wing"
wing_source = source_case / "wing.step"
te_source = source_case / "trailing_edge.json"
semiaxes = [2., .35, .35]


def make_body(name, aft_x, input_dir):
    center = [aft_x - semiaxes[0], 0., 0.]
    gmsh.clear()
    gmsh.model.add(name)
    occ = gmsh.model.occ
    body = occ.addSphere(center[0], center[1], center[2], 1.)
    occ.dilate([(3, body)], center[0], center[1], center[2], *semiaxes)
    body_copy = occ.copy([(3, body)])[0][1]
    left_box = occ.addBox(center[0] - 10., center[1] - 10., center[2] - 10., 10., 20., 20.)
    right_box = occ.addBox(center[0], center[1] - 10., center[2] - 10., 10., 20., 20.)
    left_halves, _ = occ.intersect([(3, body)], [(3, left_box)], removeObject=True, removeTool=True)
    right_halves, _ = occ.intersect([(3, body_copy)], [(3, right_box)], removeObject=True, removeTool=True)
    solids = [entity for entity in left_halves + right_halves if entity[0] == 3]
    occ.synchronize()
    assert len(solids) == 2
    path = input_dir / "body.step"
    gmsh.write(str(path))
    return center, path


gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 0)
gmsh.option.setNumber("General.NumThreads", args.ncpu)

cases = []
contact_aft = max(args.aft)
for aft in args.aft:
    sizes = args.sizes if abs(aft - contact_aft) < 1e-15 else args.sizes[:1]
    for size in sizes:
        name = "aft_" + repr(aft) + "_size_" + repr(size)
        case_dir = out / name
        input_dir = case_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wing_source, input_dir / "wing.step")
        shutil.copy2(te_source, input_dir / "trailing_edge.json")
        center, body_path = make_body(name, aft, input_dir)
        cases.append(dict(name=name, aft_x=aft, size=size, center=center,
                          body_path=str(body_path), input_dir=str(input_dir),
                          body_sha256=hashlib.sha256(body_path.read_bytes()).hexdigest()))
gmsh.finalize()
manifest = dict(radius=args.radius, semiaxes=semiaxes, cases=cases,
                wing_sha256=hashlib.sha256(wing_source.read_bytes()).hexdigest(),
                te_sha256=hashlib.sha256(te_source.read_bytes()).hexdigest(),
                verify_error_budget_py=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

env = dict(os.environ)
work = project / "runs" / "pipwork"
work.mkdir(parents=True, exist_ok=True)
env["TMP"] = env["TEMP"] = env["TMPDIR"] = str(work)


def run(command, log_path):
    started = time.perf_counter()
    process = subprocess.run([str(c) for c in command], capture_output=True, text=True, cwd=project, env=env)
    Path(log_path).write_text(process.stdout + process.stderr, encoding="utf-8")
    return process.returncode, time.perf_counter() - started


def metrics(case_dir, solve_dir):
    result = json.loads((solve_dir / "result.json").read_text(encoding="utf-8"))
    surface = json.loads((solve_dir / "surface_input.json").read_text(encoding="utf-8"))
    data = np.load(solve_dir / "fields.npz")
    reference_area = float(surface["reference_area"])
    normals = data["panel_normal"]
    area = data["panel_area"]
    coefficient = -area * normals[:, 2] / reference_area
    cp = data["Cp_raw"]
    record = dict(name=case_dir.name, aft_x=float(case_dir.name.split("_")[1]),
                  size=float(case_dir.name.split("_")[3]),
                  CL=float(result["CL"]), CM_y=float(result["CM"][1]),
                  CD_pressure=float(result["CD_pressure"]), cp_min=float(result["cp_min"]),
                  cp_max=float(result["cp_max"]),
                  condition_1_estimate=float(result["condition_1_estimate"]),
                  linear_relative_residual=float(result["linear_true_relative_residual"]),
                  wall_panels=int(result["mesh"]["wall_panels"]),
                  wake_panels=int(result["mesh"]["wake_panels"]),
                  gamma_unknowns=int(result["mesh"]["gamma_unknowns"]),
                  minimum_triangle_quality=float(result["mesh"]["minimum_triangle_quality"]),
                  wall_contact_total=float(sum(u - l for l, u in result["mesh"].get("wall_contact_intervals", []))),
                  warm_seconds=float(result["timings"]["warm_forward_seconds"]))
    field_path = solve_dir / "independent_field.npz"
    if field_path.exists():
        field = np.load(field_path)
        delta_cp = cp - field["cp"]
        record["reconstruction_delta_cp_rms"] = float(np.sqrt(np.average(delta_cp ** 2, weights=area)))
        record["reconstruction_CL_error"] = float(abs(np.sum(delta_cp * coefficient)))
        record["reconstruction_normal_velocity_rms"] = float(np.sqrt(np.average(field["normal_error"] ** 2, weights=area)))
    audit_path = solve_dir / "independent_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        independent = audit["independent_field"]
        record["audit_wall_normal_rms"] = independent["wall_normal_rms"]
        record["audit_wall_normal_max"] = independent["wall_normal_max"]
        record["audit_reconstruction_cp_rms"] = independent["reconstructed_vs_field_cp_rms"]
        record["audit_exposed_te_jump_rms"] = independent["exposed_te_jump_rms"]
        record["audit_contact_jump_rms"] = independent.get("contact_jump_rms")
        record["audit_field_cp_min"] = independent["independent_field_cp_min"]
        record["audit_field_cp_max"] = independent["independent_field_cp_max"]
        record["audit_seconds"] = audit["elapsed_seconds"]
    stations = np.asarray(surface["gamma_stations"], dtype=float)
    dof = np.asarray(surface["gamma_station_dof"], dtype=np.int64)
    knot = np.zeros(len(stations))
    knot[dof >= 0] = data["gamma"][dof[dof >= 0]]
    order = np.argsort(stations)
    lift = np.sum(0.5 * (knot[order][1:] + knot[order][:-1]) * np.diff(stations[order]))
    record["CL_circulation"] = float(2. * lift / reference_area)
    record["CL_cp_minus_circulation"] = float(record["CL"] - record["CL_circulation"])
    upper = np.asarray(surface["upper_te_faces"], dtype=np.int64)
    lower = np.asarray(surface["lower_te_faces"], dtype=np.int64)
    jump = np.abs(cp[upper] - cp[lower])
    record["te_dcp_mean"] = float(jump.mean())
    record["te_dcp_max"] = float(jump.max())
    record["te_CL_error"] = float(np.sum(jump * 0.5 * (abs(coefficient[upper]) + abs(coefficient[lower]))))
    record["surface_sha256"] = hashlib.sha256((solve_dir / "surface_input.json").read_bytes()).hexdigest()
    return record


records = []
if args.summary_only:
    for path in sorted(out.glob("*/record.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["aft_x"] = float(record["aft_x"])
        record["size"] = float(record["size"])
        records.append(record)
for case in ([] if args.summary_only else cases):
    case_dir = out / case["name"]
    cad_out = case_dir / "cad"
    solve_dir = case_dir / "solve"
    cad_command = [sys.executable, root / "cad.py",
                   "--wing-cad", Path(case["input_dir"]) / "wing.step",
                   "--body-cad", case["body_path"],
                   "--trailing-edge", Path(case["input_dir"]) / "trailing_edge.json",
                   "--size", repr(case["size"]), "--far-size", "1.5",
                   "--radius", repr(args.radius), "--wake-strips", "8",
                   "--ncpu", str(args.ncpu), "--out", cad_out]
    cad_code, cad_seconds = run(cad_command, case_dir / "cad.log")
    record = dict(name=case["name"], aft_x=case["aft_x"], size=case["size"],
                  cad_returncode=cad_code, cad_seconds=cad_seconds)
    surface_path = cad_out / "surface.json"
    if cad_code != 0 or not surface_path.exists():
        failure = cad_out / "failure.json"
        record["status"] = "failed"
        record["failure"] = json.loads(failure.read_text(encoding="utf-8")) if failure.exists() else None
        records.append(record)
        (case_dir / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        continue
    solve_command = [sys.executable, root / "solve.py", "--surface", surface_path,
                     "--out", solve_dir, "--ncpu", str(args.ncpu)]
    solve_code, solve_seconds = run(solve_command, case_dir / "solve.log")
    record.update(status="solved" if solve_code == 0 else "solve_failed",
                  solve_returncode=solve_code, solve_seconds=solve_seconds)
    if solve_code == 0:
        record.update(metrics(case_dir, solve_dir))
        if args.audit_python:
            audit_code, audit_seconds = run([args.audit_python, root / "audit.py", solve_dir,
                                             "--ncpu", str(args.ncpu), "--off-surface"],
                                            case_dir / "audit.log")
            record["audit_returncode"] = audit_code
            if audit_code == 0:
                refreshed = metrics(case_dir, solve_dir)
                for key, value in refreshed.items():
                    if key.startswith("audit_") or key in ("reconstruction_delta_cp_rms",
                                                           "reconstruction_CL_error",
                                                           "reconstruction_normal_velocity_rms"):
                        record[key] = value
    records.append(record)
    (case_dir / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({k: record.get(k) for k in ("name", "status", "CL", "cp_min", "wall_panels",
                                                 "gamma_unknowns", "audit_wall_normal_rms")}), flush=True)

solved = [r for r in records if r.get("status") == "solved"]
summary = dict(records=records)
if solved:
    sizes_by_aft = {}
    for r in solved:
        sizes_by_aft.setdefault(r["aft_x"], set()).add(r["size"])
    common = set.intersection(*sizes_by_aft.values())
    coarsest = max(common)
    at_coarse = sorted([r for r in solved if r["size"] == coarsest], key=lambda r: r["aft_x"])
    geometry = {repr(r["aft_x"]): dict(CL=r["CL"], CL_circulation=r["CL_circulation"],
                                       cp_min=r["cp_min"], wall_panels=r["wall_panels"],
                                       gamma_unknowns=r["gamma_unknowns"],
                                       wall_contact_total=r["wall_contact_total"])
                for r in at_coarse}
    refinement = []
    contact_aft = max(r["aft_x"] for r in solved)
    contact = sorted([r for r in solved if abs(r["aft_x"] - contact_aft) < 1e-15],
                     key=lambda r: -r["size"])
    for coarse, fine in zip(contact, contact[1:]):
        refinement.append(dict(coarse=coarse["size"], fine=fine["size"],
                               CL_coarse=coarse["CL"], CL_fine=fine["CL"],
                               CL_change=fine["CL"] - coarse["CL"],
                               CL_relative_change=(fine["CL"] - coarse["CL"]) / coarse["CL"],
                               circulation_coarse=coarse["CL_circulation"],
                               circulation_fine=fine["CL_circulation"],
                               circulation_change=fine["CL_circulation"] - coarse["CL_circulation"],
                               wall_normal_rms=[coarse.get("audit_wall_normal_rms"),
                                                fine.get("audit_wall_normal_rms")],
                               te_CL=[coarse["te_CL_error"], fine["te_CL_error"]],
                               wall_panels=[coarse["wall_panels"], fine["wall_panels"]]))
    base = contact[0] if contact else at_coarse[-1]
    cl_values = [row["CL"] for row in geometry.values()]
    circulation_values = [row["CL_circulation"] for row in geometry.values()]
    budget = dict(
        geometry_CL_range=float(max(cl_values) - min(cl_values)),
        geometry_CL_relative_range=float((max(cl_values) - min(cl_values)) / np.mean(cl_values)),
        geometry_circulation_range=float(max(circulation_values) - min(circulation_values)),
        geometry_circulation_relative_range=float((max(circulation_values) - min(circulation_values))
                                                  / np.mean(circulation_values)),
        discretization_CL_change=refinement[-1]["CL_change"] if refinement else None,
        discretization_CL_relative_change=refinement[-1]["CL_relative_change"] if refinement else None,
        discretization_circulation_change=refinement[-1]["circulation_change"] if refinement else None,
        reconstruction_CL_gap=base["CL_cp_minus_circulation"],
        reconstruction_CL_gap_relative=base["CL_cp_minus_circulation"] / base["CL_circulation"],
        te_CL_error=base["te_CL_error"],
        te_CL_relative=base["te_CL_error"] / base["CL"],
        te_dcp_mean=base["te_dcp_mean"],
        linear_relative_residual=base["linear_relative_residual"],
        audit_wall_normal_rms=base.get("audit_wall_normal_rms"),
        audit_field_cp_min=base.get("audit_field_cp_min"),
        pointwise_cp_caveat=("independent centroid Cp is spike-prone on coarse meshes; "
                             "compare integrals, not pointwise Cp"),
    )
    summary.update(geometry=geometry, refinement=refinement, budget=budget)

(out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary.get("budget"), indent=2), flush=True)
