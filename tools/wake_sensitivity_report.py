"""Collect numerical receipts; never promote an unfinished run to acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import jax
import numpy as np
import scipy
import torch
from threadpoolctl import threadpool_info

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
out = args.out
root = Path(__file__).resolve().parents[1]
cases = ("rect", "swept30", "elliptic")
manifest = dict(schema="dpanel.wake_sensitivity.manifest.v1", issue=113,
                head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                    capture_output=True, text=True).stdout.strip(),
                environment=dict(os=platform.platform(), cpu=platform.processor(), python=sys.version,
                                 numpy=np.__version__, scipy=scipy.__version__,
                                 jax=jax.__version__, torch=torch.__version__,
                                 collector_blas=threadpool_info()),
                physical_accuracy_accepted=False, cases={}, files={}, failures=[],
                material_passport=dict(origin_skill="experiment-agent", mode="run", date="2026-11-05",
                                       verification_status="UNVERIFIED", output_schema="exp_result_v1",
                                       scope="physical accuracy; local numerical executions have separate receipts"),
                source_provenance_note="Per-run source hashes identify actual instrumentation and audit revisions, including initially uncommitted scripts; head alone is not a claim of a clean experimental worktree.",
                source_sha256={p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted((root / "tools").glob("wake_*.py"))})
collections = {phase: dict(schema="dpanel.wake_sensitivity."+phase+".collection.v1",
                           cases={}, physical_accuracy_accepted=False) for phase in ("e1", "e2", "e3", "e4")}
for phase in ("e1", "e4"):
    collections[phase]["operator_definition"] = dict(
        unchanged="body A, J, signed-speed K, grad, wall RHS",
        moving="W; hence combined B=[A,W] and weighted reduced M=diag(sqrt(area/mean(area))) B null(J)",
        raw_receipt_note="legacy wall_operator_unchanged means body A only")
for case in cases:
    folder = out / case
    baseline = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    baseline_audit_path = folder / "baseline_geometry_audit.json"
    baseline_audit = (json.loads(baseline_audit_path.read_text(encoding="utf-8"))
                      if baseline_audit_path.exists() else None)
    manifest["cases"][case] = dict(
        baseline_manifest=case+"/manifest.json", frozen_initial_state=case+"/initial_state.npz",
        frozen_initial_state_sha256=baseline["initial_state_sha256"], spec=baseline["spec"],
        settings=baseline["settings"], command=baseline["command"], exit_code=baseline["exit_code"],
        environment=baseline["environment"], status=baseline["status"],
        pressure_initial_CL=baseline["rows"][0]["CL"],
        historical_pressure_initial_CL=baseline["historical_initial_CL"],
        initial_CL_difference=baseline["initial_CL_difference"],
        latest_CL=baseline["rows"][-1]["CL"], historical_final_CL=baseline["historical_final_CL"],
        final_CL_difference=baseline.get("final_CL_difference"),
        snapshots=[dict(path=case+"/"+r["snapshot"], sha256=r["snapshot_sha256"])
                   for r in baseline["rows"]],
        geometry_audit_complete=bool(baseline_audit and baseline_audit["baseline_status"] == "completed"
                                     and len(baseline_audit["rows"]) == len(baseline["rows"])))
    decomposition = folder / "first_step_decomposition.json"
    if decomposition.exists():
        manifest["cases"][case]["first_step_decomposition"] = json.loads(decomposition.read_text(encoding="utf-8"))
    for phase, filename in (("e1", "e1_position_sensitivity.json"), ("e4", "e4_conditioning.json")):
        path = folder / filename
        if not path.exists():
            collections[phase]["cases"][case] = dict(status="not_run")
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        item = dict(path=case+"/"+filename, status=result["status"], failures=result.get("failures"),
                    command=result["command"], seconds=result.get("seconds"), spacing=result["spacing"])
        if phase == "e1":
            audit_path = folder / "e1_geometry_audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else None
            item.update(regions=result["regions"], baseline=result["baseline"],
                        geometry_audit=case+"/e1_geometry_audit.json" if audit else None,
                        derivatives=audit["derivatives"] if audit else result["derivatives"],
                        derivative_acceptance="corrected audit" if audit else "RAW AUDIT; DO NOT ACCEPT")
            with np.load(folder/"initial_state.npz") as frozen:
                xyz = frozen["state_wake_xyz"]
            te = xyz[result["attachment_ids"]]
            te = te[np.argsort(te[:, 1])]
            dx = (xyz[:, 0]-np.interp(xyz[:, 1], te[:, 1], te[:, 0]))/baseline["spec"]["cref"]
            item["region_local_TE_dx_chord_min_median_max"] = {
                name: np.percentile(dx[ids], [0, 50, 100]).tolist()
                for name, ids in result["regions"].items() if ids}
        else:
            item["rows"] = result["rows"]
            audit_path = folder / "e1_geometry_audit.json"
            position_audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
            position_geometry = {r["name"]: r["geometry"] for r in position_audit.get("rows", [])}
            snapshot_geometry = {r["iteration"]: r["geometry"] for r in
                                 (baseline_audit or {}).get("rows", [])}
            for row in item["rows"]:
                if "wall_operator_unchanged" in row:
                    row["body_A_unchanged"] = row.pop("wall_operator_unchanged")
                    row["combined_wall_B_unchanged"] = False
                quality = position_geometry.get(row["name"])
                if row["name"] == "baseline":
                    quality = position_audit.get("initial_geometry")
                elif row["name"].startswith("baseline_iteration_"):
                    quality = snapshot_geometry.get(int(row["name"].split("_")[-1]))
                row["geometry"] = quality
                row["geometry_audit_source"] = "saved matching E1 perturbation or baseline snapshot; no extra aerodynamic solve"
                row["admissible"] = (None if quality is None else bool(
                    row.get("inner_converged", False) and row.get("finite_gamma", False)
                    and quality["min_area2"] > 0 and quality["min_reference_signed_jacobian"] > 0
                    and quality["new_or_deeper_violation_count"] == 0))
        collections[phase]["cases"][case] = item
        if result.get("failures"):
            manifest["failures"].append(dict(path=case+"/"+filename, failed_solves=result["failures"]))
    baseline_item = dict(
        variant="fully_free" if case != "rect" else "fully_free_24_faces",
        path=case+"/manifest.json", status=baseline["status"],
        iteration_limit=baseline["settings"]["iterations"],
        geometry_audit=case+"/baseline_geometry_audit.json",
        rows=[dict(iteration=r["iteration"], CL=r["CL"], CL_gamma=r["CL_gamma"], CM_y=r["CM_y"],
                   wall_rms=r["wall_rms"], jump_rms=r["jump_rms"],
                   pressure_kutta_rms=r["pressure_kutta_rms"], nonlinear=r["nonlinear"],
                   max_normal_slip=max(v["normal_slip_rms"] for v in r["wakes"].values()))
              for r in baseline["rows"]])
    for phase in ("e2", "e3"):
        items = [baseline_item]
        for path in sorted(folder.glob(phase+"_*/history.json")):
            if path.parent.name.endswith("_failure"):
                manifest["failures"].append(dict(path=path.relative_to(out).as_posix(),
                                                status="preserved_failed_attempt"))
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            item = {k: result.get(k) for k in (
                "variant", "status", "exit_code", "seconds", "iteration_limit", "description", "spacing",
                "converged", "geometry_stable", "all_states_admissible", "command",
                "attachment_ids", "prepared_root_ids", "row_ids", "constrained_ids", "error")}
            item["path"] = path.relative_to(out).as_posix()
            with np.load(path.parent/"initial_mesh.npz") as initial:
                xyz = initial["wake_xyz"]
            te = xyz[result["attachment_ids"]]
            te = te[np.argsort(te[:, 1])]
            dx = (xyz[:, 0]-np.interp(xyz[:, 1], te[:, 1], te[:, 0]))/baseline["spec"]["cref"]
            item["row_local_TE_dx_chord_min_median_max"] = {
                row: np.percentile(dx[ids], [0, 50, 100]).tolist()
                for row, ids in result["row_ids"].items() if ids}
            item["rows"] = []
            for r in result["rows"]:
                quality = r["geometry"]
                inner = bool(r["nonlinear"]["success"] and r["nonlinear"]["criteria"]["stationary"]
                             and r["jump_rms"] <= 1e-10 and r["pressure_kutta_rms"] <= 1e-4)
                item["rows"].append(dict(
                    iteration=r["iteration"], CL=r["CL"], CL_gamma=r["CL_gamma"], CM_y=r["CM_y"],
                    gamma_norm=float(np.linalg.norm(r["gamma"])), wall_rms=r["wall_rms"],
                    jump_rms=r["jump_rms"], pressure_kutta_rms=r["pressure_kutta_rms"],
                    production_inner_converged=inner, strict_inner_converged=r["inner_converged"],
                    wall_over_floor=r["nonlinear"]["wall_over_floor"],
                    max_normal_slip=max(v["normal_slip_rms"] for v in r["wakes"].values()),
                    raw_displacement_chord=r["raw_target_displacement_chord"],
                    accepted_displacement_chord=r["accepted_displacement_chord"],
                    min_reference_signed_jacobian=quality["min_reference_signed_jacobian"],
                    new_or_deeper_violation_count=quality["new_or_deeper_violation_count"],
                    attachment_max_displacement_chord=quality["attachment_max_displacement_chord"],
                    strict_admissible=r["admissible"]))
            items.append(item)
            if result["status"] == "failed":
                manifest["failures"].append(dict(path=item["path"], error=result.get("error")))
        collections[phase]["cases"][case] = items
    last = baseline["rows"][-1]
    print("%s baseline %s iter=%d CL %.9f -> %.9f" % (
        case, baseline["status"], last["iteration"], baseline["rows"][0]["CL"], last["CL"]))
    for phase in ("e2", "e3"):
        for result in collections[phase]["cases"][case][1:]:
            rows = result["rows"]
            if len(rows) > 1:
                print("  %s %-12s %s iter=%d CL1=%.6f last=%.6f slip=%.4g" % (
                    phase, result["variant"], result["status"], rows[-1]["iteration"],
                    rows[1]["CL"], rows[-1]["CL"], rows[-1]["max_normal_slip"]))
for phase, filename in (("e1", "e1_position_sensitivity.json"), ("e2", "e2_length_resolution.json"),
                         ("e3", "e3_root_rows.json"), ("e4", "e4_conditioning.json")):
    (out / filename).write_text(json.dumps(collections[phase], indent=2, allow_nan=False), encoding="utf-8")
for path in sorted(out.rglob("*")):
    if path.is_file() and path != out / "manifest.json" and path.suffix in (".json", ".log"):
        manifest["files"][path.relative_to(out).as_posix()] = dict(
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
queue_path = out / "queue.json"
queue = json.loads(queue_path.read_text(encoding="utf-8")) if queue_path.exists() else {}
manifest["queue"] = queue
manifest["process_failures"] = [r for r in queue.get("completed", [])
                                 if r.get("exit_code", 0) != 0]
kinematics = out / "uniform_axial_kinematics.json"
if kinematics.exists():
    manifest["uniform_axial_kinematics"] = json.loads(kinematics.read_text(encoding="utf-8"))
test_log = out / "full_pytest.log"
old_test_log = root / "runs/verification/native_blas/fixed_pytest.log"
if test_log.exists() and old_test_log.exists():
    old_failed = {line.split(" - ")[0] for line in old_test_log.read_text(encoding="utf-8").splitlines()
                  if line.startswith("FAILED ")}
    failed = {line.split(" - ")[0] for line in test_log.read_text(encoding="utf-8").splitlines()
              if line.startswith("FAILED ")}
    manifest["test_comparison"] = dict(
        reference="runs/verification/native_blas/fixed_pytest.log", current="full_pytest.log",
        reference_failed=sorted(old_failed), current_failed=sorted(failed),
        new_failed=sorted(failed-old_failed), resolved_failed=sorted(old_failed-failed))
expected_e2 = dict(rect={"rect_96"}, swept30={"length_5", "length_10", "length_20", "streamwise", "spanwise"},
                   elliptic={"length_5", "streamwise", "spanwise"})
missing = []
for case in cases:
    for phase, expected in (("e2", expected_e2[case]),
                            ("e3", {"freeze_1", "freeze_2", "freeze_3", "smooth"})):
        completed = {v["variant"] for v in collections[phase]["cases"][case]
                     if v["status"] == "completed"}
        missing.extend(case+"/"+phase+"_"+v for v in sorted(expected-completed))
    for phase in ("e1", "e4"):
        if collections[phase]["cases"][case]["status"] != "completed":
            missing.append(case+"/"+phase)
    if not manifest["cases"][case]["geometry_audit_complete"]:
        missing.append(case+"/baseline_geometry_audit")
    if not collections["e1"]["cases"][case].get("geometry_audit"):
        missing.append(case+"/e1_geometry_audit")
manifest["missing_completed_runs_or_audits"] = missing
manifest["execution_complete"] = bool(queue and not queue["pending"] and not queue["running"]
                                      and not queue["external_pids"]
                                      and not missing
                                      and all(v["status"] == "completed" for v in manifest["cases"].values()))
manifest["acceptance_note"] = (
    "Execution completion is not numerical or physical acceptance. Production nonlinear convergence and "
    "the stricter 1.05 wall-floor check are distinct. E2 is limited to eight updates. "
    "Unstructured projected-edge refinement cannot claim a pure tensor-grid one-axis refinement. "
    "Keep B3 and G2 T5 coupled-material evidence separate.")
(out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
print("execution_complete=%s preserved_failures=%d" % (manifest["execution_complete"], len(manifest["failures"])))
