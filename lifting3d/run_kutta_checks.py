"""Fixed-mesh node/weak Kutta comparison; no geometry generation."""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

root = Path(__file__).resolve().parent
result_root = root/"results/kutta_checks_v1"
mesh_path = root/"results/strips_sequence_v1/s24/mesh/mesh.npz"
mesh_metadata_path = mesh_path.with_name("mesh.json")
solver_path = root/"solve.py"
p2_path = root.parent/"volume/p2.py"
solver_protocol_path = root/"PROTOCOL.md"
parser = argparse.ArgumentParser()
parser.add_argument("--reuse-existing", action="store_true")
args = parser.parse_args()
if args.reuse_existing:
    assert result_root.exists(), "No completed kutta_checks_v1 results to reuse"
    prior_summary_bytes = (result_root/"summary.json").read_bytes()
    prior_summary = json.loads(prior_summary_bytes)
else:
    assert not result_root.exists(), "Preserve prior kutta_checks_v1 evidence"
    prior_summary = None
    prior_summary_bytes = None
assert mesh_path.exists(), "Fixed s24 mesh is required"
if not args.reuse_existing:
    result_root.mkdir(parents=True)

if args.reuse_existing:
    solver_bytes = (result_root/"solve.py.snapshot").read_bytes()
    p2_bytes = (result_root/"p2.py.snapshot").read_bytes()
    solver_protocol_bytes = (result_root/"PROTOCOL.md.snapshot").read_bytes()
else:
    solver_bytes = solver_path.read_bytes()
    p2_bytes = p2_path.read_bytes()
    solver_protocol_bytes = solver_protocol_path.read_bytes()
mesh_bytes = mesh_path.read_bytes()
mesh_metadata_bytes = mesh_metadata_path.read_bytes()
solver_sha256 = hashlib.sha256(solver_bytes).hexdigest()
p2_sha256 = hashlib.sha256(p2_bytes).hexdigest()
solver_protocol_sha256 = hashlib.sha256(solver_protocol_bytes).hexdigest()
mesh_file_sha256 = hashlib.sha256(mesh_bytes).hexdigest()
mesh_metadata = json.loads(mesh_metadata_bytes)

cases = [
    ("node_m0_a_m5", "node", -5., 0.),
    ("node_m0_a0", "node", 0., 0.),
    ("node_m0_a_p5", "node", 5., 0.),
    ("weak_m0_a_m5", "weak", -5., 0.),
    ("weak_m0_a0", "weak", 0., 0.),
    ("weak_m0_a_p5", "weak", 5., 0.),
    ("weak_m02_a_p5", "weak", 5., 0.2),
]

records = []
for case_name, kutta_formulation, alpha_deg, mach in cases:
    case_root = result_root/case_name
    solve_output = case_root/"solve"
    command = [
        sys.executable, str(solver_path),
        "--mesh", str(mesh_path),
        "--alpha", str(alpha_deg),
        "--mach", str(mach),
        "--ncpu", "8",
        "--kutta", kutta_formulation,
        "--out", str(solve_output),
    ]
    if args.reuse_existing:
        prior_case = next(row for row in prior_summary["cases"] if row["case"] == case_name)
        process_returncode = prior_case["returncode"]
        elapsed_seconds = prior_case["elapsed_seconds"]
    else:
        case_root.mkdir()
        started = time.perf_counter()
        process = subprocess.run(command, capture_output=True, text=True)
        elapsed_seconds = time.perf_counter()-started
        process_returncode = process.returncode
        (case_root/"run.log").write_text(process.stdout+process.stderr, encoding="utf-8")
    result_path = solve_output/"result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    field_path = solve_output/"field.npz"
    field = np.load(field_path)
    assert result["solver_sha256"] == solver_sha256, "Solver source changed during this result batch"
    assert result["p2_sha256"] == p2_sha256, "P2 source changed during this result batch"
    assert result["protocol_sha256"] == solver_protocol_sha256, "Solver protocol changed during this result batch"
    assert result["mesh_sha256"] == mesh_metadata["mesh_sha256"], "Unexpected fixed-mesh identity"
    (case_root/"solve.py.snapshot").write_bytes(solver_bytes)
    (case_root/"p2.py.snapshot").write_bytes(p2_bytes)
    (case_root/"PROTOCOL.md.snapshot").write_bytes(solver_protocol_bytes)
    cp = np.asarray(field["cp"])
    cp_finite = np.isfinite(cp)
    cp_finite_values = cp[cp_finite]
    history = result.get("history", [])
    residual_values = [row["residual_norm"] for row in history if "residual_norm" in row]
    kutta_values = [row["kutta_max"] for row in history if "kutta_max" in row]
    linear_residual_values = [row["linear_relative_residual"]
                              for row in history if "linear_relative_residual" in row]
    history_numeric_values = [value for row in history for value in row.values()
                              if isinstance(value, (int, float, np.number))]
    history_all_finite = bool(np.all(np.isfinite(history_numeric_values)))
    final_residual = residual_values[-1] if residual_values else None
    final_kutta = kutta_values[-1] if kutta_values else None
    final_residual_kutta_gate = bool(
        final_residual is not None and final_kutta is not None
        and abs(final_residual) < 1.e-9 and abs(final_kutta) < 1.e-9)
    linear_residual_gate = bool(
        len(linear_residual_values) > 0
        and np.all(np.isfinite(linear_residual_values))
        and all(abs(value) < 1.e-9 for value in linear_residual_values))
    numerical_gate = bool(history_all_finite and final_residual_kutta_gate
                          and linear_residual_gate)
    cp_summary = dict(
        shape=list(cp.shape), value_count=int(cp.size),
        finite_count=int(cp_finite.sum()), nonfinite_count=int(cp.size-cp_finite.sum()),
        minimum=float(cp_finite_values.min()) if cp_finite_values.size else None,
        maximum=float(cp_finite_values.max()) if cp_finite_values.size else None,
        raw_values_file=str(field_path.relative_to(root)),
        raw_values_sha256=hashlib.sha256(field_path.read_bytes()).hexdigest())
    record = dict(
        case=case_name, kutta_formulation=kutta_formulation,
        alpha_deg=alpha_deg, mach=mach, ncpu_requested=8,
        command=command, returncode=process_returncode,
        elapsed_seconds=elapsed_seconds,
        result_path=str(result_path.relative_to(root)),
        result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
        field_path=str(field_path.relative_to(root)),
        status=result.get("status"), converged=result.get("converged"),
        solver_sha256=result.get("solver_sha256"),
        p2_sha256=result.get("p2_sha256"),
        protocol_sha256=result.get("protocol_sha256"),
        actual_snapshots=dict(
            solver_sha256=hashlib.sha256((case_root/"solve.py.snapshot").read_bytes()).hexdigest(),
            p2_sha256=hashlib.sha256((case_root/"p2.py.snapshot").read_bytes()).hexdigest(),
            protocol_sha256=hashlib.sha256((case_root/"PROTOCOL.md.snapshot").read_bytes()).hexdigest()),
        mesh_sha256=result.get("mesh_sha256"),
        result_history=history,
        residual_values=residual_values, kutta_values=kutta_values,
        linear_relative_residual_values=linear_residual_values,
        history_all_finite=history_all_finite,
        final_residual=final_residual, final_kutta=final_kutta,
        final_residual_kutta_gate=final_residual_kutta_gate,
        linear_residual_gate=linear_residual_gate,
        cp=cp_summary,
        cl_pressure=result.get("cl_pressure"), cd_pressure=result.get("cd_pressure"),
        cl_jump=result.get("cl_jump"),
        pressure_force_coefficient=result.get("pressure_force_coefficient"),
        wall_normal_rms=result.get("wall_normal_rms"),
        wall_normal_max=result.get("wall_normal_max"),
        wall_regions=result.get("wall_regions"),
        raw_mass_by_tag=result.get("raw_mass_by_tag"),
        weak_mass_residual_norm=result.get("weak_mass_residual_norm"),
        weak_mass_residual_per_chord2=result.get("weak_mass_residual_per_chord2"),
        weak_gauge_reaction=result.get("weak_gauge_reaction"),
        prescribed_outer_mass=result.get("prescribed_outer_mass"),
        wake_diagnostics=result.get("wake_diagnostics"),
        kutta_edge_traces=result.get("kutta_edge_traces"),
        kutta_final=result.get("kutta_max"),
        potential_jump_max_error=result.get("potential_jump_max_error"),
        numerical_gate=numerical_gate,
        case_passed=bool(process_returncode == 0 and result.get("converged")
                         and result.get("kutta_formulation") == kutta_formulation
                         and cp_summary["nonfinite_count"] == 0
                         and numerical_gate))
    records.append(record)

gates = []
for formulation in ["node", "weak"]:
    zero = next(row for row in records if row["kutta_formulation"] == formulation
                and row["mach"] == 0. and row["alpha_deg"] == 0.)
    negative = next(row for row in records if row["kutta_formulation"] == formulation
                    and row["mach"] == 0. and row["alpha_deg"] == -5.)
    positive = next(row for row in records if row["kutta_formulation"] == formulation
                    and row["mach"] == 0. and row["alpha_deg"] == 5.)
    zero_gamma = np.asarray(np.load(root/zero["field_path"])["gamma"])
    negative_gamma = np.asarray(np.load(root/negative["field_path"])["gamma"])
    positive_gamma = np.asarray(np.load(root/positive["field_path"])["gamma"])
    zero_errors = dict(
        cl_pressure=abs(zero["cl_pressure"]),
        cl_jump=abs(zero["cl_jump"]),
        gamma_max=float(np.max(abs(zero_gamma))))
    symmetry_errors = dict(
        cl_pressure=abs(positive["cl_pressure"]+negative["cl_pressure"]),
        cl_jump=abs(positive["cl_jump"]+negative["cl_jump"]),
        gamma_max=float(np.max(abs(positive_gamma+negative_gamma))))
    gates.append(dict(
        formulation=formulation, zero_load_errors=zero_errors,
        zero_load_passed=all(value < 1.e-9 for value in zero_errors.values()),
        positive_negative_errors=symmetry_errors,
        positive_negative_passed=all(value < 1.e-9 for value in symmetry_errors.values())))

for record in records:
    record["zero_symmetry_gate"] = next(
        row for row in gates if row["formulation"] == record["kutta_formulation"])

solver_hashes = sorted(set(row["solver_sha256"] for row in records))
p2_hashes = sorted(set(row["p2_sha256"] for row in records))
protocol_hashes = sorted(set(row["protocol_sha256"] for row in records))
mesh_hashes = sorted(set(row["mesh_sha256"] for row in records))
same_executable_version = solver_hashes == [solver_sha256]
same_p2_version = p2_hashes == [p2_sha256]
same_solver_protocol_version = protocol_hashes == [solver_protocol_sha256]
same_mesh = len(mesh_hashes) == 1
summary = dict(
    protocol="KUTTA_RESULTS.md",
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    executable=sys.executable,
    executable_sha256=hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
    solver_sha256=solver_sha256, p2_sha256=p2_sha256,
    solver_protocol_sha256=solver_protocol_sha256,
    current_solver_file_sha256=hashlib.sha256(solver_path.read_bytes()).hexdigest(),
    current_p2_file_sha256=hashlib.sha256(p2_path.read_bytes()).hexdigest(),
    current_solver_protocol_file_sha256=hashlib.sha256(
        solver_protocol_path.read_bytes()).hexdigest(),
    mesh_path=str(mesh_path.relative_to(root)),
    mesh_file_sha256=mesh_file_sha256,
    mesh_metadata_path=str(mesh_metadata_path.relative_to(root)),
    mesh_metadata_sha256=hashlib.sha256(mesh_metadata_bytes).hexdigest(),
    mesh_sha256=mesh_metadata["mesh_sha256"],
    mesh_metadata=mesh_metadata,
    ncpu_requested=8, cases=records, zero_symmetry_gates=gates,
    observed_solver_hashes=solver_hashes, observed_p2_hashes=p2_hashes,
    observed_solver_protocol_hashes=protocol_hashes, observed_mesh_hashes=mesh_hashes,
    same_executable_version=same_executable_version,
    same_p2_version=same_p2_version,
    same_solver_protocol_version=same_solver_protocol_version,
    same_mesh=same_mesh,
    all_cases_passed=all(row["case_passed"] for row in records),
    all_zero_symmetry_passed=all(row["zero_load_passed"]
                                 and row["positive_negative_passed"] for row in gates),
    scope="Fixed-mesh implementation comparison and internal numerical checks; no absolute aerodynamic accuracy certification")
if args.reuse_existing:
    (result_root/"summary_pre_gate_correction.json").write_bytes(prior_summary_bytes)
(result_root/"solve.py.snapshot").write_bytes(solver_bytes)
(result_root/"p2.py.snapshot").write_bytes(p2_bytes)
(result_root/"PROTOCOL.md.snapshot").write_bytes(solver_protocol_bytes)
(result_root/"summary.json").write_text(
    json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
print("cases", len(records), "all cases", summary["all_cases_passed"],
      "zero/symmetry", summary["all_zero_symmetry_passed"], flush=True)
print("version hashes", same_executable_version, same_p2_version,
      same_solver_protocol_version, "mesh", same_mesh, flush=True)
for row in records:
    print(row["case"], "rc", row["returncode"], "CLp", row["cl_pressure"],
          "CLjump", row["cl_jump"], "CDp", row["cd_pressure"],
          "Cp", row["cp"]["minimum"], row["cp"]["maximum"],
          "wall RMS/max", row["wall_normal_rms"], row["wall_normal_max"],
          "residual gate", row["numerical_gate"], flush=True)
