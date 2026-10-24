"""M=0 angle synthesis from two converged same-mesh linear solutions."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
default_base = root/"results/planforms_v1/combined"
parser = argparse.ArgumentParser()
parser.add_argument("--basis0", type=Path, default=default_base/"a0p0")
parser.add_argument("--basis-beta", type=Path, default=default_base/"a5p0")
parser.add_argument("--beta", type=float, default=5.)
parser.add_argument("--mach", type=float, default=0.)
parser.add_argument("--angles", type=float, nargs="+", default=[-5., 0., 5., 2.5])
parser.add_argument("--direct-alpha", type=float, default=2.5)
parser.add_argument("--out", type=Path, default=root/"results/linear_batch_v1")
parser.add_argument("--ncpu", type=int, default=8)
args = parser.parse_args()
args.basis0 = args.basis0.resolve()
args.basis_beta = args.basis_beta.resolve()
args.out = args.out.resolve()

assert args.mach == 0., "Linear angle synthesis is restricted to M=0"
assert np.isfinite([args.beta, args.direct_alpha, *args.angles]).all()
assert abs(args.beta) > 1.e-8 and abs(np.sin(np.deg2rad(args.beta))) > 1.e-8, "Degenerate beta basis angle"
assert args.ncpu > 0 and not args.out.exists(), "Require positive NCPU and a new output directory"
assert args.direct_alpha not in (0., args.beta), "Direct comparison angle must be independent of both bases"
args.out.mkdir(parents=True)

mesh_path = args.basis0.parent/"mesh"/"mesh.npz"
metadata_path = mesh_path.with_name("mesh.json")
mesh_bytes = mesh_path.read_bytes()
mesh_sha = hashlib.sha256(mesh_bytes).hexdigest()
mesh_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
mesh = np.load(mesh_path)
snapshot_names = ["solve.py", "p2.py", "PROTOCOL.md", "schur.py", "SCHUR_PROTOCOL.md",
                  "root_wake.py", "ROOT_WAKE_PROTOCOL.md"]
result_hash_names = dict(solver_sha256="solve.py", p2_sha256="p2.py",
                         protocol_sha256="PROTOCOL.md", schur_sha256="schur.py")

basis_paths = [args.basis0, args.basis_beta]
basis_results = []
basis_fields = []
basis_hashes = []
basis_source_history = []
for basis_path in basis_paths:
    result_bytes = (basis_path/"result.json").read_bytes()
    field_bytes = (basis_path/"field.npz").read_bytes()
    result = json.loads(result_bytes)
    field = np.load(basis_path/"field.npz")
    assert result["converged"] and result["status"].startswith("converged")
    assert result["mach"] == 0. and result["mesh_sha256"] == mesh_sha
    assert abs(result["alpha_deg"] - (0. if basis_path == args.basis0 else args.beta)) < 1.e-12
    assert result["quadrature_order"] == json.loads((args.basis0/"result.json").read_text())["quadrature_order"]
    assert result["kutta_formulation"] == json.loads((args.basis0/"result.json").read_text())["kutta_formulation"]
    assert len(field["points"]) == len(mesh["points"])
    assert np.array_equal(field["points"], mesh["points"])
    assert np.array_equal(field["tets"], mesh["tets"])
    assert np.isfinite(field["phi"]).all() and np.isfinite(field["psi"]).all()
    assert np.isfinite(field["gamma"]).all() and np.isfinite(field["surface_velocity"]).all()
    assert np.array_equal(field["faces"], basis_fields[0]["faces"]) if basis_fields else True
    assert np.array_equal(field["tags"], basis_fields[0]["tags"]) if basis_fields else True
    snapshot_hashes = {}
    for source_name in snapshot_names:
        snapshot_path = basis_path/f"{source_name}.snapshot"
        if snapshot_path.exists():
            snapshot_hashes[source_name] = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    assert all(source_name in snapshot_hashes for source_name in ["solve.py", "p2.py", "PROTOCOL.md"])
    for result_key, source_name in result_hash_names.items():
        if result_key in result:
            assert source_name in snapshot_hashes, f"Missing {source_name} snapshot in {basis_path}"
            assert snapshot_hashes[source_name] == result[result_key], f"{source_name} snapshot hash mismatch in {basis_path}"
    basis_source_history.append(snapshot_hashes)
    basis_results.append(result)
    basis_fields.append(field)
    basis_hashes.append(dict(result_sha256=hashlib.sha256(result_bytes).hexdigest(),
                             field_sha256=hashlib.sha256(field_bytes).hexdigest(),
                             solver_sha256=result["solver_sha256"],
                             p2_sha256=result["p2_sha256"],
                             protocol_sha256=result["protocol_sha256"],
                             schur_sha256=result.get("schur_sha256")))

result0, result_beta = basis_results
field0, field_beta = basis_fields
for key in ["solver_sha256", "p2_sha256", "protocol_sha256", "quadrature_order",
            "kutta_formulation", "wake_model", "wake_tilt_deg", "wake_strips",
            "wake_material_coordinates", "linear_solver", "schur_sha256"]:
    assert result0.get(key) == result_beta.get(key), f"Basis mismatch: {key}"
assert basis_source_history[0] == basis_source_history[1], "Basis snapshot sets or hashes differ"
for key in ["phi", "psi", "gamma", "surface_velocity"]:
    assert field0[key].shape == field_beta[key].shape, f"Basis field shape mismatch: {key}"
for key in ["faces", "tags", "adjacent", "surface_points", "surface_normal", "surface_weight"]:
    assert np.array_equal(field0[key], field_beta[key]), f"Basis geometry mismatch: {key}"
gamma_stations = np.asarray(result0["gamma_stations"], dtype=float)
if "gamma_station_dof" in field0:
    station_dof = np.asarray(field0["gamma_station_dof"], dtype=np.int64)
else:
    station_dof = np.asarray(result0.get(
        "gamma_station_dof", np.r_[-1, np.arange(len(gamma_stations)-2), -1]), dtype=np.int64)
if "gamma_station_dof" in result0:
    assert np.array_equal(station_dof, np.asarray(result0["gamma_station_dof"], dtype=np.int64))
if "gamma_station_dof" in field_beta:
    assert np.array_equal(station_dof, np.asarray(field_beta["gamma_station_dof"], dtype=np.int64))
if "gamma_station_dof" in result_beta:
    assert np.array_equal(station_dof, np.asarray(result_beta["gamma_station_dof"], dtype=np.int64))
if "gamma_station_dof" in mesh:
    assert np.array_equal(station_dof, np.asarray(mesh["gamma_station_dof"], dtype=np.int64))
assert station_dof.shape == gamma_stations.shape
assert station_dof[0] == station_dof[-1] == -1
gamma_unknowns = int(station_dof.max()+1)
assert gamma_unknowns == len(field0["gamma"]) == len(field_beta["gamma"])
assert np.array_equal(np.unique(station_dof[station_dof >= 0]), np.arange(gamma_unknowns))
root_wake_support = result0.get("root_wake_support")
assert root_wake_support == result_beta.get("root_wake_support")
normalization_length = float(result0.get("normalization_length", mesh_metadata["chord"]))
normalization_origin = np.asarray(result0.get(
    "normalization_origin", mesh_metadata.get("normalization_origin", [0., 0., 0.])), dtype=float)
assert normalization_length > 0. and normalization_origin.shape == (3,)
assert normalization_length == float(result_beta.get("normalization_length", normalization_length))
assert np.array_equal(normalization_origin, np.asarray(result_beta.get(
    "normalization_origin", normalization_origin.tolist()), dtype=float))
assert normalization_length == float(mesh_metadata["chord"])
assert np.array_equal(normalization_origin, np.asarray(
    mesh_metadata.get("normalization_origin", [0., 0., 0.]), dtype=float))
for field in basis_fields:
    if "gamma_station_dof" in field:
        assert np.array_equal(station_dof, field["gamma_station_dof"])
assert result0["history"] and result_beta["history"]
residual0 = float(result0["history"][-1]["residual_norm"])
residual_beta = float(result_beta["history"][-1]["residual_norm"])
assert np.isfinite([residual0, residual_beta]).all()

wall = np.isin(field0["tags"], [10, 20, 30])
reference_area = float(mesh_metadata["reference_area"])
assert reference_area > 0.
online_started = time.perf_counter()
case_records = []
for alpha in args.angles:
    case_started = time.perf_counter()
    alpha_rad = np.deg2rad(alpha)
    beta_rad = np.deg2rad(args.beta)
    weight_beta = np.sin(alpha_rad)/np.sin(beta_rad)
    weight_zero = np.cos(alpha_rad)-weight_beta*np.cos(beta_rad)
    combined = {}
    for key in ["phi", "psi", "gamma", "surface_velocity"]:
        combined[key] = weight_zero*field0[key]+weight_beta*field_beta[key]
    velocity = combined["surface_velocity"]
    cp = 1.-np.einsum("fqi,fqi->fq", velocity, velocity)
    normal_velocity = np.einsum("fqi,fqi->fq", velocity, field0["surface_normal"])
    flow = np.array([np.cos(alpha_rad), 0., np.sin(alpha_rad)])
    lift_axis = np.array([-np.sin(alpha_rad), 0., np.cos(alpha_rad)])
    component_loads = {}
    pressure_force = np.zeros(3)
    for tag in [10, 20, 30]:
        selected = field0["tags"] == tag
        if not np.any(selected):
            continue
        force = np.sum(cp[selected, :, None]*field0["surface_normal"][selected]
                       *field0["surface_weight"][selected, :, None], axis=(0, 1))/reference_area
        pressure_force += force
        component_loads[str(tag)] = dict(
            force_coefficient=force.tolist(),
            cl=float(force@lift_axis),
            cd=float(force@flow),
            area=float(field0["surface_weight"][selected].sum()),
            normal_rms=float(np.sqrt(np.average(normal_velocity[selected]**2,
                                                  weights=field0["surface_weight"][selected]))))
    station_gamma = np.zeros(len(gamma_stations))
    mapped_stations = station_dof >= 0
    station_gamma[mapped_stations] = combined["gamma"][station_dof[mapped_stations]]
    cl_gamma = float(2.*np.trapezoid(station_gamma, gamma_stations)/reference_area)
    out = args.out/f"alpha_{alpha:+g}".replace("+", "p").replace("-", "m").replace(".", "p")
    out.mkdir()
    np.savez_compressed(out/"field.npz",
                        points=field0["points"], tets=field0["tets"],
                        phi=combined["phi"], psi=combined["psi"], gamma=combined["gamma"],
                        gamma_station_dof=station_dof,
                        faces=field0["faces"], tags=field0["tags"], adjacent=field0["adjacent"],
                        surface_points=field0["surface_points"], surface_velocity=velocity, cp=cp,
                        surface_normal=field0["surface_normal"], surface_weight=field0["surface_weight"],
                        surface_normal_velocity=normal_velocity)
    propagated_residual_bound = abs(weight_zero)*residual0+abs(weight_beta)*residual_beta
    synthesized_result = dict(
        status="converged_linear_superposition_mach0",
        converged=True,
        alpha_deg=float(alpha),
        mach=0.,
        method="Exact M=0 linear superposition of saved potential/Gamma/velocity basis states; Cp and pressure loads recomputed",
        history=[],
        history_note="No per-angle Newton solve or fabricated iteration history; the online path does not reassemble the target weak residual.",
        basis_weights=dict(alpha0=float(weight_zero), alpha_beta=float(weight_beta), beta_deg=float(args.beta)),
        basis_final_weak_residual_norms=dict(alpha0=residual0, alpha_beta=residual_beta),
        propagated_weak_residual_bound=float(propagated_residual_bound),
        propagated_residual_bound_definition="abs(w0)*R0 + abs(wbeta)*Rbeta; bound on weighted basis residual magnitudes, not a separately reassembled target residual",
        online_elapsed_seconds=0.,
        mesh_sha256=mesh_sha,
        solver_sha256=result0["solver_sha256"],
        linear_solver=result0.get("linear_solver", "legacy"),
        schur_sha256=result0.get("schur_sha256"),
        p2_sha256=result0["p2_sha256"],
        protocol_sha256=result0["protocol_sha256"],
        source_history_sha256=basis_source_history[0],
        quadrature_order=result0["quadrature_order"],
        kutta_formulation=result0["kutta_formulation"],
        normalization_length=normalization_length,
        normalization_origin=normalization_origin.tolist(),
        root_wake_support=root_wake_support,
        kutta_max=None,
        nodes=len(field0["points"]),
        elements=len(field0["tets"]),
        gamma_unknowns=len(combined["gamma"]),
        gamma_stations=gamma_stations.tolist(),
        gamma_station_dof=station_dof.tolist(),
        potential_jump=station_gamma.tolist(),
        cl_pressure=float(pressure_force@lift_axis),
        cd_pressure=float(pressure_force@flow),
        cl_jump=cl_gamma,
        pressure_force_coefficient=pressure_force.tolist(),
        cp_min=float(cp[wall].min()),
        cp_max=float(cp[wall].max()),
        component_loads=component_loads,
        raw_fields="field.npz stores linearly combined phi, psi, Gamma, and surface_velocity; Cp and surface_normal_velocity are recomputed; geometry, normals, weights, tags, and connectivity are copied from alpha0 basis")
    synthesized_result["online_elapsed_seconds"] = time.perf_counter()-case_started
    (out/"result.json").write_text(json.dumps(synthesized_result, indent=2), encoding="utf-8")
    case_records.append(dict(alpha_deg=float(alpha), directory=out.name,
                             weights=synthesized_result["basis_weights"],
                             propagated_residual_bound=propagated_residual_bound,
                             cl_pressure=synthesized_result["cl_pressure"],
                             cl_gamma=cl_gamma))
online_elapsed = time.perf_counter()-online_started

direct_out = args.out/"direct_alpha_2p5"
command = [sys.executable, str(root/"solve.py"), "--mesh", str(mesh_path),
           "--alpha", str(args.direct_alpha), "--mach", "0", "--kutta", result0["kutta_formulation"],
           "--quad-order", str(result0["quadrature_order"]),
           "--linear-solver", result0.get("linear_solver", "augmented"), "--ncpu", str(args.ncpu),
           "--out", str(direct_out)]
direct_started = time.perf_counter()
direct_run = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=240)
direct_elapsed = time.perf_counter()-direct_started
(args.out/"direct_alpha_2p5.stdout.log").write_text(direct_run.stdout, encoding="utf-8")
(args.out/"direct_alpha_2p5.stderr.log").write_text(direct_run.stderr, encoding="utf-8")
assert direct_run.returncode == 0, f"Direct solve failed with return code {direct_run.returncode}; see retained logs"
direct_result = json.loads((direct_out/"result.json").read_text(encoding="utf-8"))
assert direct_result["converged"] and direct_result["mesh_sha256"] == mesh_sha
direct_source_history = {}
for source_name in snapshot_names:
    snapshot_path = direct_out/f"{source_name}.snapshot"
    if snapshot_path.exists():
        direct_source_history[source_name] = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
assert all(source_name in direct_source_history for source_name in ["solve.py", "p2.py", "PROTOCOL.md"])
for result_key, source_name in result_hash_names.items():
    if result_key in direct_result:
        assert direct_source_history.get(source_name) == direct_result[result_key]
direct_same_source = direct_source_history == basis_source_history[0]
direct_same_settings = all([
    direct_result.get("linear_solver", "augmented") == result0.get("linear_solver", "augmented"),
    direct_result["kutta_formulation"] == result0["kutta_formulation"],
    direct_result["quadrature_order"] == result0["quadrature_order"],
    direct_result.get("root_wake_support") == root_wake_support,
    np.array_equal(np.asarray(direct_result["gamma_station_dof"], dtype=np.int64), station_dof)])

target_name = f"alpha_{args.direct_alpha:+g}".replace("+", "p").replace("-", "m").replace(".", "p")
linear_field = np.load(args.out/target_name/"field.npz")
direct_field = np.load(direct_out/"field.npz")
comparison = {}
for key in ["phi", "gamma", "surface_velocity", "cp"]:
    max_difference = float(np.max(np.abs(linear_field[key]-direct_field[key])))
    comparison[key] = max_difference
comparison["pressure_force_coefficient"] = float(np.max(np.abs(
    np.asarray(json.loads((args.out/target_name/"result.json").read_text())["pressure_force_coefficient"])
    -np.asarray(direct_result["pressure_force_coefficient"]))))
if direct_same_source and direct_same_settings:
    assert max(comparison.values()) < 1.e-9, "Same-source direct-vs-linear comparison exceeds tolerance"

summary = dict(
    status=("linear_batch_complete_direct_regression_passed" if direct_same_source and direct_same_settings
            else "linear_batch_complete_direct_source_mismatch_not_comparable" if not direct_same_source
            else "linear_batch_complete_direct_settings_mismatch_not_comparable"),
    mach=0.,
    mesh_sha256=mesh_sha,
    basis=dict(alpha0=args.basis0.as_posix(), alpha_beta=args.basis_beta.as_posix(),
               beta_deg=args.beta, hashes=basis_hashes,
               source_history_sha256=basis_source_history,
               solver_sha256=result0["solver_sha256"], p2_sha256=result0["p2_sha256"],
               linear_solver=result0.get("linear_solver", "legacy"),
               schur_sha256=result0.get("schur_sha256"),
               protocol_sha256=result0["protocol_sha256"],
               linear_batch_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
    synthesized_cases=case_records,
    direct_comparison=dict(alpha_deg=args.direct_alpha, returncode=direct_run.returncode,
                           elapsed_seconds=direct_elapsed, ncpu=args.ncpu,
                           source_history_sha256=direct_source_history,
                           basis_source_history_sha256=basis_source_history[0],
                           source_match=direct_same_source,
                           settings_match=direct_same_settings,
                           comparison_status=("same_source_regression" if direct_same_source and direct_same_settings
                                              else "different_source_diagnostic_only" if not direct_same_source
                                              else "different_settings_diagnostic_only"),
                           command=command, max_abs_differences=comparison,
                           acceptance_tolerance=1.e-9, direct_result_sha256=hashlib.sha256(
                               (direct_out/"result.json").read_bytes()).hexdigest()),
    timing=dict(online_combination_and_reintegration_seconds=online_elapsed,
                offline_basis_solve_seconds=[result0["elapsed_seconds"], result_beta["elapsed_seconds"]],
                offline_direct_validation_solve_seconds=direct_elapsed,
                note="Online excludes loading/output filesystem time only insofar as included in wall-clock region; offline basis/direct solves are reported separately."),
    constraints=dict(mach0_only=True, same_mesh_and_basis_sources=True,
                     direct_source_must_match_basis_for_acceptance=direct_same_source,
                     direct_settings_must_match_basis_for_acceptance=direct_same_settings,
                     target_weak_residual_reassembled=False,
                     per_angle_newton_history_fabricated=False))
(args.out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
