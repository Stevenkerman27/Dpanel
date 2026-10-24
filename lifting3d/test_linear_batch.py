"""Independent regression checks for the saved M=0 linear batch outputs."""
import argparse
import json
import hashlib
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=root/"results/linear_batch_root_v3")
args = parser.parse_args()
output = args.output.resolve()
summary = json.loads((output/"summary.json").read_text(encoding="utf-8"))
assert summary["mach"] == 0.
assert summary["direct_comparison"]["returncode"] == 0
if (summary["direct_comparison"]["source_match"]
        and summary["direct_comparison"]["settings_match"]):
    assert summary["status"] == "linear_batch_complete_direct_regression_passed"
    assert all(value < 1.e-9 for value in summary["direct_comparison"]["max_abs_differences"].values())
elif not summary["direct_comparison"]["source_match"]:
    assert summary["status"] == "linear_batch_complete_direct_source_mismatch_not_comparable"
    assert summary["direct_comparison"]["comparison_status"] == "different_source_diagnostic_only"
else:
    assert summary["status"] == "linear_batch_complete_direct_settings_mismatch_not_comparable"
    assert summary["direct_comparison"]["comparison_status"] == "different_settings_diagnostic_only"
assert summary["constraints"]["target_weak_residual_reassembled"] is False
assert summary["constraints"]["per_angle_newton_history_fabricated"] is False

base0 = Path(summary["basis"]["alpha0"])
base_beta = Path(summary["basis"]["alpha_beta"])
mesh_path = base0.parent/"mesh/mesh.npz"
mesh = np.load(mesh_path)
field0 = np.load(base0/"field.npz")
field_beta = np.load(base_beta/"field.npz")
result0 = json.loads((base0/"result.json").read_text(encoding="utf-8"))
result_beta = json.loads((base_beta/"result.json").read_text(encoding="utf-8"))
for key in ["solver_sha256", "p2_sha256", "protocol_sha256", "linear_solver",
            "schur_sha256"]:
    assert result0.get(key) == result_beta.get(key)
for basis_path, basis_result in [(base0, result0), (base_beta, result_beta)]:
    recorded_snapshot_hashes = {}
    for source_name in ["solve.py", "p2.py", "PROTOCOL.md", "schur.py",
                        "SCHUR_PROTOCOL.md", "root_wake.py", "ROOT_WAKE_PROTOCOL.md"]:
        snapshot_path = basis_path/f"{source_name}.snapshot"
        if snapshot_path.exists():
            recorded_snapshot_hashes[source_name] = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    for key, source_name in [("solver_sha256", "solve.py"), ("p2_sha256", "p2.py"),
                             ("protocol_sha256", "PROTOCOL.md"), ("schur_sha256", "schur.py")]:
        if key in basis_result:
            assert recorded_snapshot_hashes[source_name] == basis_result[key]
    assert recorded_snapshot_hashes == summary["basis"]["source_history_sha256"][0 if basis_path == base0 else 1]
assert summary["basis"]["source_history_sha256"][0] == summary["basis"]["source_history_sha256"][1]
for key in ["solver_sha256", "p2_sha256", "protocol_sha256"]:
    assert result0[key] == result_beta[key]
direct = np.load(output/"direct_alpha_2p5/field.npz")
alpha = np.deg2rad(2.5)
beta = np.deg2rad(5.)
weight_beta = np.sin(alpha)/np.sin(beta)
weight_zero = np.cos(alpha)-weight_beta*np.cos(beta)
linear_direct = np.load(output/"alpha_p2p5/field.npz")
if summary["direct_comparison"]["source_match"]:
    for key in ["phi", "gamma", "surface_velocity"]:
        synthesized = weight_zero*field0[key]+weight_beta*field_beta[key]
        assert np.max(np.abs(synthesized-direct[key])) < 1.e-9
        assert np.max(np.abs(linear_direct[key]-direct[key])) < 1.e-9
    direct_cp = 1.-np.einsum("fqi,fqi->fq", direct["surface_velocity"], direct["surface_velocity"])
    assert np.max(np.abs(direct_cp-direct["cp"])) < 1.e-9
    assert np.max(np.abs(linear_direct["cp"]-direct_cp)) < 1.e-9
    linear_result = json.loads((output/"alpha_p2p5/result.json").read_text(encoding="utf-8"))
    direct_result = json.loads((output/"direct_alpha_2p5/result.json").read_text(encoding="utf-8"))
    assert np.max(np.abs(np.asarray(linear_result["pressure_force_coefficient"])
                         -np.asarray(direct_result["pressure_force_coefficient"]))) < 1.e-9

for alpha in [-5., 0., 5., 2.5]:
    name = f"alpha_{alpha:+g}".replace("+", "p").replace("-", "m").replace(".", "p")
    case = output/name
    result = json.loads((case/"result.json").read_text(encoding="utf-8"))
    field = np.load(case/"field.npz")
    assert result["status"] == "converged_linear_superposition_mach0"
    assert result["converged"] and result["mach"] == 0.
    assert result["history"] == []
    assert np.array_equal(field["points"], mesh["points"])
    assert np.array_equal(field["tets"], mesh["tets"])
    assert np.isfinite(field["phi"]).all() and np.isfinite(field["gamma"]).all()
    assert np.isfinite(field["surface_velocity"]).all() and np.isfinite(field["cp"]).all()
    assert np.array_equal(field["gamma_station_dof"], np.asarray(result["gamma_station_dof"]))
    full_gamma = np.zeros(len(result["gamma_stations"]))
    gamma_dof = field["gamma_station_dof"] >= 0
    full_gamma[gamma_dof] = field["gamma"][field["gamma_station_dof"][gamma_dof]]
    assert np.allclose(full_gamma, result["potential_jump"], rtol=0., atol=1.e-14)
    assert result["root_wake_support"] == result0["root_wake_support"]
    assert result["normalization_length"] == result0["normalization_length"]
    assert result["normalization_origin"] == result0["normalization_origin"]
    assert np.allclose(field["cp"], 1.-np.einsum(
        "fqi,fqi->fq", field["surface_velocity"], field["surface_velocity"]), rtol=0., atol=2.e-15)

pilot_root = root/"results/root_wake_pilot_v2"
pilot_mesh = np.load(pilot_root/"mesh/mesh.npz")
pilot_field = np.load(pilot_root/"solve/field.npz")
pilot_result = json.loads((pilot_root/"solve/result.json").read_text(encoding="utf-8"))
pilot_map = np.asarray(pilot_field["gamma_station_dof"], dtype=np.int64)
assert np.array_equal(pilot_map, pilot_mesh["gamma_station_dof"])
assert np.array_equal(pilot_map, np.asarray(pilot_result["gamma_station_dof"], dtype=np.int64))
pilot_station_gamma = np.zeros(len(pilot_result["gamma_stations"]))
pilot_mapped = pilot_map >= 0
pilot_station_gamma[pilot_mapped] = pilot_field["gamma"][pilot_map[pilot_mapped]]
assert np.allclose(pilot_station_gamma, pilot_result["potential_jump"], rtol=0., atol=1.e-14)
assert pilot_result["root_wake_support"]["status"] == "constructed"
assert pilot_result["normalization_length"] == 1.
assert np.array_equal(pilot_result["normalization_origin"], [0., 0., 0.])

print("linear batch regression artifacts passed independent schema and Cp checks")
