"""Controlled B3 wake-position / operator study; production defaults are untouched.

python -m tools.wake_position_study --phase e1 --case swept30 --out runs/verification/issue102_wake_sensitivity
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh, onenormest
import torch

from panel.p1_lifting_g2 import assemble_multi, combine_wings, solve_multi
from panel.p1_pressure_kutta_torch import TorchStructuredPressureKutta
from panel.p1_steady_free_wake import reassemble_wake_only, with_wake_xyz
from tools.issue92.b3_dpanel import _wing
from tools.wake_sensitivity_geometry import audit_geometry, geometry_context, spacing


root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--phase", choices=("e1", "e4"), required=True)
parser.add_argument("--case", choices=("rect", "swept30", "elliptic"), required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
out = args.out / args.case
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
baseline_manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
spec = baseline_manifest["spec"]
wing = _wing(spec)
mesh = combine_wings([wing], reference_chord=spec["cref"], reference_area=spec["sref"],
                     reference_point=wing.reference_point)
with np.load(out / "initial_state.npz") as data:
    for key in ("xyz", "faces", "wake_xyz", "wake_faces", "gamma_hat"):
        np.testing.assert_array_equal(getattr(wing, key), data[key])
    gamma0 = data["state_gamma"].copy()
    xyz0 = data["state_wake_xyz"].copy()
with np.load(out / "iteration_001.npz") as data:
    actual = data["wake_xyz"] - xyz0
context = geometry_context(mesh)
geometry0 = audit_geometry(mesh, context)
top = context["topology"]
used, attached, rows = top["used"], top["attached"], top["rows"]
free = np.setdiff1d(used, top["graph_roots"])
regions = dict(attachment=np.intersect1d(free, attached),
               first_free_row=np.intersect1d(free, np.flatnonzero(rows == 1)),
               near_TE=np.intersect1d(free, np.flatnonzero((rows >= 1) & (rows <= 3))),
               downstream=np.intersect1d(free, np.flatnonzero(rows >= 4)), all=free)
normals = np.zeros_like(xyz0)
tri = xyz0[mesh.groups[0].wake_faces]
cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
for k in range(3):
    np.add.at(normals, mesh.groups[0].wake_faces[:, k], cross)
normals[used] /= np.linalg.norm(normals[used], axis=1)[:, None]
streamwise = np.zeros_like(xyz0)
streamwise[used, 0] = 1.
directions = dict(normal=normals, streamwise=streamwise, actual=actual)
operators0 = assemble_multi(mesh)
state0 = baseline_manifest["rows"][0]
started = time.time()
capture = {}
original_init = TorchStructuredPressureKutta.__init__


def capture_init(self, *a, **kw):
    original_init(self, *a, **kw)
    capture["problem"] = self


if args.phase == "e4":
    TorchStructuredPressureKutta.__init__ = capture_init


def write_report():
    (out / ("e1_position_sensitivity.json" if args.phase == "e1" else "e4_conditioning.json")).write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


def solve_record(changed, name, *, geometry=True):
    ops = reassemble_wake_only(changed, operators0)
    for key in ("A", "J", "K", "grad"):
        assert ops[key] is operators0[key]
    state = solve_multi(changed, 4., kutta="pressure", max_nfev=80, operators=ops)
    gamma = np.asarray(state["gamma"])
    quality = audit_geometry(changed, context, geometry0) if geometry else None
    nonlinear = state["nonlinear"]
    inner_ok = bool(nonlinear["success"] and nonlinear["criteria"]["stationary"]
                    and nonlinear["wall_over_floor"] <= 1.05
                    and state["jump_rms"] <= 1e-10 and state["pressure_kutta_rms"] <= 1e-4)
    record = {k: state[k] for k in ("CL", "CL_gamma", "CM_y", "CD_pressure", "wall_rms",
                                   "jump_rms", "pressure_kutta_rms", "nonlinear")}
    record.update(name=name, gamma=gamma.tolist(), gamma_norm=float(np.linalg.norm(gamma)),
                  gamma_relative_change=float(np.linalg.norm(gamma-gamma0)
                                              / max(np.linalg.norm(gamma0), 1e-300)),
                  gamma_shape_change=float(np.linalg.norm(
                      gamma / max(np.linalg.norm(gamma), 1e-300)
                      - gamma0 / max(np.linalg.norm(gamma0), 1e-300))),
                  finite_gamma=bool(np.isfinite(gamma).all()), inner_converged=inner_ok,
                  geometry=quality,
                  admissible=(None if quality is None else bool(
                      inner_ok and np.isfinite(gamma).all() and quality["min_area2"] > 0
                      and quality["min_reference_signed_jacobian"] > 0
                      and quality["new_or_deeper_violation_count"] == 0)),
                  W_relative_change=float(np.linalg.norm(ops["W"]-operators0["W"])
                                          / max(np.linalg.norm(operators0["W"]), 1e-300)),
                  body_A_unchanged=True, combined_wall_B_unchanged=False,
                  jump_operator_unchanged=True,
                  rhs_unchanged=True)
    return record, state, ops


def triangular_spectrum(R):
    n = int(R.shape[0])
    R = R.detach().contiguous()
    if n <= 256:
        singular = torch.linalg.svdvals(R)
        maximum, minimum = float(singular[0]), float(singular[-1])
        residual_max = residual_min = 0.
        method = "exact_dense_svd_small_QR_factor"
    else:
        def gram(v):
            x = torch.as_tensor(v, dtype=torch.float64)
            return (R.mT @ (R @ x)).numpy()

        def inverse_gram(v):
            x = torch.as_tensor(v, dtype=torch.float64).reshape(n, -1)
            a = torch.linalg.solve_triangular(R.mT, x, upper=False)
            b = torch.linalg.solve_triangular(R, a, upper=True)
            return b.numpy().reshape(np.asarray(v).shape)

        G = LinearOperator((n, n), matvec=gram, rmatvec=gram, dtype=np.float64)
        Gi = LinearOperator((n, n), matvec=inverse_gram, rmatvec=inverse_gram, dtype=np.float64)
        v0 = np.random.default_rng(20261105).normal(size=n)
        largest, vmax = eigsh(G, k=1, which="LM", v0=v0, tol=1e-8, maxiter=500)
        inverse_largest, vmin = eigsh(Gi, k=1, which="LM", v0=v0, tol=1e-8, maxiter=500)
        maximum, minimum = float(np.sqrt(largest[0])), float(1 / np.sqrt(inverse_largest[0]))
        residual_max = float(np.linalg.norm(gram(vmax[:, 0]) - largest[0] * vmax[:, 0])
                             / abs(largest[0]))
        residual_min = float(np.linalg.norm(inverse_gram(vmin[:, 0])
                                            - inverse_largest[0] * vmin[:, 0])
                             / abs(inverse_largest[0]))
        method = "Lanczos_extremes_of_QR_Gram_and_inverse_Gram_via_triangular_solves"

    def inverse(v):
        x = torch.as_tensor(v, dtype=torch.float64).reshape(n, -1)
        return torch.linalg.solve_triangular(R, x, upper=True).numpy().reshape(np.asarray(v).shape)

    def inverse_transpose(v):
        x = torch.as_tensor(v, dtype=torch.float64).reshape(n, -1)
        return torch.linalg.solve_triangular(R.mT, x, upper=False).numpy().reshape(np.asarray(v).shape)

    Ri = LinearOperator((n, n), matvec=inverse, rmatvec=inverse_transpose,
                        matmat=inverse, rmatmat=inverse_transpose, dtype=np.float64)
    np.random.seed(20261105)
    inverse_norm_estimate = float(onenormest(Ri))
    R_norm = float(R.abs().sum(dim=0).max())
    threshold = max(R.shape) * np.finfo(float).eps * maximum
    return dict(method=method, columns=n, sigma_max=maximum, sigma_min=minimum,
                condition_2_estimate=maximum / minimum,
                largest_eigenpair_relative_residual=residual_max,
                inverse_largest_eigenpair_relative_residual=residual_min,
                R_condition_1_lower_estimate=R_norm * inverse_norm_estimate,
                rank_estimate=(n if minimum > threshold else None), rank_threshold=threshold,
                rank_note="full-rank estimate only when smallest converged estimate exceeds threshold")


report = dict(
    schema="dpanel.wake_sensitivity." + args.phase + ".v1", case=args.case,
    frozen_initial_state_sha256=baseline_manifest["initial_state_sha256"],
    settings=baseline_manifest["settings"], baseline=state0, initial_geometry=geometry0,
    spacing=spacing(mesh), attachment_ids=attached.tolist(), prepared_root_ids=top["graph_roots"].tolist(),
    row_definition="unweighted mesh-edge BFS distance from all initially wall-coincident TE vertices; row 0 is attachment",
    regions={k: v.tolist() for k, v in regions.items()}, rows=[], derivatives=[],
    physical_accuracy_accepted=False, status="running", command=[sys.executable, *sys.argv],
    source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (Path(__file__), root / "tools/wake_sensitivity_geometry.py")})
report["operator_definition"] = dict(
    A="body-only operator; unchanged under wake-coordinate perturbations",
    W="moving wake-to-wall coupling; reassembled",
    B="combined wall block [A,W]; changes with W",
    M="sqrt(area/mean(area)) * [A,W] * null(J); changes with W",
    unchanged="A,J,K,grad and wall RHS",
    legacy_field_note="in early raw receipts wall_operator_unchanged denotes body A only, never combined B or M")
write_report()
if args.phase == "e1":
    for region, ids in regions.items():
        for direction_name, direction in directions.items():
            scale = float(np.linalg.norm(direction[ids], axis=1).max()) if len(ids) else 0.
            if scale < 1e-15:
                report["rows"].append(dict(region=region, direction=direction_name,
                                           status="zero_direction_or_empty_region"))
                continue
            movement = np.zeros_like(xyz0)
            movement[ids] = direction[ids] / scale
            for epsilon in (1e-5, 1e-4, 1e-3, 1e-2):
                pair = []
                for sign in (-1, 1):
                    name = "%s_%s_%g_%+d" % (region, direction_name, epsilon, sign)
                    changed = with_wake_xyz(mesh, {mesh.groups[0].name:
                                                   xyz0 + sign * epsilon * mesh.reference_chord * movement})
                    try:
                        row, state, ops = solve_record(changed, name)
                        row.update(region=region, direction=direction_name, epsilon_chord=epsilon,
                                   sign=sign, status="completed", elapsed=time.time()-started)
                    except Exception as error:
                        row = dict(name=name, region=region, direction=direction_name,
                                   epsilon_chord=epsilon, sign=sign, status="failed", error=repr(error),
                                   traceback=traceback.format_exc(), elapsed=time.time()-started)
                    report["rows"].append(row)
                    pair.append(row)
                    write_report()
                derivative = dict(region=region, direction=direction_name, epsilon_chord=epsilon,
                                  both_admissible=all(r.get("admissible", False) for r in pair))
                if all(r["status"] == "completed" for r in pair):
                    for quantity in ("CL", "CL_gamma", "CM_y"):
                        derivative["d" + quantity + "_d_epsilon_chord"] = (
                            pair[1][quantity]-pair[0][quantity]) / (2 * epsilon)
                        derivative[quantity + "_symmetric_center_shift"] = (
                            (pair[1][quantity]+pair[0][quantity])/2-state0[quantity])
                report["derivatives"].append(derivative)
                write_report()
                print("%s %s %s eps=%g dCL=%s admitted=%s" % (
                    args.case, region, direction_name, epsilon,
                    derivative.get("dCL_d_epsilon_chord"), derivative["both_admissible"]), flush=True)
else:
    selected = [("baseline", xyz0)]
    for region in ("attachment", "near_TE", "downstream"):
        for direction_name in ("normal", "actual"):
            ids = regions[region]
            direction = directions[direction_name]
            scale = float(np.linalg.norm(direction[ids], axis=1).max()) if len(ids) else 0.
            if scale < 1e-15:
                continue
            movement = np.zeros_like(xyz0)
            movement[ids] = direction[ids] / scale
            for epsilon in (1e-3, 1e-2):
                for sign in (-1, 1):
                    selected.append(("%s_%s_%g_%+d" % (region, direction_name, epsilon, sign),
                                     xyz0 + sign * epsilon * mesh.reference_chord * movement))
    for iteration in (1, 60):
        path = out / ("iteration_%03d.npz" % iteration)
        if path.exists():
            with np.load(path) as data:
                selected.append(("baseline_iteration_%d" % iteration, data["wake_xyz"].copy()))
    for name, coordinates in selected:
        try:
            changed = with_wake_xyz(mesh, {mesh.groups[0].name: coordinates})
            row, state, ops = solve_record(changed, name, geometry=False)
            problem = capture.pop("problem")
            value = torch.as_tensor(np.r_[state["mu"], state["gamma"]], dtype=torch.float64)
            F = 5 * problem.pressure_jacobian(value) @ problem.N
            augmented_R = torch.linalg.qr(torch.vstack((problem._R, F)), mode="r").R
            row.update(status="completed", operators=dict(
                weighted_wall_on_null_J=dict(
                    shape=list(problem._M.shape), spectrum=triangular_spectrum(problem._R),
                    row_norm_range=[float(problem._M.norm(dim=1).min()), float(problem._M.norm(dim=1).max())],
                    column_norm_range=[float(problem._M.norm(dim=0).min()), float(problem._M.norm(dim=0).max())]),
                pressure_residual_jacobian_on_null_J=dict(
                    shape=[int(problem._M.shape[0] + F.shape[0]), int(F.shape[1])],
                    spectrum=triangular_spectrum(augmented_R))),
                scaling=dict(wall_rows="sqrt(area / mean(area))", pressure_rows=5.,
                             column_scaling="orthonormal null(J), no additional scaling",
                             preconditioner="production QR wall factor for structured pressure update"),
                constraints=dict(J_shape=list(ops["J"].shape), J_rcond=1e-11,
                                 constrained_dofs=int(problem.N.shape[0]-problem.N.shape[1]),
                                 seed_signed_K_rank=state["linear_constrained_rank"]),
                elapsed=time.time()-started)
        except Exception as error:
            row = dict(name=name, status="failed", error=repr(error), traceback=traceback.format_exc())
        report["rows"].append(row)
        write_report()
        print("%s e4 %s %s" % (args.case, name, row["status"]), flush=True)
report.update(status="completed", seconds=time.time()-started,
              failures=sum(row["status"] == "failed" for row in report["rows"]))
write_report()
