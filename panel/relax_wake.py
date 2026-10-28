"""Research steady material-wake iteration, with all iterates and residuals retained."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--surface", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--alpha", type=float, default=5.)
parser.add_argument("--iterations", type=int, default=8)
parser.add_argument("--relaxation", type=float, default=.25)
parser.add_argument("--update", choices=("line", "face-lsq"), default="line")
parser.add_argument("--tolerance", type=float, default=1e-3)
parser.add_argument("--ncpu", type=int, default=8)
parser.add_argument("--audit-python", type=Path, required=True)
args = parser.parse_args()
args.surface, args.out = args.surface.resolve(), args.out.resolve()
assert not args.out.exists(), "Keep earlier experiments"
assert 0 < args.relaxation <= 1 and args.iterations >= 0 and args.tolerance > 0
args.out.mkdir(parents=True)
from threadpoolctl import threadpool_limits
threadpool_limits(args.ncpu)
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "panel"))
from wake_geometry import check_wake_geometry

started = time.perf_counter()
fixture = json.loads(args.surface.read_text(encoding="utf-8"))
assert "smooth_wall_contact" not in fixture["cut_kind"], "This first prototype holds TE roots only; rebuild contact fluid cuts separately"
native = (np.asarray(fixture["coordinates"]) - fixture["origin"]) / fixture["reference_chord"]
faces, wake_faces = np.asarray(fixture["faces"]), np.asarray(fixture["wake_faces"])
wall_nodes, wake_nodes = np.unique(faces), np.unique(wake_faces)
te_nodes = np.unique(fixture["trailing_edges"])
te_nodes = te_nodes[np.argsort(native[te_nodes, 1])]
root_x = np.interp(native[:, 1], native[te_nodes, 1], native[te_nodes, 0])
base = native.copy()
base[wake_nodes, 2] += np.tan(np.deg2rad(args.alpha)) * (native[wake_nodes, 0] - root_x[wake_nodes])
base[wall_nodes] = native[wall_nodes]
end_x = native[wake_nodes, 0].max()
hinge_x = float(native[wake_nodes, 0][native[wake_nodes, 0] < end_x - 1e-10].max())
rows = []
for eta in np.unique(native[wake_nodes, 1]):
    row = wake_nodes[abs(native[wake_nodes, 1] - eta) < 1e-10]
    if len(row) > 1:
        row = row[np.argsort(native[row, 0])]
        assert row[0] in te_nodes and np.all(np.diff(native[row, 0]) > 0), "Require explicit x-monotone material lines from TE"
        assert abs(native[row[-2], 0] - hinge_x) < 1e-10 and abs(native[row[-1], 0] - end_x) < 1e-10, "Far hinge must be an explicit common material-line row"
        rows.append(row)
assert np.array_equal(np.sort(np.concatenate(rows)[~np.isin(np.concatenate(rows), wall_nodes)]),
                      np.setdiff1d(wake_nodes, wall_nodes)), "All movable wake nodes need a material line"
if args.update == "face-lsq":
    from scipy.linalg import qr, solve_triangular
    # At fixed material eta, B is the affine shape derivative with respect to x.
    ref = native[wake_faces]
    determinant = ((ref[:, 1, 0] - ref[:, 0, 0]) * (ref[:, 2, 1] - ref[:, 0, 1])
                   - (ref[:, 2, 0] - ref[:, 0, 0]) * (ref[:, 1, 1] - ref[:, 0, 1]))
    shape_x = np.column_stack((ref[:, 1, 1] - ref[:, 2, 1], ref[:, 2, 1] - ref[:, 0, 1],
                               ref[:, 0, 1] - ref[:, 1, 1])) / determinant[:, None]
    B = np.zeros((len(wake_faces), len(native)))
    B[np.arange(len(wake_faces))[:, None], wake_faces] = shape_x
    free = np.setdiff1d(wake_nodes[native[wake_nodes, 0] <= hinge_x + 1e-10], wall_nodes)
    embedding = np.zeros((len(native), len(free)))
    embedding[free, np.arange(len(free))] = 1.
    fixed = np.zeros((len(native), 2))
    fixed[wall_nodes] = native[wall_nodes, 1:]
    for row in rows:
        far = row[native[row, 0] > hinge_x + 1e-10]
        anchor = row[native[row, 0] <= hinge_x + 1e-10][-1]
        embedding[far] = embedding[anchor]
        fixed[far] = (native[far, 0] - native[anchor, 0])[:, None] * [0., np.tan(np.deg2rad(args.alpha))]
    near = ref.mean(axis=1)[:, 0] <= hinge_x
    shape_free, shape_fixed = (B @ embedding)[near], (B @ fixed)[near]
offset = np.zeros_like(native)
history, commands, step_attempts = [], [], []
summary = dict(status="running", formulation="Fixed-x three-dimensional material-wake Picard iteration", update=args.update,
               initial_wake="Freestream", far_hinge_x=hinge_x, relaxation=args.relaxation,
               tolerance=args.tolerance, physical_accuracy_accepted=False,
               derivative_scope="No derivative through the nonlinear iteration",
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(args.out / "relax_wake.py.snapshot").write_bytes(Path(__file__).read_bytes())
(args.out / "surface_input.json").write_bytes(args.surface.read_bytes())
try:
    for iteration in range(args.iterations + 1):
        directory = args.out / f"iteration_{iteration:02d}"
        offset_path = args.out / f"offset_{iteration:02d}.npz"
        np.savez_compressed(offset_path, offset=offset)
        command = [sys.executable, str(root / "panel/solve.py"), "--surface", str(args.surface),
                   "--out", str(directory), "--alpha", str(args.alpha), "--wake-mode", "freestream",
                   "--wake-offset", str(offset_path), "--ncpu", str(args.ncpu)]
        process = subprocess.run(command, cwd=root, capture_output=True, text=True)
        (args.out / f"solve_{iteration:02d}.log").write_text(process.stdout + process.stderr, encoding="utf-8")
        commands.append(dict(command=command, returncode=process.returncode))
        assert process.returncode == 0, "Solve failed; preserve the iterate and log"
        command = [str(args.audit_python), str(root / "panel/audit.py"), str(directory),
                   "--wake-samples", "-1", "--ncpu", str(args.ncpu)]
        process = subprocess.run(command, cwd=root, capture_output=True, text=True)
        (args.out / f"audit_{iteration:02d}.log").write_text(process.stdout + process.stderr, encoding="utf-8")
        commands.append(dict(command=command, returncode=process.returncode))
        assert process.returncode == 0, "Independent field audit failed"
        field = np.load(directory / "fields.npz")
        audit_field = np.load(directory / "independent_field.npz")
        audit = json.loads((directory / "independent_audit.json").read_text())
        result = json.loads((directory / "result.json").read_text())
        points = field["coordinates"]
        assert np.max(abs(points - base - offset)) < 1e-12
        velocity = audit_field["wake_mean_velocity"]
        assert np.isfinite(velocity).all() and len(velocity) == len(wake_faces)
        target = points.copy()
        qr_ratio = None
        if args.update == "line":
            # Area-weighted projection avoids evaluating singular panel-edge fields.
            node_velocity, weight = np.zeros_like(points), np.zeros(len(points))
            for corner in range(3):
                np.add.at(node_velocity, wake_faces[:, corner], velocity * field["wake_area"][:, None])
                np.add.at(weight, wake_faces[:, corner], field["wake_area"])
            node_velocity[wake_nodes] /= weight[wake_nodes, None]
            for row in rows:
                for upstream, downstream in zip(row[:-1], row[1:]):
                    dx = native[downstream, 0] - native[upstream, 0]
                    if native[downstream, 0] > hinge_x + 1e-10:
                        slope = np.array([0., np.tan(np.deg2rad(args.alpha))])
                    else:
                        v = (node_velocity[upstream] + node_velocity[downstream]) / 2
                        assert v[0] > 0, "Reverse flow requires another wake parameterization"
                        slope = v[1:] / v[0]
                    target[downstream, 1:] = target[upstream, 1:] + dx * slope
        else:
            # Frozen face velocities: min sum A*|v_x*X_x|eta - v|^2 in y,z.
            # Eliminate fixed roots and the prescribed far continuation exactly.
            assert np.all(velocity[near, 0] > 0), "Reverse flow requires another wake parameterization"
            weight = np.sqrt(field["wake_area"][near])
            matrix = (weight * velocity[near, 0])[:, None] * shape_free
            rhs = weight[:, None] * (velocity[near, 1:] - velocity[near, 0, None] * shape_fixed)
            q, r, pivot = qr(matrix, mode="economic", pivoting=True)
            qr_ratio = float(np.min(abs(np.diag(r))) / np.max(abs(np.diag(r))))
            assert qr_ratio > 1e-10, "Material geometry system unresolved; do not truncate or regularize"
            values = np.empty((len(free), 2))
            values[pivot] = solve_triangular(r, q.T @ rhs)
            target[:, 1:] = fixed + embedding @ values
            target[np.setdiff1d(np.arange(len(native)), wake_nodes)] = native[np.setdiff1d(np.arange(len(native)), wake_nodes)]
            target[wall_nodes] = native[wall_nodes]
        residual = target - points
        movable = np.setdiff1d(wake_nodes, wall_nodes)
        correction = np.linalg.norm(residual[movable], axis=1)
        row_result = dict(iteration=iteration, CL=result["CL"], CD_pressure=result["CD_pressure"],
                          geometry_qr_diagonal_ratio=qr_ratio,
                          node_fixed_point_max=float(correction.max()), node_fixed_point_rms=float(np.sqrt(np.mean(correction**2))),
                          wake_tangency=audit["wake_tangency"], wall_normal_rms=audit["independent_field"]["wall_normal_rms"],
                          maximum_offset=float(np.linalg.norm(offset, axis=1).max()))
        history.append(row_result)
        print(json.dumps(row_result), flush=True)
        converged = (correction.max() <= args.tolerance
                     and audit["wake_tangency"]["near"]["total_normal_rms"] <= args.tolerance
                     and audit["wake_tangency"]["near"]["steady_pressure_jump_rms"] <= args.tolerance)
        summary.update(history=history, commands=commands, final_directory=str(directory),
                       status="converged_on_declared_metrics" if converged else "iteration_budget_exhausted")
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if converged or iteration == args.iterations:
            break
        accepted = False
        for reduction in range(9):
            step = args.relaxation / 2**reduction
            proposed = points + step * residual
            proposed[wall_nodes] = native[wall_nodes]
            np.savez_compressed(args.out / f"candidate_{iteration:02d}_{reduction:02d}.npz", coordinates=proposed)
            try:
                certificate = check_wake_geometry(proposed, faces, wake_faces, require_positive_z=False)
                step_attempts.append(dict(iteration=iteration, step=step, accepted=True, geometry=certificate))
                accepted = True
                break
            except AssertionError as error:
                step_attempts.append(dict(iteration=iteration, step=step, accepted=False, reason=str(error)))
        summary["step_attempts"] = step_attempts
        assert accepted, "No admissible geometric step; retain the failed candidates"
        offset = proposed - base
        offset[wall_nodes] = 0.
except Exception as error:
    summary.update(status="failed", error=type(error).__name__, message=str(error), history=history, commands=commands)
    raise
finally:
    summary["elapsed_seconds"] = time.perf_counter() - started
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
