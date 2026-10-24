"""Experimental streamline-graph wake: raw flow solve, material jump and explicit mesh-motion checks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent/"volume"))
import p2
from wake_velocity import sample_velocity
from wake_transport import integrate_displacement, extend_displacement
from mesh_motion import prepare_harmonic, extend_harmonic

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mesh", type=Path, required=True)
parser.add_argument("--alpha", type=float, default=5.)
parser.add_argument("--mach", type=float, default=0.)
parser.add_argument("--kutta", choices=["node", "weak"], default="weak")
parser.add_argument("--ncpu", type=int, default=8)
parser.add_argument("--maxiter", type=int, default=8)
parser.add_argument("--relaxation", type=float, default=.35)
parser.add_argument("--nx", type=int, default=41)
parser.add_argument("--neta", type=int, default=49)
parser.add_argument("--mesh-motion", choices=["blend", "harmonic"], default="blend")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
assert not args.out.exists(), "Preserve previous wake experiments"
args.out.mkdir(parents=True)
started = time.perf_counter()
source_bytes = {name: (root/name).read_bytes() for name in
                ["relax_wake.py", "wake_transport.py", "wake_velocity.py", "mesh_motion.py", "solve.py", "PROTOCOL.md"]}
history, commands = [], []
status, failure = "wake_iteration_limit", None
try:
    assert args.ncpu > 0 and args.maxiter > 0 and args.nx > 2 and args.neta > 2
    assert 0. < args.relaxation <= 1.
    threadpool_limits(args.ncpu)
    original = dict(np.load(args.mesh))
    metadata = json.loads(args.mesh.with_name("mesh.json").read_text())
    assert metadata.get("wake_strips", 0) > 0, "A conforming material-strip wake is required"
    assert metadata.get("taper_ratio", 1.) == 1. and metadata.get("quarter_chord_sweep_deg", 0.) == 0., "Graph-wake transport currently requires a rectangular planform"
    assert metadata.get("wake_tilt_deg", 0.) == 0., "Start from an undeformed flat reference wake"
    assert not metadata.get("wake_material_coordinates", False), "Restarting a deformed wake is not implemented"
    length = float(metadata["chord"])
    origin = np.array(metadata.get("normalization_origin", [0., 0., 0.]))
    reference_points = (original.get("reference_points", original["points"])-origin)/length
    assert np.max(abs(reference_points-(original["points"]-origin)/length)) < 1e-12, "Input must be the undeformed reference"
    radius, span = metadata["far_radius"]/length, metadata["span"]/length
    x = 1.+(.5+radius-1.)*np.linspace(0., 1., args.nx)**1.5
    eta = np.linspace(-span/2., span/2., args.neta)
    displacement = np.zeros((len(x), len(eta), 2))
    if args.mesh_motion == "harmonic":
        fixed, free, mesh_factor, mesh_coupling = prepare_harmonic(reference_points, original)
    solid_nodes = np.unique(original["boundary_triangles"][original["boundary_tags"] != 100])
    lattice = np.array([[i, j, k, 8-i-j-k] for i in range(9)
                        for j in range(9-i) for k in range(9-i-j)])/8.
    _, derivative = p2.p2_shape(lattice)
    for iteration in range(args.maxiter):
        target = args.out/f"{iteration:02d}"
        (target/"mesh").mkdir(parents=True)
        mesh = dict(original)
        mesh["reference_points"] = reference_points*length+origin
        points = extend_displacement(reference_points, x, eta, displacement, radius, span)
        if args.mesh_motion == "harmonic":
            points = extend_harmonic(reference_points, points, fixed, free, mesh_factor, mesh_coupling)
        assert points.shape == reference_points.shape and np.isfinite(points).all()
        solid_error = float(np.max(abs(points[solid_nodes]-reference_points[solid_nodes])))
        assert solid_error < 1e-12, "Wake update moved the solid geometry"
        minimum = np.inf
        for first in range(0, len(mesh["tets"]), 256):
            jacobian = np.einsum("eia,qib->eqab", points[mesh["tets"][first:first+256]], derivative)
            minimum = min(minimum, float(np.linalg.det(jacobian).min()))
        assert minimum > 0., "Wake update produced a nonpositive dense-sampled Jacobian"
        mesh["points"] = points*length+origin
        mesh["wake_grid_x"], mesh["wake_grid_eta"] = x, eta
        mesh["wake_displacement_grid"] = displacement
        mesh_path = target/"mesh"/"mesh.npz"
        np.savez_compressed(mesh_path, **mesh)
        current_metadata = dict(metadata)
        current_metadata.update(wake_material_coordinates=True,
            wake_model="Experimental material-coordinate streamline graph; Picard geometry iteration, no rollup certification",
            jump_representation="Piecewise linear in undeformed material eta; transported with the conforming mesh",
            min_normalized_dense_jacobian=minimum, wake_geometry_iteration=iteration,
            mesh_sha256=hashlib.sha256(mesh_path.read_bytes()).hexdigest())
        (target/"mesh"/"mesh.json").write_text(json.dumps(current_metadata, indent=2), encoding="utf-8")
        command = [sys.executable, str(root/"solve.py"), "--mesh", str(mesh_path),
                   "--alpha", str(args.alpha), "--mach", str(args.mach),
                   "--kutta", args.kutta, "--ncpu", str(args.ncpu), "--out", str(target/"solve")]
        run = subprocess.run(command, capture_output=True, text=True)
        (target/"solve.log").write_text(run.stdout+run.stderr, encoding="utf-8")
        commands.append(dict(command=command, returncode=run.returncode))
        result = json.loads((target/"solve"/"result.json").read_text())
        if run.returncode or not result["converged"]:
            status, failure = "flow_rejected", result.get("message", result["status"])
            break
        field = dict(np.load(target/"solve"/"field.npz"))
        up, low = field["tags"] == 201, field["tags"] == 202
        distance = np.linalg.norm(field["surface_points"][up, :, None]
                                  -field["surface_points"][low, None, :], axis=3)
        paired = distance.argmin(axis=2)
        assert distance.min(axis=2).max()/length < 1e-10
        lower_velocity = np.take_along_axis(field["surface_velocity"][low], paired[:, :, None], axis=1)
        average_velocity = .5*(field["surface_velocity"][up]+lower_velocity)
        normal_speed = np.sum(average_velocity*field["surface_normal"][up], axis=2)
        tangency_rms = float(np.sqrt(np.average(normal_speed**2, weights=field["surface_weight"][up])))
        velocity, recovery = sample_velocity(mesh, field, x, eta, origin, length)
        predicted = integrate_displacement(x, eta, velocity)
        fixed_point_error = float(np.max(abs(predicted-displacement)))
        history.append(dict(iteration=iteration, tangency_rms=tangency_rms,
                            tangency_max=float(np.max(abs(normal_speed))),
                            tangency_mean=float(np.average(normal_speed, weights=field["surface_weight"][up])),
                            displacement_fixed_point_max=fixed_point_error,
                            cl=result["cl_pressure"], cl_jump=result["cl_jump"], cd=result["cd_pressure"],
                            wall_rms=result["wall_normal_rms"], wall_max=result["wall_normal_max"],
                            wake=result["wake_diagnostics"], recovery=recovery,
                            min_dense_jacobian=minimum, solid_coordinate_error=solid_error,
                            displacement_max=float(np.max(abs(displacement))),
                            solver_sha256=result["solver_sha256"], mesh_sha256=result["mesh_sha256"]))
        (args.out/"progress.json").write_text(json.dumps(dict(history=history, commands=commands), indent=2), encoding="utf-8")
        print(json.dumps(history[-1]), flush=True)
        if tangency_rms < .01 and result["wake_diagnostics"]["raw_pressure_jump_rms"] < .01 and fixed_point_error < .001:
            status = "graph_wake_discrete_conditions_converged_unvalidated"
            break
        if iteration == args.maxiter-1:
            break
        step = args.relaxation
        for backtrack in range(12):
            candidate = displacement+step*(predicted-displacement)
            candidate_points = extend_displacement(reference_points, x, eta, candidate, radius, span)
            if args.mesh_motion == "harmonic":
                candidate_points = extend_harmonic(reference_points, candidate_points, fixed, free, mesh_factor, mesh_coupling)
            candidate_minimum = np.inf
            for first in range(0, len(mesh["tets"]), 256):
                jacobian = np.einsum("eia,qib->eqab", candidate_points[mesh["tets"][first:first+256]], derivative)
                candidate_minimum = min(candidate_minimum, float(np.linalg.det(jacobian).min()))
            if candidate_minimum > 0.:
                break
            step *= .5
        assert candidate_minimum > 0., "No admissible mesh-motion step"
        history[-1]["accepted_geometry_relaxation"] = step
        displacement = candidate
except Exception as exception:
    status, failure = "failed_exception", f"{type(exception).__name__}: {exception}"
summary = dict(status=status, converged=status == "graph_wake_discrete_conditions_converged_unvalidated",
               failure=failure, history=history, commands=commands, seconds=time.perf_counter()-started,
               input_mesh_sha256=hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
               source_sha256={name: hashlib.sha256(data).hexdigest() for name, data in source_bytes.items()},
               mesh_motion=args.mesh_motion,
               acceptance="Raw mean-trace wake tangency RMS<.01, raw pressure jump RMS<.01, displacement fixed point max<.001 chord",
               scope="Experimental graph wake on existing topology. Physical accuracy and rollup are not certified.")
(args.out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
for name, contents in source_bytes.items():
    (args.out/(name+".snapshot")).write_bytes(contents)
print(json.dumps(summary, indent=2))
if not summary["converged"]:
    raise SystemExit(2)
