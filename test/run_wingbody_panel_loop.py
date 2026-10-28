"""Fixed-topology thick wing/fuselage panel loop using the locked VortexAD probe.

Research driver: raw Cp -> one panel pressure integration -> component loads ->
AD and three centered-difference steps. No changes to aero.py or P2 defaults.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--surface", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--backend", type=Path, default=Path("runs/backend_probes/src/VortexAD"))
parser.add_argument("--ncpu", type=int, default=8)
parser.add_argument("--alpha", type=float, default=5.)
parser.add_argument("--bulge", type=float, default=0.)
parser.add_argument("--span", type=float, default=1.)
parser.add_argument("--gradient-rings", type=int, default=2)
parser.add_argument("--forward-only", action="store_true")
parser.add_argument("--upstream-pressure", action="store_true", help="Reproduce the original rank-deficient pressure reconstruction")
args = parser.parse_args()
args.out = args.out.resolve()
assert args.ncpu > 0 and args.span > 0 and args.bulge > -1 and args.gradient_rings >= 1
assert not args.out.exists(), "Preserve earlier experiments"
args.out.mkdir(parents=True)
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
            "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[key] = str(args.ncpu)
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["MPLCONFIGDIR"] = str((args.out / "mpl").resolve())
os.environ["TMP"] = os.environ["TEMP"] = str((args.out / "tmp").resolve())
(args.out / "tmp").mkdir()
sys.path.insert(0, str(args.backend.resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
import scipy
from scipy.sparse import csr_matrix
from scipy.linalg import lu_factor
from scipy.linalg.lapack import dgecon
import csdl_alpha as csdl
import jax
from VortexAD import PanelMethod

jax.config.update("jax_enable_x64", True)
threadpool_limits(args.ncpu)
started = time.perf_counter()
fixture = json.loads(args.surface.read_text(encoding="utf-8"))
xyz = np.asarray(fixture["coordinates"], dtype=np.float64)
faces = np.asarray(fixture["faces"], dtype=np.int64)
tags = np.asarray(fixture["component_ids"], dtype=np.int64)
edges = np.asarray(fixture["trailing_edges"], dtype=np.int64)
upper, lower = fixture["upper_te_faces"], fixture["lower_te_faces"]
seam_faces = fixture["seam_faces"]
sref, cref = fixture["reference_area"], fixture["reference_chord"]
reference = np.asarray(fixture["moment_reference"])
root = Path(__file__).resolve().parents[1]
snapshot_names = ["test/run_wingbody_panel_loop.py", "test/build_wingbody_panel_case.py"]
source_hashes = {}
for name in snapshot_names:
    content = (root / name).read_bytes()
    (args.out / (Path(name).name + ".snapshot")).write_bytes(content)
    source_hashes[name] = hashlib.sha256(content).hexdigest()
backend_files = {
    str(path.relative_to(args.backend)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted((args.backend / "VortexAD").rglob("*.py"))
}
(args.out / "backend_source_sha256.json").write_text(json.dumps(backend_files, indent=2) + "\n", encoding="utf-8")
backend_commit = subprocess.check_output(
    ["git", "-C", str(args.backend), "rev-parse", "HEAD"], text=True
).strip()
csdl_source_files = sorted(Path(csdl.__file__).parent.rglob("*.py"))
csdl_source_hash = hashlib.sha256(b"".join(path.read_bytes() for path in csdl_source_files)).hexdigest()
(args.out / "surface_input.json").write_bytes(args.surface.read_bytes())

record = csdl.Recorder(inline=False)
record.start()
alpha = csdl.Variable(value=np.array([args.alpha]), name="alpha_deg")
bulge = csdl.Variable(value=np.array([args.bulge]), name="central_thickness")
span = csdl.Variable(value=np.array([args.span]), name="span_factor")
base = csdl.Variable(value=xyz)
coordinates = csdl.Variable(value=np.zeros_like(xyz))
coordinates = coordinates.set(csdl.slice[:, 0], base[:, 0])
coordinates = coordinates.set(csdl.slice[:, 1], base[:, 1] * span)
coordinates = coordinates.set(
    csdl.slice[:, 2], base[:, 2] * (1 + bulge * csdl.exp(-(base[:, 1] / (0.5 * cref)) ** 2))
)
panel = PanelMethod({
    "V_inf": 1., "Mach": None, "alpha": alpha, "rho": 2.,
    "BC": "Dirichlet", "drag_type": "pressure", "Cp cutoff": None,
    "partition_size": 1, "reuse_AIC": False, "compressibility": False,
    "ref_area": sref, "ref_chord": cref, "moment_reference": reference,
    "nwpcl": 30.,
}, skip_geometry=True)
edge_to_faces = {}
for i, face in enumerate(faces):
    for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
        edge_to_faces.setdefault(tuple(sorted((int(a), int(b)))), []).append(i)
panel.insert_grid_data(
    coordinates,
    (xyz, {"triangle": faces}, {"triangle": fixture["neighbors"]},
     edge_to_faces, dict(enumerate(fixture["points_to_faces"]))),
    (upper, lower, [tuple(edge) for edge in edges.tolist()], np.unique(edges)),
)
names = ["CL", "CDi", "CM", "Cp", "Cp_raw", "panel_forces", "panel_area",
         "panel_center", "panel_normal", "mu", "sigma", "mu_wake", "x_w",
         "AIC_mu", "AIC_sigma", "RHS", "ql", "qm", "Qn", "panel_x_dir", "panel_y_dir"]
panel.declare_outputs(names)
outputs = panel.evaluate()
upstream_outputs = outputs.copy()

# Reconstruct the same surface-potential gradient on a larger, explicit stencil.
# Remove only actual TE adjacency. Wing/body neighbors remain coupled.
cut_pairs = {tuple(sorted(pair)) for pair in zip(upper, lower)}
adjacency = [set(row) for row in fixture["neighbors"]]
for first, second in cut_pairs:
    adjacency[first].remove(second)
    adjacency[second].remove(first)
stencils = []
for i in range(len(faces)):
    reached, frontier = {i}, {i}
    for ring in range(args.gradient_rings):
        frontier = set().union(*(adjacency[j] for j in frontier)) - reached
        reached.update(frontier)
    stencils.append(sorted(reached - {i}))
width = max(map(len, stencils))
neighbor_indices = np.array([row + [i] * (width - len(row)) for i, row in enumerate(stencils)])
flat_neighbors = neighbor_indices.ravel().tolist()
gather = csr_matrix((np.ones(len(flat_neighbors)), (np.arange(len(flat_neighbors)), flat_neighbors)),
                    shape=(len(flat_neighbors), len(faces)))
centers_var = outputs["panel_center"]
l_axis, m_axis = outputs["panel_x_dir"], outputs["panel_y_dir"]
delta = csdl.sparse.matmat(gather, centers_var.reshape((len(faces), 3))).reshape((1, len(faces), width, 3))
delta = delta - csdl.expand(centers_var, delta.shape, "ijk->ijak")
dl = csdl.sum(delta * csdl.expand(l_axis, delta.shape, "ijk->ijak"), axes=(3,))
dm = csdl.sum(delta * csdl.expand(m_axis, delta.shape, "ijk->ijak"), axes=(3,))
mu_var = outputs["mu"]
dmu = csdl.sparse.matvec(gather, mu_var.reshape((len(faces),))).reshape((1, len(faces), width))
dmu = dmu - csdl.expand(mu_var, dmu.shape, "ij->ija")
c_ll, c_mm, c_lm = csdl.sum(dl * dl, axes=(2,)), csdl.sum(dm * dm, axes=(2,)), csdl.sum(dl * dm, axes=(2,))
b_l, b_m = csdl.sum(dl * dmu, axes=(2,)), csdl.sum(dm * dmu, axes=(2,))
determinant = c_ll * c_mm - c_lm * c_lm
ql = -(b_l * c_mm - b_m * c_lm) / determinant
qm = -(b_m * c_ll - b_l * c_lm) / determinant
alpha_rad = alpha * np.pi / 180
flow_var = csdl.Variable(value=np.zeros((1, 3)))
flow_var = flow_var.set(csdl.slice[:, 0], csdl.cos(alpha_rad))
flow_var = flow_var.set(csdl.slice[:, 2], csdl.sin(alpha_rad))
flow_grid = csdl.expand(flow_var, centers_var.shape, "ij->iaj")
Q_l = csdl.sum(flow_grid * l_axis, axes=(2,)) + ql
Q_m = csdl.sum(flow_grid * m_axis, axes=(2,)) + qm
Q_n = csdl.sum(flow_grid * outputs["panel_normal"], axes=(2,)) + outputs["sigma"]
raw_cp = 1 - Q_l * Q_l - Q_m * Q_m - Q_n * Q_n
forces_var = -csdl.expand(raw_cp * outputs["panel_area"], centers_var.shape, "ij->ijk") * outputs["panel_normal"]
moment_arm = centers_var - csdl.expand(csdl.Variable(value=reference), centers_var.shape, "i->abi")
moments_var = csdl.cross(moment_arm, forces_var, axis=2)
fz, fx = csdl.sum(forces_var[:, :, 2], axes=(1,)), csdl.sum(forces_var[:, :, 0], axes=(1,))
outputs["CL"] = (fz * csdl.cos(alpha_rad) - fx * csdl.sin(alpha_rad)) / sref
outputs["CDi"] = (fz * csdl.sin(alpha_rad) + fx * csdl.cos(alpha_rad)) / sref
outputs["CM"] = csdl.sum(moments_var, axes=(1,)) / (sref * cref)
outputs["Cp"] = outputs["Cp_raw"] = raw_cp
outputs["panel_forces"], outputs["ql"], outputs["qm"], outputs["Qn"] = forces_var, ql, qm, Q_n
if args.upstream_pressure:
    outputs.update(upstream_outputs)
forces = outputs["panel_forces"]
wing_fz = csdl.sum(forces[:, list(np.flatnonzero(tags == 30)), 2])
body_fz = csdl.sum(forces[:, list(np.flatnonzero(tags == 20)), 2])
seam_fz = csdl.sum(forces[:, seam_faces, 2])
total_my = outputs["CM"][0, 1]
objective_names = ["CL", "wing_Fz", "body_Fz", "seam_Fz", "CM_y"]
objectives = [outputs["CL"], wing_fz, body_fz, seam_fz, total_my]
inputs = [alpha, bulge, span]
input_names = ["alpha_deg", "central_thickness", "span_factor"]
record.stop()
graph_seconds = time.perf_counter() - started
sim = csdl.experimental.JaxSimulator(
    recorder=record, additional_inputs=inputs,
    additional_outputs=list(outputs.values()) + objectives[1:] + [coordinates],
    gpu=False, f64=True,
)
t0 = time.perf_counter()
sim.run()
cold_seconds = time.perf_counter() - t0
t0 = time.perf_counter()
sim.run()
warm_seconds = time.perf_counter() - t0
saved = {name: np.asarray(sim[var]).copy() for name, var in outputs.items()}
saved["coordinates"] = np.asarray(sim[coordinates]).copy()
saved["faces"], saved["component_ids"] = faces, tags
objective_base = np.array([np.asarray(sim[var]).item() for var in objectives])
np.savez_compressed(args.out / "fields.npz", **saved)
nonfinite = {name: int(np.count_nonzero(~np.isfinite(value))) for name, value in saved.items()
             if not np.isfinite(value).all()}
if nonfinite:
    failure = dict(status="nonfinite_fields", upstream_pressure=args.upstream_pressure,
                   nonfinite_counts=nonfinite, source_sha256=source_hashes,
                   backend_commit=backend_commit, surface_sha256=hashlib.sha256(args.surface.read_bytes()).hexdigest())
    (args.out / "failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(failure, indent=2), flush=True)
assert not nonfinite, "Raw failed fields saved; no clipping or regularization applied"
assert np.array_equal(saved["Cp"], saved["Cp_raw"]), "Raw pressure must feed the loads"

# Independently integrate the exported raw Cp, with one common reference point.
centers, normals = saved["panel_center"].squeeze(), saved["panel_normal"].squeeze()
area, cp = saved["panel_area"].reshape(-1), saved["Cp_raw"].reshape(-1)
triangles_actual = saved["coordinates"][faces]
triangle_quality = 4 * np.sqrt(3) * area / np.sum(
    (triangles_actual - triangles_actual[:, [1, 2, 0]]) ** 2, axis=(1, 2)
)
actual_mesh = dict(
    minimum_triangle_quality=float(triangle_quality.min()),
    minimum_area=float(area.min()),
    volume=float(np.sum(centers * normals * area[:, None]) / 3),
    normalized_vector_area_closure=float(np.linalg.norm((normals * area[:, None]).sum(axis=0)) / area.sum()),
)
assert actual_mesh["volume"] > 0 and actual_mesh["minimum_area"] > 0
assert actual_mesh["normalized_vector_area_closure"] < 1e-12
pressure_force = -cp[:, None] * normals * area[:, None]  # rho=2, U=1, q=1
moment = np.cross(centers - reference, pressure_force)
component_loads = {}
for name, tag in (("wing", 30), ("body", 20)):
    selection = tags == tag
    component_loads[name] = dict(force=pressure_force[selection].sum(axis=0).tolist(),
                                 moment=moment[selection].sum(axis=0).tolist(),
                                 cp_min=float(cp[selection].min()), cp_max=float(cp[selection].max()))
force, total_moment = pressure_force.sum(axis=0), moment.sum(axis=0)
angle = np.deg2rad(args.alpha)
flow = np.array([np.cos(angle), 0., np.sin(angle)])
lift_axis = np.array([-np.sin(angle), 0., np.cos(angle)])
independent_cl, independent_cd = float(force @ lift_axis / sref), float(force @ flow / sref)
A, rhs, mu = saved["AIC_mu"].squeeze(), saved["RHS"].reshape(-1), saved["mu"].reshape(-1)
lu, piv = lu_factor(A)
rcond, info = dgecon(lu, np.linalg.norm(A, 1))
assert info == 0
closure = dict(
    exported_panel_force_max_abs=float(np.max(abs(pressure_force - saved["panel_forces"].squeeze()))),
    CL_abs=float(abs(independent_cl - saved["CL"].item())),
    CD_pressure_abs=float(abs(independent_cd - saved["CDi"].item())),
    CM_max_abs=float(np.max(abs(total_moment / (sref * cref) - saved["CM"].reshape(3)))),
    component_force_max_abs=float(np.max(abs(
        np.sum([v["force"] for v in component_loads.values()], axis=0) - force))),
    component_moment_max_abs=float(np.max(abs(
        np.sum([v["moment"] for v in component_loads.values()], axis=0) - total_moment))),
)
result = dict(
    status="forward_complete", surface_sha256=hashlib.sha256(args.surface.read_bytes()).hexdigest(),
    source_sha256=source_hashes, mesh=fixture["audit"], actual_mesh=actual_mesh, ncpu_requested=args.ncpu,
    environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                     jax=jax.__version__, csdl_alpha=getattr(csdl, "__version__", "unknown"),
                     backend=str(args.backend.resolve()), backend_commit=backend_commit,
                     backend_source_manifest="backend_source_sha256.json",
                     csdl_python_source_sha256=csdl_source_hash, threadpools=threadpool_info()),
    method="VortexAD locked source-doublet Dirichlet + fixed freestream-aligned wake; no Cp cutoff",
    references=dict(area=sref, chord=cref, moment=reference.tolist(), density=2., speed=1., dynamic_pressure=1.),
    parameters=dict(zip(input_names, [args.alpha, args.bulge, args.span])),
    timings=dict(graph_seconds=graph_seconds, compile_and_first_seconds=cold_seconds,
                 warm_forward_seconds=warm_seconds),
    CL=independent_cl, CD_pressure=independent_cd, CM=(total_moment / (sref * cref)).tolist(),
    force=force.tolist(), moment=total_moment.tolist(), components=component_loads,
    cp_min=float(cp.min()), cp_max=float(cp.max()), cp_below_minus5=int(sum(cp < -5)),
    load_closure=closure, raw_cp_equal_used_cp=bool(np.array_equal(saved["Cp"], saved["Cp_raw"])),
    linear_true_relative_residual=float(np.linalg.norm(A @ mu - rhs) / np.linalg.norm(rhs)),
    condition_1_estimate=float(1 / rcond),
    normalized_source_flux=float(np.sum(area * saved["sigma"].reshape(-1)) / area.sum()),
    physical_accuracy_accepted=False,
    scope="One fixed-topology joined wing/body family; CAD Boolean/remeshing is not differentiated",
    upstream_numerics="Original fixed collocation offset / kernel small constants retained; not claimed removed",
    pressure_reconstruction=dict(method="Unweighted affine least squares in each panel tangent plane",
                                 rings=args.gradient_rings, max_neighbors=width,
                                 cutoff=None, regularization=None, excluded_edges="Declared TE only",
                                 stencil_min_neighbors=min(map(len, stencils))),
    objective_values=dict(zip(objective_names, objective_base.tolist())),
)
(args.out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, allow_nan=False), flush=True)
assert max(closure.values()) < 1e-10
assert result["linear_true_relative_residual"] < 1e-10

if not args.forward_only:
    # A separate scalar-output simulator avoids a full panel-pressure Jacobian.
    derivative_sim = csdl.experimental.JaxSimulator(
        recorder=record, additional_inputs=inputs, additional_outputs=objectives, gpu=False, f64=True,
    )
    t0 = time.perf_counter()
    derivatives = derivative_sim.compute_totals()
    result["timings"]["derivative_compile_and_first_seconds"] = time.perf_counter() - t0
    derivative_rows = []
    base_values = [args.alpha, args.bulge, args.span]
    for parameter_index, variable in enumerate(inputs):
        ad = np.array([np.asarray(derivatives[(objective, variable)]).item() for objective in objectives])
        assert np.isfinite(ad).all()
        for step in (1e-3, 1e-4, 1e-5):
            sim[variable] = np.array([base_values[parameter_index] + step])
            sim.run()
            plus = np.array([np.asarray(sim[obj]).item() for obj in objectives])
            sim[variable] = np.array([base_values[parameter_index] - step])
            sim.run()
            minus = np.array([np.asarray(sim[obj]).item() for obj in objectives])
            fd = (plus - minus) / (2 * step)
            absolute = abs(ad - fd)
            relative = absolute / np.maximum(abs(ad), 1e-10)
            passed = np.all(absolute <= 2e-7 + 2e-4 * abs(ad))
            row = dict(parameter=input_names[parameter_index], step=step, objectives=objective_names,
                       ad=ad.tolist(), fd=fd.tolist(), abs_error=absolute.tolist(),
                       relative_error=relative.tolist(), passed=bool(passed))
            derivative_rows.append(row)
            print(json.dumps(row), flush=True)
        sim[variable] = np.array([base_values[parameter_index]])
    result["geometry_and_angle_derivatives"] = derivative_rows
    result["derivatives_passed"] = all(row["passed"] for row in derivative_rows)
    result["status"] = "loop_complete" if result["derivatives_passed"] else "derivative_mismatch"
    (args.out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    assert result["derivatives_passed"], "Keep the failed directional derivative evidence"
print("Completed:", args.out, flush=True)
