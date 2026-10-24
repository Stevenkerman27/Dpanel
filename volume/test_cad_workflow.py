"""Create translated/scaled STEP spheres and an open surface; verify the complete closed-CAD path."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent/"audit"/"runtime"/"python"))
import gmsh
import p2

out = root/"results"/"cad_workflow_v1"
assert not out.exists(), "Preserve previous CAD evidence"
out.mkdir(parents=True)
commands, records = [], []
for name, scale in [("sphere", 1.), ("sphere_large", 1000.), ("open_plane", 1.)]:
    target = out/name
    target.mkdir()
    gmsh.initialize()
    gmsh.model.add(name)
    if name == "open_plane":
        gmsh.model.occ.addRectangle(0., 0., 0., 1., 1.)
    else:
        gmsh.model.occ.addSphere(2.*scale, -3.*scale, scale, scale)
    gmsh.model.occ.synchronize()
    gmsh.write(str(target/"input.step"))
    gmsh.finalize()
    command = [sys.executable, str(root/"build_cad_mesh.py"), "--cad", str(target/"input.step"),
               "--size", ".12", "--out", str(target/"mesh")]
    run = subprocess.run(command, capture_output=True, text=True)
    (target/"mesh.log").write_text(run.stdout+run.stderr, encoding="utf-8")
    commands.append(dict(command=command, returncode=run.returncode))
    if name == "open_plane":
        failure = json.loads((target/"mesh"/"failure.json").read_text())
        records.append(dict(test="open_surface_explicit_rejection",
                            passed=run.returncode == 2 and "no closed solid volumes" in failure.get("message", "")))
        continue
    records.append(dict(test=name+"_mesh_valid", passed=run.returncode == 0))
    if run.returncode:
        continue
    command = [sys.executable, str(root/"solve_p2.py"), "--mesh", str(target/"mesh"/"mesh.npz"),
               "--mach", "0", "--alpha", "0", "--sref", "1", "--ncpu", "8", "--out", str(target/"solve")]
    run = subprocess.run(command, capture_output=True, text=True)
    (target/"solve.log").write_text(run.stdout+run.stderr, encoding="utf-8")
    commands.append(dict(command=command, returncode=run.returncode))
    result = json.loads((target/"solve"/"result.json").read_text())
    records.append(dict(test=name+"_solve_status", passed=run.returncode == 0 and result["status"] == "converged"))
    if run.returncode:
        continue
    field = np.load(target/"solve"/"field.npz")
    wall = field["boundary_tags"] == 10
    shape, _ = p2.triangle_shape(p2.triangle_quadrature(3)[0])
    coordinates = np.einsum("qi,bia->bqa", shape, field["points"][field["boundary_triangles"][wall]])
    radial = coordinates/np.linalg.norm(coordinates, axis=2)[:, :, None]
    exact_cp = 1.-2.25*(1.-radial[:, :, 0]**2)
    weight = field["surface_area_weight"][wall]
    cp_rms = float(np.sqrt(np.average((field["surface_cp"][wall]-exact_cp)**2, weights=weight)))
    normal_rms = float(np.sqrt(np.average(field["surface_normal_velocity"][wall]**2, weights=weight)))
    load_error = max(abs(result["loads"][key]) for key in ["lift_over_sref", "drag_over_sref"])
    metadata = json.loads((target/"mesh"/"mesh.json").read_text())
    mapping_error = max(float(np.max(abs(np.array(metadata["native_origin"])/scale-[2., -3., 1.]))),
                        abs(metadata["native_length"]/(2.*scale)-1.))
    records.extend([
        dict(test=name+"_raw_sphere_cp", error=cp_rms, passed=cp_rms < .05),
        dict(test=name+"_raw_wall_normal", error=normal_rms, passed=normal_rms < .05),
        dict(test=name+"_pressure_load", error=load_error, passed=load_error < .005),
        dict(test=name+"_native_coordinate_mapping", error=mapping_error, passed=mapping_error < 1e-6),
    ])
    (target/"sphere_accuracy.json").write_text(json.dumps(dict(cp_rms=cp_rms, normal_rms=normal_rms,
         load_error=load_error, native_coordinate_mapping_error=mapping_error,
         reference="Infinite-domain incompressible sphere; finite uniform-Dirichlet box remains part of error"), indent=2),
         encoding="utf-8")
summary = dict(records=records, commands=commands, passed=sum(row["passed"] for row in records),
               total=len(records), driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               scope="Specified translated/scaled STEP spheres and expected open-surface rejection; not arbitrary CAD certification")
(out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(out/"test.py.snapshot").write_bytes(Path(__file__).read_bytes())
print(json.dumps(summary, indent=2))
assert summary["passed"] == summary["total"]
