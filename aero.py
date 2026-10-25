"""Run one explicit research case and retain mesh/solver commands, logs and acceptance status."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--export-surface", action="store_true", help="Export raw lifting surface samples and Tri6 geometry as VTU")
args = parser.parse_args()
root = Path(__file__).resolve().parent
if json.loads(args.case.read_text(encoding="utf-8")).get("method") == "panel-cut":
    command = [sys.executable, str(root / "panel" / "run_case.py"), "--case", str(args.case.resolve()),
               "--out", str(args.out.resolve())]
    if args.export_surface:
        command.append("--export-surface")
    sys.exit(subprocess.run(command, cwd=root).returncode)
assert not args.out.exists() or not any(args.out.iterdir()), "Output directory must be empty"
args.out.mkdir(parents=True, exist_ok=True)
started = time.perf_counter()
manifest = dict(status="failed", converged=False, commands=[], case_path=str(args.case.resolve()),
                driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
try:
    contents = args.case.read_bytes()
    manifest["case_sha256"] = hashlib.sha256(contents).hexdigest()
    (args.out/"case.json").write_bytes(contents)
    case = json.loads(contents)
    assert not set(case)-{"method", "geometry", "flow", "numerics"}, "Unknown case keys"
    method = case["method"]
    lifting_method = method in ["wing-p2", "lifting-cad-p2"]
    assert method in ["airfoil-p2", "wing-p2", "lifting-cad-p2", "closed-p2"], "Unsupported method"
    assert not args.export_surface or lifting_method, "Surface VTU export requires a lifting method"
    geometry = case["geometry"]
    flow = case.get("flow", {})
    numerics = case.get("numerics", {})
    assert not set(flow)-{"alpha", "mach", "beta"}, "Unknown flow keys"
    assert not set(numerics)-{"ncpu", "quadrature", "boundary", "center", "sref", "flux-audit", "kutta", "moment-center", "linear-solver"}, "Unknown numerics keys"
    ncpu = numerics.get("ncpu", 8)
    assert isinstance(ncpu, int) and ncpu > 0
    mesh_command = None
    if "mesh" in geometry:
        assert set(geometry) == {"mesh"}, "Existing mesh excludes additional geometry parameters"
        mesh = Path(geometry["mesh"])
        mesh = mesh if mesh.is_absolute() else args.case.resolve().parent/mesh
        mesh = mesh.resolve()
        assert mesh.is_file(), f"Mesh does not exist: {mesh}"
    else:
        mesh = (args.out/"mesh"/"mesh.npz").resolve()
        script = (root/"volume"/"build_cad_mesh.py" if method == "closed-p2" else
                  root/"lifting3d"/"build_cad_mesh.py" if method == "lifting-cad-p2" else
                  root/("airfoil2d" if method == "airfoil-p2" else "lifting3d")/"build_mesh.py")
        mesh_command = [sys.executable, str(script), "--out", str(mesh.parent)]
        allowed = ({"cad", "size", "radius", "far-size", "curvature"} if method == "closed-p2" else
                   {"wing-cad", "body-cad", "trailing-edge", "size", "radius",
                    "far-size", "wake-strips", "curvature"} if method == "lifting-cad-p2" else
                   {"shape", "airfoil", "skiprows", "size", "radius", "far-size", "epsilon", "eta", "thickness",
                    "te-exponent", "mirror", "scale", "translate"} if method == "airfoil-p2"
                   else {"size", "span", "thickness", "taper", "sweep", "radius", "far-size", "curvature",
                         "body-length", "body-radius", "body-aft", "tip-size", "wake-tilt", "wake-strips",
                         "background-view", "root-wake"})
        assert not set(geometry)-allowed, "Unsupported geometry parameters"
        if method == "closed-p2":
            assert "cad" in geometry, "Closed P2 geometry requires a CAD input or an existing mesh"
            mesh_command += ["--ncpu", str(ncpu)]
        if method == "lifting-cad-p2":
            assert "wing-cad" in geometry and "trailing-edge" in geometry, (
                "CAD lifting geometry requires wing-cad and trailing-edge inputs")
            mesh_command += ["--ncpu", str(ncpu)]
        for key, value in geometry.items():
            if key == "mirror":
                assert isinstance(value, bool)
                if value:
                    mesh_command.append("--mirror")
            elif key == "translate":
                assert len(value) == 2
                mesh_command += ["--translate"]+list(map(str, value))
            else:
                if key in ["airfoil", "cad", "background-view", "wing-cad",
                           "body-cad", "trailing-edge"]:
                    path = Path(value)
                    value = path if path.is_absolute() else args.case.resolve().parent/path
                mesh_command += ["--"+key, str(value)]
    target = (args.out/"solution").resolve()
    script = root/{"airfoil-p2": "airfoil2d", "wing-p2": "lifting3d",
                   "lifting-cad-p2": "lifting3d", "closed-p2": "volume"}[method]
    script = script/("solve_p2.py" if method == "closed-p2" else "solve.py")
    solver_command = [sys.executable, str(script), "--mesh", str(mesh), "--out", str(target),
                      "--ncpu", str(ncpu), "--alpha", str(flow.get("alpha", 0.)),
                      "--mach", str(flow.get("mach", 0.))]
    if method == "airfoil-p2":
        assert "beta" not in flow and not set(numerics)-{"ncpu", "quadrature", "boundary", "center"}
        solver_command += ["--boundary", str(numerics.get("boundary", "uniform")),
                           "--quadrature", str(numerics.get("quadrature", 4))]
        if "center" in numerics:
            assert len(numerics["center"]) == 2
            solver_command += ["--center"]+list(map(str, numerics["center"]))
    elif lifting_method:
        assert not set(flow)-{"mach", "alpha"}
        assert not set(numerics)-{"ncpu", "quadrature", "flux-audit", "kutta", "moment-center", "linear-solver"}, "Wing solver uses natural farfield and mesh Sref"
        solver_command += ["--quad-order", str(numerics.get("quadrature", 3))]
        if method == "lifting-cad-p2":
            assert numerics.get("kutta", "weak") == "weak", "CAD lifting currently requires weak Kutta"
            solver_command += ["--kutta", "weak"]
        elif "kutta" in numerics:
            solver_command += ["--kutta", str(numerics["kutta"])]
        if "linear-solver" in numerics:
            solver_command += ["--linear-solver", str(numerics["linear-solver"])]
        if "flux-audit" in numerics:
            assert isinstance(numerics["flux-audit"], bool)
            if numerics["flux-audit"]:
                solver_command.append("--flux-audit")
    else:
        assert not set(numerics)-{"ncpu", "quadrature", "boundary", "sref"}
        solver_command += ["--quad-order", str(numerics.get("quadrature", 3)),
                           "--boundary", str(numerics.get("boundary", "uniform")),
                           "--sref", str(numerics.get("sref", 1.)), "--beta", str(flow.get("beta", 0.))]
    audit_mesh = mesh
    if lifting_method and "mesh" in geometry:
        audit_dir = args.out/"geometry_audit"
        audit_dir.mkdir()
        audit_mesh = (audit_dir/"mesh.npz").resolve()
        shutil.copy2(mesh, audit_mesh)
        shutil.copy2(mesh.with_name("mesh.json"), audit_mesh.with_name("mesh.json"))
    mesh_audit = ([sys.executable, str(root/"lifting3d"/"test_mesh.py"), "--mesh", str(audit_mesh)]
                  if lifting_method else None)
    for name, command in [("mesh", mesh_command), ("mesh-audit", mesh_audit), ("solve", solver_command)]:
        if command is None:
            continue
        step_started = time.perf_counter()
        run = subprocess.run(command, capture_output=True, text=True, cwd=root)
        (args.out/(name+".log")).write_text(run.stdout+run.stderr, encoding="utf-8")
        manifest["commands"].append(dict(stage=name, command=command, returncode=run.returncode,
                                         elapsed_seconds=time.perf_counter()-step_started))
        if name == "mesh-audit":
            manifest["geometry_accepted"] = run.returncode == 0
            tests_path = audit_mesh.with_name("tests.json")
            if tests_path.exists():
                manifest["geometry_checks"] = json.loads(tests_path.read_text())
        if run.returncode:
            manifest["status"] = name+"_rejected"
            break
    if mesh.exists():
        manifest["mesh_sha256"] = hashlib.sha256(mesh.read_bytes()).hexdigest()
        if lifting_method:
            mesh_metadata = json.loads(mesh.with_name("mesh.json").read_text())
            manifest["geometry_diagnostics"] = {key: mesh_metadata.get(key) for key in
                ["min_dense_jacobian", "points", "tets", "gamma_unknowns", "root_wake_support",
                 "cad_root_reconciliation", "wake_strips", "actual_wake_strips",
                 "cad_components", "component_tags", "trailing_edge_metadata"]}
        if method == "closed-p2" and "cad" in geometry:
            mesh_metadata = json.loads(mesh.with_name("mesh.json").read_text())
            manifest["coordinate_mapping"] = {key: mesh_metadata[key] for key in
                                              ["native_origin", "native_length", "coordinate_system", "native_area_mapping"]}
    result_path = target/"result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        manifest["solver_status"] = result["status"]
        manifest["converged"] = result.get("converged", result.get("convergence", {}).get("converged", False))
        if manifest["converged"]:
            manifest["status"] = "numerically_converged"
            if lifting_method:
                manifest["raw_diagnostics"] = {key: result[key] for key in
                    ["wall_normal_rms", "wall_normal_max", "cp_min", "cp_max", "wake_diagnostics",
                     "cl_pressure", "cl_jump", "component_loads", "weak_mass_residual_norm",
                     "raw_mass_by_tag"]}
                manifest["physical_accuracy_accepted"] = False
                load_path = (args.out/"pressure_loads.json").resolve()
                command = [sys.executable, str(root/"lifting3d"/"loads.py"), "--run", str(target),
                           "--mesh", str(mesh), "--out", str(load_path)]
                if "moment-center" in numerics:
                    assert len(numerics["moment-center"]) == 3
                    command += ["--center"]+list(map(str, numerics["moment-center"]))
                step_started = time.perf_counter()
                integrated = subprocess.run(command, capture_output=True, text=True, cwd=root)
                (args.out/"loads.log").write_text(integrated.stdout+integrated.stderr, encoding="utf-8")
                manifest["commands"].append(dict(stage="loads", command=command, returncode=integrated.returncode,
                    elapsed_seconds=time.perf_counter()-step_started))
                if integrated.returncode == 0:
                    manifest["pressure_loads"] = str(load_path)
                else:
                    manifest["status"] = "numerically_converged_loads_failed"
                if args.export_surface:
                    export_dir = (args.out/"surface").resolve()
                    command = [sys.executable, str(root/"lifting3d"/"export_surface.py"),
                               "--run", str(target), "--mesh", str(mesh), "--out", str(export_dir)]
                    step_started = time.perf_counter()
                    exported = subprocess.run(command, capture_output=True, text=True, cwd=root)
                    (args.out/"export.log").write_text(exported.stdout+exported.stderr, encoding="utf-8")
                    manifest["commands"].append(dict(stage="export", command=command,
                        returncode=exported.returncode, elapsed_seconds=time.perf_counter()-step_started))
                    if exported.returncode == 0:
                        manifest["surface_export"] = str(export_dir/"export_metadata.json")
                    else:
                        manifest["status"] = "numerically_converged_export_failed"
    manifest["method"] = method
    manifest["validation"] = (
        "2D mapped-airfoil accuracy and NACA consistency tested; individual case accuracy needs its own reference"
        if method == "airfoil-p2" else
        "3D lifting accuracy validation in progress" if lifting_method else
        "Closed-body reference; no circulation, Kutta or lifting wake")
except Exception as exception:
    manifest.update(status="invalid_case_or_runtime_exception", exception_type=type(exception).__name__,
                    message=str(exception))
manifest["elapsed_seconds"] = time.perf_counter()-started
(args.out/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
(args.out/"aero.py.snapshot").write_bytes(Path(__file__).read_bytes())
print(json.dumps(manifest, indent=2))
if not manifest["converged"] or manifest["status"] != "numerically_converged":
    raise SystemExit(2)
