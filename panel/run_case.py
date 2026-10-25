"""Run one explicit-cut panel case from a surface fixture or closed CAD solids."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--case", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--export-surface", action="store_true")
args = parser.parse_args()
args.out = args.out.resolve()
assert not args.out.exists() or not any(args.out.iterdir()), "Preserve previous runs"
args.out.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parents[1]
started = time.perf_counter()
manifest = dict(status="failed", commands=[], geometry_accepted=False,
                physical_accuracy_accepted=False, case_path=str(args.case.resolve()),
                driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
stage = "case_validation"
try:
    contents = args.case.read_bytes()
    (args.out / "case.json").write_bytes(contents)
    manifest["case_sha256"] = hashlib.sha256(contents).hexdigest()
    case = json.loads(contents)
    assert case["method"] == "panel-cut"
    assert not set(case) - {"method", "geometry", "flow", "numerics"}
    geometry = case["geometry"]
    flow, numerics = case.get("flow", {}), case.get("numerics", {})
    assert not set(flow) - {"alpha", "mach"}
    assert flow.get("mach", 0.) == 0., "This panel method solves incompressible Laplace flow"
    assert not set(numerics) - {"ncpu", "central-thickness", "span-factor", "derivatives"}
    ncpu = numerics.get("ncpu", 8)
    assert isinstance(ncpu, int) and ncpu > 0
    commands = []
    if "surface" in geometry:
        assert set(geometry) == {"surface"}
        surface = Path(geometry["surface"])
        surface = surface if surface.is_absolute() else args.case.resolve().parent / surface
    elif "mesh" in geometry:
        assert set(geometry) == {"mesh"}
        mesh = Path(geometry["mesh"])
        mesh = mesh if mesh.is_absolute() else args.case.resolve().parent / mesh
        surface = args.out / "surface.json"
        commands.append(("surface", [sys.executable, str(root / "panel" / "build_surface.py"),
                                    "--mesh", str(mesh.resolve()), "--out", str(surface)]))
    else:
        assert {"wing-cad", "trailing-edge"}.issubset(geometry)
        assert not set(geometry) - {"wing-cad", "body-cad", "trailing-edge", "size", "far-size", "radius", "wake-strips"}
        surface = args.out / "mesh" / "surface.json"
        command = [sys.executable, str(root / "panel" / "cad.py"), "--out", str(surface.parent), "--ncpu", str(ncpu)]
        for key, value in geometry.items():
            if key in ("wing-cad", "body-cad", "trailing-edge"):
                path = Path(value)
                value = path if path.is_absolute() else args.case.resolve().parent / path
            command += ["--" + key, str(value)]
        commands.append(("cad_surface", command))
    target = args.out / "solution"
    command = [sys.executable, str(root / "panel" / "solve.py"), "--surface", str(surface.resolve()),
               "--out", str(target), "--alpha", str(flow.get("alpha", 5.)), "--ncpu", str(ncpu),
               "--bulge", str(numerics.get("central-thickness", 0.)),
               "--span", str(numerics.get("span-factor", 1.))]
    if numerics.get("derivatives", False):
        command.append("--derivatives")
    commands.append(("solve", command))
    if args.export_surface:
        commands.append(("export_surface", [sys.executable, str(root / "panel" / "export.py"), str(target)]))
    for stage, command in commands:
        t0 = time.perf_counter()
        process = subprocess.run(command, capture_output=True, text=True, cwd=root)
        (args.out / (stage + ".log")).write_text(process.stdout + process.stderr, encoding="utf-8")
        manifest["commands"].append(dict(stage=stage, command=command, returncode=process.returncode,
                                         elapsed_seconds=time.perf_counter() - t0))
        if process.returncode:
            raise RuntimeError(stage + " failed; see retained log")
    result = json.loads((target / "result.json").read_text())
    manifest.update(status=result["status"], geometry_accepted=result["geometry_robustness_accepted"],
                    linear_true_relative_residual=result["linear_true_relative_residual"],
                    objectives=result["objectives"], components=result["components"],
                    derivatives_passed=result.get("derivatives_passed"),
                    surface_sha256=result["surface_sha256"], solution=str(target),
                    physical_accuracy_accepted=result["physical_accuracy_accepted"])
except Exception as error:
    manifest.update(status=stage + "_rejected", exception=type(error).__name__, message=str(error))
finally:
    manifest["elapsed_seconds"] = time.perf_counter() - started
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, allow_nan=False), flush=True)
assert manifest["geometry_accepted"] and manifest["status"] in ("forward_complete", "loop_complete")
