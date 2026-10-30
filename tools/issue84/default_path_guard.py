"""Issue #84 items A1/A2/A4: targeted guards for the default-path promises.

The contract asks for evidence, not just a green test list, that

  A1  RegularizedP1SheetField can be built from a single part (the constructor
      used to reference its part lists before assigning them),
  A2  the regularized unit test no longer touches a nonexistent grad_face,
  A4  analytic_edges stays the default and its outputs are untouched, the RM
      path still demands an explicit core radius, and no report claims physical
      accuracy.

Every check prints PASS/FAIL with the observed value, so this file doubles as the
reproduction recipe. It builds its own one-triangle stub, so it needs no solve.
"""
from __future__ import annotations

import argparse
import inspect
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from panel.p1_coupled_material_wake import TotalVelocity
from panel.p1_regularized_sheet_velocity import RegularizedP1SheetField

ROOT = Path(__file__).resolve().parents[2]


def stub():
    xyz = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    faces = np.array([[0, 1, 2]])
    # dof_face is (n_faces, 3): the three nodal dofs of each face
    mesh = SimpleNamespace(xyz=xyz, faces=faces, groups=[], dof_face=np.array([[0, 1, 2]]))
    state = {"mu": np.array([0.1, 0.2, 0.3]), "gamma": np.zeros(0), "alpha_deg": 5.0}
    parts = [(xyz, faces, np.array([[0.1, 0.2, 0.3]]))]  # values are per-face nodal
    return mesh, state, parts


def check_a1(parts):
    field = RegularizedP1SheetField(parts, core_radius=0.01)
    area2 = np.asarray(field._triangle_area2)
    value = field(np.array([0.25, 0.25, 0.1]), order=16)
    ok = area2.shape == (1,) and area2[0] > 0 and np.isfinite(value).all()
    return {
        "check": "A1 single-part constructor + evaluate",
        "status": "PASS" if ok else "FAIL",
        "n_triangles": int(len(field.triangles)),
        "triangle_area2": area2.tolist(),
        "velocity": np.asarray(value).tolist(),
    }


def check_a2(path):
    text = Path(path).read_text(encoding="utf-8")
    hits = len(re.findall(r"grad_face", text))
    gamma = len(re.findall(r"varying\.gamma", text))
    ok = hits == 0 and gamma > 0
    return {
        "check": "A2 regularized test uses a real attribute",
        "status": "PASS" if ok else "FAIL",
        "file": str(path),
        "grad_face_occurrences": hits,
        "varying_gamma_occurrences": gamma,
    }


def check_a4_default(mesh, state):
    signature = inspect.signature(TotalVelocity.__init__)
    default = signature.parameters["method"].default
    field = TotalVelocity(mesh, state)
    value = field(np.array([0.3, 0.3, 0.2]))
    ok = default == "analytic_edges" and field.method == "analytic_edges" and np.isfinite(value).all()
    return {
        "check": "A4a analytic_edges is the default evaluator",
        "status": "PASS" if ok else "FAIL",
        "signature_default": default,
        "instance_method": field.method,
        "velocity": np.asarray(value).tolist(),
        "has_regularized_attribute": hasattr(field, "_regularized"),
    }


def check_a4_core(mesh, state, parts):
    outcomes = {}
    try:
        TotalVelocity(mesh, state, method="rm_regularized_p1")
        outcomes["TotalVelocity_without_core"] = "no error (BAD)"
    except (ValueError, TypeError) as exc:
        outcomes["TotalVelocity_without_core"] = f"{type(exc).__name__}: {exc}"
    try:
        RegularizedP1SheetField(parts)
        outcomes["RegularizedP1SheetField_without_core"] = "no error (BAD)"
    except (ValueError, TypeError) as exc:
        outcomes["RegularizedP1SheetField_without_core"] = f"{type(exc).__name__}: {exc}"
    ok = all("error" not in v and "BAD" not in v for v in outcomes.values())
    return {
        "check": "A4b RM still requires an explicit positive core",
        "status": "PASS" if ok else "FAIL",
        "errors": outcomes,
    }


def check_a4_accuracy(package_dir):
    suspicious = []
    total = 0
    files = 0
    for path in sorted(Path(package_dir).glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "physical_accuracy_accepted" not in text:
            continue
        files += 1
        for line_no, line in enumerate(text.splitlines(), 1):
            if "physical_accuracy_accepted" not in line:
                continue
            total += 1
            after = line.split("physical_accuracy_accepted", 1)[1]
            if re.search(r"True", after):
                suspicious.append(f"{path.name}:{line_no}: {line.strip()}")
    ok = not suspicious
    return {
        "check": "A4c no report claims physical accuracy",
        "status": "PASS" if ok else "FAIL",
        "files_scanned_with_flag": files,
        "occurrences": total,
        "true_occurrences": suspicious,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.root

    mesh, state, parts = stub()
    probes = [
        ("A1 single-part constructor", lambda: check_a1(parts)),
        ("A2 no grad_face in test", lambda: check_a2(root / "test" / "test_p1_regularized_sheet_velocity.py")),
        ("A4a default method", lambda: check_a4_default(mesh, state)),
        ("A4b explicit core", lambda: check_a4_core(mesh, state, parts)),
        ("A4c accuracy flag", lambda: check_a4_accuracy(root / "panel")),
    ]
    results = []
    for name, probe in probes:
        try:
            results.append(probe())
        except Exception as exc:  # a broken probe is a failed check, not a crash
            results.append({"check": name, "status": "FAIL",
                            "error": f"{type(exc).__name__}: {exc}"})
    report = {
        "schema": "dpanel.issue84.default_path_guard.v1",
        "root": str(root),
        "python": __import__("sys").version.split()[0],
        "numpy": np.__version__,
        "checks": results,
        "all_pass": all(r["status"] == "PASS" for r in results),
    }
    for item in results:
        print(f'{item["status"]}  {item["check"]}')
        for key, value in item.items():
            if key not in ("check", "status"):
                print(f"        {key}: {json.dumps(value)}")
    print("all_pass:", report["all_pass"])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
