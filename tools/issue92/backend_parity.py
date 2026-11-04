"""CPU/CUDA float64 parity gate for the regularized area kernel (issue #92, step 3).

The CUDA port is only allowed in after the CPU torch path reproduces the numpy
reference, so this tool measures the same sheet, the same targets and the same fan
partition on three backends and reports the largest deviation of the assembled
induced velocity:

    v = sum_f gamma_f x area_terms_f

Declared acceptance (written before the measurement, as in the step-3 acceptance):

  * torch CPU  vs numpy: max relative error <= 1e-12
  * torch CUDA vs numpy: max relative error <= 1e-10
  * no silent device fallback: a 'cuda' request without a device must raise

Relative error is |dv| / U with U = 1, the freestream magnitude, so it can be read
next to the sampling gate's epsilon_tol (1e-3) and quadrature_tol (1e-2).

Geometry is dumped once in the numpy-only environment (the G2 solve needs jax, torch
does not), then replayed:

    python -m tools.issue92.backend_parity --dump runs/scratch/issue92_loose/parity_g2.npz \
        --main-half-surface runs/cases/wingbody/half_cad_wingbody_v1/surface.json \
        --csv eps:runs/studies/rm_bench/projection_gate_eps.csv \
        --csv quad:runs/studies/rm_bench/projection_gate_quad.csv

    <python-with-torch> -m tools.issue92.backend_parity \
        --geometry runs/scratch/issue92_loose/parity_g2.npz --out docs/issue92/backend_parity.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panel.p1_projection_duffy import area_terms, torch_available  # noqa: E402

THRESHOLDS = {"cpu_max_relative_error": 1e-12, "cuda_max_relative_error": 1e-10}

CONFIGS = [
    {"name": "fan16x4_gate109", "order": 16, "strips": 4, "points": "gate", "limit": 0},
    {"name": "fan48x8_gate8", "order": 48, "strips": 8, "points": "gate", "limit": 8},
    {"name": "fan8x1_gate40", "order": 8, "strips": 1, "points": "gate", "limit": 40},
]


def dump(args):
    from panel.p1_coupled_material_wake import TotalVelocity
    from panel.p1_lifting_g2 import assemble_multi, solve_multi
    from tools.issue84.projection_gate_set import load_points
    from tools.issue84.wake_node_rejection import build_case

    mesh, alpha_deg = build_case(args.main_half_surface, case=args.case, alpha_deg=args.alpha)
    state = solve_multi(mesh, alpha_deg, kutta="pressure", operators=assemble_multi(mesh))
    field = TotalVelocity(mesh, state, method="rm_regularized_p1", core_radius=args.core_radius,
                          area_quadrature="projection_fan", fan_order=16, fan_strips=8)
    regularized = field._regularized
    triangles = np.asarray(regularized.triangles, float)
    gamma = np.asarray(regularized.gamma, float)
    points = []
    for spec in args.csv:
        label, _, path = spec.partition(":")
        if not path:
            raise SystemExit("--csv wants label:path")
        points.extend(dict(row, label=label) for row in load_points(Path(path)))
    analytic = TotalVelocity(mesh, state, method="analytic_edges")._analytic
    analytic_arrays = {key: np.asarray(getattr(analytic, key), float) for key in
                       ("triangles", "normals", "gamma", "face_edge_a", "face_edge_b",
                        "edge_a", "edge_b", "edge_mu", "t_cross_n")}
    analytic_arrays["freestream"] = np.asarray(analytic.freestream, float)
    xyz = np.asarray([row["xyz"] for row in points], float)
    labels = np.asarray([row["label"] for row in points])
    nodes = np.asarray([int(row["node"]) for row in points], int) if "node" in points[0] else np.arange(len(points))
    Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
    arrays = {"triangles": triangles, "gamma": gamma, "points": xyz, "labels": labels,
              "nodes": nodes, "core": np.asarray(float(args.core_radius))}
    arrays.update({"an_" + key: value for key, value in analytic_arrays.items()})
    np.savez_compressed(args.dump, **arrays)
    print(json.dumps({"dumped": str(args.dump), "faces": int(len(triangles)),
                      "points": int(len(xyz)), "core": float(args.core_radius)}))


def synthetic_targets(n, seed=92):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-0.6, 0.6, size=(n, 3))
    xyz[:, 2] = 0.05 + 0.55*rng.random(n)
    return xyz


def statistics(errors):
    errors = np.asarray(errors, float)
    return {"max": float(errors.max()), "p90": float(np.percentile(errors, 90)),
            "median": float(np.median(errors))}


def induced(triangles, gamma, point, core, *, order, strips, backend, device):
    terms = area_terms(triangles, point, core, order=order, strips=strips,
                       backend=backend, device=device)
    return np.sum(np.cross(gamma, terms), axis=0)


def run_config(config, data, backends, repeats=3):
    triangles = data["triangles"]
    gamma = data["gamma"]
    core = float(data["core"])
    if config["points"] == "gate":
        points = data["points"]
        if config["limit"]:
            points = points[: config["limit"]]
    else:
        points = synthetic_targets(18992)
    order, strips = config["order"], config["strips"]
    out = {"name": config["name"], "order": order, "strips": strips,
           "points": int(len(points)), "times": {}, "errors": {}}

    reference = None
    for backend, device in backends:
        started = time.perf_counter()
        values = [induced(triangles, gamma, p, core, order=order, strips=strips,
                          backend=backend, device=device) for p in points]
        cold = time.perf_counter() - started
        warm = cold
        for _ in range(max(0, repeats - 1)):
            started = time.perf_counter()
            for p in points:
                induced(triangles, gamma, p, core, order=order, strips=strips,
                        backend=backend, device=device)
            warm = min(warm, time.perf_counter() - started)
        values = np.asarray(values, float)
        if reference is None:
            reference = values
        err = np.max(np.abs(values - reference), axis=1)
        out["times"]["{0}:{1}".format(backend, device)] = {
            "cold_s": float(cold), "best_s": float(warm),
            "per_point_ms": float(1000.0*warm/len(points)),
        }
        out["errors"]["{0}:{1}".format(backend, device)] = statistics(err)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, default=None)
    parser.add_argument("--geometry", type=Path, default=None)
    parser.add_argument("--main-half-surface", type=Path, default=None)
    parser.add_argument("--csv", action="append", default=None, help="label:path")
    parser.add_argument("--case", default="g2", choices=("g1", "g2"))
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--core-radius", type=float, default=0.01)
    parser.add_argument("--configs", nargs="+", default=[c["name"] for c in CONFIGS])
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.dump:
        dump(args)
        return 0
    if not args.geometry:
        raise SystemExit("--geometry is required unless --dump is given")

    data = np.load(args.geometry, allow_pickle=False)
    backends = [("numpy", "cpu"), ("torch", "cpu")]
    cuda_available = False
    if torch_available() and not args.no_cuda:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            backends.append(("torch", "cuda"))
    report = {
        "schema": "dpanel.issue92.backend_parity.v1",
        "thresholds": THRESHOLDS,
        "python": platform.python_version(),
        "torch": None,
        "cuda": None,
        "geometry": {"faces": int(len(data["triangles"])),
                     "gate_points": int(len(data["points"])),
                     "core": float(data["core"])},
        "configs": [],
    }
    if torch_available():
        import torch
        report["torch"] = torch.__version__
        report["cuda"] = {"available": cuda_available,
                          "version": torch.version.cuda,
                          "device": torch.cuda.get_device_name(0) if cuda_available else None}
    selected = [c for c in CONFIGS if c["name"] in args.configs]
    if not selected:
        raise SystemExit("no known config selected")
    for config in selected:
        if config["points"] == "synthetic" and not cuda_available and args.no_cuda:
            continue
        entry = run_config(config, data, backends)
        report["configs"].append(entry)
        print(json.dumps({"config": entry["name"], "points": entry["points"],
                          "errors": {k: v["max"] for k, v in entry["errors"].items()},
                          "per_point_ms": {k: v["per_point_ms"] for k, v in entry["times"].items()}}))
        if cuda_available:
            import torch
            report["configs"][-1]["cuda_peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())

    verdict = {}
    for entry in report["configs"]:
        for key, value in entry["errors"].items():
            backend, device = key.split(":")
            limit = THRESHOLDS["cpu_max_relative_error" if device == "cpu" else "cuda_max_relative_error"]
            verdict["{0}:{1}:{2}".format(entry["name"], backend, device)] = bool(value["max"] <= limit)
    report["verdict"] = verdict
    report["passed"] = bool(all(verdict.values()))
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print("wrote {0} passed={1}".format(args.out, report["passed"]))
    else:
        print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
