"""Throughput and parity of the analytic (production) wake evaluator on CPU and CUDA.

The production path is panel/p1_analytic_sheet_velocity.py: pure closed form, ~1.1 ms
per target on the G2 sheet, and it is what a coupled iteration spends its sampling time
on. This tool measures the torch float64 port against that reference on the same dumped
sheet: parity on the 109 gate nodes, then throughput on a full 18,992-target sample -
the number of targets one real G2 coupled update evaluates.

Declared acceptance before the measurement:

  * torch CPU  vs numpy: max relative error <= 1e-12
  * torch CUDA vs numpy: max relative error <= 1e-10

    <python-with-torch> -m tools.issue92.analytic_bench \
        --geometry runs/scratch/issue92_loose/parity_g2.npz --out docs/issue92/analytic_torch_bench.json
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

from panel.p1_analytic_sheet_velocity import AnalyticSheetField  # noqa: E402
from panel.p1_torch_analytic_sheet import GEOMETRY_KEYS, TorchAnalyticSheetField  # noqa: E402

THRESHOLDS = {"cpu_max_relative_error": 1e-12, "cuda_max_relative_error": 1e-10}
BATCH_TARGETS = 18992


def load(path):
    data = np.load(path, allow_pickle=False)
    arrays = {key: data["an_" + key] for key in GEOMETRY_KEYS}
    freestream = data["an_freestream"]
    return arrays, freestream, data["points"]


def synthetic_targets(n, seed=92):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform([-0.6, -0.6, 0.02], [1.4, 0.6, 1.2], size=(n, 3))
    return xyz


def relative_error(a, b):
    scale = np.maximum(np.linalg.norm(b, axis=-1), 1e-12)
    return float(np.max(np.linalg.norm(a - b, axis=-1)/scale))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--batch-targets", type=int, default=BATCH_TARGETS)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args(argv)

    import torch
    arrays, freestream, gate_points = load(args.geometry)
    reference = AnalyticSheetField.from_arrays(arrays, freestream=freestream)
    cuda_available = bool(torch.cuda.is_available()) and not args.no_cuda
    devices = ["cpu"] + (["cuda"] if cuda_available else [])

    report = {
        "schema": "dpanel.issue92.analytic_torch_bench.v1",
        "thresholds": THRESHOLDS,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": {"available": cuda_available,
                 "version": torch.version.cuda,
                 "device": torch.cuda.get_device_name(0) if cuda_available else None},
        "sheet": {"faces": int(len(arrays["triangles"])),
                  "boundary_edges": int(len(arrays["edge_a"])),
                  "face_edges": int(len(arrays["face_edge_a"]))},
        "gate_points": int(len(gate_points)),
        "backends": {},
    }

    # Gate nodes sit on the sheet boundary by construction, where the analytic kernel
    # is genuinely singular (the numpy reference raises). Those points are excluded
    # from the parity set and counted, never silently dropped.
    probes = np.concatenate([gate_points + np.array([0.0, 0.0, sign*0.05])
                             for sign in (1.0, -1.0)])
    valid, values = [], []
    for p in probes:
        try:
            values.append(reference(p))
        except FloatingPointError:
            continue
        valid.append(p)
    valid = np.asarray(valid)
    expected = np.asarray(values)
    report["gate_points_evaluable"] = int(len(valid))
    report["gate_points_singular"] = int(len(probes) - len(valid))
    report["gate_probe"] = "gate nodes offset by +-0.05 along z; the raw nodes lie on the sheet boundary where the analytic kernel is singular"
    if not len(valid):
        raise SystemExit("no gate point is evaluable by the analytic reference")
    started = time.perf_counter()
    for p in valid:
        reference(p)
    report["numpy_gate_seconds"] = float(time.perf_counter() - started)

    for device in devices:
        field = TorchAnalyticSheetField.from_arrays(arrays, freestream=freestream, device=device)
        values = field.velocity(valid)
        entry = {"gate_max_relative_error": relative_error(values, expected),
                 "gate_bitwise_equal": bool(np.array_equal(values, expected))}
        limit = THRESHOLDS["cpu_max_relative_error" if device == "cpu" else "cuda_max_relative_error"]
        entry["accepted"] = bool(entry["gate_max_relative_error"] <= limit)
        started = time.perf_counter()
        field.velocity(valid, chunk=args.chunk)
        entry["gate_seconds"] = float(time.perf_counter() - started)
        points = synthetic_targets(args.batch_targets)
        started = time.perf_counter()
        field.velocity(points, chunk=args.chunk)
        entry["batch_seconds"] = float(time.perf_counter() - started)
        entry["batch_per_point_ms"] = float(1000.0*entry["batch_seconds"]/len(points))
        if device == "cuda":
            entry["cuda_peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
            del field
            torch.cuda.empty_cache()
        report["backends"][device] = entry
        print(json.dumps({"device": device, **entry}))

    points = synthetic_targets(args.batch_targets)
    started = time.perf_counter()
    for p in points:
        reference(p)
    numpy_batch = float(time.perf_counter() - started)
    report["numpy_batch_seconds"] = numpy_batch
    report["numpy_batch_per_point_ms"] = float(1000.0*numpy_batch/len(points))
    for device, entry in report["backends"].items():
        entry["speedup_vs_numpy"] = float(numpy_batch/entry["batch_seconds"])
    report["passed"] = bool(all(entry["accepted"] for entry in report["backends"].values()))
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print("wrote {0} passed={1} numpy_batch_s={2:.1f}".format(args.out, report["passed"], numpy_batch))
    else:
        print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
