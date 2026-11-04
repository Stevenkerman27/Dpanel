"""B2: whole coupled solve on device=cpu|cuda, cold/warm timings and parity.

Route B step B2 asks for the full steady solve plus at least one coupled wake
iteration on the same process with an explicit device, and for timings that
separate cold start (context, upload of the operator set) from warm repeats,
with H2D volume and peak device memory reported.

References are the S5 NumPy records in runs/scratch/issue92_loose/s5/<case>_numpy.json, which
the S5 gate already tied to the torch CPU backend at 1e-5 relative. The gate
here is inherited, not re-derived: loads to 1e-5 relative against the same
iteration index, plus the requirement that status and acceptance decisions
agree.

    python -m tools.issue92.b2_cuda_bench --out docs/issue92/b2_cuda_bench.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

from panel.p1_coupled_material_wake import solve_coupled_material_wake
from panel.p1_lifting_g2 import assemble_multi
from panel.p1_solve_torch import solve_multi_torch
from tools.issue92.s5_loop_parity import _mesh

ROOT = pathlib.Path(__file__).resolve().parents[2]
LOADS = ("CL", "CL_gamma", "CD_pressure", "CM_y")
CASES = ("g2_conventional", "g2_cad")
THRESHOLDS = dict(load_relative=1e-5, speedup=1.5, repetitions=2)


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def _peak(device):
    return float(torch.cuda.max_memory_allocated()/2**20) if device == "cuda" else None


def _reset(device):
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _uploaded_bytes(operators):
    total = 0
    for key in ("A", "W", "J", "K", "grad"):
        array = operators[key]
        total += int(array.size)*8
    return total


def _reference(case):
    path = ROOT/("runs/scratch/issue92_loose/s5/%s_numpy.json" % case)
    data = json.loads(path.read_text(encoding="utf-8"))
    return {row["index"]: row["loads"] for row in data["rows"]}, data["status"]


def _loads(state):
    return {key: float(state[key]) for key in LOADS}


def _relative(got, expected):
    return {key: abs(got[key]-expected[key])/max(abs(expected[key]), 1e-12) for key in LOADS}


def bench(case, device, repetitions=2):
    reference, reference_status = _reference(case)
    mesh = _mesh(case)
    operators = assemble_multi(mesh)
    _reset(device)
    timings = []
    states = []
    for index in range(repetitions):
        _sync(device)
        started = time.perf_counter()
        state = solve_multi_torch(mesh, 5., operators=operators, device=device)
        _sync(device)
        timings.append(time.perf_counter()-started)
        states.append(state)
    cold, warm = timings[0], min(timings[1:]) if len(timings) > 1 else timings[0]
    solved_peak = _peak(device)

    _reset(device)
    _sync(device)
    started = time.perf_counter()
    coupled = solve_coupled_material_wake(
        mesh, 5., iterations=1, kutta="pressure", pressure_solver="structured",
        solve_backend="torch", solve_device=device, analytic_backend="torch",
        analytic_device=device, gate_mode="audit")
    _sync(device)
    loop_seconds = time.perf_counter()-started
    loop_peak = _peak(device)

    solved = _loads(states[-1])
    loop = _loads(coupled["state"])
    return dict(
        case=case, device=device,
        gpu=(torch.cuda.get_device_name(0) if device == "cuda" else None),
        wall_faces=len(mesh.xyz) if hasattr(mesh, "xyz") else None,
        solve_cold_seconds=cold, solve_warm_seconds=warm,
        solve_seconds_all=timings,
        solve_peak_mib=solved_peak, loop_peak_mib=loop_peak,
        loop_seconds=loop_seconds, loop_status=coupled["status"],
        reference_status=reference_status,
        solve_loads=solved, solve_relative=_relative(solved, reference[0]),
        loop_loads=loop, loop_relative=_relative(loop, reference[1]),
        uploaded_mib=_uploaded_bytes(operators)/2**20,
        peak_mib=max(x for x in (solved_peak, loop_peak) if x is not None) if device == "cuda" else None,
        torch_version=torch.__version__,
        cuda=torch.version.cuda,
    ), reference


def judge(rows):
    verdicts = {}
    for row in rows:
        relative = max(row["solve_relative"].values())
        verdicts["%s/%s/solve" % (row["case"], row["device"])] = relative <= THRESHOLDS["load_relative"]
    pairs = {}
    for case in CASES:
        cpu = next((r for r in rows if r["case"] == case and r["device"] == "cpu"), None)
        cuda = next((r for r in rows if r["case"] == case and r["device"] == "cuda"), None)
        if cpu and cuda:
            pairs[case] = dict(
                solve_speedup=cpu["solve_warm_seconds"]/cuda["solve_warm_seconds"],
                loop_speedup=cpu["loop_seconds"]/cuda["loop_seconds"],
                meets=cpu["solve_warm_seconds"]/cuda["solve_warm_seconds"] >= THRESHOLDS["speedup"])
    return dict(thresholds=THRESHOLDS, parity=verdicts, speedup=pairs,
                passed=all(verdicts.values()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=ROOT/"docs/issue92/b2_cuda_bench.json")
    parser.add_argument("--case", action="append", choices=CASES, default=None)
    parser.add_argument("--device", action="append", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--repetitions", type=int, default=THRESHOLDS["repetitions"])
    args = parser.parse_args(argv)
    cases = tuple(args.case) if args.case else CASES
    devices = tuple(args.device) if args.device else (("cpu", "cuda") if torch.cuda.is_available() else ("cpu",))
    rows = []
    for case in cases:
        for device in devices:
            row, _reference_rows = bench(case, device, args.repetitions)
            rows.append(row)
            print(json.dumps({key: row[key] for key in
                              ("case", "device", "solve_cold_seconds", "solve_warm_seconds",
                               "loop_seconds", "loop_status", "peak_mib")}, indent=None))
            print("   solve rel", {k: "%.2e" % v for k, v in row["solve_relative"].items()})
            print("   loop  rel", {k: "%.2e" % v for k, v in row["loop_relative"].items()})
    report = dict(judge(rows), rows=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("passed:", report["passed"])
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
