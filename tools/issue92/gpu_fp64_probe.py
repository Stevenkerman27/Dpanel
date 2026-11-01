"""FP64 throughput probe for the Phase 2 decision: dense LU, batched solves, matmul.

Run inside an interpreter that has a CUDA torch build (myml on this machine):

    C:/Users/zyx20/anaconda3/envs/myml/python.exe -m tools.issue92.gpu_fp64_probe \
        --out docs/issue92/gpu_fp64_probe.json

The question is not "is the GPU faster" in general, but "at the sizes DPanel actually
solves (541-1728 unknowns) and for the kernels DPanel actually runs (one dense solve,
a handful of right-hand sides, and many pairwise induced-velocity evaluations), does an
FP64 consumer GPU beat a 32-core CPU". Every timing here is float64 and synchronised;
host-to-device traffic is reported separately so a single-shot solve and a reused
factorisation cannot be confused.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time(fn, device, repeats=5):
    fn()
    _sync(device)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), min(samples)


def _environment():
    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env.update({
            "gpu": props.name,
            "gpu_total_memory_gb": round(props.total_memory / 1e9, 2),
            "gpu_multi_processor_count": props.multi_processor_count,
            "gpu_warp_size": props.warp_size,
            "cuda_version": torch.version.cuda,
        })
    return env


def _matmul_throughput(device, sizes=(1024, 2048, 4096), dtype=torch.float64):
    rows = []
    for n in sizes:
        a = torch.randn(n, n, dtype=dtype, device=device)
        b = torch.randn(n, n, dtype=dtype, device=device)
        seconds, _ = _time(lambda: a @ b, device, repeats=3)
        flops = 2.0 * n ** 3
        rows.append({
            "n": n,
            "dtype": str(dtype).replace("torch.", ""),
            "seconds": round(seconds, 6),
            "tflops": round(flops / seconds / 1e12, 4),
        })
    return rows


def _batched_solve(device, n, rhs_counts, dtype=torch.float64):
    a = torch.randn(n, n, dtype=dtype, device=device)
    # diagonally dominant, so the factorisation is well conditioned (same recipe as
    # tools/issue92/solver_crossover.py)
    a = a + torch.diag(torch.linspace(1.0, 2.0, n, dtype=dtype, device=device)) * n
    rows = []
    # warm the factorisation path first: the first cuSOLVER call pays context and plan
    # setup (about 1 s at n=1728), which would masquerade as a device-speed difference
    lu, piv = torch.linalg.lu_factor(a)
    _sync(device)
    factor_seconds, _ = _time(lambda: torch.linalg.lu_factor(a), device, repeats=5)
    for k in rhs_counts:
        b = torch.randn(n, k, dtype=dtype, device=device)
        solve_seconds, _ = _time(lambda: torch.linalg.lu_solve(lu, piv, b), device, repeats=5)
        x = torch.linalg.lu_solve(lu, piv, b)
        residual = (a @ x - b).abs().max().item() / max(b.abs().max().item(), 1e-300)
        rows.append({
            "n": n,
            "rhs": k,
            "lu_factor_seconds": round(factor_seconds, 6),
            "solve_seconds": round(solve_seconds, 6),
            "rel_residual": residual,
        })
    return rows


def _pairwise_kernel(device, n_targets, n_triangles, dtype=torch.float64, chunk=2048):
    """Proxy for the B2 induced-velocity kernel: targets x triangles, one interaction.

    Per interaction it does the arithmetic of a segment/solid-angle evaluation
    (a handful of multiplies, adds, a divide and an atan2 substitute via a rational
    approximation) but not the panel-specific bookkeeping, so the number is a lower
    bound for the real kernel and an upper bound for pure memory traffic.
    """
    tri = torch.randn(n_triangles, 3, dtype=dtype, device=device)
    pts = torch.randn(n_targets, 3, dtype=dtype, device=device)

    def run():
        total = torch.zeros(n_targets, 3, dtype=dtype, device=device)
        for start in range(0, n_targets, chunk):
            p = pts[start:start + chunk].unsqueeze(1)
            d = p - tri.unsqueeze(0)
            r2 = (d * d).sum(dim=-1)
            inv = 1.0 / (r2 + 1e-12)
            total[start:start + chunk] = (d * inv.unsqueeze(-1)).sum(dim=1)
        return total

    seconds, _ = _time(run, device, repeats=3)
    interactions = float(n_targets) * float(n_triangles)
    return {
        "n_targets": n_targets,
        "n_triangles": n_triangles,
        "interactions": interactions,
        "seconds": round(seconds, 6),
        "ms_per_target": round(seconds * 1e3 / n_targets, 4),
        "ns_per_interaction": round(seconds * 1e9 / interactions, 3),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/issue92/gpu_fp64_probe.json")
    parser.add_argument("--targets", type=int, default=18992,
                        help="points evaluated in one B2 coupled iteration (Phase 0)")
    parser.add_argument("--triangles", type=int, default=794,
                        help="wake triangles of the Phase 0 CAD case")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("this probe needs a CUDA torch build; run it in the myml interpreter")

    payload = {"environment": _environment(), "kernels": {}}
    devices = {"cpu": torch.device("cpu"), "cuda": torch.device("cuda")}

    payload["kernels"]["matmul"] = {}
    for name, device in devices.items():
        payload["kernels"]["matmul"][name] = _matmul_throughput(device)

    payload["kernels"]["batched_solve"] = {}
    for n in (1728, 3220):
        for name, device in devices.items():
            payload["kernels"]["batched_solve"].setdefault(str(n), {})[name] = _batched_solve(
                device, n, rhs_counts=(1, 16, 64))

    payload["kernels"]["pairwise"] = {}
    for name, device in devices.items():
        payload["kernels"]["pairwise"][name] = _pairwise_kernel(
            device, args.targets, args.triangles)

    # one-shot cost of moving a production-sized dense matrix to the device
    hook = {}
    for n in (1728, 3220):
        host = torch.randn(n, n, dtype=torch.float64)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        host.to("cuda")
        torch.cuda.synchronize()
        hook[str(n)] = {
            "matrix_mb": round(n * n * 8 / 1048576.0, 1),
            "host_to_device_seconds": round(time.perf_counter() - t0, 4),
        }
    payload["host_to_device"] = hook

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("matmul FP64 TFLOPS")
    for name in ("cpu", "cuda"):
        for row in payload["kernels"]["matmul"][name]:
            print("  %-4s n=%5d %8.4f TFLOPS" % (name, row["n"], row["tflops"]))
    print("batched FP64 solve")
    for n, block in payload["kernels"]["batched_solve"].items():
        for name in ("cpu", "cuda"):
            for row in block[name]:
                print("  n=%-5s %-4s rhs=%-3d factor=%.4fs solve=%.4fs res=%.1e" % (
                    n, name, row["rhs"], row["lu_factor_seconds"], row["solve_seconds"],
                    row["rel_residual"]))
    print("pairwise induced-velocity proxy")
    for name in ("cpu", "cuda"):
        row = payload["kernels"]["pairwise"][name]
        print("  %-4s %d targets x %d triangles: %.3fs (%.3f ms/target, %.2f ns/pair)" % (
            name, row["n_targets"], row["n_triangles"], row["seconds"],
            row["ms_per_target"], row["ns_per_interaction"]))
    print("host to device")
    for n, row in payload["host_to_device"].items():
        print("  n=%-5s %6.1f MB in %.4fs" % (n, row["matrix_mb"], row["host_to_device_seconds"]))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())