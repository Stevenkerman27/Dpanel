"""Issue #92 Phase 0: crossover between CPU dense LU, GPU dense LU and matrix-free GMRES.

Measured on the same float64 inputs and the same deterministic diagonally dominant
matrices, so the comparison isolates the linear-algebra primitive. The real fixture
unknown counts are printed from docs/issue92/phase0_profile.json so the measured
crossover can be read against the sizes DPanel actually assembles.

Rules:
  * every backend reports the residual, so a fast wrong answer is visible;
  * CUDA timings synchronize and separate cold start from warm runs;
  * no silent fallback: if CUDA is unavailable the row is reported as skipped.

Usage:
  python -m tools.issue92.solver_crossover --out docs/issue92/solver_crossover.json
  python -m tools.issue92.solver_crossover --sizes 256 512 1024 --warm 2
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("PANEL_ROOT", Path.cwd()))

import numpy as np

from panel.velocity_profile import peak_rss_mb

DEFAULT_SIZES = (96, 192, 385, 769, 1537, 3073, 6145)


def _torch():
    try:
        import torch
        return torch
    except Exception:
        pass
    for site in (ROOT / "runs/torch_site", ROOT.parent / "torch_site",
                 ROOT.parent / "runs/torch_site", ROOT.parent.parent / "runs/torch_site"):
        if site.exists() and str(site) not in sys.path:
            sys.path.insert(0, str(site))
            try:
                import torch
                return torch
            except Exception:
                sys.path.remove(str(site))
    return None


def _matrix(n, seed=20261101):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n)) / np.sqrt(n)
    a += np.eye(n) * (2.0 + np.arange(n) / max(n, 1))  # diagonally dominant, well conditioned
    return np.ascontiguousarray(a)


def _residual(a, x, b):
    scale = np.linalg.norm(b) or 1.0
    return float(np.linalg.norm(a @ x - b) / scale)


def _version(name):
    try:
        return __import__(name).__version__
    except Exception as exc:
        return "unavailable: " + type(exc).__name__


def _timed(func, warm=3, repeat=3):
    """Returns (cold_seconds, warm_median, warm_min) for one backend."""
    t0 = time.perf_counter()
    value = func()
    cold = time.perf_counter() - t0
    for _ in range(max(0, warm - 1)):
        func()
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        value = func()
        samples.append(time.perf_counter() - t0)
    return cold, statistics.median(samples), min(samples), value


def cpu_lu(a, b):
    import scipy.linalg as sla
    lu, piv = sla.lu_factor(a, check_finite=False)
    return sla.lu_solve((lu, piv), b, check_finite=False)


def gmres_dense(a, b, tol=1e-10):
    import scipy.sparse.linalg as spla
    calls = {"n": 0}

    def matvec(x):
        calls["n"] += 1
        return a @ x

    x, info = spla.gmres(spla.LinearOperator((a.shape[0], a.shape[0]), matvec=matvec),
                         b, rtol=tol, atol=0.0, restart=200, maxiter=200)
    return x, info, calls["n"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Issue #92 Phase 0 solver crossover")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/issue92/solver_crossover.json")
    parser.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES))
    parser.add_argument("--warm", type=int, default=3)
    parser.add_argument("--backends", nargs="+", default=["cpu", "torch_cpu", "cuda"],
                        choices=("cpu", "torch_cpu", "cuda"),
                        help="run each backend family in its own process; a CUDA crash in one "
                             "must not destroy the CPU numbers")
    parser.add_argument("--merge", nargs="+", type=Path, default=None,
                        help="merge per-backend json files into --out instead of measuring")
    args = parser.parse_args(argv)
    if args.merge:
        return _merge(args.merge, args.out)
    torch = _torch()
    device = "cuda" if ("cuda" in args.backends and torch is not None
                        and torch.cuda.is_available()) else "cpu"
    payload = {"schema": "dpanel.issue92.solver_crossover.v1",
               "environment": {"python": sys.version.split()[0],
                               "platform": platform.platform(),
                               "numpy": np.__version__,
                               "scipy": _version("scipy"),
                               "torch": None if torch is None else torch.__version__,
                               "device": device,
                               "cuda_available": bool(torch is not None and torch.cuda.is_available()),
                               "cpu_count": os.cpu_count()},
               "rows": []}
    if torch is not None and torch.cuda.is_available():
        payload["environment"]["gpu"] = torch.cuda.get_device_name(0)
        payload["environment"]["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    for n in args.sizes:
        a = _matrix(n)
        b = np.cos(np.arange(n) * 0.017) + 0.5
        row = {"n": n, "bytes_dense_mb": round(a.nbytes / 1048576.0, 2), "backends": {}}
        if "cpu" in args.backends:
            cold, med, best, x = _timed(lambda: cpu_lu(a, b), args.warm)
            row["backends"]["cpu_lu"] = {"cold_seconds": round(cold, 4),
                                         "warm_median": round(med, 4),
                                         "warm_min": round(best, 4),
                                         "residual": _residual(a, x, b)}
        if torch is not None:
            at = torch.from_numpy(a)
            bt = torch.from_numpy(b)
            if "torch_cpu" in args.backends:
                cold, med, best, x = _timed(
                    lambda: torch.linalg.solve(at, bt).numpy(), args.warm)
                row["backends"]["torch_cpu"] = {
                    "cold_seconds": round(cold, 4), "warm_median": round(med, 4),
                    "warm_min": round(best, 4), "residual": _residual(a, x, b)}
            if ("cuda" in args.backends and device == "cuda"
                    and torch.cuda.is_available()):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                transfer_start = time.perf_counter()
                at_gpu = at.to("cuda")
                bt_gpu = bt.to("cuda")
                torch.cuda.synchronize()
                transfer = time.perf_counter() - transfer_start

                def solve_gpu():
                    return torch.linalg.solve(at_gpu, bt_gpu)

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                xg = solve_gpu()
                torch.cuda.synchronize()
                cold = time.perf_counter() - t0
                samples = []
                for _ in range(max(1, args.warm)):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    xg = solve_gpu()
                    torch.cuda.synchronize()
                    samples.append(time.perf_counter() - t0)
                peak_vram = torch.cuda.max_memory_allocated() / 1048576.0
                xh = xg.detach().cpu().numpy()
                row["backends"]["cuda_lu"] = {
                    "cold_seconds": round(cold, 4),
                    "warm_median": round(statistics.median(samples), 4),
                    "warm_min": round(min(samples), 4),
                    "residual": _residual(a, xh, b),
                    "host_to_device_seconds": round(transfer, 4),
                    "peak_vram_mb": round(peak_vram, 1)}
                del at_gpu, bt_gpu, xg
                torch.cuda.empty_cache()
        if "cpu" in args.backends:
            t0 = time.perf_counter()
            xg, info, n_matvec = gmres_dense(a, b)
            row["backends"]["gmres_dense"] = {
                "seconds": round(time.perf_counter() - t0, 4), "iterations": n_matvec,
                "converged": int(info) == 0, "residual": _residual(a, xg, b),
                "matvec_ms": round(1000.0 * (time.perf_counter() - t0) / max(n_matvec, 1), 4)}
        payload["rows"].append(row)
        parts = ["n=%-5d" % n]
        for key in ("cpu_lu", "torch_cpu", "cuda_lu"):
            if key in row["backends"]:
                parts.append("%s=%7.3fs" % (key, row["backends"][key]["warm_median"]))
        if "gmres_dense" in row["backends"]:
            parts.append("gmres=%7.3fs(%d)" % (row["backends"]["gmres_dense"]["seconds"],
                                               row["backends"]["gmres_dense"]["iterations"]))
        print(" ".join(parts))
        payload["peak_rss_mb"] = round(peak_rss_mb(), 1)
    payload["verdict"] = _verdict(payload["rows"])
    payload["fixture_sizes"] = _fixture_sizes()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["verdict"], indent=1))
    print("wrote " + str(args.out))
    return payload


def _merge(paths, out):
    rows = []
    environment = {}
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        environment.update({k: v for k, v in payload.get("environment", {}).items()
                            if v is not None})
        by_n = {row["n"]: row for row in rows}
        for row in payload["rows"]:
            if row["n"] in by_n:
                by_n[row["n"]]["backends"].update(row["backends"])
            else:
                rows.append(row)
    rows.sort(key=lambda row: row["n"])
    merged = {"schema": "dpanel.issue92.solver_crossover.v1", "merged_from":
              [str(Path(p).name) for p in paths], "environment": environment,
              "rows": rows, "verdict": _verdict(rows), "fixture_sizes": _fixture_sizes()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(json.dumps(merged["verdict"], indent=1))
    print("merged " + str(len(rows)) + " rows -> " + str(out))
    return merged


def _verdict(rows):
    """First size where each alternative beats the CPU dense LU baseline."""
    def first_win(key):
        for row in rows:
            base = row["backends"].get("cpu_lu", {}).get("warm_median")
            other = row["backends"].get(key, {}).get("warm_median")
            if base and other and other < base:
                return {"n": row["n"], "speedup": round(base / other, 3)}
        return None
    out = {}
    for key in ("torch_cpu", "torch_cuda", "cuda_lu"):
        out[key + "_first_win_over_cpu_lu"] = first_win(key)
    if rows:
        last = rows[-1]["backends"]
        if "cuda_lu" in last and "cpu_lu" in last:
            out["largest_measured_speedup_cuda_lu"] = round(
                last["cpu_lu"]["warm_median"] / last["cuda_lu"]["warm_median"], 3)
        if "gmres_dense" in last and "cpu_lu" in last:
            out["gmres_vs_cpu_lu_at_largest"] = round(
                last["gmres_dense"]["seconds"] / last["cpu_lu"]["warm_median"], 3)
    return out


def _fixture_sizes():
    path = ROOT / "docs/issue92/phase0_profile.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for name, record in payload.get("fixtures", {}).items():
        counters = record.get("counters") or {}
        if "matrix_shape" in counters:
            out[name] = {"n_wall_faces": counters["n_wall_faces"],
                         "unknowns": counters["matrix_shape"][1],
                         "rows": counters["matrix_shape"][0],
                         "A_bytes_mb": counters.get("A_bytes_mb")}
    return out


if __name__ == "__main__":
    main()