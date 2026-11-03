"""CUDA float64 run of the batched influence kernel (issue 92, Phase 1 step 1).

The CPU parity harness writes the NumPy reference matrices; this script runs the same
kernel on the GPU, compares against those references and reports transfer time, kernel
time, peak VRAM and the end-to-end assembly time. It deliberately imports nothing from
the SciPy/SciPy-LAPACK side of the package: mixing an OpenBLAS-heavy process with CUDA
torch on this host trips a silent access violation, so the GPU run stays in its own
process and reads fixtures plus reference arrays from disk.

Run with the CUDA interpreter:
  C:/Users/zyx20/anaconda3/envs/myml/python.exe -m tools.issue92.influence_cuda
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

from panel.p1_batched_influence import influence_matrix, source_invariants

# tools.issue92.fixtures imports the Wing builders, which pull SciPy/OpenBLAS into the
# process; mixing that with CUDA torch on this host trips an access violation with no
# Python traceback. The GPU runner therefore reads the npz plus the manifest directly
# and re-verifies the file digest, which keeps the frozen-fixture guarantee.
FIXTURES = ROOT / "docs/issue92/fixtures"


def _load_fixture(name, arrays_dir=None):
    import hashlib

    arrays_dir = Path(arrays_dir) if arrays_dir is not None else FIXTURES
    path = arrays_dir / (name + ".npz")
    manifest_path = FIXTURES.parent / "fixtures_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["fixtures"][name]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["npz_sha256"]:
        raise ValueError("fixture npz digest drift for " + name)
    with np.load(path) as data:
        return {key: data[key] for key in data.files}, entry


def _compare(reference, value):
    reference = np.asarray(reference, float)
    value = np.asarray(value, float)
    diff = float(np.max(np.abs(reference - value)))
    scale = float(np.max(np.abs(reference)))
    return {"shape": list(reference.shape), "max_abs": diff,
            "max_rel_to_reference_scale": diff / scale if scale else None}


def _to_device(src, torch, device):
    out = {}
    bytes_moved = 0
    for key, value in src.items():
        moved = value.to(device)
        bytes_moved += moved.numel() * moved.element_size()
        out[key] = moved
    return out, bytes_moved


def _run(name, arrays, reference, torch, repeat):
    device = "cuda"
    tri = np.asarray(arrays["xyz"], float)[np.asarray(arrays["faces"], int)]
    centers = tri.mean(axis=1)
    dofs = np.asarray(arrays["dof_face"], int)
    ndof = int(arrays["ndof_mu"])
    nface = len(tri)

    src_cpu = source_invariants(tri, device="cpu")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    src_gpu, bytes_moved = _to_device(src_cpu, torch, device)
    torch.cuda.synchronize()
    t_transfer = time.perf_counter() - t0

    def run_a():
        return influence_matrix(tri, centers, self_faces=np.arange(nface), face_dofs=dofs,
                                ndof=ndof, device=device, src=src_gpu)

    run_a()
    torch.cuda.synchronize()
    t_cold = time.perf_counter() - t0
    best = float("inf")
    out_a = None
    for _ in range(max(1, repeat)):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out_a = run_a()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - start)

    wake_xyz = np.asarray(arrays["wake_xyz"], float)
    wake_faces = np.asarray(arrays["wake_faces"], int)
    gamma = torch.as_tensor(np.asarray(arrays["gamma_hat"], float), device=device)
    wake_src_cpu = source_invariants(wake_xyz[wake_faces], device="cpu")
    wake_src, wake_bytes = _to_device(wake_src_cpu, torch, device)

    def run_w():
        # The contraction stays on the device: taking the coefficients back to NumPy and
        # multiplying there would put a BLAS call in a process with a live CUDA context,
        # which crashes with an access violation on this host.
        coefficients = influence_matrix(wake_xyz[wake_faces], centers, face_dofs=wake_faces,
                                        ndof=len(wake_xyz), device=device, src=wake_src,
                                        return_tensor=True)
        return coefficients @ gamma

    run_w()
    torch.cuda.synchronize()
    best_w = float("inf")
    for _ in range(max(1, repeat)):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out_w = run_w()
        torch.cuda.synchronize()
        best_w = min(best_w, time.perf_counter() - start)

    return {"n_faces": int(nface), "n_wake_triangles": int(len(wake_faces)),
            "seconds": {"transfer_sources": t_transfer, "cold_assembly": t_cold,
                        "warm_A": best, "warm_W": best_w},
            "transfer_mb": bytes_moved / 1048576.0,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 1048576.0,
            "A": _compare(reference["A"], out_a),
            "W": _compare(reference["W"], out_w.detach().cpu().numpy())}


def main(argv=None):
    parser = argparse.ArgumentParser(description="CUDA parity for the batched influence kernel")
    parser.add_argument("--fixtures", default="g1_elliptic,g1_wingbody_cad")
    parser.add_argument("--reference", type=Path, default=ROOT / "runs/scratch/issue92_loose/reference")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/issue92/influence_cuda.json")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args(argv)
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this interpreter")
    payload = {"status": "ok", "device": torch.cuda.get_device_name(0),
               "torch": torch.__version__, "cuda": torch.version.cuda,
               "python": platform.python_version(), "repeat": args.repeat,
               "total_vram_mb": torch.cuda.get_device_properties(0).total_memory / 1048576.0,
               "fixtures": {}}
    for name in [item.strip() for item in args.fixtures.split(",") if item.strip()]:
        arrays, _entry = _load_fixture(name)
        reference_path = args.reference / (name + "_reference.npz")
        if not reference_path.exists():
            payload["fixtures"][name] = {"status": "reference-missing"}
            continue
        with np.load(reference_path) as data:
            reference = {key: data[key] for key in data.files}
        record = _run(name, arrays, reference, torch, args.repeat)
        record["status"] = "ok"
        payload["fixtures"][name] = record
        print(name, "A max_abs=%.3e" % record["A"]["max_abs"],
              "W max_abs=%.3e" % record["W"]["max_abs"],
              "warmA=%.3f s" % record["seconds"]["warm_A"],
              "transfer=%.3f s" % record["seconds"]["transfer_sources"],
              "peak=%.0f MB" % record["peak_vram_mb"], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())