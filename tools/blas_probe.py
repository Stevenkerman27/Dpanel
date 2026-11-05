"""Isolate native BLAS failures from geometry and JAX.

  python -m tools.blas_probe --case lu --size 4612 --out runs/verification/blas_lu
  python -m tools.blas_probe --case gemm --size 1024 --out runs/verification/blas_gemm

Each core/repetition runs in a fresh interpreter because a bad kernel kills it.
This Windows diagnostic uses the current environment's OpenBLAS DLL directly.
"""
import argparse
import ctypes
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case", choices=["lu", "gemm", "jax", "kernel"], default="lu")
parser.add_argument("--size", type=int, default=4612)
parser.add_argument("--cores", nargs="+", default=["Haswell", "Sandybridge"])
parser.add_argument("--repeat", type=int, default=3)
parser.add_argument("--threads", type=int, default=1)
parser.add_argument("--timeout", type=float, default=60)
parser.add_argument("--dll", type=Path, help="candidate DLL to test before installation")
parser.add_argument("--gemm-q", type=int, help="diagnostic override for the measured DLL only")
parser.add_argument("--matrix", choices=["identity", "random"], default="identity")
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
args = parser.parse_args()
if sys.platform != "win32" or min(args.size, args.repeat, args.threads, args.timeout) <= 0:
    parser.error("requires Windows and positive size/repeat/threads/timeout")
if args.gemm_q is not None and not 1 <= args.gemm_q <= 512:
    parser.error("--gemm-q must be between 1 and 512")
if not args.worker and args.case == "kernel" and args.cores != ["Haswell"]:
    parser.error("the direct Haswell kernel test requires --cores Haswell")
args.out.mkdir(parents=True, exist_ok=True)
dll_path = (args.dll or Path(sys.prefix) / "Library/bin/openblas.dll").resolve()
dll_sha = hashlib.sha256(dll_path.read_bytes()).hexdigest()
original_sha = "61333f15528c05e0684e1efcdaebc44b4cceb3cae5fcd265f9370a654715289c"
fixed_sha = "e0d23e1c32e26f05719d98d92c9aa26a303a83ba0b66463099677459e48223e8"

if args.worker:
    faulthandler.enable()
    import numpy as np

    dll_search = os.add_dll_directory(str(Path(sys.prefix) / "Library/bin"))
    blas = ctypes.CDLL(str(dll_path))
    blas.openblas_get_config.restype = ctypes.c_char_p
    blas.openblas_get_corename.restype = ctypes.c_char_p
    tuning = None
    if args.gemm_q is not None and dll_sha != original_sha:
        raise ValueError("diagnostic table writes require the original measured DLL")
    if dll_sha in (original_sha, fixed_sha):
        # dgemm_p/q/r + unroll fields in this exact DLL's exported gotoblas table.
        table = ctypes.c_void_p.in_dll(blas, "gotoblas").value
        tuning = list((ctypes.c_int * 6).from_address(table + 0x2f0))
        if args.gemm_q is not None:
            ctypes.c_int.from_address(table + 0x2f4).value = args.gemm_q
    elif args.gemm_q is not None:
        raise ValueError("unknown DLL: do not write a hardcoded table offset")
    print(json.dumps(dict(config=blas.openblas_get_config().decode(),
                          core=blas.openblas_get_corename().decode(),
                          size=args.size, case=args.case, tuning_before=tuning,
                          tuning_after=list((ctypes.c_int * 6).from_address(table + 0x2f0)) if tuning else None,
                          dll=str(dll_path), dll_sha256=dll_sha,
                          threads=args.threads,
                          gemm_q_override=args.gemm_q)), flush=True)
    started = time.perf_counter()
    operation_seconds = None
    max_abs_error = None
    n = ctypes.c_int(args.size)
    pointer = ctypes.POINTER(ctypes.c_double)
    if args.case == "kernel":
        # Constant packed inputs make the internal kernel's expected result K.
        a = np.ones(4 * args.size, dtype=np.float64)
        b = np.ones(24 * args.size, dtype=np.float64)
        c = np.zeros(4 * 24, dtype=np.float64)
        kernel = blas.dgemm_kernel_HASWELL
        kernel.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                           ctypes.c_double, pointer, pointer, pointer, ctypes.c_int64]
        kernel.restype = None
        kernel(4, 24, args.size, 1., a.ctypes.data_as(pointer),
               b.ctypes.data_as(pointer), c.ctypes.data_as(pointer), 4)
        correct = np.array_equal(c, np.full(4 * 24, args.size))
    elif args.case == "lu":
        if args.matrix == "random":
            a = np.asfortranarray(np.random.default_rng(20261105).standard_normal((args.size, args.size)) / np.sqrt(args.size))
            a[np.diag_indices(args.size)] += 2.
        else:
            a = np.eye(args.size, dtype=np.float64, order="F")
        rhs = a.sum(axis=1)
        pivots = np.zeros(args.size, dtype=np.int32)
        info = ctypes.c_int()
        operation_started = time.perf_counter()
        blas.dgetrf_(ctypes.byref(n), ctypes.byref(n), a.ctypes.data_as(pointer),
                     ctypes.byref(n), pivots.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                     ctypes.byref(info))
        operation_seconds = time.perf_counter() - operation_started
        factor_info = info.value
        nrhs = ctypes.c_int(1)
        blas.dgetrs_(b"N", ctypes.byref(n), ctypes.byref(nrhs), a.ctypes.data_as(pointer),
                     ctypes.byref(n), pivots.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                     rhs.ctypes.data_as(pointer), ctypes.byref(n), ctypes.byref(info))
        max_abs_error = float(np.max(np.abs(rhs - 1.)))
        correct = factor_info == 0 and info.value == 0 and max_abs_error < 1e-10
    elif args.case == "gemm":
        a = np.eye(args.size, dtype=np.float64, order="F")
        c = np.zeros_like(a, order="F")
        one, zero = ctypes.c_double(1.), ctypes.c_double(0.)
        blas.dgemm_(b"N", b"N", ctypes.byref(n), ctypes.byref(n), ctypes.byref(n),
                    ctypes.byref(one), a.ctypes.data_as(pointer), ctypes.byref(n),
                    a.ctypes.data_as(pointer), ctypes.byref(n), ctypes.byref(zero),
                    c.ctypes.data_as(pointer), ctypes.byref(n))
        correct = np.array_equal(c, a)
    else:
        if dll_path != (Path(sys.prefix) / "Library/bin/openblas.dll").resolve():
            raise ValueError("JAX must test the installed DLL, not a staged DLL")
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        x = np.asarray(jnp.linalg.solve(jnp.eye(args.size), jnp.ones(args.size)))
        correct = np.array_equal(x, np.ones(args.size))
    print(json.dumps(dict(correct=bool(correct), seconds=time.perf_counter() - started,
                          operation_seconds=operation_seconds,
                          max_abs_error=max_abs_error)), flush=True)
    sys.exit(0 if correct else 1)

report = dict(python=sys.version, executable=sys.executable, case=args.case,
              size=args.size, matrix=args.matrix, dll=str(dll_path),
              threads=args.threads,
              gemm_q_override=args.gemm_q,
              dll_sha256=dll_sha, runs=[])
for core in args.cores:
    for repeat in range(args.repeat):
        log = args.out / ("%s_%d.log" % (core, repeat))
        env = dict(os.environ, OPENBLAS_NUM_THREADS=str(args.threads),
                   OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                   PYTHONIOENCODING="utf-8")
        if core == "auto":
            env.pop("OPENBLAS_CORETYPE", None)
        else:
            env["OPENBLAS_CORETYPE"] = core
        command = [sys.executable, "-X", "faulthandler", "-m", "tools.blas_probe",
                   "--worker", "--case", args.case, "--size", str(args.size),
                   "--out", str(args.out), "--matrix", args.matrix,
                   "--dll", str(dll_path), "--threads", str(args.threads)]
        if args.gemm_q is not None:
            command.extend(["--gemm-q", str(args.gemm_q)])
        with log.open("w", encoding="utf-8") as stream:
            try:
                child = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                       env=env, stdout=stream, stderr=subprocess.STDOUT,
                                       timeout=args.timeout)
                exit_code = child.returncode
            except subprocess.TimeoutExpired:
                stream.write("\nTimeout after %g seconds\n" % args.timeout)
                exit_code = 124
        report["runs"].append(dict(core=core, repeat=repeat, exit_code=exit_code,
                                   exit_hex="0x%08X" % (exit_code & 0xFFFFFFFF),
                                   log=str(log)))
        (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    codes = [row["exit_hex"] for row in report["runs"] if row["core"] == core]
    print("%s %s n=%d: %s" % (core, args.case, args.size, ", ".join(codes)), flush=True)
sys.exit(0 if all(row["exit_code"] == 0 for row in report["runs"]) else 1)
