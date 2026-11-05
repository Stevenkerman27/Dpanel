"""Check installed SciPy BLAS/LAPACK across precisions and K-block boundaries.

  python -m tools.blas_verify_runtime --out runs/verification/native_blas/fixed_scipy.json

Run in a fresh process, without OPENBLAS_CORETYPE or a diagnostic table write.
NumPy's separate wheel runtime supplies the GEMM reference.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import linalg


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
dll = Path(sys.prefix) / "Library/bin/openblas.dll"
blas = ctypes.CDLL(str(dll))
blas.openblas_get_corename.restype = ctypes.c_char_p
report = dict(dll=str(dll), dll_sha256=hashlib.sha256(dll.read_bytes()).hexdigest(),
              core=blas.openblas_get_corename().decode(), checks=[])
rng = np.random.default_rng(20261105)
for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
    tolerance = 1e-4 if dtype in [np.float32, np.complex64] else 1e-10
    for k in [127, 128, 129, 255, 256, 257, 513, 1024]:
        a = rng.standard_normal((73, k))
        b = rng.standard_normal((55, k))
        if np.issubdtype(dtype, np.complexfloating):
            a = a + 1j * rng.standard_normal(a.shape)
            b = b + 1j * rng.standard_normal(b.shape)
        a = np.asfortranarray(a, dtype=dtype)
        b = np.asfortranarray(b, dtype=dtype)
        a *= 0.02
        b *= 0.02
        gemm = linalg.blas.get_blas_funcs("gemm", (a, b))
        c = gemm(1, a, b, trans_b=2)
        error = float(np.max(np.abs(c - a @ b.conj().T)))
        report["checks"].append(dict(operation="gemm", dtype=np.dtype(dtype).name,
                                     k=k, max_abs_error=error, correct=error < tolerance))
    a = rng.standard_normal((513, 513))
    if np.issubdtype(dtype, np.complexfloating):
        a = a + 1j * rng.standard_normal(a.shape)
    a = np.asfortranarray(a, dtype=dtype)
    a /= np.sqrt(513)
    a[np.diag_indices(513)] += 2
    expected = np.ones(513, dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        expected += 0.5j
    rhs = a @ expected
    x = linalg.solve(a, rhs)
    if x.dtype != dtype:
        raise ValueError("LAPACK test unexpectedly changed precision")
    error = float(np.max(np.abs(x - expected)))
    report["checks"].append(dict(operation="solve", dtype=np.dtype(dtype).name,
                                 size=513, max_abs_error=error, correct=error < tolerance))

report["correct"] = all(row["correct"] for row in report["checks"])
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(dict(core=report["core"], checks=len(report["checks"]),
                      correct=report["correct"], out=str(args.out))))
sys.exit(0 if report["correct"] else 1)
