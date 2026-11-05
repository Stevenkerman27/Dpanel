"""Backport the NO_AVX512 Ryzen-tuning guard to one measured Windows DLL.

Stage and test before installing:
  python -m tools.blas_runtime_fix --out runs/verification/native_blas/runtime_fix
  python -m tools.blas_probe --dll runs/verification/native_blas/runtime_fix/openblas.dll --cores auto --out runs/verification/native_blas/candidate
  python -m tools.blas_runtime_fix --out runs/verification/native_blas/runtime_fix --install

This is a local binary hotfix, not a new upstream/conda package. Only the exact
conda-forge 0.3.34 pthreads_h877e47f_0 DLL is accepted. Installing a different
OpenBLAS package requires checking that its source includes the upstream fix.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--install", action="store_true")
args = parser.parse_args()
if sys.platform != "win32":
    parser.error("requires the measured Windows x64 runtime")

original_sha = "61333f15528c05e0684e1efcdaebc44b4cceb3cae5fcd265f9370a654715289c"
fixed_sha = "e0d23e1c32e26f05719d98d92c9aa26a303a83ba0b66463099677459e48223e8"
target = Path(sys.prefix) / "Library/bin/openblas.dll"
original = target.read_bytes()
current_sha = hashlib.sha256(original).hexdigest()
if current_sha == fixed_sha:
    print("Runtime already has the verified NO_AVX512 local hotfix.")
    sys.exit(0)
if current_sha != original_sha:
    raise ValueError("Unknown runtime SHA-256; refusing to modify it: " + current_sha)

# Each JE +0x44 skips the four Ryzen P/Q overrides when AVX512 is absent.
# Change JE to JMP in all 11 compiled initializers, equivalent to excluding this
# tuning block with #ifndef NO_AVX512. Kernel code and ABI are untouched.
offsets = [0x449e53, 0x4db2bb, 0x5ce0d0, 0x6ad167, 0x813d34, 0x905bbb,
           0x97d81b, 0x9f6e1b, 0xa6761b, 0xb87cc1, 0xbdfc41]
signature = bytes.fromhex("f7c300000100744448b88001000000020000488905")
if original.count(signature) != len(offsets):
    raise ValueError("Unexpected number of Ryzen tuning blocks")
fixed = bytearray(original)
for offset in offsets:
    if original[offset - 6:offset - 6 + len(signature)] != signature:
        raise ValueError("Unexpected instruction at file offset " + hex(offset))
    fixed[offset] = 0xeb
if hashlib.sha256(fixed).hexdigest() != fixed_sha:
    raise ValueError("Candidate checksum differs from the verified hotfix")

args.out.mkdir(parents=True, exist_ok=True)
candidate = args.out.resolve() / "openblas.dll"
if candidate == target.resolve():
    raise ValueError("--out must be a staging directory outside Library/bin")
candidate.write_bytes(fixed)
receipt = dict(target=str(target), candidate=str(candidate), original_sha256=original_sha,
               fixed_sha256=fixed_sha, changed_bytes=len(offsets),
               file_offsets=[hex(offset) for offset in offsets],
               upstream_commit="02f585444a0fba2505cfda77cc6684de008ea477",
               scope="NO_AVX512 guard; local binary backport, not upstream build",
               installed=False)
(args.out / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
if args.install:
    # Replace the directory entry, never mutate a conda package-cache hardlink.
    # Windows rejects replacement if another process has the old DLL mapped.
    temporary = target.with_name("openblas.no_avx512_hotfix.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        temporary.write_bytes(fixed)
        if hashlib.sha256(target.read_bytes()).hexdigest() != original_sha:
            raise ValueError("Runtime changed during installation")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    receipt["installed"] = hashlib.sha256(target.read_bytes()).hexdigest() == fixed_sha
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    if not receipt["installed"]:
        raise ValueError("Installed runtime checksum mismatch")
print(json.dumps(dict(candidate=str(candidate), installed=receipt["installed"],
                      fixed_sha256=fixed_sha, changed_bytes=len(offsets))))
