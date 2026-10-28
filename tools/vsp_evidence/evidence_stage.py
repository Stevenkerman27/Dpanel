"""Stage the issue-49 VSPAERO evidence tree under docs/issue49_vspaero/ (committed).

`evidence_build.py` produces the numeric tables from runs/; this script copies the RAW
files that back them into the repository and writes `sha256.json` over the committed tree,
so the hashes a reviewer verifies are the hashes of the files they can actually read.

Usage:  .conda_env\\python.exe runs\\vsp_conv\\evidence_stage.py
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "runs" / "vsp_conv"
DEST = ROOT / "docs" / "issue49_vspaero"

LEVELS = ["l1", "l2", "l3", "l4"]
DPANEL = ["dp_s020", "dp_s014", "dp_s010", "dp_s007", "dp_s005"]
RAW_FILES = [
    "trap.vspaero", "trap.vspgeom", "trap.vkey", "trap.polar", "trap.lod",
    "trap.history", "trap.tec.dat", "trap.group.0", "trap.group.1",
    "run.txt", "stdout.log",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> int:
    copied = 0
    for tag in LEVELS:
        for name in RAW_FILES:
            if copy(SRC / "evidence" / tag / name, DEST / "raw" / tag / name):
                copied += 1
    for case_dir in sorted(p for p in SRC.glob("api_*") if p.is_dir()):
        for name in ("ladder.json", "trap.vspaero", "trap.polar", "trap.lod",
                     "trap.group.0", "trap.group.1", "stdout.log"):
            if copy(case_dir / name, DEST / "ladders" / case_dir.name / name):
                copied += 1
    for tag in DPANEL:
        if copy(SRC / tag / "case.json", DEST / "dpanel" / (tag + ".json")):
            copied += 1
    for fig in sorted((SRC / "figs").glob("*.png")):
        if copy(fig, DEST / "figs" / fig.name):
            copied += 1
    entries = []
    for path in sorted(DEST.rglob("*")):
        if not path.is_file() or path.name in ("sha256.json", "README.md"):
            continue
        rel = path.relative_to(ROOT).as_posix()
        entries.append(dict(path=rel, bytes=path.stat().st_size, sha256=sha256(path)))
    (DEST / "sha256.json").write_text(__import__("json").dumps(entries, indent=1), encoding="utf-8")
    total = sum(e["bytes"] for e in entries)
    print(f"staged {copied} files, manifest {len(entries)} entries, {total / 1e6:.1f} MB")
    for e in entries:
        if e["bytes"] > 200_000:
            print(f"  {e['bytes'] / 1e6:7.2f} MB  {e['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())