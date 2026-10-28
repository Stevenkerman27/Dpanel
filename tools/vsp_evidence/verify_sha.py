"""Verify docs/issue49_vspaero/sha256.json against the files on disk.

Exit code 0 only when every recorded hash matches and no recorded file is missing.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs" / "issue49_vspaero" / "sha256.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ok = bad = missing = 0
    for entry in entries:
        path = ROOT / entry["path"]
        if not path.is_file():
            print("MISSING", entry["path"]); missing += 1; continue
        actual = sha256(path)
        if actual == entry["sha256"]:
            ok += 1
        else:
            print("MISMATCH", entry["path"], entry["sha256"][:16], actual[:16]); bad += 1
    print("verified %d  mismatch %d  missing %d  entries %d" % (ok, bad, missing, len(entries)))
    return 0 if bad == 0 and missing == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())