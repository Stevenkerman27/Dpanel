"""Parsers for VSPAERO output files: `.polar`, `.history`, `.lod`, `.tec.dat`.

All four were reverse-engineered against VSPAERO 3.52 output; the parsing is
column-name driven (never positional) so extra columns do not break it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


def _table_after(header_predicate, lines: Sequence[str]) -> List[Dict[str, float]]:
    index = next((i for i, line in enumerate(lines) if header_predicate(line)), None)
    if index is None:
        raise ValueError("Could not locate the table header")
    columns = lines[index].split()
    rows: List[Dict[str, float]] = []
    for line in lines[index + 1:]:
        fields = line.split()
        if len(fields) != len(columns):
            continue
        try:
            rows.append({name: float(value) for name, value in zip(columns, fields)})
        except ValueError:
            continue
    return rows


def read_polar(path) -> List[Dict[str, float]]:
    """One dict per alpha.  Header line starts with `Beta `.

    In thick-panel mode only `CLwtot` / `CDwtot` (wake-induced, Kutta-Joukowski)
    are trustworthy; `CLi` / `CDi` can be wrong by orders of magnitude when the
    chordwise clustering is bad (see `vspapi.health`).
    """
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    return _table_after(lambda line: line.strip().startswith("Beta "), lines)


def read_history(path) -> List[Dict[str, float]]:
    """Wake-iteration history: one dict per iteration per alpha."""
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    return _table_after(
        lambda line: line.split()[:3] == ["Iter", "Mach", "AoA"], lines)


def read_lod(path, n_alpha: Optional[int] = None) -> List[dict]:
    """Spanwise strip loads.

    VSPAERO writes one block of `TrailVort` 1..N strips **per alpha**, with no
    alpha column, so the blocks must be split by alpha count.
    """
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    rows = _table_after(
        lambda line: line.split()[:3] == ["Iter", "VortexSheet", "TrailVort"], lines)
    if not rows:
        return []
    if n_alpha is None:
        # a new block starts whenever TrailVort goes back to 1
        starts = [0] + [i for i in range(1, len(rows))
                        if rows[i]["TrailVort"] <= rows[i - 1]["TrailVort"]]
        blocks = [rows[s:(starts[i + 1] if i + 1 < len(starts) else len(rows))]
                  for i, s in enumerate(starts)]
    else:
        if n_alpha <= 0 or len(rows) % n_alpha:
            raise ValueError(f"{len(rows)} strip rows not divisible by {n_alpha} alpha values")
        per = len(rows) // n_alpha
        blocks = [rows[k * per:(k + 1) * per] for k in range(n_alpha)]
    out = []
    for block in blocks:
        cl = np.array([r["Cl"] for r in block])
        cd = np.array([r["Cd"] for r in block])
        dy = np.array([r["dSpan"] for r in block])
        out.append({
            "strips": len(block),
            "sum_cl_dspan": float((cl * dy).sum()),
            "sum_cd_dspan": float((cd * dy).sum()),
            "cl_max": float(cl.max()),
            "cl": cl,
            "cd": cd,
            "dspan": dy,
            "y": np.array([r["Yavg"] for r in block]),
        })
    return out


def read_tecplot(path) -> dict:
    """Parse the single-zone BLOCK-format tecplot file VSPAERO writes.

    Returns nodal coordinates, one cell-centred scalar (Cp) and the triangle
    connectivity.  There is **no wake zone** in this file: only the wall.
    """
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    variables: List[str] = []
    zone: dict = {}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.upper().startswith("VARIABLES"):
            variables = [v.strip().strip('"') for v in stripped.split("=", 1)[1].split(",")]
        elif stripped.upper().startswith("ZONE") and not zone:
            zone = {"index": i}
            for token in stripped.split():
                if "=" in token:
                    key, _, value = token.partition("=")
                    zone[key.strip().upper()] = value.strip().strip('"').rstrip(",")
    if not zone:
        raise ValueError(f"No ZONE record in {path}")
    n_points = int(zone["N"])
    n_cells = int(zone["E"])
    start = zone["index"] + 1
    while start < len(lines):
        try:
            float(lines[start].split()[0])
            break
        except (ValueError, IndexError):
            start += 1
    # every scalar block is one value per line, the connectivity is three ints
    # per line, so flattening token-wise is the layout-agnostic reading
    tokens = [t for line in lines[start:] for t in line.split()]
    values = np.array([float(t) for t in tokens], dtype=float)
    # x, y, z (nodal) then the cell-centred scalar
    expected = 3 * n_points + n_cells + 3 * n_cells
    if values.size != expected:
        raise ValueError(f"{path}: expected {expected} numbers, found {values.size}")
    xyz = values[: 3 * n_points].reshape(3, n_points).T
    cp = values[3 * n_points: 3 * n_points + n_cells]
    faces = values[-3 * n_cells:].astype(int).reshape(n_cells, 3) - 1
    return {"variables": variables, "points": xyz, "cell_values": cp, "faces": faces}