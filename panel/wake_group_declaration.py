"""Parse and validate a dpanel.half_wake_groups_declaration.v1 file.

Kept free of gmsh imports so the declaration contract can be tested without
a mesher: cad_half_groups.py uses it to build the OCC wake sheets, and the
unit tests exercise it directly. All coordinates in the declaration and in
the returned structures are physical SI metres; the OCC builder normalizes
them by the reference chord itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCHEMA = "dpanel.half_wake_groups_declaration.v1"
REQUIRED = ("group_id", "points", "z")


def positive_trailing_edge(te, *, tolerance=1e-9):
    """Restrict a full-span declared TE to y>=0 and pin the root point.

    The returned polyline is intended for a *sheet-relative* frame: callers
    that must satisfy a z=0-only helper should shift z first and shift back.
    """
    te = np.asarray(te, dtype=float)
    if te.ndim != 2 or te.shape[1] != 3 or len(te) < 2:
        raise ValueError("Trailing edge must be an (n, 3) polyline")
    if not np.isfinite(te).all():
        raise ValueError("Trailing edge must be finite")
    if np.any(np.diff(te[:, 1]) <= 0):
        raise ValueError("Trailing edge must have strictly increasing y")
    if te[0, 1] > tolerance or te[-1, 1] <= tolerance:
        raise ValueError("Full CAD trailing edge must span positive and negative y")
    positive = te[te[:, 1] > tolerance]
    if len(positive) < 1:
        raise ValueError("Missing positive-y trailing-edge stations")
    x0 = float(np.interp(0., te[:, 1], te[:, 0]))
    root = np.asarray([x0, 0., float(np.interp(0., te[:, 1], te[:, 2]))])
    at0 = te[np.abs(te[:, 1]) <= tolerance]
    if len(at0) and np.max(np.linalg.norm(at0-root, axis=1)) > tolerance:
        raise ValueError("Declared TE does not pass through a unique point at y=0")
    result = np.vstack((root, positive))
    if np.any(np.diff(result[:, 1]) <= 0):
        raise ValueError("Nonmonotone positive-y trailing edge")
    return result


def merged_stations(te_y, strips, *, tolerance=1e-9):
    """Station union without float near-duplicates.

    np.unique keeps 0.3999999999999999 and 0.4 apart, which would create a
    zero-width wake strip and an OCC addLine failure.
    """
    values = np.asarray(te_y, dtype=float).reshape(-1)
    if values.size < 2 or np.any(np.diff(values) <= 0):
        raise ValueError("Trailing-edge y values must be strictly increasing")
    grid = np.linspace(0., values[-1], int(strips) + 1)
    merged = np.unique(np.round(np.r_[grid, values], 12))
    if merged[0] < -tolerance or merged[-1] > values[-1] + tolerance:
        raise ValueError("Trailing-edge y values must span 0..tip")
    keep = np.r_[True, np.diff(merged) > tolerance]
    return merged[keep]


def read_wake_groups(path, *, chord, origin):
    """Read a declaration into normalized per-group records.

    Each record carries both the normalized TE (divided by the reference
    chord and shifted by ``origin``) used by the OCC builder and the physical
    TE used for the exported contract.
    """
    path = Path(path)
    declaration = json.loads(path.read_text(encoding="utf-8"))
    schema = declaration.get("schema")
    if schema not in (None, SCHEMA):
        raise ValueError(f"Unsupported wake-group declaration schema: {schema!r}")
    if chord <= 0 or not np.isfinite(chord):
        raise ValueError("Reference chord must be positive and finite")
    origin = np.asarray(origin, dtype=float)
    if origin.shape != (3,) or abs(origin[1]) > 1e-12 or not np.isfinite(origin).all():
        raise ValueError("Declared origin must be symmetric and finite")
    items = declaration.get("groups") or []
    if not items:
        raise ValueError("Wake-group declaration contains no groups")
    identifiers = [str(item.get("group_id")) for item in items]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate wake group_id")
    groups = []
    for item in items:
        missing = [key for key in REQUIRED if key not in item]
        if missing:
            raise ValueError(f"Wake group declaration is missing {missing}")
        physical = np.asarray(item["points"], dtype=float)
        if physical.ndim != 2 or physical.shape[1] != 3 or len(physical) < 2:
            raise ValueError(f"Wake group {item['group_id']!r} has an invalid TE polyline")
        z = float(item["z"])
        if not np.isfinite(z):
            raise ValueError(f"Wake group {item['group_id']!r} has a non-finite z")
        normalized = (physical-origin)/chord
        normalized_z = (z-origin[2])/chord
        shifted = normalized.copy()
        shifted[:, 2] -= normalized_z
        positive = positive_trailing_edge(shifted)
        positive[:, 2] += normalized_z
        groups.append(dict(
            group_id=str(item["group_id"]),
            role=item.get("role"),
            kind=str(item.get("kind", "attached_lifting_TE")),
            root_condition=str(item.get("root_condition", "symmetry_plane")),
            buried_inside_component=item.get("buried_inside_component"),
            source_component_ids=tuple(int(v) for v in item.get("source_component_ids", ())),
            normalized_te=positive,
            physical_te=positive*chord + origin,
            normalized_z=normalized_z,
            physical_z=z,
        ))
    planes = [round(float(group["normalized_z"]), 9) for group in groups]
    if len(set(planes)) != len(planes):
        raise ValueError("Two wake groups share the same z plane")
    return declaration, groups