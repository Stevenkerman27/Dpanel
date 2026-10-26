"""Opt-in geometric mesh-size callback; no flow solution or CAD topology edits.

Coordinates in the feature JSON are SI (original CAD frame); the Gmsh model
uses (xyz-origin)/reference_chord. This policy is only a *request* to Gmsh:
actual triangle sizes and wall/TE conformity must be audited after meshing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCHEMA = "dpanel.mesh_features.v1"


def _polyline(points, name):
    p = np.asarray(points, dtype=float)
    if (p.ndim != 2 or p.shape[1] != 3 or len(p) < 2 or
            not np.isfinite(p).all()):
        raise ValueError(f"{name}: expected finite (N>=2,3) polyline")
    edges = np.diff(p, axis=0)
    length2 = np.einsum("ij,ij->i", edges, edges)
    if np.any(length2 < 1e-20):
        raise ValueError(f"{name}: zero-length polyline segment")
    return p[:-1].copy(), edges, length2


def _distance(point, segments):
    start, delta, length2 = segments
    rel = point - start
    frac = np.clip(np.einsum("ij,ij->i", rel, delta) / length2, 0., 1.)
    error = rel - frac[:, None] * delta
    return float(np.sqrt(np.min(np.einsum("ij,ij->i", error, error))))


class FeatureSizePolicy:
    """Deterministic C1 distance bands for explicit LE/TE/tip curves.

    The callback returns the original lc outside each band. Inside a band it
    only *reduces* the old value; it never loosens existing background fields.
    """

    def __init__(self, declaration, *, origin, chord, wall_size):
        if declaration.get("schema") != SCHEMA or declaration.get("units") != "m":
            raise ValueError("Require dpanel.mesh_features.v1 with SI metre coordinates")
        origin = np.asarray(origin, dtype=float)
        if (origin.shape != (3,) or not np.isfinite(origin).all() or
                not np.isfinite([chord, wall_size]).all() or
                chord <= 0 or wall_size <= 0):
            raise ValueError("Invalid mesh-feature normalization")
        entries = declaration.get("features")
        if not isinstance(entries, list) or not 1 <= len(entries) <= 16:
            raise ValueError("Require 1 to 16 named sizing features")
        seen = set()
        self.wall_size = float(wall_size)
        self.features = []
        budget = 0
        for item in entries:
            if not isinstance(item, dict):
                raise ValueError("Sizing feature must be a JSON object")
            name = item.get("name")
            if (not isinstance(name, str) or not name or name in seen):
                raise ValueError("Sizing feature names must be unique and nonempty")
            seen.add(name)
            ratio = float(item.get("target_ratio", float("nan")))
            radius = float(item.get("radius_chords", float("nan")))
            if (not np.isfinite([ratio, radius]).all() or
                    not (1 / 8 <= ratio <= 1) or not radius > 0):
                raise ValueError(f"{name}: target_ratio in [1/8,1], radius_chords > 0 required")
            points = np.asarray(item.get("points"), dtype=float)
            segments = _polyline((points - origin) / chord, name)
            budget += len(points) - 1
            if budget > 1000:
                raise ValueError("Mesh size callback exceeds 1000 polyline segments")
            self.features.append((name, ratio, radius, segments))

    def size(self, xyz, lc):
        point = np.asarray(xyz, dtype=float)
        if (point.shape != (3,) or not np.isfinite(point).all() or
                not np.isfinite(lc) or lc <= 0):
            raise ValueError("Invalid Gmsh mesh-size callback argument")
        result = float(lc)
        for _, ratio, radius, line in self.features:
            distance = _distance(point, line)
            if distance >= radius:
                continue
            # C1 Hermite easing at both ends of each influence band.
            t = 1 - distance / radius
            weight = t * t * (3 - 2 * t)
            size = self.wall_size * (1 - (1 - ratio) * weight)
            result = min(result, size)
        return result

    def __call__(self, dim, tag, x, y, z, lc):
        # Applied on CAD curves AND surfaces: a dense surface interior cannot
        # compensate for a coarse physical leading/trailing-edge curve mesh.
        return self.size((x, y, z), lc)


def read_policy(path, *, origin, chord, wall_size):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return FeatureSizePolicy(data, origin=origin, chord=chord, wall_size=wall_size)
