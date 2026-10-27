"""Reader / checker for VSPAERO `.vspgeom` surface meshes.

Format (v3)::

    # vspgeom v3 | 1
    1                                  <- (line 1) version marker line count
    <npoints> <npanels> <symmetry>     <- line 2
    x y z                              <- npoints lines
    <k> <i1> ... <ik>                  <- npanels lines, 1-based node ids

Panels may be triangles, quads or mixed.  Everything here is pure NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class VspGeom:
    points: np.ndarray                      # (n, 3) float64
    panels: List[np.ndarray]                # 0-based node ids, per polygon
    symmetry: int = 1
    path: Optional[Path] = None
    header: Sequence[str] = field(default_factory=tuple)
    extra_lines: int = 0

    # ------------------------------------------------------------------ info
    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_panels(self) -> int:
        return len(self.panels)

    def panel_types(self) -> dict:
        counts: dict = {}
        for panel in self.panels:
            counts[len(panel)] = counts.get(len(panel), 0) + 1
        return dict(sorted(counts.items()))

    def area(self) -> float:
        total = 0.0
        for panel in self.panels:
            q = self.points[panel]
            if len(q) == 3:
                total += 0.5 * float(np.linalg.norm(np.cross(q[1] - q[0], q[2] - q[0])))
            else:
                for j in range(1, len(q) - 1):
                    total += 0.5 * float(np.linalg.norm(np.cross(q[j] - q[0], q[j + 1] - q[0])))
        return total

    def bbox(self) -> tuple:
        return self.points.min(axis=0), self.points.max(axis=0)


def read_vspgeom(path) -> VspGeom:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines or not lines[0].lstrip().startswith("# vspgeom"):
        raise ValueError(f"Not a vspgeom file: {path}")
    n_points, n_panels, symmetry = (int(v) for v in lines[2].split()[:3])
    first = 3
    points = np.array(
        [[float(v) for v in lines[first + i].split()[:3]] for i in range(n_points)],
        dtype=float,
    )
    cursor = first + n_points
    # VSPAERO 3.55 echoes the panel count on its own line before the records
    header_fields = lines[cursor].split() if cursor < len(lines) else []
    if len(header_fields) == 1 and header_fields[0].isdigit() \
            and int(header_fields[0]) == n_panels:
        cursor += 1
    panels: List[np.ndarray] = []
    for i in range(n_panels):
        fields = lines[cursor + i].split()
        k = int(fields[0])
        ids = np.array([int(v) - 1 for v in fields[1:1 + k]], dtype=int)
        if len(ids) != k:
            raise ValueError(f"Panel {i} in {path} declares {k} nodes but lists {len(ids)}")
        if ids.min() < 0 or ids.max() >= n_points:
            raise ValueError(f"Panel {i} in {path} references a node outside [1, {n_points}]")
        panels.append(ids)
    extra = lines[cursor + n_panels:]
    return VspGeom(points=points, panels=panels, symmetry=symmetry, path=path,
                   header=tuple(lines[:3]), extra_lines=len(extra))


def write_vspgeom(geom: VspGeom, path, *, symmetry: Optional[int] = None) -> Path:
    path = Path(path)
    flag = geom.symmetry if symmetry is None else int(symmetry)
    out = ["# vspgeom v3 | 1", "1", f"{geom.n_points} {geom.n_panels} {flag}"]
    out += ["%.10g %.10g %.10g" % tuple(p) for p in geom.points]
    out += ["%d %s" % (len(panel), " ".join(str(int(i) + 1) for i in panel))
            for panel in geom.panels]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


def measure(geom: VspGeom) -> dict:
    lo, hi = geom.bbox()
    return {
        "points": geom.n_points,
        "panels": geom.n_panels,
        "symmetry": int(geom.symmetry),
        "panel_types": geom.panel_types(),
        "area": geom.area(),
        "bbox_min": [float(v) for v in lo],
        "bbox_max": [float(v) for v in hi],
    }


def mesh_quality(geom: VspGeom, *, min_edge: float = 1e-9) -> dict:
    """Edge-length / aspect statistics used by DEP-OPT's geometry gate."""
    edges_all = []
    min_edge_seen = float("inf")
    max_aspect = 0.0
    degenerate = 0
    for panel in geom.panels:
        q = geom.points[panel]
        edges = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)
        edges_all.append(edges)
        current_min = float(edges.min())
        min_edge_seen = min(min_edge_seen, current_min)
        if current_min <= min_edge:
            degenerate += 1
            continue
        max_aspect = max(max_aspect, float(edges.max() / current_min))
    edges_all = np.concatenate(edges_all) if edges_all else np.zeros(0)
    areas = np.array([_panel_area(geom.points[p]) for p in geom.panels])
    return {
        "min_edge": min_edge_seen,
        "max_aspect": max_aspect,
        "degenerate_panels": degenerate,
        "edge_min": float(edges_all.min()) if edges_all.size else 0.0,
        "edge_median": float(np.median(edges_all)) if edges_all.size else 0.0,
        "edge_max": float(edges_all.max()) if edges_all.size else 0.0,
        "area_min": float(areas.min()) if areas.size else 0.0,
        "area_median": float(np.median(areas)) if areas.size else 0.0,
        "area_max": float(areas.max()) if areas.size else 0.0,
    }


def _panel_area(q: np.ndarray) -> float:
    if len(q) == 3:
        return 0.5 * float(np.linalg.norm(np.cross(q[1] - q[0], q[2] - q[0])))
    total = 0.0
    for j in range(1, len(q) - 1):
        total += 0.5 * float(np.linalg.norm(np.cross(q[j] - q[0], q[j + 1] - q[0])))
    return total


def open_boundary_edges(geom: VspGeom) -> np.ndarray:
    """Edges used by exactly one panel (0-based node id pairs)."""
    counts: dict = {}
    for panel in geom.panels:
        n = len(panel)
        for j in range(n):
            a, b = int(panel[j]), int(panel[(j + 1) % n])
            key = (a, b) if a < b else (b, a)
            counts[key] = counts.get(key, 0) + 1
    return np.array([k for k, c in counts.items() if c == 1], dtype=int).reshape(-1, 2)


def validate_mesh(geom: VspGeom, *, min_edge: float = 1e-9,
                  max_aspect: float = 1.0e3) -> dict:
    quality = mesh_quality(geom, min_edge=min_edge)
    if quality["degenerate_panels"]:
        raise RuntimeError(
            f"{geom.path}: {quality['degenerate_panels']} panels with edge <= {min_edge:g} m")
    if quality["max_aspect"] > max_aspect:
        raise RuntimeError(
            f"{geom.path}: aspect ratio {quality['max_aspect']:.6g} > {max_aspect:g}")
    return quality


def mirror_about_y0(geom: VspGeom, *, tol: float = 1.0e-9,
                    symmetry_flag: int = 0) -> VspGeom:
    """Mirror a half model into a full one, welding the y = 0 nodes.

    Nodes on the symmetry plane are shared; every other node gets a mirrored
    twin and mirrored panels reverse their winding so normals stay outward.
    """
    points = geom.points
    on_plane = np.abs(points[:, 1]) <= tol
    extra = [i for i in range(geom.n_points) if not on_plane[i]]
    mirror_of = np.arange(geom.n_points)
    mirror_of[np.asarray(extra, dtype=int)] = geom.n_points + np.arange(len(extra))
    new_points = np.vstack([
        points,
        np.column_stack([points[extra][:, 0], -points[extra][:, 1], points[extra][:, 2]]),
    ]) if extra else points.copy()
    panels = [np.asarray(p, dtype=int) for p in geom.panels]
    panels += [mirror_of[np.asarray(p, dtype=int)][::-1] for p in geom.panels]
    return VspGeom(points=new_points, panels=panels, symmetry=int(symmetry_flag),
                   path=None, header=geom.header)