"""Read-only Gmsh triangle quality audit and explicit opt-in resource gates.

The input is the actual mesh AFTER OCC meshing. No geometry, triangle ordering,
pressure field or solution is changed. Both counts and quality are checked:
requesting a smaller sizing field alone is not an accuracy guarantee.
"""
from __future__ import annotations

import numpy as np


def _positive_int(value, label):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _limit_float(value, label, *, bottom, top):
    if value is None:
        return None
    try:
        n = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite and within bounds") from exc
    if not np.isfinite(n) or not (bottom < n < top):
        raise ValueError(f"{label} must be finite and within bounds")
    return n


def _triangle_geometry(vertices, faces, label):
    x, raw = np.asarray(vertices, dtype=np.float64), np.asarray(faces)
    if (x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all()
            or raw.ndim != 2 or raw.shape[1] != 3 or not len(raw)
            or not np.issubdtype(raw.dtype, np.integer)):
        raise ValueError(f"{label}: invalid triangle geometry")
    f = raw.astype(np.int64, copy=False)
    if f.min() < 0 or f.max() >= len(x):
        raise ValueError(f"{label}: triangle references invalid vertices")
    tri = x[f]
    e01, e12, e20 = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 1], tri[:, 0]-tri[:, 2]
    lengths = np.stack([np.linalg.norm(e, axis=1) for e in (e01, e12, e20)], axis=1)
    area = np.linalg.norm(np.cross(e01, -e20), axis=1)/2
    if (np.any(lengths <= 0) or np.any(area <= 0)
            or np.any(area <= 1e-14*np.max(lengths, axis=1)**2)):
        raise ValueError(f"{label}: degenerate or nearly collinear triangle")
    pairs = ((-e20, e01), (-e01, e12), (-e12, e20))
    angles = np.stack([np.degrees(np.arctan2(
        np.linalg.norm(np.cross(u, v), axis=1), np.einsum("ij,ij->i", u, v)))
        for u, v in pairs], axis=1)
    shape = 4*np.sqrt(3)*area / np.sum(lengths**2, axis=1)
    return f, area, angles, lengths, shape


def _shared_area_growth(faces, area):
    edge_to_face = {}
    for i, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (int(min(a,b)), int(max(a,b)))
            rows = edge_to_face.setdefault(edge, [])
            rows.append(i)
            if len(rows) > 2:
                raise ValueError("Nonmanifold wall edge in mesh quality audit")
    ratios = []
    for rows in edge_to_face.values():
        if len(rows) == 2:
            a,b = area[rows[0]],area[rows[1]]
            ratios.append(float(max(a,b)/min(a,b)))
    return np.asarray(ratios, dtype=float)


def audit_mesh_quality(vertices, wall_faces, wake_faces, *, max_wall_panels=None,
                       max_wake_panels=None, max_total_panels=None,
                       min_wall_angle_deg=None, max_wall_growth_p99=None):
    """Audit actual triangles; reject explicit limits without changing the mesh.

    Size/angle statistics are output in the coordinate units of vertices.
    max_wall_growth_p99 is a ratio of areas across shared physical wall edges.
    """
    caps = {
        "max_wall_panels": _positive_int(max_wall_panels,"max_wall_panels"),
        "max_wake_panels": _positive_int(max_wake_panels,"max_wake_panels"),
        "max_total_panels": _positive_int(max_total_panels,"max_total_panels"),
    }
    min_angle = _limit_float(min_wall_angle_deg, "min_wall_angle_deg", bottom=0, top=60)
    max_growth = _limit_float(max_wall_growth_p99, "max_wall_growth_p99", bottom=1, top=float("inf"))
    wall = np.asarray(wall_faces)
    wake = np.asarray(wake_faces)
    if wall.ndim != 2 or wall.shape[1] != 3 or wake.ndim != 2 or wake.shape[1] != 3:
        raise ValueError("Mesh quality audit requires (N,3) face arrays")
    count = dict(wall_panels=int(len(wall)),wake_panels=int(len(wake)),
                 total_panels=int(len(wall)+len(wake)))
    for label,limit_key in (("wall_panels","max_wall_panels"),
                            ("wake_panels","max_wake_panels"),
                            ("total_panels","max_total_panels")):
        limit=caps[limit_key]
        if limit is not None and count[label]>limit:
            raise ValueError(f"Mesh budget exceeded: {label}={count[label]} > {limit_key}={limit}")
    wf, wa, wang, wl, wq = _triangle_geometry(vertices,wall,"wall")
    _, za, zang, zl, zq = _triangle_geometry(vertices,wake,"wake")
    grows=_shared_area_growth(wf,wa)
    result=dict(counts=count,
        wall=dict(min_angle_deg=float(wang.min()), angle_p01_deg=float(np.percentile(wang.min(axis=1),1)),
                  worst_angle_face=int(np.argmin(wang.min(axis=1))),
                  min_shape_quality=float(wq.min()),
                  shape_quality_p01=float(np.percentile(wq,1)),
                  max_edge_ratio=float((wl.max(axis=1)/wl.min(axis=1)).max()),
                  p99_edge_ratio=float(np.percentile(wl.max(axis=1)/wl.min(axis=1),99)),
                  p99_neighbor_area_ratio=float(np.percentile(grows,99)) if len(grows) else None,
                  neighbor_edge_pairs=int(len(grows)), area_total=float(wa.sum())),
        wake=dict(min_angle_deg=float(zang.min()),
                  min_shape_quality=float(zq.min()),area_total=float(za.sum()),
                  p99_edge_ratio=float(np.percentile(zl.max(axis=1)/zl.min(axis=1),99))),
        requested_gates={**caps,"min_wall_angle_deg":min_angle,
                        "max_wall_growth_p99":max_growth},
        status="geometry_quality_only_not_physical_accuracy")
    if min_angle is not None and result["wall"]["min_angle_deg"] < min_angle:
        raise ValueError("Wall mesh angle gate failed: min_angle_deg="
                         f"{result['wall']['min_angle_deg']:.6g} < {min_angle:.6g}")
    if max_growth is not None:
        actual=result["wall"]["p99_neighbor_area_ratio"]
        if actual is None:
            raise ValueError("Wall mesh growth gate requires neighboring wall faces")
        if actual > max_growth:
            raise ValueError("Wall mesh growth gate failed: p99_neighbor_area_ratio="
                             f"{actual:.6g} > {max_growth:.6g}")
    return result
