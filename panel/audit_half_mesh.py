"""Read-only XZ symmetry audit of a Dpanel planar-cut surface fixture.

Usage: python -m panel.audit_half_mesh --surface cases/panel_cut/single_wing.json \
        --out runs/half_symmetry_audit_single.json

Topology audit only: this does not change meshes, average AIC, or validate
physical pressure accuracy. Vertex, wall face, wake face, and TE correspondence
are checked separately, rather than inferring symmetry from gamma stations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def _faces(value, label):
    a = np.asarray(value, dtype=np.int64)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"{label} must have shape [N,3]")
    return a


def _face_pairing(faces, vertex_map):
    """Return exact reflected triangle counterpart IDs, or -1 if missing."""
    lookup = {}
    for i, face in enumerate(faces):
        key = tuple(sorted(map(int, face)))
        if key in lookup:
            raise ValueError("Duplicate triangles in surface mesh")
        lookup[key] = i
    return np.array([
        lookup.get(tuple(sorted(int(vertex_map[j]) for j in face)), -1)
        for face in faces
    ], dtype=np.int64)


def _face_report(name, x, faces, vertex_map, vertex_valid, tolerance):
    tri = x[faces]
    y = tri[:, :, 1]
    on_plane = np.all(np.abs(y) <= tolerance, axis=1)
    crossing = (y.min(axis=1) < -tolerance) & (y.max(axis=1) > tolerance)
    centered = np.abs(y.mean(axis=1)) <= tolerance
    centroids = tri.mean(axis=1)
    target = centroids * np.array([1., -1., 1.])
    dist, nearest = cKDTree(centroids).query(target)
    result = {
        "count": int(len(faces)),
        "plane_face_count": int(on_plane.sum()),
        "cross_plane_face_count": int(crossing.sum()),
        "centroid_on_plane_count": int(centered.sum()),
        "centroid_self_nearest_count": int((nearest == np.arange(len(faces))).sum()),
        "centroid_reflection_median_distance": float(np.median(dist)),
        "centroid_reflection_max_distance": float(np.max(dist)),
        "centroid_matches_within_tolerance": int((dist <= tolerance).sum()),
    }
    if not vertex_valid:
        result.update(exact_face_pairs=None, normals_mirror_max_error=None,
                      face_map_involution=None, map_available=False)
        return result, None
    matched = _face_pairing(faces, vertex_map)
    match_valid = matched >= 0
    a = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(a, axis=1)
    if np.any(area == 0):
        raise ValueError(f"Degenerate {name} face")
    normal = a / area[:, None]
    mismatch = np.linalg.norm(
        normal[matched[match_valid]] - normal[match_valid] * [1., -1., 1.], axis=1
    )
    result.update(
        map_available=True,
        exact_face_pairs=int(match_valid.sum()),
        face_map_involution=bool(match_valid.all() and np.array_equal(matched[matched], np.arange(len(faces)))),
        normals_mirror_max_error=float(mismatch.max()) if mismatch.size else None,
        unmatched_face_ids=np.flatnonzero(~match_valid)[:12].tolist(),
    )
    return result, matched


def audit_half_surface(surface, *, relative_tolerance=1e-10):
    """Diagnose exact, oriented mirror connectivity in an immutable fixture."""
    if surface.get("schema") != "dpanel.planar_cut.v1":
        raise ValueError("Require dpanel.planar_cut.v1 surface fixture")
    x = np.asarray(surface["coordinates"], dtype=np.float64)
    faces = _faces(surface["faces"], "faces")
    wake = _faces(surface["wake_faces"], "wake_faces")
    te = np.asarray(surface["trailing_edges"], dtype=np.int64)
    upper = np.asarray(surface["upper_te_faces"], dtype=np.int64)
    lower = np.asarray(surface["lower_te_faces"], dtype=np.int64)
    tags = np.asarray(surface["component_ids"], dtype=np.int64)
    y = np.asarray(surface["gamma_stations"], dtype=np.float64)
    if (x.ndim != 2 or x.shape[1] != 3 or len(x) == 0 or
            len(faces) == 0 or len(wake) == 0 or
            te.ndim != 2 or te.shape[1] != 2 or len(te) != len(upper) or
            len(te) != len(lower) or len(tags) != len(faces) or
            y.ndim != 1 or len(y) < 2):
        raise ValueError("Invalid surface array shapes")
    for label, ids in (("faces", faces), ("wake_faces", wake), ("trailing_edges", te)):
        if np.min(ids) < 0 or np.max(ids) >= len(x):
            raise ValueError(f"{label} index out of range")
    if (not np.isfinite(x).all() or not np.isfinite(y).all() or
            not np.all(np.diff(y) > 0) or
            np.any(upper < 0) or np.any(upper >= len(faces)) or
            np.any(lower < 0) or np.any(lower >= len(faces))):
        raise ValueError("Nonfinite, unordered, or out-of-range fixture")
    if relative_tolerance <= 0:
        raise ValueError("Tolerance must be positive")
    # Mirrors about the normalized fixture's physical y=0 plane only.
    scale = max(float(np.ptp(x, axis=0).max()), 1.0)
    tol = relative_tolerance * scale
    reflected = x * np.array([1., -1., 1.])
    distance, pair = cKDTree(x).query(reflected)
    paired = distance <= tol
    bijective = bool(len(np.unique(pair)) == len(x))
    involution = bool(np.array_equal(pair[pair], np.arange(len(x))))
    vertex_valid = bool(paired.all() and bijective and involution)
    result = {
        "status": "blocked",
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": tol,
        "vertices": {
            "count": int(len(x)),
            "paired": int(paired.sum()),
            "bijective": bijective,
            "involution": involution,
            "median_distance": float(np.median(distance)),
            "maximum_distance": float(distance.max()),
            "unpaired_ids": np.flatnonzero(~paired)[:12].tolist(),
        },
        "notes": [
            "Centroid self-nearest is not proof of a physical symmetry-plane cap.",
            "This audit does not modify Gmsh triangles or certify aerodynamic loads.",
        ],
    }
    wall, wall_map = _face_report("wall", x, faces, pair, vertex_valid, tol)
    wake_report, wake_map = _face_report("wake", x, wake, pair, vertex_valid, tol)
    result["wall"] = wall
    result["wake"] = wake_report

    # Partial exact matches remain useful on broken meshes, but never turn a
    # non-bijective vertex map into an accepted global symmetry map.
    def partial_exact_triangles(triangles):
        lookup = {tuple(sorted(map(int, face))): i for i, face in enumerate(triangles)}
        return int(sum(
            all(paired[k] for k in face)
            and tuple(sorted(int(pair[k]) for k in face)) in lookup
            for face in triangles
        ))

    wall["partial_exact_face_pairs"] = partial_exact_triangles(faces)
    wake_report["partial_exact_face_pairs"] = partial_exact_triangles(wake)

    sdist = np.abs(y[:, None] + y[None, :])
    spair = np.argmin(sdist, axis=1)
    valid_station = sdist[np.arange(len(y)), spair] <= tol
    station_ok = bool(valid_station.all() and np.array_equal(spair[spair], np.arange(len(y))))
    result["stations"] = {
        "count": int(len(y)), "mirror_compatible": station_ok,
        "paired": int(valid_station.sum()),
        "max_reflection_error": float(sdist[np.arange(len(y)), spair].max()),
    }
    te_good = 0
    te_missed = []
    if vertex_valid and wall_map is not None and np.all(wall_map >= 0):
        te_lookup = {tuple(sorted(map(int, edge))): i for i, edge in enumerate(te)}
        if len(te_lookup) != len(te):
            raise ValueError("Repeated trailing-edge edges")
        for i, edge in enumerate(te):
            reflected_edge = tuple(sorted((int(pair[edge[0]]), int(pair[edge[1]]))))
            j = te_lookup.get(reflected_edge)
            if (j is not None and wall_map[upper[i]] == upper[j]
                    and wall_map[lower[i]] == lower[j]):
                te_good += 1
            else:
                te_missed.append(int(i))
    # Report geometry-only TE counterparts even when the *full* vertex/wall
    # reflection is invalid, to separate bad banks from missing edge geometry.
    te_lookup_partial = {tuple(sorted(map(int, edge))): i for i, edge in enumerate(te)}
    wall_lookup_partial = {tuple(sorted(map(int, face))): i for i, face in enumerate(faces)}
    te_edge_geometry = 0
    te_edge_banks_partial = 0
    for i, edge in enumerate(te):
        if not all(paired[k] for k in edge):
            continue
        reflected_edge = tuple(sorted(int(pair[k]) for k in edge))
        j = te_lookup_partial.get(reflected_edge)
        if j is None:
            continue
        te_edge_geometry += 1
        uface, lface = faces[upper[i]], faces[lower[i]]
        if not all(paired[k] for k in (*uface, *lface)):
            continue
        reflected_upper = tuple(sorted(int(pair[k]) for k in uface))
        reflected_lower = tuple(sorted(int(pair[k]) for k in lface))
        if (wall_lookup_partial.get(reflected_upper) == upper[j]
                and wall_lookup_partial.get(reflected_lower) == lower[j]):
            te_edge_banks_partial += 1

    result["trailing_edges"] = {
        "geometry_edges_reflected_even_if_unpaired_mesh": int(te_edge_geometry),
        "partial_exact_banks": int(te_edge_banks_partial),
        "count": int(len(te)),
        "paired_with_same_upper_lower_banks": int(te_good),
        "full_pairing_auditable": bool(vertex_valid and wall_map is not None and np.all(wall_map >= 0)),
        "unmatched_edge_ids": te_missed[:12] if vertex_valid else None,
    }
    component_ok = bool(wall_map is not None and np.all(wall_map >= 0)
                        and np.array_equal(tags[wall_map], tags))
    result["component_tags_reflection_compatible"] = component_ok
    reasons = []
    if not vertex_valid:
        reasons.append("No complete bijective vertex reflection, including wake/TE seam nodes")
    if wall_map is None or not wall["face_map_involution"]:
        reasons.append("Wall triangle connectivity lacks exact reflected counterparts")
    if wall["normals_mirror_max_error"] is not None and wall["normals_mirror_max_error"] > 1e-8:
        reasons.append("Wall outward normals are not mirror equivariant")
    if wall["plane_face_count"]:
        reasons.append("Wall contains triangles wholly on the symmetry plane: inspect artificial caps")
    if wall["cross_plane_face_count"]:
        reasons.append("Wall contains triangles crossing the symmetry plane: split before mirroring")
    if not wake_report["face_map_involution"]:
        reasons.append("Wake triangle connectivity lacks exact reflected counterparts")
    if wake_report["normals_mirror_max_error"] is not None and wake_report["normals_mirror_max_error"] > 1e-8:
        reasons.append("Wake normals are not mirror equivariant")
    if not station_ok:
        reasons.append("Circulation stations are not exactly mirrored")
    if te_good != len(te):
        reasons.append("TE edges and upper/lower banks are not mirror matched (or unauditable)")
    if not component_ok:
        reasons.append("Component labels are not mirror equivariant")
    result["reasons"] = reasons
    result["status"] = "topology_symmetric" if not reasons else "blocked"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tol", type=float, default=1e-10)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite old audit: {args.out}")
    fixture = json.loads(args.surface.read_text(encoding="utf-8"))
    report = audit_half_surface(fixture, relative_tolerance=args.tol)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    # blocked is a valid diagnostic result, not a solver or validation crash.


if __name__ == "__main__":
    main()
