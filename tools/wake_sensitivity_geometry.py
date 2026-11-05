"""Geometry-only measurements for the controlled steady-wake experiments."""
from collections import deque

import numpy as np
from scipy.spatial import cKDTree

from panel.p1_geometry_contact import coplanar_overlap_area
from panel.p1_steady_free_wake import prepare_wake_material
from panel.shed_groups import _segment_triangle_t
from tools.issue84.wake_node_rejection import point_triangle_distance


def attachment_rows(mesh):
    group = mesh.groups[0]
    used = np.unique(group.wake_faces)
    wall_ids = np.unique(mesh.faces)
    distance, nearest = cKDTree(mesh.xyz[wall_ids]).query(group.wake_xyz[used])
    attached = used[distance <= 1e-10 * mesh.reference_chord]
    matched_wall = {int(n): int(wall_ids[k]) for n, k, d in zip(used, nearest, distance)
                    if d <= 1e-10 * mesh.reference_chord}
    adjacency = {int(n): set() for n in used}
    for a, b, c in group.wake_faces:
        for p, q in ((a, b), (b, c), (c, a)):
            adjacency[int(p)].add(int(q))
            adjacency[int(q)].add(int(p))
    rows = np.full(len(group.wake_xyz), -1, int)
    rows[attached] = 0
    queue = deque(int(n) for n in attached)
    while queue:
        node = queue.popleft()
        for other in adjacency[node]:
            if rows[other] < 0:
                rows[other] = rows[node] + 1
                queue.append(other)
    if np.any(rows[used] < 0):
        raise ValueError("A wake component has no physical wall attachment")
    material = prepare_wake_material(group)
    return dict(used=used, attached=attached, wall_map=matched_wall, rows=rows,
                graph_roots=material.roots, adjacency=adjacency)


def geometry_context(mesh):
    group = mesh.groups[0]
    tri = group.wake_xyz[group.wake_faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    wall = mesh.xyz[mesh.faces]
    wall_edges = np.sort(np.concatenate((mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]],
                                         mesh.faces[:, [2, 0]])), axis=1)
    _, edge_count = np.unique(wall_edges, axis=0, return_counts=True)
    return dict(topology=attachment_rows(mesh), cross=cross,
                area2=np.linalg.norm(cross, axis=1), wall=wall,
                wall_lo=wall.min(axis=1), wall_hi=wall.max(axis=1),
                wall_closed=bool(np.all(edge_count == 2)),
                initial_xyz=group.wake_xyz.copy(), initial_faces=group.wake_faces.copy())


def audit_geometry(mesh, context, baseline=None, *, distance=True):
    group = mesh.groups[0]
    tri = group.wake_xyz[group.wake_faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    tol = 1e-10 * mesh.reference_chord
    violations, tested = [], 0
    for face_id, triangle in enumerate(tri):
        candidates = np.flatnonzero(
            np.all(triangle.min(axis=0) <= context["wall_hi"] + tol, axis=1)
            & np.all(context["wall_lo"] <= triangle.max(axis=0) + tol, axis=1))
        for wall_id in candidates:
            common = set(context["topology"]["wall_map"].get(int(n), -1)
                         for n in group.wake_faces[face_id]) & set(mesh.faces[wall_id])
            common.discard(-1)
            tested += 1
            other = context["wall"][wall_id]
            overlap = coplanar_overlap_area(triangle, other, tol)
            intersection = []
            for source, target in ((triangle, other), (other, triangle)):
                for a, b in zip(source, np.roll(source, -1, axis=0)):
                    t = _segment_triangle_t(a, b, target, tol)
                    if t is not None:
                        intersection.append(a + np.clip(t, 0., 1.) * (b-a))
            wall_normal = np.cross(other[1]-other[0], other[2]-other[0])
            wall_normal /= np.linalg.norm(wall_normal)
            plane = (triangle-other[0]) @ wall_normal
            wake_normal = cross[face_id] / np.linalg.norm(cross[face_id])
            reverse_plane = (other-triangle[0]) @ wake_normal
            off_contact = intersection
            if intersection and common:
                points = np.asarray(intersection)
                shared = mesh.xyz[sorted(common)]
                contact_distance = np.linalg.norm(points[:, None] - shared[None], axis=2).min(axis=1)
                if len(shared) == 2:
                    edge = shared[1] - shared[0]
                    t = np.clip((points-shared[0]) @ edge / (edge @ edge), 0., 1.)
                    contact_distance = np.linalg.norm(points-shared[0]-t[:, None]*edge, axis=1)
                off_contact = points[contact_distance > tol].tolist()
            transverse = bool(off_contact and (
                (plane.min() < -tol and plane.max() > tol)
                or (reverse_plane.min() < -tol and reverse_plane.max() > tol)))
            if overlap > tol**2 or transverse:
                violations.append(dict(wake_face=face_id, wall_face=int(wall_id),
                                       kind="coplanar_overlap" if overlap > tol**2 else "transverse",
                                       depth=float(max(0., -plane.min())),
                                       intersection_points=[p.tolist() for p in intersection],
                                       overlap_area=float(overlap),
                                       declared_wall_ids=[int(n) for n in sorted(common)]))
    initial = {} if baseline is None else {
        (r["wake_face"], r["wall_face"], r["kind"]): r["depth"]
        for r in baseline["violations"]}
    new = [r for r in violations
           if (r["wake_face"], r["wall_face"], r["kind"]) not in initial
           or r["depth"] > initial[(r["wake_face"], r["wall_face"], r["kind"])] + 10 * tol]
    top = context["topology"]
    attached = top["attached"]
    movement = np.linalg.norm(group.wake_xyz - context["initial_xyz"], axis=1)
    free = np.setdiff1d(top["used"], attached)
    free_distance = None
    inside = []
    if context["wall_closed"]:
        # A closed outward wall gives winding 1 inside and 0 outside. Exact
        # retained wall vertices are attachment, not interior samples.
        for start in range(0, len(top["used"]), 32):
            ids = top["used"][start:start+32]
            points = group.wake_xyz[ids]
            r = context["wall"][None] - points[:, None, None, :]
            lengths = np.linalg.norm(r, axis=3)
            a, b, c = r[:, :, 0], r[:, :, 1], r[:, :, 2]
            det = np.einsum("bfi,bfi->bf", a, np.cross(b, c))
            denom = (lengths.prod(axis=2)
                     + np.einsum("bfi,bfi->bf", a, b) * lengths[:, :, 2]
                     + np.einsum("bfi,bfi->bf", b, c) * lengths[:, :, 0]
                     + np.einsum("bfi,bfi->bf", c, a) * lengths[:, :, 1])
            winding = abs((2*np.arctan2(det, denom)).sum(axis=1) / (4*np.pi))
            for node, value, minimum in zip(ids, winding, lengths.min(axis=(1, 2))):
                if minimum > tol and value > .5 + 1e-8:
                    inside.append(int(node))
    initial_inside = set() if baseline is None else set(baseline.get("inside_wall_node_ids", []))
    new_inside = sorted(set(inside)-initial_inside)
    if distance and len(free):
        free_distance = np.inf
        for start in range(0, len(free), 32):
            values = point_triangle_distance(group.wake_xyz[free[start:start+32]], context["wall"])
            free_distance = min(free_distance, float(values.min()))
    # A retained attached node at a wall vertex proves the global distance is
    # exactly zero, even when all other wake vertices have positive distance.
    global_distance = None
    for node in attached:
        wall_node = top["wall_map"][int(node)]
        if np.array_equal(group.wake_xyz[node], mesh.xyz[wall_node]):
            global_distance = 0.
            break
    return dict(
        min_area2=float(np.linalg.norm(cross, axis=1).min()),
        min_reference_signed_jacobian=float(np.min(
            np.einsum("fi,fi->f", cross, context["cross"]) / context["area2"]**2)),
        min_wake_wall_distance=global_distance,
        min_nonattachment_vertex_wall_distance=free_distance,
        distance_definition="global minimum is certified zero by a retained wall vertex; extra metric is vertex-to-wall-triangle",
        attachment_max_displacement_chord=float(movement[attached].max() / mesh.reference_chord),
        attachment_ids=attached.tolist(), prepared_root_ids=top["graph_roots"].tolist(),
        violation_count=len(violations), violations=violations,
        wall_closed=context["wall_closed"], inside_wall_node_ids=inside,
        new_inside_wall_node_ids=new_inside if baseline is not None else [],
        new_or_deeper_violation_count=(len(new)+len(new_inside)) if baseline is not None else 0,
        new_or_deeper_violations=new if baseline is not None else [],
        pairs_tested=tested, tolerance=tol,
        collision_method="six actual segment/triangle tests, coplanar area, and closed-wall solid-angle winding")


def spacing(mesh):
    group = mesh.groups[0]
    f, x = group.wake_faces, group.wake_xyz
    edges = np.unique(np.sort(np.concatenate((f[:, [0, 1]], f[:, [1, 2]],
                                               f[:, [2, 0]])), axis=1), axis=0)
    d = abs(x[edges[:, 0]] - x[edges[:, 1]])
    return dict(faces=len(f), used_nodes=len(np.unique(f)), gamma_dofs=group.gamma_hat.shape[1],
                edge_lengths=np.percentile(np.linalg.norm(d, axis=1), [0, 10, 50, 90, 100]).tolist(),
                dx=np.percentile(d[:, 0], [0, 10, 50, 90, 100]).tolist(),
                dy=np.percentile(d[:, 1], [0, 10, 50, 90, 100]).tolist())
