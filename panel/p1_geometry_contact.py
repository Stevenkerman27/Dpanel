"""Topology-aware contact and penetration classification for the material wake.

The coupled wake loop moves a wake sheet under a velocity field and validates
the candidate geometry before accepting it.  The sheet is attached to the
wall: its root nodes are wall vertices, so wake triangles share vertices and
edges with wall triangles by construction.  A boolean convex-hull SAT test
cannot tell that declared contact apart from a transverse penetration, and it
rejected every candidate update on the CAD case (docs/ISSUE84_T4_G2_UPDATE.md).

This module separates the two questions:

* declared_contact_pairs builds the declared contact set ONCE from reference
  topology.  Identity is (group, wake face index, wall face index, shared
  vertex ids); vertex indices are stable through geometry updates, so the map
  is transferred without ever matching coordinates.
* classify_swept_wall returns a signed classification with a penetration
  depth: 'disjoint', 'contact' (touching within tolerance), 'penetration'
  (transverse overlap deeper than tolerance) or 'coplanar_overlap' (positive
  in-plane area).

Declared contact only excuses CONTACT AT THE SHARED SIMPLEX.  Penetration or
coplanar overlap of a declared pair is still a violation, so a deforming sheet
cannot hide inside its own attachment.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

__all__ = [
    "DECLARED_VERTEX",
    "DECLARED_EDGE",
    "ContactClass",
    "declared_contact_pairs",
    "swept_hull",
    "classify_swept_wall",
    "coplanar_overlap_area",
    "triangle_triangle_intersection",
    "classify_new_triangle",
]

DECLARED_VERTEX = "vertex"
DECLARED_EDGE = "edge"


@dataclass(frozen=True)
class ContactClass:
    """Signed classification of one swept wake triangle against one wall triangle."""

    kind: str             # disjoint | contact | penetration | coplanar_overlap
    depth: float          # minimum translation depth (0 for contact)
    area: float           # coplanar overlap area (0 when not coplanar)
    declared: str | None  # None | 'vertex' | 'edge'
    violation: bool       # True when the pair must refuse the candidate

    def as_dict(self):
        return dict(kind=self.kind, depth=float(self.depth), area=float(self.area),
                    declared=self.declared, violation=bool(self.violation))


def swept_hull(old_tri, new_tri, shrink=0.0):
    """Six points bounding the linear sweep of a triangle (old -> new)."""
    t0 = np.asarray(old_tri, float).reshape(3, 3)
    t1 = np.asarray(new_tri, float).reshape(3, 3)
    if shrink:
        t0 = (1.-shrink)*t0 + shrink*t0.mean(axis=0)
        t1 = (1.-shrink)*t1 + shrink*t1.mean(axis=0)
    return np.concatenate((t0, t1), axis=0)


def declared_contact_pairs(wake_faces, wall_faces, group_labels=None):
    """Declared wake/wall contact pairs from shared vertex ids, computed once.

    wake_faces is a sequence of (n, 3) triangle arrays (one per wake group),
    wall_faces the mesh wall connectivity.  Vertex ids are the only identity
    used, so the map survives arbitrary geometry updates while connectivity and
    face numbering stay fixed.
    """
    wall_faces = np.asarray(wall_faces, int)
    vertex_to_wall = {}
    for j, face in enumerate(wall_faces):
        for v in face:
            vertex_to_wall.setdefault(int(v), set()).add(j)
    if group_labels is None:
        labels = list(range(len(wake_faces)))
    elif np.isscalar(group_labels):
        labels = [int(group_labels)]*len(wake_faces)
    else:
        labels = list(group_labels)
    pairs = {}
    for label, faces in zip(labels, wake_faces):
        faces = np.asarray(faces, int)
        for i, face in enumerate(faces):
            shared = set()
            for v in face:
                shared |= vertex_to_wall.get(int(v), set())
            for j in sorted(shared):
                common = np.intersect1d(face, wall_faces[j])
                if len(common) == 0:
                    continue
                kind = DECLARED_EDGE if len(common) >= 2 else DECLARED_VERTEX
                key = (label, int(i), int(j))
                previous = pairs.get(key)
                if previous is None or (previous[0] == DECLARED_VERTEX and kind == DECLARED_EDGE):
                    pairs[key] = (kind, tuple(int(v) for v in common))
    return pairs


def _axes(p, q):
    """SAT axes for two point sets: face normals and edge-edge cross products."""
    def edges_normals(x):
        ep = np.asarray([x[j]-x[i] for i, j in combinations(range(len(x)), 2)])
        np_ = np.asarray([np.cross(x[j]-x[i], x[k]-x[i])
                          for i, j, k in combinations(range(len(x)), 3)]).reshape(-1, 3)
        return ep, np_
    ep, np_ = edges_normals(p)
    eq, nq = edges_normals(q)
    normals = np.concatenate((np_, nq))
    blocks = [normals, np.cross(ep[:, None], eq[None, :]).reshape(-1, 3)]
    if len(normals):
        blocks.append(np.cross(normals[:, None], np.concatenate((ep, eq))[None]).reshape(-1, 3))
    axes = np.concatenate(blocks)
    norm = np.linalg.norm(axes, axis=1)
    keep = norm > 1e-30
    return axes[keep]/norm[keep, None]


def _dominant_normal(points):
    """Unit normal of the flattest direction of a nearly flat point set."""
    centred = np.asarray(points, float)-np.asarray(points, float).mean(axis=0)
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    n = vt[-1]
    norm = np.linalg.norm(n)
    return n/norm if norm > 0 else np.array([0., 0., 1.])


def coplanar_overlap_area(p, q, tol):
    """Positive overlap area of two coplanar triangles, projected to 2D."""
    p = np.asarray(p, float).reshape(-1, 3)
    q = np.asarray(q, float).reshape(3, 3)
    np_ = _dominant_normal(p)
    nq = np.cross(q[1]-q[0], q[2]-q[0])
    nq_norm = np.linalg.norm(nq)
    if nq_norm <= 0:
        return 0.
    nq = nq/nq_norm
    if abs(float(np.dot(np_, nq))) < 1.-1e-9:
        return 0.
    if abs(float(np.dot(q[0]-p.mean(axis=0), np_))) > tol:
        return 0.
    helper = np.array([1., 0., 0.]) if abs(np_[0]) < .9 else np.array([0., 1., 0.])
    e1 = helper-np_*np.dot(helper, np_)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(np_, e1)
    origin = p.mean(axis=0)

    def to2d(x):
        d = np.asarray(x, float)-origin
        return np.stack((d@e1, d@e2), axis=1)

    hull_pts = _plane_hull_2d(to2d(p))
    if len(hull_pts) < 3:
        return 0.
    clipped = _clip_polygon(to2d(q), hull_pts)
    if len(clipped) < 3:
        return 0.
    x, y = clipped[:, 0], clipped[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1))-np.dot(y, np.roll(x, -1)))/2.)


def _plane_hull_2d(points):
    """Convex hull (counter-clockwise) of projected points."""
    pts = np.unique(np.round(np.asarray(points, float), 12), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])

    lower = []
    for pt in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0:
            lower.pop()
        lower.append(pt)
    upper = []
    for pt in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 0:
            upper.pop()
        upper.append(pt)
    return np.asarray(lower[:-1]+upper[:-1], float)


def _clip_polygon(subject, clip):
    """Sutherland-Hodgman clip of a convex polygon by a CCW convex polygon."""
    def inside(pt, a, b):
        return (b[0]-a[0])*(pt[1]-a[1])-(b[1]-a[1])*(pt[0]-a[0]) >= -1e-18

    def intersect(p1, p2, a, b):
        d = np.array([p2[0]-p1[0], p2[1]-p1[1]])
        e = np.array([b[0]-a[0], b[1]-a[1]])
        denom = d[0]*e[1]-d[1]*e[0]
        if abs(denom) < 1e-30:
            return np.asarray(p2, float)
        ap = np.array([a[0]-p1[0], a[1]-p1[1]])
        t = (ap[0]*e[1]-ap[1]*e[0])/denom
        return np.array([p1[0]+t*d[0], p1[1]+t*d[1]])

    output = [np.asarray(pt, float) for pt in subject]
    for i in range(len(clip)):
        a, b = clip[i], clip[(i+1) % len(clip)]
        if not output:
            return np.zeros((0, 2))
        current, output = output, []
        prev = current[-1]
        for pt in current:
            if inside(pt, a, b):
                if not inside(prev, a, b):
                    output.append(intersect(prev, pt, a, b))
                output.append(pt)
            elif inside(prev, a, b):
                output.append(intersect(prev, pt, a, b))
            prev = pt
    return np.asarray(output, float) if output else np.zeros((0, 2))


def triangle_triangle_intersection(p, q, tol=0.):
    """Moller triangle/triangle intersection, degeneracy aware.

    Returns (kind, points): kind in 'none', 'point', 'segment', 'coplanar'.
    """
    p = np.asarray(p, float).reshape(3, 3)
    q = np.asarray(q, float).reshape(3, 3)
    n2 = np.cross(q[1]-q[0], q[2]-q[0])
    n2n = np.linalg.norm(n2)
    if n2n <= 1e-30:
        return "none", None
    n2 = n2/n2n
    d = (p-q[0])@n2
    if np.all(d > tol) or np.all(d < -tol):
        return "none", None
    n1 = np.cross(p[1]-p[0], p[2]-p[0])
    n1n = np.linalg.norm(n1)
    if n1n <= 1e-30:
        return "none", None
    n1 = n1/n1n
    d2 = (q-p[0])@n1
    if np.all(d2 > tol) or np.all(d2 < -tol):
        return "none", None
    direction = np.cross(n1, n2)
    dn = np.linalg.norm(direction)
    if dn <= 1e-12:
        area = coplanar_overlap_area(p, q, tol)
        return ("coplanar", None) if area > tol*tol else ("none", None)
    direction = direction/dn

    def interval(tri):
        proj = (tri-p[0])@direction
        return float(proj.min()), float(proj.max())

    pmin, pmax = interval(p)
    qmin, qmax = interval(q)
    lo, hi = max(pmin, qmin), min(pmax, qmax)
    if lo-hi > tol:
        return "none", None
    if hi-lo <= tol:
        return "point", np.array([np.mean([lo, hi])])
    return "segment", np.array([lo, hi])


def _point_inside_triangle_plane(pt, tri, tol):
    """True when the in-plane projection of pt lies inside tri."""
    tri = np.asarray(tri, float)
    n = np.cross(tri[1]-tri[0], tri[2]-tri[0])
    nn = np.linalg.norm(n)
    if nn <= 1e-30:
        return False
    n = n/nn
    d = float(np.dot(pt-tri[0], n))
    proj = pt-d*n
    for a, b in ((0, 1), (1, 2), (2, 0)):
        e = tri[b]-tri[a]
        v = proj-tri[a]
        if float(np.dot(np.cross(e, v), n)) < -tol*max(1., float(np.linalg.norm(e))):
            return False
    return True


def _plane_distance(points, wall_tri):
    """Signed distance of points to the wall triangle plane, and its unit normal."""
    wall_tri = np.asarray(wall_tri, float)
    n = np.cross(wall_tri[1]-wall_tri[0], wall_tri[2]-wall_tri[0])
    norm = np.linalg.norm(n)
    if norm <= 1e-30:
        return np.zeros(len(points)), np.array([0., 0., 1.])
    n = n/norm
    return (np.asarray(points, float)-wall_tri[0])@n, n


def _crossing_points(points, distances, tol):
    """Points where the point-set edges cross the wall plane (strictly)."""
    points = np.asarray(points, float)
    out = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            di, dj = distances[i], distances[j]
            if di*dj < 0 and abs(di) > tol and abs(dj) > tol:
                t = di/(di-dj)
                out.append(points[i]+t*(points[j]-points[i]))
    return out


def _distance_to_region(points, region):
    """Distance from each point to a point (k=1) or a segment (k=2)."""
    pts = np.asarray(points, float).reshape(-1, 3)
    region = np.asarray(region, float).reshape(-1, 3)
    if len(region) == 1:
        return np.linalg.norm(pts-region[0], axis=1)
    if len(region) == 2:
        a, b = region
        ab = b-a
        length2 = float(ab@ab)
        if length2 <= 0.:
            return np.linalg.norm(pts-a, axis=1)
        t = np.clip((pts-a)@ab/length2, 0., 1.)
        return np.linalg.norm(pts-(a+t[:, None]*ab), axis=1)
    return np.full(len(pts), np.inf)


def _classify_set(points, wall_tri, *, tol, declared, area_tol,
                  declared_region=None, graze_tol=0.):
    """Signed classification of a point set against a wall triangle.

    A violation requires a point of the set to be on the FAR side of the wall
    plane (signed distance below -tol) whose in-plane projection falls inside
    the wall triangle, or an edge crossing the plane inside the wall triangle.
    Declared contact merely permits touching (|distance| <= tol) without any
    protrusion, so a deforming sheet cannot pass through its own attachment.

    A sheet leaving a shared edge or vertex necessarily grazes its own wall
    triangle while the shared simplex moves.  declared_region (the current
    coordinates of the shared simplex) plus graze_tol confines that allowance to
    a corridor around the declared simplex: crossings further inside the wall
    triangle, and every crossing deeper than graze_tol, are still violations.
    """
    points = np.asarray(points, float)
    wall_tri = np.asarray(wall_tri, float)
    area_tol = (tol*tol if area_tol is None else area_tol)
    # No AABB early exit here: a set entirely behind the wall plane (deep
    # penetration) has a disjoint AABB along the normal yet still penetrates.
    # The caller already filters far-apart pairs by bounds.
    area = coplanar_overlap_area(points, wall_tri, tol)
    if area > area_tol:
        return ContactClass("coplanar_overlap", 0., area, declared, True)
    distance, _normal = _plane_distance(points, wall_tri)
    behind = distance < -tol
    if not np.any(behind):
        if np.all(distance > tol):
            return ContactClass("disjoint", 0., 0., declared, False)
        return ContactClass("contact", 0., 0., declared, declared is None)
    depth = float(-distance.min())
    offending = [points[i] for i in np.nonzero(behind)[0]
                 if _point_inside_triangle_plane(points[i], wall_tri, tol)]
    offending.extend(pt for pt in _crossing_points(points, distance, tol)
                     if _point_inside_triangle_plane(pt, wall_tri, tol))
    grazing = 0
    if offending and declared is not None and declared_region is not None and graze_tol > 0.:
        offending = np.asarray(offending, float).reshape(-1, 3)
        beyond = _distance_to_region(offending, declared_region) > graze_tol
        grazing = int(len(offending)-int(np.count_nonzero(beyond)))
        offending = offending[beyond]
    if len(offending) == 0:
        if grazing:
            return ContactClass("contact", depth, 0., declared, False)
        return ContactClass("disjoint", depth, 0., declared, False)
    kind = "transverse" if np.any(distance > tol) else "penetration"
    return ContactClass(kind, depth, 0., declared, True)


def classify_swept_wall(hull, wall_tri, *, tol, declared=None, area_tol=None,
                        declared_region=None, graze_tol=0.):
    """Classify a swept wake hull against one wall triangle.

    declared is 'vertex' or 'edge' when reference topology declares the pair in
    contact.  Declared pairs are excused only while they touch, and while any
    protrusion stays inside graze_tol of declared_region (the shared simplex at
    its current coordinates).
    """
    return _classify_set(hull, wall_tri, tol=tol, declared=declared, area_tol=area_tol,
                         declared_region=declared_region, graze_tol=graze_tol)


def classify_new_triangle(new_tri, wall_tri, *, tol, declared=None, area_tol=None,
                          declared_region=None, graze_tol=0.):
    """Fine classification of the CANDIDATE triangle against a wall triangle."""
    return _classify_set(new_tri, wall_tri, tol=tol, declared=declared, area_tol=area_tol,
                         declared_region=declared_region, graze_tol=graze_tol)


def _bounds_overlap(p, q, tol):
    return not (np.any(p.max(axis=0) < q.min(axis=0)-tol)
                or np.any(q.max(axis=0) < p.min(axis=0)-tol))
