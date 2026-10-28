"""3D solid-angle free terms at mesh nodes (issue #56, research only).

The free term of a doublet boundary-integral operator at a node is a *3D* solid angle,
not a sum of in-plane sector angles. This module measures both and shows where they
disagree, using analytic oracles (flat sheet, right tetrahedron corner, icosphere).
"""

from __future__ import annotations

import numpy as np

FLOAT = np.float64
FOUR_PI = 4.0 * np.pi


def triangle_solid_angle(triangle, point):
    """Solid angle (signed, steradians) of a flat triangle seen from ``point``.

    Van Oosterom-Strackee: ``Omega = 2 atan2(r1.(r2 x r3), |r1||r2||r3| + ...)``.
    The sign follows the triangle orientation: positive when the triangle is seen
    counter-clockwise, i.e. from the side its normal points to.
    """
    r = np.asarray(triangle, dtype=FLOAT) - np.asarray(point, dtype=FLOAT)
    lengths = np.linalg.norm(r, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("Point coincides with a triangle vertex; probe an offset point")
    numerator = float(np.dot(r[0], np.cross(r[1], r[2])))
    denominator = float(
        lengths[0] * lengths[1] * lengths[2]
        + np.dot(r[0], r[1]) * lengths[2]
        + np.dot(r[1], r[2]) * lengths[0]
        + np.dot(r[2], r[0]) * lengths[1]
    )
    return 2.0 * float(np.arctan2(numerator, denominator))


def incident_faces(faces, node):
    faces = np.asarray(faces, dtype=np.int64)
    return np.nonzero(np.any(faces == int(node), axis=1))[0]


def node_normal(vertices, faces, node):
    vertices = np.asarray(vertices, dtype=FLOAT)
    faces = np.asarray(faces, dtype=np.int64)
    rows = incident_faces(faces, node)
    triangle = vertices[faces[rows]]
    normals = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    total = normals.sum(axis=0)
    length = float(np.linalg.norm(total))
    if length == 0.0:
        raise ValueError("Incident normals cancel at node %d" % int(node))
    return total / length


def sector_angle_free_term(vertices, faces, node):
    """Sum of in-plane sector angles at ``node``, divided by 4pi (the old rule)."""
    vertices = np.asarray(vertices, dtype=FLOAT)
    faces = np.asarray(faces, dtype=np.int64)
    total = 0.0
    for row in incident_faces(faces, node):
        triangle = vertices[faces[row]]
        local = list(faces[row]).index(int(node))
        a = triangle[(local + 1) % 3] - triangle[local]
        b = triangle[(local + 2) % 3] - triangle[local]
        cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        total += float(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return total / FOUR_PI


def solid_angle_free_term(vertices, faces, node, *, offset=1e-6):
    """Sum of 3D solid angles of the incident faces, seen from a probe point at ``node``."""
    vertices = np.asarray(vertices, dtype=FLOAT)
    faces = np.asarray(faces, dtype=np.int64)
    normal = node_normal(vertices, faces, node)
    probe = vertices[node] + offset * normal
    total = 0.0
    for row in incident_faces(faces, node):
        total += triangle_solid_angle(vertices[faces[row]], probe)
    return total / FOUR_PI


def tetrahedron():
    """Right tetrahedron: corner at the origin, the other three vertices on the axes."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=FLOAT
    )
    faces = np.array(
        [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int64
    )
    return vertices, faces


def flat_sheet():
    """Two unit triangles sharing an edge, lying in the z = 0 plane."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=FLOAT
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    return vertices, faces


def icosphere(level):
    """Subdivided icosahedron with outward-facing triangles (same rule as p1_dirichlet)."""
    from .p1_dirichlet import icosphere as _icosphere

    return _icosphere(level)


def node_free_term_table(vertices, faces, *, offset=1e-6):
    """Compact table of sector-angle and 3D solid-angle free terms per used node."""
    vertices = np.asarray(vertices, dtype=FLOAT)
    faces = np.asarray(faces, dtype=np.int64)
    nodes = np.unique(faces)
    rows = []
    for node in nodes:
        rows.append(
            {
                "node": int(node),
                "sector": float(sector_angle_free_term(vertices, faces, node)),
                "solid": float(solid_angle_free_term(vertices, faces, node, offset=offset)),
            }
        )
    sector = np.array([row["sector"] for row in rows])
    solid = np.array([row["solid"] for row in rows])
    return {
        "nodes": len(rows),
        "sector_min": float(sector.min()),
        "sector_max": float(sector.max()),
        "solid_min": float(solid.min()),
        "solid_max": float(solid.max()),
        "max_abs_difference": float(np.max(np.abs(solid - sector))),
        "rows": rows,
    }


def outward_orient(vertices, faces):
    """Flip faces so that every normal points away from the body centroid."""
    vertices = np.asarray(vertices, dtype=FLOAT)
    faces = np.asarray(faces, dtype=np.int64).copy()
    centre = vertices[np.unique(faces)].mean(axis=0)
    for row, face in enumerate(faces):
        triangle = vertices[face]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        if float(np.dot(normal, triangle.mean(axis=0) - centre)) < 0.0:
            faces[row] = face[::-1]
    return faces


def dihedral_angles(vertices, faces, node):
    """Pairwise angles between the outward normals of the faces incident on ``node``."""
    vertices = np.asarray(vertices, dtype=FLOAT)
    faces = np.asarray(faces, dtype=np.int64)
    rows = incident_faces(faces, node)
    normals = []
    for row in rows:
        triangle = vertices[faces[row]]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normals.append(normal / np.linalg.norm(normal))
    angles = []
    for i in range(len(normals)):
        for j in range(i + 1, len(normals)):
            cosine = float(np.clip(np.dot(normals[i], normals[j]), -1.0, 1.0))
            angles.append(float(np.arccos(cosine)))
    return np.array(angles)


def cone_solid_angle(vertices, faces, node):
    """Solid angle of the body cone at a trihedral corner (spherical excess).

    For three faces meeting at a corner the interior cone subtends
    ``Omega = alpha + beta + gamma - pi`` where the angles are the dihedral angles
    between the outward normals. This is the quantity that the classical corner
    coefficient ``c = 1 - Omega/(4 pi)`` needs; it is *not* the sum of in-plane
    sector angles, and it is not a sum of signed panel solid angles either.
    """
    angles = dihedral_angles(vertices, faces, node)
    if len(angles) != 3:
        raise ValueError("Corner formula needs exactly three incident faces, got %d" % len(angles))
    omega = float(angles.sum()) - np.pi
    return {
        "omega": omega,
        "cone_fraction": omega / FOUR_PI,
        "interior_coefficient": omega / FOUR_PI,
        "exterior_coefficient": 1.0 - omega / FOUR_PI,
        "dihedral_angles": [float(a) for a in angles],
    }


def assembled_row_sum(vertices, faces, node, *, side="interior", offset=1e-6):
    """Sum of the signed solid angles of *all* panels, probed off the node.

    ``side="interior"`` steps along ``-normal`` (into the body) and gives ``+1`` for a
    closed surface, ``side="exterior"`` gives ``0``. This is the constant-density
    identity of the doublet layer and the only free-term statement that survived
    measurement; the incident-fan sums do not (see the module docstring).
    """
    vertices = np.asarray(vertices, dtype=FLOAT)
    faces = np.asarray(faces, dtype=np.int64)
    normal = node_normal(vertices, faces, node)
    sign = -1.0 if side == "interior" else 1.0
    probe = vertices[node] + sign * offset * normal
    return float(sum(triangle_solid_angle(vertices[face], probe) for face in faces) / FOUR_PI)


def incident_fan_sum(vertices, faces, node, *, side="exterior", offset=1e-6):
    """Signed solid-angle sum over the incident faces only (the refuted rule)."""
    vertices = np.asarray(vertices, dtype=FLOAT)
    faces = np.asarray(faces, dtype=np.int64)
    normal = node_normal(vertices, faces, node)
    sign = -1.0 if side == "interior" else 1.0
    probe = vertices[node] + sign * offset * normal
    total = 0.0
    for row in incident_faces(faces, node):
        total += triangle_solid_angle(vertices[faces[row]], probe)
    return float(total / FOUR_PI)
