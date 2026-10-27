"""P1 doublet wake sheet and Kutta rows for the G1 lifting step (issue #49, priority d).

Research-only.  Nothing in here is imported by the production solver.

Conventions fixed by this module (all measured by the tests, not assumed):

* a doublet sheet carries a potential *jump* equal to its density, with the sign of the
  panel normal: the potential on the ``+n`` side minus the ``-n`` side equals ``+mu`` when
  the triangle winding follows its normal (right hand rule);
* mirroring a sheet in a symmetry plane flips the normal, so the mirror image that keeps
  the potential field symmetric must carry ``-mu`` (this is the ``my -> -my`` rule in
  ``panel/half_native.py`` ``W = (dw + wi) @ basis + (my - mi) @ slope``);
* the Kutta row of one trailing edge station is ``mu_upper - mu_lower - Gamma = 0``, which
  is ``kutta @ mu - mass @ gamma`` in the production assembly.
"""

import numpy as np

from panel.p1_dirichlet import panel_doublet_coefficients

__all__ = [
    "sheet_potential",
    "jump_across_sheet",
    "mirror_nodes",
    "kutta_rows",
    "wake_transport_rows",
    "dof_audit",
]


def _as_float64(array, name):
    out = np.ascontiguousarray(array, dtype=np.float64)
    if out.ndim < 1 or out.size == 0:
        raise ValueError("%s must be a non-empty array" % name)
    return out


def _triangles(vertices, faces):
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be a (m, 3) integer array")
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("faces index outside the vertex array")
    return vertices[faces]


def sheet_potential(vertices, faces, density, targets):
    """Potential of a P1 doublet sheet at ``targets`` (targets must be off the sheet)."""
    vertices = _as_float64(vertices, "vertices").reshape(-1, 3)
    targets = _as_float64(targets, "targets").reshape(-1, 3)
    density = _as_float64(density, "density").reshape(-1)
    if len(density) != len(vertices):
        raise ValueError("density must have one entry per vertex")
    triangles = _triangles(vertices, faces)
    potential = np.zeros(len(targets))
    for j, triangle in enumerate(triangles):
        nodes = np.asarray(faces[j], dtype=np.int64)
        coefficients = panel_doublet_coefficients(targets, triangle)
        potential += coefficients @ density[nodes]
    return potential


def jump_across_sheet(vertices, faces, density, *, heights=(1e-3, 1e-4, 1e-5)):
    """Signed potential jump across each panel: ``phi(+n) - phi(-n)`` at fixed heights."""
    vertices = _as_float64(vertices, "vertices").reshape(-1, 3)
    triangles = _triangles(vertices, faces)
    centroids = triangles.mean(axis=1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    rows = []
    for height in heights:
        above = sheet_potential(vertices, faces, density, centroids + height * normals)
        below = sheet_potential(vertices, faces, density, centroids - height * normals)
        rows.append({
            "height": float(height),
            "above": above,
            "below": below,
            "jump": above - below,
            "mean": 0.5 * (above + below),
        })
    return {"rows": rows, "centroids": centroids, "normals": normals}


def mirror_nodes(vertices, faces, *, axis=1):
    """Mirror a sheet in the plane ``x_axis = 0`` keeping the same connectivity."""
    vertices = _as_float64(vertices, "vertices").reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64)
    mirrored = vertices.copy()
    mirrored[:, axis] = -mirrored[:, axis]
    return mirrored, faces.copy()


def kutta_rows(te_upper, te_lower, wake_first):
    """Kutta rows of a P1 wall/wake system.

    Each station gives one row ``mu_upper - mu_lower - Gamma_te = 0``: ``te_upper`` and
    ``te_lower`` are wall node indices, ``wake_first`` the wake node on the same station
    (the first wake row downstream of the trailing edge).
    """
    te_upper = np.asarray(te_upper, dtype=np.int64).reshape(-1)
    te_lower = np.asarray(te_lower, dtype=np.int64).reshape(-1)
    wake_first = np.asarray(wake_first, dtype=np.int64).reshape(-1)
    if not (len(te_upper) == len(te_lower) == len(wake_first)):
        raise ValueError("Kutta rows need one upper, lower and wake node per station")
    return {"upper": te_upper, "lower": te_lower, "wake": wake_first}


def wake_transport_rows(wake_stations):
    """Rows that tie a wake station to the next station downstream (Gamma is constant)."""
    wake_stations = [np.asarray(station, dtype=np.int64).reshape(-1) for station in wake_stations]
    if len(wake_stations) < 2:
        raise ValueError("wake transport needs at least two stations")
    widths = {len(station) for station in wake_stations}
    if len(widths) != 1:
        raise ValueError("all wake stations must have the same number of nodes")
    return {"stations": wake_stations, "n_stations": len(wake_stations)}


def dof_audit(n_wall_nodes, n_wake_nodes, n_stations, n_wall_rows, *, closed_body=False):
    """Unknown/equation bookkeeping of the P1 wall plus wake system."""
    unknown = int(n_wall_nodes) + int(n_wake_nodes)
    rows = int(n_wall_rows) + int(n_stations) + int(n_wake_nodes) - int(n_stations)
    audit = {
        "unknowns": unknown,
        "wall_rows": int(n_wall_rows),
        "kutta_rows": int(n_stations),
        "wake_rows": int(n_wake_nodes) - int(n_stations),
        "equations": rows,
        "square": unknown == rows,
        "gauge_required": not closed_body,
    }
    if unknown != rows:
        raise ValueError("P1 wall plus wake system is not square: %d unknowns, %d equations"
                         % (unknown, rows))
    return audit
