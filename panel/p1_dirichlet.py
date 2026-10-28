"""P1 doublet Dirichlet solve for closed non-lifting bodies (issue #49, priority d, G0).

This module is the solver-level companion of :mod:`panel.p1_gate_checks` and
:mod:`panel.p1_doublet_moment`.  It assembles the *nodal* P1 doublet operator,
solves the interior Dirichlet problem for a closed body with a piecewise linear
doublet layer, and recovers a per-panel Cp from the panel-wise constant surface
gradient of that layer.  It is research-only: nothing here touches the production
solver in :mod:`panel.kernel` / :mod:`panel.half_native`.

Formulation actually validated here (see docs/ISSUE49_P1_G0_SPHERE.md):

    D[mu] = -phi_inf,      mu, phi_inf evaluated on the body surface,

i.e. the doublet-only (direct, Morino-style) Dirichlet formulation with *no*
source term.  For a closed body the source term is redundant: a single layer with
density n.U_inf reproduces one third of the freestream potential on a sphere, and
including it shifts the doublet solution to mu = -phi_inf, which is not the surface
potential and gives O(1) wrong pressures.  That variant is kept as an explicit,
tested negative result (``sigma_sign`` argument) instead of being silently dropped.

Collocation points are the panel centroids, not the mesh vertices.  Vertex
collocation would need the free term to be the solid angle of the *curved* mesh at
the vertex; the analytic triangle formula degenerates exactly on the surface and the
assembled row sums stop being -1 (measured: -0.816 / -0.911 / -0.957 for 80 / 320 /
1280 faces).  At centroids the free term is the smooth limit -B/2 and the row sum of
the assembled operator is -1 to 5.6e-16 at every level, which is the discrete
statement that a uniform doublet layer produces the constant potential -1 inside.
The system is overdetermined (one row per panel, one column per node, M ~ 2N for a
closed mesh) and is solved in the least squares sense.
"""

import numpy as np

__all__ = [
    "icosphere",
    "panel_doublet_coefficients",
    "self_free_row",
    "assemble_p1_doublet",
    "solve_p1_dirichlet",
    "face_cp_from_mu",
    "sphere_case",
]

_FOUR_PI = 4.0 * np.pi
_SELF_FREE_TOTAL = -0.5


def self_free_row():
    """Free-term coefficients of one panel at its own centroid (-B/2 per basis)."""
    return np.full(3, _SELF_FREE_TOTAL / 3.0)


def _as_float64(array, name):
    out = np.ascontiguousarray(array, dtype=np.float64)
    if out.ndim < 1 or out.size == 0:
        raise ValueError("%s must be a non-empty array" % name)
    return out


def icosphere(level=2, radius=1.0):
    """Geodesic sphere from an icosahedron by edge bisection + projection."""
    if level < 0:
        raise ValueError("icosphere level must be non-negative")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("icosphere radius must be positive and finite")
    t = (1.0 + np.sqrt(5.0)) / 2.0
    base = np.array([[-1.0, t, 0.0], [1.0, t, 0.0], [-1.0, -t, 0.0], [1.0, -t, 0.0],
                     [0.0, -1.0, t], [0.0, 1.0, t], [0.0, -1.0, -t], [0.0, 1.0, -t],
                     [t, 0.0, -1.0], [t, 0.0, 1.0], [-t, 0.0, -1.0], [-t, 0.0, 1.0]], dtype=np.float64)
    faces = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                      [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                      [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                      [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]], dtype=np.int64)
    vertices = base / np.linalg.norm(base, axis=1)[:, None]
    for _ in range(level):
        edges = {}
        midpoint_ids = {}
        new_faces = []
        next_id = len(vertices)
        extra = []

        def midpoint(a, b):
            nonlocal next_id
            key = (a, b) if a < b else (b, a)
            if key in midpoint_ids:
                return midpoint_ids[key]
            point = 0.5 * (vertices[a] + vertices[b])
            extra.append(point / np.linalg.norm(point))
            midpoint_ids[key] = next_id
            next_id += 1
            return midpoint_ids[key]

        for face in faces:
            a, b, c = int(face[0]), int(face[1]), int(face[2])
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
        vertices = np.vstack([vertices, np.array(extra, dtype=np.float64)])
        faces = np.array(new_faces, dtype=np.int64)
    return vertices * radius, faces


def _panel_geometry(triangle):
    e1 = triangle[1] - triangle[0]
    e2 = triangle[2] - triangle[0]
    cross = np.cross(e1, e2)
    area2 = np.linalg.norm(cross)
    if not np.isfinite(area2) or area2 <= 0.0:
        raise ValueError("Degenerate triangle: zero area")
    normal = cross / area2
    g1 = np.cross(e2, normal) / area2
    g2 = np.cross(normal, e1) / area2
    basis_gradients = np.stack((-g1 - g2, g1, g2))
    return normal, basis_gradients


def panel_doublet_coefficients(targets, triangle):
    """Exact P1 doublet coefficients of one flat triangle at off-panel targets.

    Returns an array of shape ``(len(targets), 3)``: entry ``[i, a]`` is the
    doublet potential at ``targets[i]`` of the P1 basis function associated with
    vertex ``a`` of ``triangle``, using the same arctan2 solid angle kernel and the
    same linear moment term as :mod:`panel.kernel` and :mod:`panel.p1_doublet_moment`.
    Targets lying on the panel plane inside the triangle are singular; the caller
    must supply their self free term (-B/2) instead.
    """
    targets = _as_float64(targets, "targets").reshape(-1, 3)
    triangle = _as_float64(triangle, "triangle").reshape(3, 3)
    normal, basis_gradients = _panel_geometry(triangle)
    center = triangle.mean(axis=0)
    r = triangle[None, :, :] - targets[:, None, :]
    lengths = np.linalg.norm(r, axis=2)
    if np.any(lengths <= 0.0):
        raise ValueError("Target coincides with a panel vertex; use the self free term")
    r0, r1, r2 = r[:, 0], r[:, 1], r[:, 2]
    l0, l1, l2 = lengths[:, 0], lengths[:, 1], lengths[:, 2]
    det = np.sum(r0 * np.cross(r1, r2), axis=1)
    den = l0 * l1 * l2 + (r0 * r1).sum(1) * l2 + (r1 * r2).sum(1) * l0 + (r2 * r0).sum(1) * l1
    doublet = -2.0 * np.arctan2(det, den) / _FOUR_PI
    height = (targets - triangle[0]) @ normal
    edge_grad = np.zeros((len(targets), 3))
    for k, q in ((0, 1), (1, 2), (2, 0)):
        edge = triangle[q] - triangle[k]
        length = float(np.linalg.norm(edge))
        outward = np.cross(edge / length, normal)
        dot = (r[:, k] * r[:, q]).sum(1)
        prod = lengths[:, k] * lengths[:, q]
        cross = np.cross(r[:, k], r[:, q])
        opposite = dot < 0.0
        plus = np.where(opposite, (cross * cross).sum(1) / np.where(opposite, prod - dot, 1.0), prod + dot)
        gap = 2.0 * plus / (lengths[:, k] + lengths[:, q] + length)
        edge_grad -= outward[None, :] * np.log1p(2.0 * length / gap)[:, None]
    moment = (targets - center) * doublet[:, None] + height[:, None] * edge_grad / _FOUR_PI
    return doublet[:, None] / 3.0 + moment @ basis_gradients.T


def assemble_p1_doublet(vertices, faces):
    """Assemble the nodal P1 doublet operator collocated at panel centroids.

    Returns a dict with ``matrix`` (n_panels x n_vertices), ``centroids``,
    ``triangles``, ``normals``, ``areas`` and ``row_sums``.
    """
    vertices = _as_float64(vertices, "vertices").reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (n_panels, 3)")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("faces reference vertices out of range")
    triangles = vertices[faces]
    normals = np.zeros((len(faces), 3))
    basis_gradients = np.zeros((len(faces), 3, 3))
    for j in range(len(faces)):
        normal, gradients = _panel_geometry(triangles[j])
        normals[j] = normal
        basis_gradients[j] = gradients
    areas = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                          triangles[:, 2] - triangles[:, 0]), axis=1)
    centroids = triangles.mean(axis=1)
    matrix = np.zeros((len(faces), len(vertices)))
    for j in range(len(faces)):
        coefficients = panel_doublet_coefficients(centroids, triangles[j])
        coefficients[j] = self_free_row()
        for k in range(3):
            matrix[:, faces[j][k]] += coefficients[:, k]
    return {"matrix": matrix, "centroids": centroids, "triangles": triangles,
            "normals": normals, "basis_gradients": basis_gradients, "areas": areas,
            "row_sums": matrix.sum(axis=1)}


def solve_p1_dirichlet(vertices, faces, freestream, *, sigma_sign=0.0):
    """Solve the closed-body Dirichlet problem in the P1 doublet unknowns.

    ``sigma_sign=0`` (default) is the validated doublet-only formulation
    ``D[mu] = -phi_inf``.  ``sigma_sign=+/-1`` adds the single layer
    ``sigma = sign * (n . U_inf)`` and is kept only as a tested negative result.
    """
    vertices = _as_float64(vertices, "vertices").reshape(-1, 3)
    freestream = _as_float64(freestream, "freestream").reshape(3)
    sigma_sign = float(sigma_sign)
    if sigma_sign not in (0.0, 1.0, -1.0):
        raise ValueError("sigma_sign must be 0, +1 or -1")
    state = assemble_p1_doublet(vertices, faces)
    matrix = state["matrix"]
    phi_inf = state["centroids"] @ freestream
    rhs = -phi_inf
    if sigma_sign != 0.0:
        single = _p0_single_matrix(state["triangles"], state["centroids"], state["normals"])
        sigma = sigma_sign * (state["normals"] @ freestream)
        rhs = rhs - single @ sigma
    mu, residuals, rank, singular = np.linalg.lstsq(matrix, rhs, rcond=None)
    residual = float(np.abs(matrix @ mu - rhs).max())
    cp = face_cp_from_mu(vertices, faces, mu, speed=float(np.linalg.norm(freestream)))
    return {"mu": mu, "cp": cp, "residual": residual, "rank": int(rank),
            "singular_values": singular, "matrix": matrix,
            "centroids": state["centroids"], "triangles": state["triangles"],
            "normals": state["normals"], "areas": state["areas"],
            "row_sums": state["row_sums"], "sigma_sign": sigma_sign,
            "freestream": freestream}


def face_cp_from_mu(vertices, faces, mu, *, speed=1.0):
    """Cp per panel from the panel-wise constant gradient of the P1 potential."""
    vertices = _as_float64(vertices, "vertices").reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64)
    mu = _as_float64(mu, "mu").reshape(-1)
    if len(mu) != len(vertices):
        raise ValueError("mu must have one entry per vertex")
    triangles = vertices[faces]
    gradients = np.zeros((len(faces), 3))
    for j, face in enumerate(faces):
        triangle = triangles[j]
        cross = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        area2 = float(np.linalg.norm(cross))
        if area2 <= 0.0:
            raise ValueError("Degenerate triangle: zero area")
        normal = cross / area2
        gradient = np.zeros(3)
        for k in range(3):
            a = triangle[(k + 1) % 3] - triangle[(k + 2) % 3]
            gradient += mu[face[k]] * np.cross(normal, a) / area2
        gradients[j] = gradient
    return 1.0 - np.sum(gradients * gradients, axis=1) / (speed * speed)


def _p0_single_matrix(triangles, targets, normals):
    matrix = np.zeros((len(targets), len(triangles)))
    for j in range(len(triangles)):
        triangle = triangles[j]
        normal = normals[j]
        r = triangle[None, :, :] - targets[:, None, :]
        lengths = np.linalg.norm(r, axis=2)
        r0, r1, r2 = r[:, 0], r[:, 1], r[:, 2]
        l0, l1, l2 = lengths[:, 0], lengths[:, 1], lengths[:, 2]
        det = np.sum(r0 * np.cross(r1, r2), axis=1)
        den = l0 * l1 * l2 + (r0 * r1).sum(1) * l2 + (r1 * r2).sum(1) * l0 + (r2 * r0).sum(1) * l1
        omega = -2.0 * np.arctan2(det, den)
        height = (targets - triangle[0]) @ normal
        edge_sum = np.zeros(len(targets))
        for k, q in ((0, 1), (1, 2), (2, 0)):
            edge = triangle[q] - triangle[k]
            length = float(np.linalg.norm(edge))
            outward = np.cross(edge / length, normal)
            dot = (r[:, k] * r[:, q]).sum(1)
            prod = lengths[:, k] * lengths[:, q]
            cross = np.cross(r[:, k], r[:, q])
            opposite = dot < 0.0
            plus = np.where(opposite, (cross * cross).sum(1) / np.where(opposite, prod - dot, 1.0), prod + dot)
            gap = 2.0 * plus / (lengths[:, k] + lengths[:, q] + length)
            edge_sum += (r[:, k] @ outward) * np.log1p(2.0 * length / gap)
        matrix[:, j] = (edge_sum - height * omega) / _FOUR_PI
    return matrix


def sphere_case(level=2, radius=1.0, speed=1.0, sigma_sign=0.0):
    """G0 check: uniform flow past a sphere, compared with the exact surface Cp."""
    vertices, faces = icosphere(level, radius=radius)
    freestream = np.array([speed, 0.0, 0.0])
    solution = solve_p1_dirichlet(vertices, faces, freestream, sigma_sign=sigma_sign)
    triangles = vertices[faces]
    face_centers = triangles.mean(axis=1)
    cos_face = np.clip(face_centers[:, 0] / radius, -1.0, 1.0)
    exact_face = 1.0 - 2.25 * (1.0 - cos_face ** 2)
    error = solution["cp"] - exact_face
    node_cp = np.zeros(len(vertices))
    node_count = np.zeros(len(vertices))
    for j, face in enumerate(faces):
        node_cp[face] += solution["cp"][j]
        node_count[face] += 1.0
    node_cp /= np.maximum(node_count, 1.0)
    cos_node = np.clip(vertices[:, 0] / radius, -1.0, 1.0)
    exact_node = 1.0 - 2.25 * (1.0 - cos_node ** 2)
    fit = np.linalg.lstsq(np.stack([cos_node, np.ones_like(cos_node)], axis=1),
                          solution["mu"] / (speed * radius), rcond=None)[0]
    return {"level": level, "vertices": len(vertices), "faces": len(faces),
            "mu_alpha": float(fit[0]), "mu_beta": float(fit[1]),
            "row_sum_mean": float(solution["row_sums"].mean()),
            "row_sum_spread": float(np.ptp(solution["row_sums"])),
            "least_squares_residual": solution["residual"],
            "cp_rms": float(np.sqrt((error ** 2).mean())),
            "cp_max": float(np.abs(error).max()),
            "cp_rms_node": float(np.sqrt(((node_cp - exact_node) ** 2).mean())),
            "cp_nose": float(node_cp[np.argmax(cos_node)]),
            "cp_tail": float(node_cp[np.argmin(cos_node)]),
            "area_total": float(solution["areas"].sum()),
            "solution": solution}
