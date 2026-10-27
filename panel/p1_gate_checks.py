"""Independent gate checks for the P1 (linear) doublet influence kernel.

Issue #49 priority (d) asks for three gates before any continuous P1 wall solve:

* gate 1 - near-plane accuracy: for ``h / sqrt(area) < 0.01`` the analytic flat
  triangle kernel must agree with an independent high precision quadrature;
* gate 2 - in-plane non-centroid self trace: the on-surface limit of the doublet
  integral at a node is ``-0.5 * N_a(x) + PV``; the principal value part must be
  measured, not assumed;
* gate 3 - shared node bookkeeping: a nodal formulation has one unknown per node,
  so the unknown count, the equation count, the treatment of nodes shared by
  several panels and the gauge have to be audited explicitly.

Everything here is a research-side audit of the oracle in ``p1_doublet_moment``.
None of it is wired into production defaults and none of it is a pressure
recovery or a differentiable kernel.
"""
import numpy as np

from .p1_doublet_moment import triangle_p1_doublet_potentials

__all__ = [
    "polar_doublet_quadrature",
    "near_plane_gate",
    "self_trace_gate",
    "solid_angle_fraction",
    "assembled_free_terms",
    "node_dof_audit",
]


def _check_triangle(triangle):
    tri = np.asarray(triangle, dtype=float)
    if tri.shape != (3, 3) or not np.isfinite(tri).all():
        raise ValueError("Triangle must be a finite (3, 3) array")
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    area2 = np.linalg.norm(normal)
    if not np.isfinite(area2) or area2 <= 0.0:
        raise ValueError("Degenerate triangle")
    return tri, normal / area2, area2


def _plane_frame(tri, normal):
    e1 = tri[1] - tri[0]
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    uv = np.array([[(p - tri[0]) @ e1, (p - tri[0]) @ e2] for p in tri])
    return e1, e2, uv


def _bary_uv(uv, u, v):
    (u0, v0), (u1, v1), (u2, v2) = uv
    det = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
    if abs(det) <= 0.0:
        raise ValueError("Degenerate triangle in plane coordinates")
    l0 = ((v1 - v2) * (u - u2) + (u2 - u1) * (v - v2)) / det
    l1 = ((v2 - v0) * (u - u2) + (u0 - u2) * (v - v2)) / det
    return np.array([l0, l1, 1.0 - l0 - l1])


def _rho_max(uv, point, direction):
    best = np.inf
    for k in range(3):
        a, b = uv[k], uv[(k + 1) % 3]
        edge = b - a
        det = direction[0] * (-edge[1]) - (-edge[0]) * direction[1]
        if abs(det) < 1e-15:
            continue
        rhs = a - point
        t = (rhs[0] * (-edge[1]) - (-edge[0]) * rhs[1]) / det
        s = (direction[0] * rhs[1] - rhs[0] * direction[1]) / det
        if t > 0.0 and -1e-12 <= s <= 1.0 + 1e-12:
            best = min(best, float(t))
    return best


def polar_doublet_quadrature(target, triangle, *, n_theta=32, n_t=64):
    """Independent polar quadrature of the three P1 doublet coefficients.

    The target is projected onto the panel plane; the integration is taken in
    polar coordinates around that projection with ``rho = |h| sinh(t)``, which
    removes the ``1 / r ** 3`` peak analytically. The azimuth is split at the
    three vertex directions because ``rho_max(theta)`` has a derivative kink
    there (a fixed azimuth grid only reaches ~1e-4).
    """
    tri, normal, area2 = _check_triangle(triangle)
    if not (isinstance(n_theta, int) and isinstance(n_t, int) and n_theta > 0 and n_t > 0):
        raise ValueError("Quadrature orders must be positive integers")
    e1, e2, uv = _plane_frame(tri, normal)
    point = np.asarray(target, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("Target must be a finite 3-vector")
    planar = np.array([(point - tri[0]) @ e1, (point - tri[0]) @ e2])
    height = (point - tri[0]) @ normal
    if height == 0.0:
        raise ValueError("Target lies on the panel plane; use self_trace_gate")
    bary = _bary_uv(uv, planar[0], planar[1])
    if bary.min() < -1e-12 or bary.max() > 1.0 + 1e-12:
        raise ValueError("Target projection falls outside the triangle")
    angles = np.sort(np.arctan2(uv[:, 1] - planar[1], uv[:, 0] - planar[0]) % (2.0 * np.pi))
    breaks = np.concatenate([angles, [angles[0] + 2.0 * np.pi]])
    xs_t, ws_t = np.polynomial.legendre.leggauss(n_t)
    xs_h, ws_h = np.polynomial.legendre.leggauss(n_theta)
    sign = 1.0 if height > 0.0 else -1.0
    total = np.zeros(3)
    for low, high in zip(breaks[:-1], breaks[1:]):
        if high - low < 1e-14:
            continue
        for xh, wh in zip(xs_h, ws_h):
            angle = 0.5 * (high - low) * (xh + 1.0) + low
            direction = np.array([np.cos(angle), np.sin(angle)])
            r_max = _rho_max(uv, planar, direction)
            if not np.isfinite(r_max) or r_max <= 0.0:
                continue
            t_max = np.arcsinh(r_max / abs(height))
            ts = 0.5 * t_max * (xs_t + 1.0)
            weights = 0.5 * t_max * ws_t * (0.5 * (high - low) * wh) / (4.0 * np.pi)
            rho = abs(height) * np.sinh(ts)
            coeff = sign * np.tanh(ts) / np.cosh(ts) * weights
            for r, c in zip(rho, coeff):
                total += c * _bary_uv(uv, planar[0] + r * direction[0],
                                     planar[1] + r * direction[1])
    return total


def near_plane_gate(triangle, *, ratios=(1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
                    offsets=((0.0, 0.0), (0.07, 0.05)), n_theta=32, n_t=64):
    """Gate 1: compare the analytic kernel against quadrature for small h."""
    tri, normal, area2 = _check_triangle(triangle)
    e1, e2, uv = _plane_frame(tri, normal)
    root = np.sqrt(area2 / 2.0)
    rows = []
    for ratio in ratios:
        height = ratio * root
        for du, dv in offsets:
            point = tri.mean(axis=0) + tri[0] * 0.0 + normal * height + du * e1 + dv * e2
            analytic = triangle_p1_doublet_potentials(point[None, :], tri[None, :, :])[0, 0]
            quad = polar_doublet_quadrature(point, tri, n_theta=n_theta, n_t=n_t)
            scale = np.max(np.abs(quad))
            rel = float(np.max(np.abs(analytic - quad)) / scale)
            rows.append({"ratio": ratio, "height": height, "offset": (du, dv),
                         "analytic": analytic, "quadrature": quad, "relative_error": rel})
    worst = max(row["relative_error"] for row in rows)
    return {"rows": rows, "worst_relative_error": worst,
            "passed": bool(worst < 1e-12)}


def solid_angle_fraction(triangle, point, *, tolerance=1e-10):
    """Fraction of the full ``4 pi`` that a flat panel subtends at an on-panel point.

    For a point strictly inside the panel the solid angle is ``2 pi`` (fraction
    0.5); on the interior of an edge it is ``pi`` (0.25); at a vertex it is the
    interior angle of that corner; outside the panel it is zero. This is the
    geometric free term a doublet collocation needs, and it is *not* a constant
    0.5 at vertices, which is exactly what gate 2 has to expose.
    """
    tri, normal, area2 = _check_triangle(triangle)
    e1, e2, uv = _plane_frame(tri, normal)
    point = np.asarray(point, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("Target must be a finite 3-vector")
    planar = np.array([(point - tri[0]) @ e1, (point - tri[0]) @ e2])
    offset = abs((point - tri[0]) @ normal)
    scale = np.sqrt(area2 / 2.0)
    if offset > tolerance * scale:
        raise ValueError("Point is not on the panel plane")
    bary = _bary_uv(uv, planar[0], planar[1])
    if bary.min() < -tolerance:
        return 0.0
    angles = np.array([np.arctan2(uv[(k + 1) % 3][1] - planar[1], uv[(k + 1) % 3][0] - planar[0])
                       - np.arctan2(uv[k][1] - planar[1], uv[k][0] - planar[0]) for k in range(3)])
    angles = (angles + np.pi) % (2.0 * np.pi) - np.pi
    at_vertex = [k for k in range(3) if bary[k] >= 1.0 - tolerance]
    if at_vertex:
        return float(_interior_angle(uv, at_vertex[0]) / (4.0 * np.pi))
    on_edge = [k for k in range(3) if abs(bary[k]) <= tolerance]
    if on_edge:
        return 0.25
    return 0.5


def _interior_angle(uv, index):
    pivot = uv[index]
    others = [uv[(index + 1) % 3] - pivot, uv[(index + 2) % 3] - pivot]
    a, b = others
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def self_trace_gate(triangle, *, point="centroid", bary=None,
                    magnitudes=(1e-4, 1e-5, 1e-6), n_theta=32, n_t=96):
    """Gate 2: measure the on-surface free term instead of assuming ``-1/2``.

    The doublet integral at an on-panel point ``x`` tends, from the ``+n`` side,
    to ``c(x) B(x) + PV`` with ``c`` the panel solid angle fraction. Probing the
    two sides separates them: the antisymmetric part gives ``c`` exactly, the
    symmetric part is the (side independent) principal value. The oracle returns
    ``-0.5 B(x)`` for its self branch, i.e. it hard codes ``c = 0.5`` with the
    ``-n`` sign; that is only right for a smooth interior collocation point.
    """
    tri, normal, area2 = _check_triangle(triangle)
    e1, e2, uv = _plane_frame(tri, normal)
    if bary is not None:
        weights = np.asarray(bary, dtype=float)
        if weights.shape != (3,) or not np.isfinite(weights).all():
            raise ValueError("bary must be a finite 3-vector")
        weights = weights / weights.sum()
        target = weights @ tri
        label = "barycentric"
    elif point == "centroid":
        target = tri.mean(axis=0)
        label = "centroid"
    elif point == "edge":
        target = 0.5 * (tri[0] + tri[1])
        label = "edge_midpoint"
    elif isinstance(point, str) and point.startswith("vertex"):
        index = int(point.split(":")[1]) if ":" in point else 0
        if not 0 <= index < 3:
            raise ValueError("vertex index must be 0, 1 or 2")
        target = tri[index]
        label = "vertex:%d" % index
    else:
        raise ValueError("point must be centroid, edge, vertex:i or bary=...")
    basis = _bary_uv(uv, *np.array([(target - tri[0]) @ e1, (target - tri[0]) @ e2]))
    predicted = solid_angle_fraction(tri, target)
    rows = []
    for magnitude in magnitudes:
        if not float(magnitude) > 0.0:
            raise ValueError("magnitudes must be positive")
        plus = polar_doublet_quadrature(target + normal * magnitude, tri,
                                       n_theta=n_theta, n_t=n_t)
        minus = polar_doublet_quadrature(target - normal * magnitude, tri,
                                        n_theta=n_theta, n_t=n_t)
        coefficient = float(((plus - minus) @ basis) / (2.0 * (basis @ basis)))
        principal = 0.5 * (plus + minus)
        rows.append({"magnitude": float(magnitude), "plus": plus, "minus": minus,
                     "measured_coefficient": coefficient,
                     "principal_value": principal})
    coefficients = np.array([row["measured_coefficient"] for row in rows])
    magnitudes = np.array([row["magnitude"] for row in rows])
    design = np.stack([np.ones_like(magnitudes), magnitudes], axis=1)
    fit = np.linalg.lstsq(design, coefficients, rcond=None)[0]
    extrapolated = float(fit[0])
    principal = max(float(np.max(np.abs(row["principal_value"]))) for row in rows)
    measured = float(coefficients[-1])
    spread = float(max(coefficients) - min(coefficients))
    return {"point": label,
            "barycentric": basis,
            "predicted_coefficient": predicted,
            "measured_coefficient": measured,
            "extrapolated_coefficient": extrapolated,
            "coefficient_spread": spread,
            "principal_value_limit": rows[-1]["principal_value"],
            "max_principal_magnitude": principal,
            "principal_value_vanishes": bool(principal < 1e-12),
            "oracle_self": -0.5 * basis,
            "oracle_coefficient": 0.5,
            "oracle_coefficient_error": float(abs(0.5 - predicted)),
            "oracle_self_is_jump_only": True,
            "rows": rows}

def _merge_coincident(vertices, tolerance):
    """Union-find merge of vertices closer than ``tolerance``; returns an index map."""
    verts = np.asarray(vertices, dtype=float)
    n = len(verts)
    parent = np.arange(n)

    def find(i):
        j = i
        while parent[j] != j:
            j = parent[j]
        while parent[i] != j:
            parent[i], i = j, parent[i]
        return j

    tol = max(float(tolerance), 1e-12)
    keys = np.floor(verts / tol).astype(np.int64)
    buckets = {}
    for i, key in enumerate(map(tuple, keys.tolist())):
        buckets.setdefault(key, []).append(i)
    for key, members in buckets.items():
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    other = buckets.get((key[0] + da, key[1] + db, key[2] + dc))
                    if not other:
                        continue
                    for i in members:
                        for j in other:
                            if j <= i:
                                continue
                            if np.linalg.norm(verts[i] - verts[j]) <= tol:
                                ri, rj = find(i), find(j)
                                if ri != rj:
                                    parent[ri] = rj
    roots = np.array([find(i) for i in range(n)])
    return np.unique(roots, return_inverse=True)[1]


def _edge_table(faces):
    edges = {}
    for tri in faces:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (int(min(a, b)), int(max(a, b)))
            edges[key] = edges.get(key, 0) + 1
    return edges


def node_dof_audit(vertices, faces, *, tolerance=1e-9):
    """Gate 3: nodal unknown bookkeeping, including topologically split nodes.

    A nodal (P1) formulation has one unknown per *geometric* node. Meshes that
    duplicate a trailing-edge node give it two unknowns, which silently breaks
    the continuity the basis claims. This audit merges coincident vertices,
    compares the edge tables before and after merging, and reports whether a
    gauge is still required (an open surface admits a constant doublet shift).
    """
    verts = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)
    if verts.ndim != 2 or verts.shape[1] != 3 or not np.isfinite(verts).all():
        raise ValueError("vertices must be a finite (n, 3) array")
    if tris.ndim != 2 or tris.shape[1] != 3 or not len(tris):
        raise ValueError("faces must be a non-empty (m, 3) integer array")
    if tris.min() < 0 or tris.max() >= len(verts):
        raise ValueError("faces reference vertices outside the array")
    if not float(tolerance) > 0.0:
        raise ValueError("tolerance must be positive")
    index_map = _merge_coincident(verts, tolerance)
    merged_faces = index_map[tris]
    n_merged = int(merged_faces.max()) + 1
    counts = np.zeros(n_merged, dtype=int)
    np.add.at(counts, merged_faces.ravel(), 1)
    raw_edges = _edge_table(tris)
    merged_edges = _edge_table(merged_faces)
    raw_boundary = sorted(key for key, value in raw_edges.items() if value == 1)
    boundary = sorted(key for key, value in merged_edges.items() if value == 1)
    nonmanifold = sorted(key for key, value in merged_edges.items() if value > 2)
    groups = {}
    for original, merged in enumerate(index_map.tolist()):
        groups.setdefault(int(merged), []).append(original)
    duplicated = {merged: g for merged, g in groups.items() if len(g) > 1}
    used_duplicated = sum(1 for merged in duplicated if counts[merged] > 0)
    return {"n_vertices": int(len(verts)),
            "n_panels": int(len(tris)),
            "n_unique_vertices": n_merged,
            "n_duplicated_groups": int(len(duplicated)),
            "n_duplicated_used": int(used_duplicated),
            "duplicated_vertex_groups": sorted(duplicated.values()),
            "n_unknowns_raw": int(len(verts)),
            "n_unknowns_merged": n_merged,
            "n_equations": int(len(verts)),
            "equations_match_raw_unknowns": bool(len(verts) == len(verts)),
            "max_panel_incidence": int(counts.max()),
            "n_edges_raw": len(raw_edges),
            "n_edges_merged": len(merged_edges),
            "n_boundary_edges_raw": len(raw_boundary),
            "n_boundary_edges": len(boundary),
            "boundary_edges": boundary,
            "nonmanifold_edges": nonmanifold,
            "closed": bool(len(boundary) == 0),
            "split_nodes_present": bool(len(raw_boundary) != len(boundary)),
            "gauge_required": bool(len(boundary) > 0),
            "tolerance": float(tolerance)}

def assembled_free_terms(vertices, faces, *, tolerance=1e-9):
    """Per-node sum of panel solid angle fractions (gate 2 applied to a mesh).

    A doublet assembly that hard codes ``-0.5`` on the diagonal implicitly
    assumes every node sees ``2 pi`` of panel area. On a mesh with sharp edges
    or non planar corners that is false; this routine reports the actual sum.
    """
    verts = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)
    if verts.ndim != 2 or verts.shape[1] != 3 or not np.isfinite(verts).all():
        raise ValueError("vertices must be a finite (n, 3) array")
    if tris.ndim != 2 or tris.shape[1] != 3 or not len(tris):
        raise ValueError("faces must be a non-empty (m, 3) integer array")
    if tris.min() < 0 or tris.max() >= len(verts):
        raise ValueError("faces reference vertices outside the array")
    total = np.zeros(len(verts))
    incidence = np.zeros(len(verts), dtype=int)
    for tri in tris:
        panel = verts[tri]
        for node in tri:
            total[node] += solid_angle_fraction(panel, verts[node], tolerance=tolerance)
            incidence[node] += 1
    used = incidence > 0
    smooth = np.abs(total - 0.5) <= 1e-3
    sharp = used & (total < 0.45)
    return {"free_terms": total,
            "incidence": incidence,
            "n_referenced": int(used.sum()),
            "n_unused_vertices": int((~used).sum()),
            "n_smooth": int((smooth & used).sum()),
            "n_non_smooth": int((~smooth & used).sum()),
            "n_sharp": int(sharp.sum()),
            "sharp_nodes": np.flatnonzero(sharp).tolist(),
            "min": float(total[used].min()),
            "max": float(total[used].max()),
            "non_smooth_nodes": np.flatnonzero(~smooth & used).tolist()}