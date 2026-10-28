"""Material nodes on a piecewise-linear doublet sheet: PV velocity and 3D advancement.

Phase A (panel/wake_velocity.py) defines the velocity of a piecewise-linear
doublet sheet from the exact solid angle and its boundary integrals.  Phase B
uses that definition to move *material nodes*, instead of the probe-and-gradient
estimator in panel/p1_steady_free_wake.py (two potential gradients on either
side of each face centroid, then a dy/dx march that requires Vx > 0).

The estimator here is the one approved for Phase B: PV plus the local term,
i.e. the mean of the two one-sided limits

    u_node = 1/2 [ u(x + eps*n) + u(x - eps*n) ],   eps -> 0

which is the principal value of the sheet.  On a vortex sheet u+ - u- = gamma x n,
so the mean is the material velocity and the difference is the jump the sheet
carries; both are reported.  The limit is taken with an explicit eps ladder and
its convergence is measured, never assumed.

What this module deliberately does not do: it does not solve for mu or Gamma
(that is the caller's job; Phase B2 re-solves them every step), it does not
claim a steady wake exists, and it does not add a vortex core, cutoff or probe
floor.  Invalid steps are rejected and counted, never silently clamped into a
plausible-looking wake.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .wake_velocity import sheet_velocity_sum


_EPS_LADDER = (1.0, 0.5, 0.25, 0.125)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def face_geometry(vertices, faces):
    """Unit normals, areas, centroids and edge lengths of the sheet."""
    v = np.asarray(vertices, float)
    f = np.asarray(faces, np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("vertices must be (n,3) and faces (m,3)")
    if len(f) == 0:
        raise ValueError("sheet has no faces")
    tri = v[f]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    a2 = np.linalg.norm(cross, axis=1)
    if np.any(a2 <= 1e-14):
        raise ValueError("sheet has a degenerate (zero area) triangle")
    normals = cross / a2[:, None]
    centers = tri.mean(axis=1)
    edges = np.stack(
        [
            np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
            np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
            np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
        ],
        axis=1,
    )
    return normals, 0.5 * a2, centers, edges, tri


def nodal_normals(vertices, faces, normals=None, areas=None):
    """Area-weighted unit normal at every node; unused nodes get zero."""
    v = np.asarray(vertices, float)
    f = np.asarray(faces, np.int64)
    if normals is None or areas is None:
        normals, areas, _c, _e, _t = face_geometry(v, f)
    acc = np.zeros_like(v)
    for k in range(3):
        np.add.at(acc, f[:, k], normals * areas[:, None])
    length = np.linalg.norm(acc, axis=1)
    out = np.zeros_like(v)
    used = length > 0.0
    out[used] = acc[used] / length[used, None]
    return out, used


def nodal_edge_scale(vertices, faces):
    """Shortest incident edge at every node: the local length scale for eps."""
    _n, _a, _c, edges, _t = face_geometry(vertices, faces)
    scale = np.full(len(np.asarray(vertices, float)), np.inf)
    f = np.asarray(faces, np.int64)
    face_min = edges.min(axis=1)
    for k in range(3):
        np.minimum.at(scale, f[:, k], face_min)
    scale[~np.isfinite(scale)] = 0.0
    return scale


def shape_gradients(vertices, faces):
    """Constant gradients of the three linear shape functions per face.

    Returns (m,3,3): grad[i, a] is grad(phi_a) on face i, so a linear nodal
    field mu has grad(mu) = sum_a mu[face[i,a]] * grad[i,a].  Same construction
    as the analytic P1 triangle potential used by the wall solve.
    """
    normals, areas, _c, _e, tri = face_geometry(vertices, faces)
    a2 = 2.0 * areas
    g1 = np.cross(tri[:, 2] - tri[:, 0], normals) / a2[:, None]
    g2 = np.cross(normals, tri[:, 1] - tri[:, 0]) / a2[:, None]
    return np.stack((-g1 - g2, g1, g2), axis=1)


# --------------------------------------------------------------------------- #
# sheet
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WakeSheet:
    """Triangular material sheet carrying a piecewise-linear doublet strength.

    mu_face is the strength at each face centroid and grad_face its in-plane
    gradient.  Both are kept because the P1 wake solve delivers a per-face
    constant strength while a nodal projection gives a continuous P1 sheet; the
    velocity kernel takes exactly this pair.
    """

    vertices: np.ndarray
    faces: np.ndarray
    mu_face: np.ndarray
    grad_face: np.ndarray
    roots: np.ndarray
    freestream: np.ndarray
    reference_xyz: np.ndarray
    name: str = "wake"

    @property
    def n_nodes(self):
        return int(len(self.vertices))

    @property
    def n_faces(self):
        return int(len(self.faces))

    def triangles(self, vertices=None):
        v = self.vertices if vertices is None else np.asarray(vertices, float)
        return v[np.asarray(self.faces, np.int64)]

    def with_vertices(self, vertices):
        v = np.asarray(vertices, float)
        if v.shape != self.vertices.shape:
            raise ValueError("wake vertices must keep their shape")
        if not np.isfinite(v).all():
            raise ValueError("non-finite wake vertices")
        return replace(self, vertices=v.copy())

    def with_strength(self, mu_face=None, grad_face=None):
        """Replace the strength pair; geometry and material labels are untouched."""
        mu = self.mu_face if mu_face is None else np.asarray(mu_face, float)
        gr = self.grad_face if grad_face is None else np.asarray(grad_face, float)
        if mu.shape != (self.n_faces,) or gr.shape != (self.n_faces, 3):
            raise ValueError("strength arrays do not match the sheet")
        return replace(self, mu_face=mu, grad_face=gr)


def sheet_from_nodal(vertices, faces, mu_node, roots=(), freestream=(1.0, 0.0, 0.0),
                     name="wake", reference_xyz=None):
    """Continuous piecewise-linear sheet from nodal doublet strengths."""
    v = np.asarray(vertices, float)
    f = np.asarray(faces, np.int64)
    mu = np.asarray(mu_node, float)
    if mu.shape != (len(v),):
        raise ValueError("mu_node must have one value per vertex")
    grad = shape_gradients(v, f)
    mu_face = mu[f].mean(axis=1)
    grad_face = np.einsum("fa,fai->fi", mu[f], grad, optimize=True)
    roots = np.asarray(roots, np.int64)
    ref = v.copy() if reference_xyz is None else np.asarray(reference_xyz, float)
    return WakeSheet(v, f, mu_face, grad_face, roots,
                     np.asarray(freestream, float), ref.copy(), name)


def sheet_from_face_constant(vertices, faces, mu_faces, roots=(),
                             freestream=(1.0, 0.0, 0.0), name="wake",
                             reference_xyz=None):
    """Piecewise-constant sheet, the representation the P1 wake solve produces."""
    v = np.asarray(vertices, float)
    f = np.asarray(faces, np.int64)
    mu = np.asarray(mu_faces, float)
    if mu.shape != (len(f),):
        raise ValueError("mu_faces must have one value per face")
    ref = v.copy() if reference_xyz is None else np.asarray(reference_xyz, float)
    return WakeSheet(v, f, mu, np.zeros((len(f), 3)), np.asarray(roots, np.int64),
                     np.asarray(freestream, float), ref.copy(), name)


# --------------------------------------------------------------------------- #
# node velocity: PV + local term
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NodeVelocity:
    """One-sided limits, their mean (PV) and the jump at every node."""

    plus: np.ndarray
    minus: np.ndarray
    pv: np.ndarray
    freestream: np.ndarray
    eps: np.ndarray
    convergence: float
    nodal_normal: np.ndarray
    convergence_array: np.ndarray = None


def node_velocity(sheet, *, eps_fraction=0.05, ladder=_EPS_LADDER,
                  eps_min_fraction=1e-3, nodes=None):
    """Velocity at material nodes as the mean of the two one-sided limits.

    eps is eps_fraction times the shortest incident edge at each node, refined
    along the ladder; the reported convergence is the largest change in the last
    refinement relative to |U|, so a caller can refuse a wake whose node velocity
    has not converged in eps.
    """
    if not (0.0 < eps_fraction < 0.5):
        raise ValueError("eps_fraction must be in (0, 0.5)")
    if len(ladder) < 2:
        raise ValueError("eps ladder needs at least two levels")
    factors = np.asarray(ladder, float)
    if np.any(np.diff(factors) >= 0.0):
        raise ValueError("eps ladder must be strictly decreasing")
    normals, used = nodal_normals(sheet.vertices, sheet.faces)
    scale = nodal_edge_scale(sheet.vertices, sheet.faces)
    tri = sheet.triangles()
    free = np.asarray(sheet.freestream, float)
    sel = np.arange(sheet.n_nodes) if nodes is None else np.asarray(nodes, np.int64)

    plus = np.zeros((sheet.n_nodes, 3))
    minus = np.zeros((sheet.n_nodes, 3))
    eps_out = np.zeros(sheet.n_nodes)
    converged = np.zeros(sheet.n_nodes)
    for i in sel:
        i = int(i)
        if not used[i]:
            raise ValueError("node %d is not part of any wake face" % i)
        eps = eps_fraction * max(scale[i], eps_min_fraction * scale.max())
        eps_out[i] = eps * factors[-1]
        p = sheet.vertices[i]
        pv_prev = None
        for level, factor in enumerate(factors):
            h = eps * factor
            up = sheet_velocity_sum(p + h * normals[i], tri, sheet.mu_face,
                                    sheet.grad_face) + free
            um = sheet_velocity_sum(p - h * normals[i], tri, sheet.mu_face,
                                    sheet.grad_face) + free
            pv = 0.5 * (up + um)
            if level == len(factors) - 1:
                plus[i] = up
                minus[i] = um
                if pv_prev is not None:
                    converged[i] = float(np.linalg.norm(pv - pv_prev) / max(
                        np.linalg.norm(free), 1e-30))
            pv_prev = pv
    chosen = sel[used[sel]]
    return NodeVelocity(
        plus=plus,
        minus=minus,
        pv=0.5 * (plus + minus),
        freestream=free,
        eps=eps_out,
        convergence=float(converged[chosen].max()) if len(chosen) else 0.0,
        nodal_normal=normals,
        convergence_array=converged,
    )


# --------------------------------------------------------------------------- #
# step gates
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StepGates:
    """Every reason a material node step may be refused.

    A discrete doublet sheet near a tip or a junction can produce a velocity that
    no mesh can follow.  A refused step leaves the node where it was and is
    counted, so the caller sees a stalled wake instead of a plausible-looking one.
    """

    min_speed: float = 1e-6
    max_area_ratio: float = 4.0
    max_fold_deg: float = 89.0
    max_step_fraction: float = 0.5
    wall_triangles: object | None = None
    wall_margin: float = 0.0
    check_intersections: bool = True
    intersection_pairs_limit: int = 200000
    max_eps_convergence: float = float("inf")


@dataclass(frozen=True)
class AdvanceResult:
    sheet: object
    accepted: np.ndarray
    reasons: dict
    stats: dict = field(default_factory=dict)


def _point_triangle_distance(p, tri):
    """Distance from p to each triangle and the nearest point, vectorized."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac = b - a, c - a
    ap = p[None, :] - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p[None, :] - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p[None, :] - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = va + vb + vc
    safe = np.where(np.abs(denom) > 0.0, denom, 1.0)
    u = vb / safe
    v = vc / safe
    nearest = a + ab * u[:, None] + ac * v[:, None]
    inside = (va >= 0.0) & (vb >= 0.0) & (vc >= 0.0)
    for i in np.nonzero(~inside)[0]:
        best = None
        bestd = np.inf
        for e0, e1 in ((a[i], b[i]), (b[i], c[i]), (c[i], a[i])):
            edge = e1 - e0
            t = np.clip(np.dot(p - e0, edge) / max(float(np.dot(edge, edge)), 1e-30), 0.0, 1.0)
            q = e0 + t * edge
            d = float(np.linalg.norm(p - q))
            if d < bestd:
                bestd, best = d, q
        nearest[i] = best
    return np.linalg.norm(nearest - p[None, :], axis=1), nearest


def _segment_hits_triangle(p0, p1, tri, eps=1e-12):
    """Moller-Trumbore: does the segment p0 -> p1 cross the triangle?"""
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    d = np.broadcast_to(p1 - p0, e1.shape)
    h = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(det) > eps
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = p0 - tri[:, 0]
    u = inv * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1)
    v = inv * np.einsum("ij,ij->i", d, q)
    t = inv * np.einsum("ij,ij->i", e2, q)
    return ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9) \
        & (t > 1e-9) & (t < 1.0 - 1e-9)


def advance(sheet, velocity, step_length, *, mode="arc_length", relaxation=1.0,
            gates=None, eps_convergence=None):
    """Move material nodes by one arc-length or pseudo-time step.

    mode='arc_length' moves each node step_length along its own streamline;
    mode='pseudo_time' moves it by step_length * u / |U|ref, so a node in slow
    flow advances slowly.  Neither requires Vx > 0, unlike the dy/dx march this
    replaces.
    """
    g = StepGates() if gates is None else gates
    if mode not in ("arc_length", "pseudo_time"):
        raise ValueError("mode must be 'arc_length' or 'pseudo_time'")
    if not (0.0 < relaxation <= 1.0):
        raise ValueError("relaxation must be in (0,1]")
    v = np.asarray(velocity, float)
    if v.shape != (sheet.n_nodes, 3):
        raise ValueError("velocity must be (n_nodes,3)")
    speed = np.linalg.norm(v, axis=1)
    uref = max(float(np.linalg.norm(sheet.freestream)), 1e-30)
    reasons = dict.fromkeys(("root", "nonfinite", "speed", "step", "area",
                             "fold", "wall", "intersection", "convergence"), 0)

    movable = np.ones(sheet.n_nodes, bool)
    movable[np.asarray(sheet.roots, np.int64)] = False
    reasons["root"] = int((~movable).sum())

    dirn = np.zeros_like(v)
    ok = speed > g.min_speed * uref
    dirn[ok] = v[ok] / speed[ok, None]
    reasons["speed"] = int((movable & ~ok).sum())
    movable = movable & ok

    if eps_convergence is not None:
        conv = np.asarray(eps_convergence, float)
        if conv.shape != (sheet.n_nodes,):
            raise ValueError("eps_convergence must be (n_nodes,)")
        shaky = movable & (conv > g.max_eps_convergence)
        reasons["convergence"] = int(shaky.sum())
        movable = movable & ~shaky

    step = step_length * dirn if mode == "arc_length" else (step_length / uref) * v
    median_edge = float(np.median(_edge_lengths(sheet)))
    limit = g.max_step_fraction * median_edge
    over = movable & (np.linalg.norm(step, axis=1) > limit)
    if over.any():
        step[over] *= (limit / np.linalg.norm(step[over], axis=1))[:, None]
        reasons["step"] = int(over.sum())
    length = np.linalg.norm(step, axis=1)
    cand = sheet.vertices + relaxation * step
    bad = movable & ~np.isfinite(cand).all(axis=1)
    reasons["nonfinite"] = int(bad.sum())
    movable = movable & ~bad
    cand[~movable] = sheet.vertices[~movable]

    faces = np.asarray(sheet.faces, np.int64)
    normals0, areas0, _c0, _e0, _t0 = face_geometry(sheet.vertices, faces)
    normals1, areas1, _c1, _e1, _t1 = face_geometry(cand, faces)
    ratio = areas1 / np.maximum(areas0, 1e-300)
    area_bad = (ratio > g.max_area_ratio) | (ratio < 1.0 / g.max_area_ratio)
    cos_fold = np.einsum("ij,ij->i", normals0, normals1)
    fold_bad = cos_fold < np.cos(np.deg2rad(g.max_fold_deg))
    for name, mask in (("area", area_bad), ("fold", fold_bad)):
        if mask.any():
            hit = np.zeros(sheet.n_nodes, bool)
            hit[faces[mask].ravel()] = True
            hit = hit & movable
            reasons[name] = int(hit.sum())
            movable = movable & ~hit
            cand[~movable] = sheet.vertices[~movable]

    if g.wall_triangles is not None and movable.any():
        wall = np.asarray(g.wall_triangles, float)
        if wall.ndim != 3 or wall.shape[1:] != (3, 3):
            raise ValueError("wall_triangles must be (m,3,3)")
        wall_normal = np.cross(wall[:, 1] - wall[:, 0], wall[:, 2] - wall[:, 0])
        for i in np.nonzero(movable)[0]:
            if _segment_hits_triangle(sheet.vertices[i], cand[i], wall).any():
                movable[i] = False
                reasons["wall"] += 1
                continue
            dist, nearest = _point_triangle_distance(cand[i], wall)
            side = np.einsum("ij,ij->i", cand[i][None, :] - nearest, wall_normal)
            if np.any((dist < max(g.wall_margin, 1e-12)) & (side <= 0.0)):
                movable[i] = False
                reasons["wall"] += 1
        cand[~movable] = sheet.vertices[~movable]

    if g.check_intersections and movable.any():
        tri0 = sheet.triangles()
        n_pairs = int(movable.sum()) * len(faces)
        if n_pairs <= g.intersection_pairs_limit:
            incident = [set() for _ in range(sheet.n_nodes)]
            for fi, (a, b, c) in enumerate(faces):
                for n in (a, b, c):
                    incident[int(n)].add(fi)
            for i in np.nonzero(movable)[0]:
                pool = np.ones(len(faces), bool)
                pool[list(incident[int(i)])] = False
                if not pool.any():
                    continue
                if _segment_hits_triangle(sheet.vertices[i], cand[i], tri0[pool]).any():
                    movable[i] = False
                    reasons["intersection"] += 1
            cand[~movable] = sheet.vertices[~movable]
        else:
            reasons["intersection"] = -n_pairs

    stats = dict(
        moved=int(movable.sum()),
        step_max=float(length[movable].max()) if movable.any() else 0.0,
        step_mean=float(length[movable].mean()) if movable.any() else 0.0,
        area_min=float(areas1.min()),
        area_max=float(areas1.max()),
        area_ratio_min=float(ratio.min()),
        area_ratio_max=float(ratio.max()),
    )
    return AdvanceResult(sheet.with_vertices(cand), movable, reasons, stats)


def boundary_nodes(sheet):
    """Mask of nodes on the sheet boundary (an edge used by a single face).

    A doublet sheet whose strength does not vanish at its lateral edge carries a
    concentrated vortex line there, so the inviscid node velocity diverges as the
    eps ladder is refined.  Those are exactly the nodes whose velocity must be
    refused rather than averaged.
    """
    faces = np.asarray(sheet.faces, np.int64)
    counts = {}
    for a, b, c in faces:
        for e in ((min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(c, a), max(c, a))):
            counts[e] = counts.get(e, 0) + 1
    mask = np.zeros(sheet.n_nodes, bool)
    for (a, b), n in counts.items():
        if n == 1:
            mask[a] = True
            mask[b] = True
    return mask


def _edge_lengths(sheet):
    _n, _a, _c, edges, _t = face_geometry(sheet.vertices, sheet.faces)
    return edges.ravel()

# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
def sheet_diagnostics(sheet, vel, *, bands=3):
    """Area-weighted slip, two-sided Cp jump and mesh quality on the sheet.

    Slip is the normal component of the node PV velocity divided by |U|, the
    kinematic quantity a converged wake would drive to zero.  The Cp jump uses
    the sign convention of the half-wake audit,
    DeltaCp = -2 * Vmean . (Vplus - Vminus) / U^2, measured not assumed.
    """
    if bands < 1:
        raise ValueError("bands must be >= 1")
    normals, areas, centers, edges, _tri = face_geometry(sheet.vertices, sheet.faces)
    faces = np.asarray(sheet.faces, np.int64)
    uref = max(float(np.linalg.norm(vel.freestream)), 1e-30)
    face_v = vel.pv[faces].mean(axis=1)
    slip = np.einsum("fi,fi->f", face_v, normals) / uref
    plus_f = vel.plus[faces].mean(axis=1)
    minus_f = vel.minus[faces].mean(axis=1)
    # Audit convention: DeltaCp = Cp_plus - Cp_minus = -2 Vmean.(Vplus-Vminus)/U^2.
    # The two expressions below are algebraically identical; their difference is
    # reported as an identity residual instead of being assumed to vanish.
    dCp = (np.sum(minus_f ** 2, axis=1) - np.sum(plus_f ** 2, axis=1)) / uref ** 2
    audit = -2.0 * np.einsum("fi,fi->f", 0.5 * (plus_f + minus_f), plus_f - minus_f) / uref ** 2
    jump = plus_f - minus_f
    normal_jump = np.einsum("fi,fi->f", jump, normals) / uref

    direction = np.asarray(sheet.freestream, float)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-30)
    s = centers @ direction
    order = np.sort(s)
    cuts = [order[int(len(s) * k / bands)] for k in range(1, bands)]
    band_id = np.searchsorted(np.asarray(cuts, float), s, side="right")

    def _agg(mask):
        w = areas[mask]
        if not len(w):
            return dict(panels=0, slip_rms=None, slip_max=None, dcp_rms=None,
                        dcp_max=None, normal_jump_max=None, audit_rms=None)
        return dict(
            panels=int(mask.sum()),
            slip_rms=float(np.sqrt(np.sum(w * slip[mask] ** 2) / w.sum())),
            slip_max=float(np.max(np.abs(slip[mask]))),
            dcp_rms=float(np.sqrt(np.sum(w * dCp[mask] ** 2) / w.sum())),
            dcp_max=float(np.max(np.abs(dCp[mask]))),
            normal_jump_max=float(np.max(np.abs(normal_jump[mask]))),
            audit_rms=float(np.sqrt(np.sum(w * audit[mask] ** 2) / w.sum())),
        )

    return dict(
        overall=_agg(np.ones(len(faces), bool)),
        bands=[_agg(band_id == b) for b in range(bands)],
        quality=dict(
            area_min=float(areas.min()),
            area_max=float(areas.max()),
            aspect_p90=float(np.percentile(edges.max(axis=1) / edges.min(axis=1), 90)),
            edge_min=float(edges.min()),
            extent=float(s.max() - s.min()),
        ),
        eps_convergence=float(vel.convergence),
        eps_median=float(np.median(vel.eps[vel.eps > 0])),
        dcp_identity_residual=float(np.max(np.abs(dCp - audit))),
    )


def probe_velocity_face_centers(sheet, *, probe_fraction=0.055):
    """The estimator this module replaces, evaluated with the same exact kernel.

    p1_steady_free_wake.mean_wake_velocity takes two gradients of the P1
    potential at a height h = probe_fraction * sqrt(area) on either side of each
    face centroid and averages them.  A gradient of the potential at a point is
    the exact velocity there, so its mean is the mean of the two one-sided
    velocities at +/-h; that is what is computed here, so the two estimators can
    be compared on one and the same strength field without a finite-difference
    truncation term.
    """
    normals, areas, centers, _edges, _tri = face_geometry(sheet.vertices, sheet.faces)
    h = probe_fraction * np.sqrt(areas)
    tri = sheet.triangles()
    free = np.asarray(sheet.freestream, float)
    out = np.zeros((len(areas), 3))
    for i in range(len(areas)):
        p = centers[i]
        n = normals[i]
        up = sheet_velocity_sum(p + h[i] * n, tri, sheet.mu_face, sheet.grad_face) + free
        um = sheet_velocity_sum(p - h[i] * n, tri, sheet.mu_face, sheet.grad_face) + free
        out[i] = 0.5 * (up + um)
    return out


def grid_sheet(nx=6, ny=4, *, chord=1.0, span=1.0, sweep_deg=0.0, camber=0.0,
               mu0=0.3, mu_y=0.0, loading="elliptic", half=False,
               freestream=(1.0, 0.0, 0.0), name="grid"):
    """A regular (nx, ny) wake strip with a smooth doublet strength.

    Nodes are indexed i + j*(nx+1); the first row (i=0) is the TE attachment and
    is returned as the fixed root set.  loading='elliptic' tapers the strength to
    zero at the lateral edges (a free edge with a strength jump would carry an
    artificial vortex line); loading='linear' keeps a linear spanwise profile for
    tests that need an exact analytic gradient.
    """
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1")
    xs = np.linspace(0.0, chord, nx + 1)
    eta = np.linspace(0.0, 1.0, ny + 1) if half else np.linspace(-1.0, 1.0, ny + 1)
    nodes = []
    for j in range(ny + 1):
        y = span * eta[j]
        for i in range(nx + 1):
            x = xs[i] + abs(y) * np.tan(np.deg2rad(sweep_deg))
            z = camber * np.sin(np.pi * xs[i] / chord) * eta[j]
            nodes.append((x, y, z))
    v = np.asarray(nodes, float)
    faces = []
    for j in range(ny):
        for i in range(nx):
            a = i + j * (nx + 1)
            b = a + 1
            c = a + (nx + 1)
            d = c + 1
            faces.append((a, b, d))
            faces.append((a, d, c))
    f = np.asarray(faces, np.int64)
    if loading == "linear":
        mu = mu0 + mu_y * v[:, 1]
    elif loading == "elliptic":
        # Zero circulation at a free lateral edge: without this the sheet ends in
        # a concentrated vortex line and the inviscid node velocity diverges
        # there, which is a property of the model, not of the discretisation.
        mu = mu0 * np.sqrt(np.clip(1.0 - (v[:, 1] / span) ** 2, 0.0, None))
    else:
        raise ValueError("loading must be 'linear' or 'elliptic'")
    roots = np.arange(nx + 1)
    return sheet_from_nodal(v, f, mu, roots=roots, freestream=freestream, name=name)
