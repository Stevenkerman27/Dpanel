"""Right-half multi-shed-group (v2) assembly, validation and dense reference solve.

This module implements Step B of the v2 contract in
`docs/ISSUE39_SHED_GROUPS_V2.md`: several independent lifting trailing edges,
each with its own finite horizontal wake, its own circulation stations and its
own weak Kutta/mass rows, coupled only through the wall doublet intensity.

Nothing here changes the v1 path: `panel.half_native.solve_half_fields` is
untouched, and a single group must reproduce it to round-off (test F1). No
sideslip, no vertical/inclined wake, no separation vortex, no GPU, no claim of
physical accuracy: every report carries `physical_accuracy_accepted=False`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .differentiable import linear_solve
from .half_native import _AXIAL_REFLECT, _REFLECT, _half_loads, _mirror_triangles
from .kernel import triangle_potential

GROUP_KINDS = ("attached_lifting_TE", "prescribed_sharp_separation")
ROLES = ("lifting", "non_lifting", "unclassified")
ROOT_CONDITIONS = ("symmetry_plane", "buried_in_solid")
SCHEMA_V2 = "dpanel.half_shed_groups.v2"
SCHEMA_V1 = "dpanel.half_planar_cut.v1"


def _tol(vertices):
    x = np.asarray(vertices, dtype=float)
    return 1e-10 * max(float(np.ptp(x, axis=0).max()), 1.0)


@dataclass
class ShedGroup:
    """Declared trailing-edge shedding group (numpy, host side only)."""

    group_id: str
    wall_faces: np.ndarray
    wake_faces: np.ndarray
    shed_edges: np.ndarray
    upper_te: np.ndarray
    lower_te: np.ndarray
    stations: np.ndarray
    station_dof: np.ndarray
    kind: str = "attached_lifting_TE"
    role: str | None = None
    source_component_ids: tuple = ()
    root_condition: str = "symmetry_plane"
    buried_inside_component: int | None = None
    wake_z: float | None = None
    # derived, filled by validate_shed_groups
    hat_map: np.ndarray | None = None
    wake_interval: np.ndarray | None = None
    te_interval: np.ndarray | None = None
    audit: dict = field(default_factory=dict)

    @property
    def ng(self):
        return 0 if self.station_dof is None else int(np.max(self.station_dof)) + 1


class ShedGroupsTopology(NamedTuple):
    groups: tuple
    wall_faces: object
    component_ids: object
    neighbor_ghost_ids: object
    audit: dict


def _edges_of(faces):
    f = np.asarray(faces, dtype=np.int64)
    return np.concatenate(
        (f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]), axis=0
    )


def _edge_multiset(faces):
    edges = _edges_of(faces)
    keys = np.sort(edges, axis=1)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    keys = keys[order]
    uniq, counts = np.unique(keys, axis=0, return_counts=True)
    return {tuple(int(v) for v in key): int(c) for key, c in zip(uniq, counts)}


def _wake_edges(wake_faces):
    return {tuple(sorted(int(v) for v in edge)) for edge in _edges_of(wake_faces)}


def _strip_index(stations, y, tol):
    idx = np.searchsorted(stations, y, side="right") - 1
    if np.any(idx < 0) or np.any(idx >= len(stations) - 1):
        return idx, False
    ok = (y >= stations[idx] - tol) & (y <= stations[idx + 1] + tol)
    return idx, bool(np.all(ok))


def _triangle_aabb_intersect(tri_a, tri_b, tol):
    lo_a = tri_a.min(axis=0)
    hi_a = tri_a.max(axis=0)
    lo_b = tri_b.min(axis=0)
    hi_b = tri_b.max(axis=0)
    return np.all(lo_a <= hi_b + tol) and np.all(lo_b <= hi_a + tol)


def _segment_triangle_t(p, q, tri, tol):
    """Moller-Trumbore segment/triangle test; returns the segment parameter."""
    e1 = tri[1] - tri[0]
    e2 = tri[2] - tri[0]
    d = q - p
    h = np.cross(d, e2)
    a = float(np.dot(e1, h))
    if abs(a) < 1e-14:
        return None
    f = 1.0 / a
    s = p - tri[0]
    u = f * float(np.dot(s, h))
    if u < -tol or u > 1.0 + tol:
        return None
    qv = np.cross(s, e1)
    v = f * float(np.dot(d, qv))
    if v < -tol or u + v > 1.0 + tol:
        return None
    t = f * float(np.dot(e2, qv))
    return t if -tol <= t <= 1.0 + tol else None


def validate_shed_groups(vertices, wall_faces, component_ids, groups, *,
                         component_roles=None, prototype_neighbors=None,
                         max_wake_wall_tests=20000):
    """Validate D1-D12 and return a frozen topology, or raise ValueError.

    Every rejection message is the exact English string pinned by section 7 of
    the v2 contract; tests assert on those strings.
    """
    x = np.asarray(vertices, dtype=float)
    f = np.asarray(wall_faces, dtype=np.int32)
    tags = np.asarray(component_ids, dtype=np.int32)
    tol = _tol(x)
    if x.ndim != 2 or x.shape[1] != 3 or len(x) == 0:
        raise ValueError("Shed group surface needs (n,3) coordinates")
    if f.ndim != 2 or f.shape[1] != 3 or len(f) == 0:
        raise ValueError("Shed group surface needs (m,3) wall faces")
    if tags.shape != (len(f),):
        raise ValueError("wall.component_ids must match the wall face count")
    if np.any(x[:, 1] < -tol):
        raise ValueError("Half vertices must lie at y >= 0")
    if np.any(x[f].mean(axis=1)[:, 1] <= tol):
        raise ValueError("Wall triangles must lie on the positive side")
    if not groups:
        return ShedGroupsTopology(
            groups=(),
            wall_faces=jnp.asarray(f),
            component_ids=jnp.asarray(tags),
            neighbor_ghost_ids=jnp.zeros((len(f), 0), dtype=jnp.int32),
            audit=dict(closed_body=True, groups=[], wake_faces=0),
        )

    roles = dict(component_roles or {})
    wall_edge_faces = {}
    for i, face in enumerate(f):
        for a, b in zip(face, np.roll(face, -1)):
            wall_edge_faces.setdefault(tuple(sorted((int(a), int(b)))), []).append(i)
    manifold = _edge_multiset(f)

    seen_edges = {}
    reports = []
    for g in groups:
        if g.kind not in GROUP_KINDS:
            raise ValueError(f"Unknown shed group kind {g.kind!r}")
        if g.role is not None and g.role not in ROLES:
            raise ValueError(f"Unknown shed group role {g.role!r}")
        if g.root_condition not in ROOT_CONDITIONS:
            raise ValueError(f"Unknown shed group root condition {g.root_condition!r}")
        e = np.asarray(g.shed_edges, dtype=np.int32).reshape(-1, 2)
        up = np.asarray(g.upper_te, dtype=np.int32).reshape(-1)
        lo = np.asarray(g.lower_te, dtype=np.int32).reshape(-1)
        w = np.asarray(g.wake_faces, dtype=np.int32).reshape(-1, 3)
        ys = np.asarray(g.stations, dtype=float)
        dof = np.asarray(g.station_dof, dtype=np.int32)
        if len(e) == 0 or len(w) == 0 or len(up) != len(e) or len(lo) != len(e):
            raise ValueError("Shed group needs shed edges, banks and wake faces")
        if ys.ndim != 1 or len(ys) < 2 or dof.shape != ys.shape:
            raise ValueError("Shed group stations and DOFs must match")
        if not np.isfinite(ys).all() or np.any(np.diff(ys) <= 0):
            raise ValueError("Shed group stations must be finite and increasing")
        if abs(float(ys[0])) > tol:
            raise ValueError("Shed group stations must start at y=0")
        if np.any(dof < -1) or dof[-1] != -1:
            raise ValueError("Group circulation must vanish at the outboard tip")
        ng = int(dof.max()) + 1
        if ng < 1 or sorted(set(dof[dof >= 0].tolist())) != list(range(ng)):
            raise ValueError("Group circulation DOFs must be contiguous and present")
        if min(int(v) for v in e.min(axis=0)) < 0 or int(e.max()) >= len(x):
            raise ValueError("Shed edge index out of range")
        if min(int(w.min()), int(e.min())) < 0 or max(int(w.max()), int(e.max())) >= len(x):
            raise ValueError("Group node index out of range")
        if np.any(up == lo):
            raise ValueError("Shed upper and lower banks must be distinct faces")
        if min(int(up.min()), int(lo.min())) < 0 or max(int(up.max()), int(lo.max())) >= len(f):
            raise ValueError("Shed bank face index out of range")
        if np.any(x[w].mean(axis=1)[:, 1] <= tol):
            raise ValueError("Group wake triangles must lie on the positive side")
        wake_edges = _wake_edges(w)
        for k, edge in enumerate(e):
            pair = tuple(sorted(int(v) for v in edge))
            if manifold.get(pair) != 2:
                raise ValueError(
                    "Shed edge must be a real wall edge shared by the group wake")
            if pair not in wall_edge_faces:
                raise ValueError(
                    "Shed edge must be a real wall edge shared by the group wake")
            if not set(pair).issubset(set(int(v) for v in f[int(up[k])])):
                raise ValueError("Shed edge must be shared by its upper bank face")
            if not set(pair).issubset(set(int(v) for v in f[int(lo[k])])):
                raise ValueError("Shed edge must be shared by its lower bank face")
            if pair not in wake_edges:
                raise ValueError(
                    "Shed edge must be a real wall edge shared by the group wake")
            if pair in seen_edges:
                raise ValueError("Duplicate shed edge across groups")
            seen_edges[pair] = g.group_id
        wake_z = x[w][:, :, 2]
        if float(np.ptp(wake_z)) > tol:
            raise ValueError("Only horizontal constant-z wakes are supported in v2")
        z0 = float(wake_z.mean())
        if g.wake_z is not None and abs(float(g.wake_z) - z0) > tol:
            raise ValueError("Group wake is not planar at its declared origin_z")
        cross = np.cross(x[w][:, 1] - x[w][:, 0], x[w][:, 2] - x[w][:, 0])
        if np.any(cross[:, 2] <= 0):
            raise ValueError("Group wake triangles must have +z normals")
        # stations/DOf -> basis
        hat = np.zeros((len(ys), ng), dtype=float)
        rows = np.flatnonzero(dof >= 0)
        hat[rows, dof[rows]] = 1.0
        wc = x[w].mean(axis=1)[:, 1]
        wi = np.searchsorted(ys, wc, side="right") - 1
        if np.any(wi < 0) or np.any(wi >= len(ys) - 1):
            raise ValueError("Group wake/TE left the fixed circulation strips")
        gauss = np.array([0.5 - np.sqrt(3.0) / 6.0, 0.5 + np.sqrt(3.0) / 6.0])
        ends = x[e]
        te_y = ends[:, 0, 1, None] * (1 - gauss) + ends[:, 1, 1, None] * gauss
        ti = np.searchsorted(ys, te_y, side="right") - 1
        if (np.any(ti < 0) or np.any(ti >= len(ys) - 1) or np.any(ti[:, 0] != ti[:, 1])):
            raise ValueError("Group wake/TE left the fixed circulation strips")
        for k in range(len(w)):
            if (x[w[k]][:, 1].min() < ys[wi[k]] - tol
                    or x[w[k]][:, 1].max() > ys[wi[k] + 1] + tol):
                raise ValueError("Group wake/TE left the fixed circulation strips")
        for k in range(len(e)):
            if (te_y[k].min() < ys[ti[k, 0]] - tol
                    or te_y[k].max() > ys[ti[k, 0] + 1] + tol):
                raise ValueError("Group wake/TE left the fixed circulation strips")
        # D11: legality is topological, never a CAD component id
        if g.kind == "attached_lifting_TE":
            if g.role == "non_lifting":
                raise ValueError(
                    "Shed group role contradicts the declared component roles")
            declared = [roles[c] for c in g.source_component_ids if c in roles]
            if g.role is not None and declared and any(r != g.role for r in declared):
                raise ValueError(
                    "Shed group role contradicts the declared component roles")
            if not declared and any(c not in roles for c in g.source_component_ids):
                raise ValueError(
                    "Attached lifting shed group requires a verified physical TE and bank topology")
        # D9 root condition
        root_y = np.abs(x[e][:, :, 1]).min()
        if g.root_condition == "symmetry_plane":
            if root_y > tol:
                raise ValueError("Symmetry-plane root must touch the y=0 plane")
        else:
            if root_y <= tol:
                raise ValueError("Buried root must not lie on the symmetry plane")
            if g.buried_inside_component is None:
                raise ValueError("Buried root needs an explicit buried component id")
            inner = f[tags == int(g.buried_inside_component)]
            if len(inner) == 0:
                raise ValueError("Buried root is not inside the declared component")
            box_lo = x[inner].reshape(-1, 3).min(axis=0) - tol
            box_hi = x[inner].reshape(-1, 3).max(axis=0) + tol
            root_nodes = x[np.unique(e)]
            if np.any(root_nodes < box_lo) or np.any(root_nodes > box_hi):
                raise ValueError("Buried root is not inside the declared component")
        # D10 wake must not cut other wall panels (AABB prefilter + edge hits)
        tested = 0
        hits = 0
        banks = set(int(v) for v in up) | set(int(v) for v in lo)
        for k in range(len(w)):
            tri = x[w[k]]
            wnodes = set(int(v) for v in w[k])
            tested_here = 0
            for j in range(len(f)):
                if j in banks or wnodes & set(int(v) for v in f[j]):
                    continue
                if not _triangle_aabb_intersect(tri, x[f[j]], tol):
                    continue
                tested_here += 1
                for a, b in zip(tri, np.roll(tri, -1)):
                    t = _segment_triangle_t(a, b, x[f[j]], tol)
                    if t is not None and tol < t < 1.0 - tol:
                        hits += 1
                        break
            if tested_here + tested > max_wake_wall_tests:
                raise ValueError("Group wake/wall test budget exhausted")
            tested += tested_here
        if hits:
            raise ValueError("Group wake intersects a wall panel")
        g.hat_map = hat
        g.wake_interval = wi
        g.te_interval = ti
        g.wake_z = z0
        g.audit = dict(
            group_id=g.group_id,
            kind=g.kind,
            role=g.role,
            root_condition=g.root_condition,
            circulation_unknowns=ng,
            stations=len(ys),
            shed_edges=len(e),
            wake_faces=len(w),
            wake_z=z0,
            root_y=float(root_y),
            wake_wall_tests=tested,
            root_containment=("symmetry_plane" if g.root_condition == "symmetry_plane"
                              else "aabb_and_edge_distance"),
        )
        reports.append(g.audit)

    # D12: the wake sheet is a disk. Interior edges carry exactly two triangles;
    # a boundary edge is legal only on the declared TE, on the symmetry root
    # line, or on the downstream truncation of the finite wake.
    for g in groups:
        counts = _edge_multiset(g.wake_faces)
        if any(c > 2 for c in counts.values()):
            raise ValueError("Group wake has a non-manifold interior edge")
        if any(c < 1 for c in counts.values()):
            raise ValueError("Group wake has an unused edge")
        shed = {tuple(sorted(int(v) for v in edge)) for edge in np.asarray(g.shed_edges)}
        trunc = float(np.max(x[np.asarray(g.wake_faces)].reshape(-1, 3)[:, 0])) - tol
        tip = float(ys[-1]) - tol
        for (a, b), count in counts.items():
            if count == 2 or (a, b) in shed:
                continue
            if abs(float(x[a, 1])) <= tol and abs(float(x[b, 1])) <= tol:
                continue
            if float(x[a, 1]) >= tip and float(x[b, 1]) >= tip:
                continue
            if float(x[a, 0]) >= trunc and float(x[b, 0]) >= trunc:
                continue
            raise ValueError(
                "Group wake boundary is not a declared TE, root line, tip line or downstream truncation")
    if prototype_neighbors is None:
        neighbor_ghost_ids = np.zeros((len(f), 0), dtype=np.int32)
    else:
        neighbor_ghost_ids = np.asarray(prototype_neighbors, dtype=np.int32)
    return ShedGroupsTopology(
        groups=tuple(groups),
        wall_faces=jnp.asarray(f),
        component_ids=jnp.asarray(tags),
        neighbor_ghost_ids=neighbor_ghost_ids,
        audit=dict(closed_body=False, groups=reports,
                   wake_faces=int(sum(len(g.wake_faces) for g in groups))),
    )


class _WallTags(NamedTuple):
    """Minimal duck-typed topology consumed by panel.half_native._half_loads."""

    faces: object
    component_ids: object
    neighbor_ghost_ids: object


def group_operators(vertices, group, centers):
    """Wake doublet operator, weak Kutta row and circulation mass of one group."""
    x = jnp.asarray(vertices, dtype=float)
    y = jnp.asarray(np.asarray(group.stations, dtype=float))
    hat = jnp.asarray(np.asarray(group.hat_map, dtype=float))
    wake = x[jnp.asarray(np.asarray(group.wake_faces, dtype=np.int32))]
    wi = jnp.asarray(np.asarray(group.wake_interval, dtype=np.int32))
    dy = y[wi + 1] - y[wi]
    frac = (wake.mean(axis=1)[:, 1] - y[wi]) / dy
    basis = (1.0 - frac[:, None]) * hat[wi] + frac[:, None] * hat[wi + 1]
    slope = (hat[wi + 1] - hat[wi]) / dy[:, None]
    edges = np.asarray(group.shed_edges, dtype=np.int32)
    up = np.asarray(group.upper_te, dtype=np.int32)
    lo = np.asarray(group.lower_te, dtype=np.int32)
    ends = x[jnp.asarray(edges)]
    gauss = jnp.asarray([0.5 - np.sqrt(3.0) / 6.0, 0.5 + np.sqrt(3.0) / 6.0])
    te_y = ends[:, 0, 1, None] * (1.0 - gauss) + ends[:, 1, 1, None] * gauss
    ti = jnp.asarray(np.asarray(group.te_interval, dtype=np.int32))
    tf = (te_y - y[ti]) / (y[ti + 1] - y[ti])
    te_basis = (1.0 - tf[..., None]) * hat[ti] + tf[..., None] * hat[ti + 1]
    length = jnp.linalg.norm(ends[:, 1] - ends[:, 0], axis=1)
    weights = jnp.broadcast_to(length[:, None] / 2.0, te_y.shape)
    mass = jnp.einsum("eq,eqi,eqj->ij", weights, te_basis, te_basis)
    integrals = jnp.einsum("eq,eqi->ei", weights, te_basis)
    measure = integrals.sum(axis=0)
    incidence = jnp.zeros((len(edges), len(centers)), dtype=x.dtype)
    rows = jnp.arange(len(edges))
    incidence = incidence.at[rows, jnp.asarray(up)].add(1.0)
    incidence = incidence.at[rows, jnp.asarray(lo)].add(-1.0)
    kutta = (integrals.T @ incidence) / measure[:, None]
    circulation_mass = mass / measure[:, None]
    dw, my = triangle_potential(centers, wake, source=False, linear_y=True)
    di, mi = triangle_potential(centers, _mirror_triangles(wake),
                                source=False, linear_y=True)
    # As in v1: an even circulation strip makes the image slope enter negated.
    wake_operator = (dw + di) @ basis + (my - mi) @ slope
    return dict(wake=wake, basis=basis, slope=slope, te_basis=te_basis,
                weights=weights, kutta=kutta, mass=circulation_mass, measure=measure,
                integrals=integrals, raw_mass=mass, incidence=incidence,
                wake_operator=wake_operator)


def assemble_half_groups(vertices, topology, freestream):
    """Assemble the v2 dense reference operator; no full-mesh image is built."""
    if not jax.config.x64_enabled:
        raise ValueError("Explicitly enable JAX float64")
    x = jnp.asarray(vertices, dtype=float)
    wall = jnp.asarray(np.asarray(topology.wall_faces, dtype=np.int32))
    tri = x[wall]
    av = jnp.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) / 2.0
    normals = av / jnp.linalg.norm(av, axis=1)[:, None]
    flow = jnp.asarray(freestream, dtype=float)
    sigma = -(normals) @ flow
    centers = tri.mean(axis=1)
    sd, dd = triangle_potential(centers, tri, interior_diagonal=True)
    si, di = triangle_potential(centers, _mirror_triangles(tri))
    doublet = dd + di
    single = sd + si
    ops = [group_operators(vertices, g, centers) for g in topology.groups]
    sizes = [int(o["wake_operator"].shape[1]) for o in ops]
    nf = len(tri)
    blocks = [[doublet] + [o["wake_operator"] for o in ops]]
    for k, o in enumerate(ops):
        row = [-o["kutta"]]
        for j in range(len(ops)):
            row.append(o["mass"] if j == k
                       else jnp.zeros((sizes[k], sizes[j]), dtype=x.dtype))
        blocks.append(row)
    matrix = jnp.block(blocks) if ops else doublet
    rhs = jnp.concatenate([single @ sigma]
                          + [jnp.zeros((n,), dtype=x.dtype) for n in sizes])
    offsets = []
    start = nf
    for n in sizes:
        offsets.append((start, start + n))
        start += n
    return dict(matrix=matrix, rhs=rhs, sigma=sigma, single=single,
                doublet=doublet, centers=centers, normals=normals, flow=flow,
                operators=ops, sizes=sizes, wall=(0, nf), group_slices=offsets,
                unknowns=nf + sum(sizes), wall_panels=nf)


def pressure_neighbors(vertices, topology, contact_cuts=(), *, rings=2,
                       max_rings=6):
    """Cut-aware wall stencil over all shed edges, shared by every group."""
    from .half_native import _pressure_neighbors_from_half

    if topology.groups:
        edges = np.concatenate(
            [np.asarray(g.shed_edges, dtype=np.int32) for g in topology.groups],
            axis=0)
    else:
        edges = np.zeros((0, 2), dtype=np.int32)
    return _pressure_neighbors_from_half(
        np.asarray(vertices, dtype=float),
        np.asarray(topology.wall_faces, dtype=np.int32),
        edges, rings, contact_cuts, max_rings=max_rings)


def solve_half_groups(vertices, topology, freestream, density, reference_area,
                      reference_chord, reference_point, *, contact_cuts=(),
                      solve=None):
    """Dense reference solve of the v2 system; returns loads and per-group rows."""
    assembled = assemble_half_groups(vertices, topology, freestream)
    solver = linear_solve if solve is None else solve
    state = solver(assembled["matrix"], assembled["rhs"])
    nf = assembled["wall_panels"]
    mu = state[:nf]
    gammas = [state[a:b] for a, b in assembled["group_slices"]]
    neighbors = pressure_neighbors(vertices, topology, contact_cuts)
    tags = _WallTags(
        faces=jnp.asarray(np.asarray(topology.wall_faces, dtype=np.int32)),
        component_ids=jnp.asarray(np.asarray(topology.component_ids, dtype=np.int32)),
        neighbor_ghost_ids=jnp.asarray(np.asarray(neighbors, dtype=np.int32)),
    )
    loads = _half_loads(jnp.asarray(vertices, dtype=float), tags,
                        assembled["flow"], density, reference_area,
                        reference_chord, reference_point, mu)
    groups = []
    for g, op, gamma in zip(topology.groups, assembled["operators"], gammas):
        up = jnp.asarray(np.asarray(g.upper_te, dtype=np.int32))
        lo = jnp.asarray(np.asarray(g.lower_te, dtype=np.int32))
        jump = mu[up] - mu[lo]
        kutta = op["kutta"] @ mu - op["mass"] @ gamma
        groups.append(dict(
            group_id=g.group_id,
            kind=g.kind,
            role=g.role,
            root_condition=g.root_condition,
            circulation_unknowns=int(op["wake_operator"].shape[1]),
            gamma=gamma,
            gamma_sum=float(jnp.sum(gamma)),
            gamma_max_abs=float(jnp.max(jnp.abs(gamma))) if len(gamma) else 0.0,
            te_jump=jump,
            te_jump_rms=float(jnp.sqrt(jnp.mean(jump * jump))) if len(jump) else 0.0,
            te_jump_max_abs=float(jnp.max(jnp.abs(jump))) if len(jump) else 0.0,
            kutta_residual_rms=(float(jnp.sqrt(jnp.mean(kutta * kutta)))
                                if len(kutta) else 0.0),
            wake_doublet=op["basis"] @ gamma,
            wake_transport=op["slope"] @ gamma,
            wake_z=float(g.wake_z),
            wake_faces=len(g.wake_faces),
            shed_edges=len(g.shed_edges),
        ))
    out = dict(loads)
    out.update(
        matrix_half=assembled["matrix"], rhs_half=assembled["rhs"],
        mu_half=mu, gamma_groups=gammas, sigma_half=assembled["sigma"],
        linear_residual_half=assembled["matrix"] @ state - assembled["rhs"],
        unknowns=assembled["unknowns"], wall_panels=nf,
        group_slices=assembled["group_slices"], group_rows=groups,
        physical_accuracy_accepted=False,
    )
    return out

