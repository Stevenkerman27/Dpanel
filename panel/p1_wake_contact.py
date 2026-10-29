"""Conforming P1 potential traces at declared wake/wall contact cuts.

A wake doublet sheet carries a potential jump. At its boundary on a smooth
body the wall trace has the same jump, with the oriented edge signs cancelling
its line vortex. This is a branch compatibility condition, not another Kutta
condition and not a change to the impermeable-body boundary condition.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import replace
import numpy as np


def _edge_table(faces):
    table = defaultdict(list)
    for f, row in enumerate(faces):
        for a, b in zip(row, np.roll(row, -1)):
            a, b = int(a), int(b)
            table[tuple(sorted((a, b)))].append((f, a, b))
    return table


def split_wall_trace_fans(xyz, faces, wake_faces, cut_edges, cut_kind, gamma_hat):
    """Split vertex fans only across declared cuts, with oriented jump pairs.

    All triangles refer to the same geometric vertex table. Every declared cut
    must have exactly two oppositely wound wall faces and one incident wake
    face. The plus bank is the wall face whose edge orientation opposes the
    wake edge. Thus mu_plus - mu_minus = Gamma cancels the artificial line
    vortex independently of global coordinates or the sign of normal_z.
    """
    x = np.asarray(xyz, float)
    F = np.asarray(faces, int)
    W = np.asarray(wake_faces, int)
    E = np.asarray(cut_edges, int)
    kinds = np.asarray(cut_kind, str)
    H = np.asarray(gamma_hat, float)
    if (x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all()
            or F.ndim != 2 or F.shape[1] != 3 or W.ndim != 2 or W.shape[1] != 3
            or E.ndim != 2 or E.shape[1] != 2 or kinds.shape != (len(E),)
            or H.ndim != 2 or H.shape[0] != len(x) or not np.isfinite(H).all()):
        raise ValueError('Invalid wall/wake/cut arrays')
    for name, arr in (('wall', F), ('wake', W), ('cut', E)):
        if not arr.size or np.any(arr < 0) or np.any(arr >= len(x)):
            raise ValueError(f'Invalid {name} vertex references')
        if any(len(set(map(int, row))) != len(row) for row in arr):
            raise ValueError(f'Repeated vertex in {name}')
    wall_edges, wake_edges = _edge_table(F), _edge_table(W)
    if any(len(v) != 2 or v[0][1:] != v[1][1:][::-1] for v in wall_edges.values()):
        raise ValueError('Full mirrored wall must be closed and consistently oriented')
    cuts = {}
    for edge, kind in zip(E, kinds):
        key = tuple(sorted(map(int, edge)))
        if kind not in ('trailing_edge', 'smooth_wall_contact'):
            raise ValueError(f'Unsupported cut kind {kind}')
        if key in cuts and cuts[key] != kind:
            raise ValueError('Conflicting cut declarations')
        cuts[key] = kind
    shared = set(wall_edges) & set(wake_edges)
    if shared != set(cuts):
        raise ValueError('Declared cuts must equal the actual conforming wall/wake edges')
    for key in cuts:
        if len(wake_edges[key]) != 1:
            raise ValueError(f'Wall contact {key} must be a wake boundary edge')
    # One connected local face fan per trace at a geometric vertex.
    incident = defaultdict(list)
    adjacent = defaultdict(list)
    for f, row in enumerate(F):
        for vertex in row:
            incident[int(vertex)].append(f)
    for (a, b), pair in wall_edges.items():
        if (a, b) not in cuts:
            f, g = pair[0][0], pair[1][0]
            adjacent[a].append((f, g))
            adjacent[b].append((f, g))
    used = sorted(incident)
    compact = {v: i for i, v in enumerate(used)}
    dof_face = np.empty_like(F)
    next_dof = len(used)
    for v in used:
        parent = {f: f for f in incident[v]}
        def root(f):
            while parent[f] != f:
                parent[f] = parent[parent[f]]
                f = parent[f]
            return f
        for f, g in adjacent[v]:
            rf, rg = root(f), root(g)
            if rf != rg:
                parent[max(rf, rg)] = min(rf, rg)
        fans = defaultdict(list)
        for f in incident[v]:
            fans[root(f)].append(f)
        if len(fans) > 2:
            raise ValueError(f'Branched cut at vertex {v}: more than two wall traces')
        for k, (_, members) in enumerate(sorted(fans.items())):
            dof = compact[v] if k == 0 else next_dof
            if k:
                next_dof += 1
            for f in members:
                dof_face[f, np.flatnonzero(F[f] == v)[0]] = dof
    pairs, te_nodes, contact_nodes = {}, set(), set()
    te_face_banks = defaultdict(lambda: (set(), set()))
    edge_records = []
    unsplit_tips = set()
    for key, kind in cuts.items():
        _, wa, wb = wake_edges[key][0]
        wall_pair = wall_edges[key]
        plus = [q[0] for q in wall_pair if q[1:] == (wb, wa)]
        minus = [q[0] for q in wall_pair if q[1:] == (wa, wb)]
        if len(plus) != 1 or len(minus) != 1:
            raise ValueError(f'Cannot orient cut banks at edge {key}')
        fp, fm = plus[0], minus[0]
        endpoints = []
        for v in key:
            dp = int(dof_face[fp, np.flatnonzero(F[fp] == v)[0]])
            dm = int(dof_face[fm, np.flatnonzero(F[fm] == v)[0]])
            endpoints.append((dp, dm))
            if dp == dm:
                if np.max(np.abs(H[v]), initial=0.) > 1e-12:
                    raise ValueError(f'Unsplit cut endpoint {v} has nonzero Gamma basis')
                unsplit_tips.add(v)
                continue
            if v in pairs and pairs[v] != (dp, dm):
                raise ValueError(f'Inconsistent oriented bank pairing at vertex {v}')
            pairs[v] = (dp, dm)
            (te_nodes if kind == 'trailing_edge' else contact_nodes).add(v)
            if kind == 'trailing_edge':
                te_face_banks[v][0].add(fp)
                te_face_banks[v][1].add(fm)
        edge_records.append(dict(vertices=list(key), kind=kind,
                                 plus_face=fp, minus_face=fm,
                                 endpoint_pairs=[list(p) for p in endpoints]))
    # At a TE/contact junction one algebraic jump row is enough. Do not add
    # the contact constraint twice and never add Kutta rows to smooth contact.
    contact_nodes -= te_nodes
    te = np.array(sorted(te_nodes, key=lambda v: (x[v, 1], v)), int)
    cn = np.array(sorted(contact_nodes), int)
    def pair_array(nodes):
        return np.asarray([pairs[int(v)] for v in nodes], int).reshape(-1, 2)
    audit = dict(wall_geometric_nodes=len(used), wall_potential_dofs=next_dof,
                 split_nodes=next_dof-len(used), te_jump_rows=len(te),
                 contact_jump_rows=len(cn), unsplit_zero_gamma_tips=sorted(unsplit_tips),
                 contact_edges=sum(k == 'smooth_wall_contact' for k in cuts.values()),
                 trailing_edges=sum(k == 'trailing_edge' for k in cuts.values()),
                 oriented_edge_records=edge_records,
                 contact_constraints_are_not_kutta=True)
    return dict(dof_face=dof_face, ndof_mu=next_dof, te_geo=te,
                te_pairs=pair_array(te),
                te_face_pairs=tuple((np.array(sorted(te_face_banks[v][0]), int),
                                     np.array(sorted(te_face_banks[v][1]), int)) for v in te),
                contact_geo=cn,
                contact_pairs=pair_array(cn), contact_gamma_hat=H[cn].copy(),
                trace_audit=audit)


def with_declared_contact_cuts(wing, data):
    """Upgrade a mirrored CAD Wing using its explicit cut_edges/cut_kind.

    The no-contact path returns the exact original object, keeping the three
    existing healthy CAD fixtures and generated wings unchanged.
    """
    kinds = np.asarray(data.get('cut_kind', []), str)
    if not np.any(kinds == 'smooth_wall_contact'):
        return wing
    half_x = np.asarray(data['coordinates'], float)
    edges = np.asarray(data.get('cut_edges', []), int)
    if edges.shape != (len(kinds), 2):
        raise ValueError('cut_edges and cut_kind do not match')
    tol = 1e-10*max(1., float(np.ptp(half_x, axis=0).max()))
    other = np.flatnonzero(half_x[:, 1] > tol)
    mirror = np.arange(len(half_x), dtype=int)
    mirror[other] = len(half_x)+np.arange(len(other))
    expected = np.vstack((half_x, half_x[other]*[1., -1., 1.]))
    if expected.shape != wing.xyz.shape or not np.allclose(expected, wing.xyz, atol=tol, rtol=0.):
        raise ValueError('Contact-cut adapter requires the unchanged mirrored geometry')
    out = split_wall_trace_fans(
        wing.xyz, wing.faces, wing.wake_faces,
        np.vstack((edges, mirror[edges])), np.tile(kinds, 2), wing.gamma_hat)
    te = out['te_geo']
    pairs = out['te_pairs']
    return replace(wing, dof_face=out['dof_face'], ndof_mu=out['ndof_mu'],
                   te_geo=te, te_up=pairs[:, 0], te_down=pairs[:, 1],
                   te_pairs=pairs, te_pair_y=wing.xyz[te, 1],
                   te_face_pairs=out['te_face_pairs'],
                   contact_geo=out['contact_geo'], contact_pairs=out['contact_pairs'],
                   contact_gamma_hat=out['contact_gamma_hat'], trace_audit=out['trace_audit'])


def contact_jump_report(mesh, mu, gamma, slices):
    """Report contact continuity separately from TE pressure or speed Kutta."""
    mu, gamma = np.asarray(mu, float), np.asarray(gamma, float)
    result = {}
    for group in mesh.groups:
        pairs = getattr(group, 'contact_pairs', None)
        if pairs is None or not len(pairs):
            continue
        pairs = np.asarray(pairs, int)
        residual = mu[pairs[:, 0]]-mu[pairs[:, 1]]-group.contact_gamma_hat@gamma[slices[group.name]]
        result[group.name] = dict(rows=len(pairs), rms=float(np.sqrt(np.mean(residual**2))),
                                 max_abs=float(np.max(np.abs(residual))),
                                 condition='wake_body_potential_jump_not_kutta')
    return result
