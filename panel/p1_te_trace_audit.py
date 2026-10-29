"""Read-only continuous-P1 TE potential-trace topology audit (Issue #71).

An intentional TE cut permits different potentials at the SAME geometric
vertex on opposite banks. Two thick-wing chordwise vertices on the upper and
lower surface occupy different 3-D locations and must never be merged.
This audit neither alters an aerodynamic solve nor certifies Kutta physics.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np


def _basis(y, knots):
    y, knots = np.asarray(y, float).reshape(-1), np.asarray(knots, float).reshape(-1)
    if len(knots) < 3 or np.any(np.diff(knots) <= 0):
        raise ValueError("gamma stations must be strictly increasing")
    if np.any(y < knots[0]-1e-10) or np.any(y > knots[-1]+1e-10):
        raise ValueError("TE station outside gamma interpolation range")
    H = np.zeros((len(y), len(knots)))
    for k in range(len(knots)-1):
        hits = (y >= knots[k]-1e-12) & (y <= knots[k+1]+1e-12)
        t = np.clip((y[hits]-knots[k])/(knots[k+1]-knots[k]), 0., 1.)
        H[hits, k] = 1.-t
        H[hits, k+1] = t
    return H[:, 1:-1]


def audit_te_trace(wing, *, coordinate_tol=None):
    """Check G1 Wing TE DOF/geometry/wake consistency without changing data."""
    X = np.asarray(wing.xyz, float)
    F = np.asarray(wing.faces, int)
    D = np.asarray(wing.dof_face, int)
    bank = np.asarray(wing.face_bank, int)
    pairs = np.asarray(wing.te_pairs, int).reshape(-1, 2)
    yte = np.asarray(wing.te_pair_y, float).reshape(-1)
    te_geo = set(map(int, np.asarray(wing.te_geo, int).reshape(-1)))
    wake_X = np.asarray(wing.wake_xyz, float)
    wake_F = np.asarray(wing.wake_faces, int)
    hat = np.asarray(wing.gamma_hat, float)
    knots = np.asarray(wing.gamma_y, float)
    errors, warnings = [], []
    def err(code, detail):
        errors.append(dict(code=code, detail=str(detail)))
    if (X.ndim != 2 or X.shape[1] != 3 or F.ndim != 2 or F.shape[1] != 3
            or F.shape != D.shape or bank.shape != (len(F),)
            or np.any(F < 0) or np.any(F >= len(X))
            or np.any(D < 0) or np.any(D >= int(wing.ndof_mu))
            or not np.isfinite(X).all() or not np.isin(bank, (-1, 0, 1)).all()
            or len(pairs) != len(yte)):
        raise ValueError("incompatible wall geometry/DOF/TE")
    if (wake_X.ndim != 2 or wake_X.shape[1] != 3 or wake_F.ndim != 2
            or wake_F.shape[1] != 3 or np.any(wake_F < 0)
            or np.any(wake_F >= len(wake_X))
            or hat.shape != (len(wake_X), len(knots)-2)
            or not np.isfinite(wake_X).all() or not np.isfinite(hat).all()):
        raise ValueError("incompatible wake geometry/material Gamma basis")
    if np.any(pairs < 0) or np.any(pairs >= int(wing.ndof_mu)):
        raise ValueError("TE pair references invalid DOF")
    atol = (1e-9*float(getattr(wing, "chord", 1.))
            if coordinate_tol is None else float(coordinate_tol))
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError("invalid coordinate tolerance")
    dof_geom, geom_dof, dof_bank = defaultdict(set), defaultdict(set), defaultdict(set)
    incidence = defaultdict(list)
    for face_idx, (geo, df) in enumerate(zip(F, D)):
        for v, m in zip(geo, df):
            dof_geom[int(m)].add(int(v))
            geom_dof[int(v)].add(int(m))
            dof_bank[int(m)].add(int(bank[face_idx]))
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = int(geo[i]), int(geo[j])
            m, n = int(df[i]), int(df[j])
            incidence[tuple(sorted((a, b)))].append((face_idx, {a:m, b:n}))
    for m, vs in dof_geom.items():
        if len(vs) > 1:
            err("dof_merges_distinct_geometry", f"dof={m}, physical_vertices={sorted(vs)}")
    unreferenced = sorted(set(range(int(wing.ndof_mu)))-set(dof_geom))
    if unreferenced:
        err("unused_potential_dof", unreferenced[:20])
    declared = {}
    for k, (upper, lower) in enumerate(pairs):
        upper, lower = int(upper), int(lower)
        if upper == lower:
            err("unsplit_te_pair", f"row={k}, dof={upper}")
            continue
        gu, gl = dof_geom.get(upper, set()), dof_geom.get(lower, set())
        if len(gu) != 1 or gu != gl:
            err("te_pair_not_same_geometry",
                f"row={k}, up_geo={sorted(gu)}, down_geo={sorted(gl)}")
            continue
        v = next(iter(gu))
        if v not in te_geo:
            err("te_pair_not_declared_te_geometry", f"row={k}, vertex={v}")
        if v in declared and declared[v] != (upper, lower):
            err("multiple_te_pair_for_vertex", f"vertex={v}")
        declared[v] = (upper, lower)
        if 1 not in dof_bank[upper] or -1 not in dof_bank[lower]:
            err("missing_upper_or_lower_bank_support",
                f"row={k}, up={sorted(dof_bank[upper])}, low={sorted(dof_bank[lower])}")
        if -1 in dof_bank[upper] or 1 in dof_bank[lower]:
            err("te_split_crosses_bank", f"row={k}")
        if abs(X[v,1]-yte[k]) > atol:
            err("te_station_coordinate_mismatch", f"row={k}, y={X[v,1]}/{yte[k]}")
    te_declared_count = len(declared)
    contact_pairs = getattr(wing, "contact_pairs", None)
    contact_vertices = np.asarray(getattr(wing, "contact_geo", None)
                                  if getattr(wing, "contact_geo", None) is not None else [], int)
    contact_edges = set()
    if getattr(wing, "trace_audit", None):
        contact_edges = {tuple(sorted(e["vertices"])) for e in
                         wing.trace_audit.get("oriented_edge_records", [])
                         if e["kind"] == "smooth_wall_contact"}
    if contact_pairs is not None:
        contact_pairs = np.asarray(contact_pairs, int).reshape(-1,2)
        if len(contact_pairs) != len(contact_vertices):
            err("contact_pair_geometry_length_mismatch", len(contact_pairs))
        else:
            for v, (upper, lower) in zip(contact_vertices, contact_pairs):
                gu, gl = dof_geom.get(int(upper), set()), dof_geom.get(int(lower), set())
                if upper == lower or gu != {int(v)} or gl != {int(v)}:
                    err("contact_pair_not_split_same_geometry", f"vertex={v}")
                    continue
                declared[int(v)] = (int(upper), int(lower))
    for v, ms in geom_dof.items():
        if len(ms)>1 and (v not in declared or ms!=set(declared[v])):
            err("unexpected_potential_cut_vertex", f"vertex={v}, dofs={sorted(ms)}")
        if v in declared and ms!=set(declared[v]):
            err("declared_te_pair_unsupported", f"vertex={v}")
    seam = []
    for key, faces in incidence.items():
        if len(faces)>2:
            err("nonmanifold_wall_edge", f"edge={key}, incidence={len(faces)}")
        elif len(faces)==2:
            (fa, da), (fb, db) = faces
            if da != db:
                if key not in contact_edges and sorted((int(bank[fa]), int(bank[fb]))) != [-1, 1]:
                    err("cut_across_non_te_bank_edge", f"edge={key}, faces={fa},{fb}")
                if not any(v in declared for v in key):
                    err("seam_disconnected_from_declared_te", key)
                seam.append(key)
    degree = defaultdict(int)
    for a,b in seam:
        degree[a]+=1
        degree[b]+=1
    if any(v>2 for v in degree.values()):
        err("te_seam_branches", {k:v for k,v in degree.items() if v>2})
    for v in declared:
        if degree[v] == 0:
            err("declared_te_pair_has_no_seam_edge", f"vertex={v}")
    # A nodal jump can be satisfied on a TE segment for ALL possible Gamma
    # only if the wake hat basis has no unmatched interior knot on that edge.
    for a,b in seam:
        ya,yb = sorted((float(X[a,1]),float(X[b,1])))
        mid = knots[(knots > ya+atol) & (knots < yb-atol)]
        if len(mid):
            err("te_edge_crosses_gamma_knot",
                f"edge={(a,b)}, interior_gamma_knots={mid.tolist()}")
    seam_nodes = sorted({x for e in seam for x in e})
    wake_to_wall = {}
    for idx in np.unique(wake_F):
        if seam_nodes:
            distances = np.linalg.norm(X[seam_nodes]-wake_X[int(idx)], axis=1)
            hits = np.flatnonzero(distances <= atol)
            if len(hits)==1:
                wake_to_wall[int(idx)] = seam_nodes[int(hits[0])]
            elif len(hits)>1:
                err("ambiguous_wake_root_match", f"wake_node={idx}")
    wake_seam = set()
    for f in wake_F:
        for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):
            if int(a) in wake_to_wall and int(b) in wake_to_wall:
                wake_seam.add(tuple(sorted((wake_to_wall[int(a)],wake_to_wall[int(b)]))))
    for edge in seam:
        if edge not in wake_seam:
            err("te_cut_edge_missing_from_wake_root", edge)
    try:
        H = _basis(yte, knots)
        for row,(upper,lower) in enumerate(pairs):
            vertex_set = dof_geom.get(int(upper), set())
            if len(vertex_set) != 1:
                continue
            v = next(iter(vertex_set))
            for node, gv in wake_to_wall.items():
                if gv==v and np.max(np.abs(hat[node]-H[row])) > 1e-9:
                    err("wake_root_gamma_hat_incompatible_with_te_jump",
                        f"wake_node={node}, te_row={row}")
        for node, v in wake_to_wall.items():
            if abs(X[v,1]-knots[0]) <= atol or abs(X[v,1]-knots[-1]) <= atol:
                if np.max(np.abs(hat[node]))>1e-9:
                    err("nonzero_tip_wake_gamma_hat", f"wake_node={node}")
    except ValueError as exc:
        err("invalid_gamma_stations", exc)
    if len(pairs) != te_declared_count:
        warnings.append("TE pair rows not one-to-one with geometry")
    return dict(ok=not errors, errors=errors, warnings=warnings,
                wall_faces=int(len(F)), wall_geometric_vertices=int(len(geom_dof)),
                wall_potential_dofs=int(len(dof_geom)),
                te_pair_rows=int(len(pairs)), split_te_vertices=int(te_declared_count),
                split_contact_vertices=int(len(contact_vertices)),
                seam_edges=int(len(seam)),
                wake_attached_seam_edges=int(len(set(seam)&wake_seam)),
                seam_geometry_edges=[list(e) for e in seam],
                te_jump_algebraic_only=True, pressure_kutta_verified=False,
                physical_accuracy_accepted=False)


def audit_declared_cad_te(data):
    """Audit SOURCE half-surface TE pairing before G1 drops source metadata.

    The adapter currently zip()s face arrays and accepts any nonempty shared
    vertex intersection. A sharp TE pair must share an actual declared edge.
    """
    f = np.asarray(data["faces"], int)
    up = np.asarray(data["upper_te_faces"], int).reshape(-1)
    lo = np.asarray(data["lower_te_faces"], int).reshape(-1)
    edges = np.asarray(data["trailing_edges"], int)
    errors = []
    def err(code, detail):
        errors.append(dict(code=code, detail=str(detail)))
    if f.ndim != 2 or f.shape[1] != 3 or edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("invalid CAD faces or TE edge array")
    if len(up)!=len(lo):
        err("cad_te_face_pairs_length_mismatch", f"upper={len(up)}, lower={len(lo)}")
    if np.any(up<0) or np.any(lo<0) or np.any(up>=len(f)) or np.any(lo>=len(f)):
        raise ValueError("CAD TE face index out of range")
    edge_set = {tuple(sorted(map(int,e))) for e in edges}
    found = []
    for k in range(min(len(up),len(lo))):
        U,L = f[up[k]],f[lo[k]]
        common = np.intersect1d(U,L)
        if len(common)!=2:
            err("cad_te_pair_not_shared_edge", f"row={k}, shared={common.tolist()}")
            continue
        key = tuple(sorted(map(int,common)))
        if key not in edge_set:
            err("cad_te_pair_not_declared_edge", f"row={k}, edge={key}")
        def oriented(row):
            for a,b in ((row[0],row[1]),(row[1],row[2]),(row[2],row[0])):
                if tuple(sorted(map(int,(a,b))))==key:
                    return int(a),int(b)
            raise AssertionError("shared endpoints must form an edge")
        if oriented(U)!=oriented(L)[::-1]:
            err("cad_te_edge_winding_mismatch", f"row={k}, edge={key}")
        found.append(key)
    if len(found)!=len(set(found)):
        err("cad_te_edge_repeated_in_face_pairing",
            f"found={len(found)}, unique={len(set(found))}")
    return dict(ok=not errors, errors=errors, source_face_pairs=min(len(up),len(lo)),
                valid_shared_edges=len(found), declared_edge_count=len(edge_set),
                does_not_modify_input=True, physical_accuracy_accepted=False)
