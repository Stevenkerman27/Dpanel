"""Group-aware half-native panel fixture from one positive-y OCC fluid mesh.

This is the v2 counterpart of half_surface_mesh.build_half_surface. It accepts
SEVERAL horizontal wake sheets, each with its own plane z and its own trailing
edge polyline, and returns both the surface record and the dpanel v2 shed-group
declaration. No wall clipping, no welding of distinct nodes, no invented TE.
"""
from __future__ import annotations

import numpy as np

from .half_surface_mesh import half_contact_station_map
from .half_surface_mesh import _orient_open_half_wall
from .wake_group_declaration import merged_stations


def _wall_edge_map(wall):
    edges = {}
    for i, face in enumerate(wall):
        for a, b in zip(face, np.roll(face, -1)):
            edges.setdefault(tuple(sorted((int(a), int(b)))), []).append(int(i))
    return edges


def _wake_edge_map(wake):
    edges = {}
    for i, face in enumerate(wake):
        for a, b in zip(face, np.roll(face, -1)):
            edges.setdefault(tuple(sorted((int(a), int(b)))), []).append(int(i))
    return edges


def build_half_groups_surface(points, wall_faces, wall_tags, groups, *,
                              reference_area, reference_chord, reference_point,
                              freestream, density, wake_length, wake_strips,
                              contact_intervals=None):
    """Build the v2 group surface plus its shed-group declaration.

    groups must be a list of dicts with keys group_id, wake_faces, te, z,
    and optionally role, kind, root_condition, buried_inside_component,
    source_component_ids. te is the FULL-span declared TE polyline in physical
    units; the positive half is extracted here.
    """
    xyz = np.asarray(points, dtype=np.float64)
    wall = np.asarray(wall_faces, dtype=np.int64)
    tags = np.asarray(wall_tags, dtype=np.int64)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or
            wall.ndim != 2 or wall.shape[1] != 3 or len(wall) == 0 or
            len(tags) != len(wall)):
        raise ValueError("Invalid half CAD wall mesh shape")
    if not groups:
        raise ValueError("A group surface needs at least one declared shed group")
    if not np.isfinite(xyz).all():
        raise ValueError("Nonfinite CAD coordinates")
    used = np.unique(np.r_[wall.ravel(), np.concatenate(
        [np.asarray(g["wake_faces"], dtype=np.int64).ravel() for g in groups])])
    remap = np.full(len(xyz), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    xyz = xyz[used].copy()
    wall = remap[wall]
    scale = max(float(np.ptp(xyz, axis=0).max()), float(reference_chord), 1.)
    tol = 1e-8 * scale
    if np.any(xyz[:, 1] < -tol):
        raise ValueError("OCC half domain contains negative-y nodes")
    ynear = abs(xyz[:, 1]) <= tol
    plane_correction = float(np.max(abs(xyz[ynear, 1]))) if np.any(ynear) else 0.
    xyz[ynear, 1] = 0.
    if not np.any(ynear):
        raise ValueError("Half CAD wall has no symmetry seam nodes")
    wall, seam, volumes = _orient_open_half_wall(xyz, wall, tol)
    wall_edges = _wall_edge_map(wall)
    contacts = np.zeros((0, 2)) if contact_intervals is None else np.asarray(
        contact_intervals, dtype=float).reshape(-1, 2)

    cuts, cut_kind, cut_group = [], [], []
    declared = []
    group_wakes = []
    for group in groups:
        gid = str(group["group_id"])
        z_plane = float(group["z"])
        te = np.asarray(group["te"], dtype=np.float64)
        if (te.ndim != 2 or te.shape[1] != 3 or len(te) < 2
                or not np.isfinite(te).all() or np.any(np.diff(te[:, 1]) <= 0)
                or not np.allclose(te[:, 2], z_plane, rtol=0, atol=tol)):
            raise ValueError(f"Nonfinite or non-planar declared TE for group {gid}")
        wake = remap[np.asarray(group["wake_faces"], dtype=np.int64)]
        if not np.any(wake >= 0):
            raise ValueError(f"Group {gid} declares no wake triangle")
        wake = wake.reshape(-1, 3)
        if any(len(v) > 2 for v in _wake_edge_map(wake).values()):
            raise ValueError(f"Non-manifold wake edge in group {gid}")
        tri = xyz[wake]
        av = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        invert = av[:, 2] < 0.
        wake[invert] = wake[invert][:, [0, 2, 1]]
        tri = xyz[wake]
        av = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        if (np.any(av[:, 2] <= 0) or np.ptp(tri[:, :, 2]) > tol
                or np.max(abs(av[:, :2])) > 1e-9 * scale * scale
                or abs(float(tri[0, 0, 2]) - z_plane) > tol):
            raise ValueError(f"Group {gid} wake must be planar at its declared z and +z oriented")
        # Downstream extension, exactly like the v1 builder: two triangles per
        # boundary edge on the current downstream edge of this group wake.
        far_x = float(tri[:, :, 0].max())
        wake_end_x = far_x + float(wake_length)
        boundary = [e for e, inc in _wake_edge_map(wake).items() if len(inc) == 1]
        outflow = [edge for edge in boundary
                   if np.all(abs(xyz[np.asarray(edge), 0] - far_x) <= tol)]
        if not outflow:
            raise ValueError(f"No downstream wake boundary edge in group {gid}")
        far_nodes = sorted({int(v) for edge in outflow for v in edge})
        mapping = {node: len(xyz) + k for k, node in enumerate(far_nodes)}
        additions = []
        for node in far_nodes:
            p = xyz[node].copy()
            p[0] = wake_end_x
            additions.append(p)
        extension = []
        for a, b in outflow:
            if xyz[a, 1] > xyz[b, 1]:
                a, b = b, a
            extension.extend(((a, mapping[a], mapping[b]),
                              (a, mapping[b], b)))
        xyz = np.vstack((xyz, np.asarray(additions, dtype=np.float64)))
        wake = np.vstack((wake, np.asarray(extension, dtype=np.int64)))
        new_av = np.cross(xyz[wake][:, 1] - xyz[wake][:, 0],
                          xyz[wake][:, 2] - xyz[wake][:, 0])
        if np.any(new_av[:, 2] <= 0):
            raise ValueError(f"Extended wake triangles must stay +z oriented in {gid}")
        te_positive = te[te[:, 1] >= -tol]
        stations = merged_stations(te_positive[:, 1], wake_strips)
        if abs(stations[0]) > tol:
            raise ValueError(f"Group {gid} stations must start at y=0")
        dof = half_contact_station_map(stations, contacts, tolerance=tol)
        te_edges, upper, lower = [], [], []
        for edge, incident in _wake_edge_map(wake).items():
            if len(incident) != 1:
                continue
            banks = wall_edges.get(edge)
            if banks is None:
                continue
            if len(banks) != 2:
                raise ValueError("Physical wake/wall edge must have exactly two wall banks")
            v = xyz[np.asarray(edge)]
            bank_z = xyz[wall[banks]].mean(axis=1)[:, 2] - z_plane
            if not (bank_z.min() < 0 and bank_z.max() > 0):
                raise ValueError(f"Wake/wall cut must have distinct upper and lower banks in {gid}")
            edge_order = tuple(sorted(edge, key=lambda index: xyz[index, 1]))
            diff = abs(v[:, 0] - np.interp(v[:, 1], te[:, 1], te[:, 0]))
            on_te = bool(np.max(diff) <= tol)
            if not on_te:
                continue
            cuts.append(list(edge_order))
            cut_kind.append("trailing_edge")
            cut_group.append(gid)
            te_edges.append(list(edge_order))
            upper.append(int(banks[int(np.argmax(bank_z))]))
            lower.append(int(banks[int(np.argmin(bank_z))]))
        if not te_edges:
            raise ValueError(f"No exposed physical TE edge shared with the wake of {gid}")
        centers = xyz[wake].mean(axis=1)
        wi = np.searchsorted(stations, centers[:, 1], side="right") - 1
        if np.any(wi < 0) or np.any(wi >= len(stations) - 1):
            raise ValueError(f"Wake leaves the circulation stations of {gid}")
        if (np.any(xyz[wake][:, :, 1].min(axis=1) < stations[wi] - tol) or
                np.any(xyz[wake][:, :, 1].max(axis=1) > stations[wi + 1] + tol)):
            raise ValueError(f"Wake triangles cross a circulation strip of {gid}")
        declared.append(dict(
            group_id=gid,
            kind=str(group.get("kind", "attached_lifting_TE")),
            role=group.get("role"),
            root_condition=str(group.get("root_condition", "symmetry_plane")),
            buried_inside_component=group.get("buried_inside_component"),
            source_component_ids=[int(v) for v in group.get("source_component_ids", ())],
            wake_z=float(z_plane),
            wall_faces=None,
            wake_faces=None,
            shed_edges=te_edges,
            upper_te=upper,
            lower_te=lower,
            stations=[float(v) for v in stations],
            station_dof=[int(v) for v in dof],
        ))
        group_wakes.append(wake)
    if len({tuple(item) for item in cuts}) != len(cuts):
        raise ValueError("Duplicate wake/wall contact cut edges across groups")
    ref = np.asarray(reference_point, dtype=float)
    flow = np.asarray(freestream, dtype=float)
    if (ref.shape != (3,) or flow.shape != (3,)
            or abs(ref[1]) > tol or abs(flow[1]) > 1e-12 * np.linalg.norm(flow)):
        raise ValueError("Only symmetric reference and zero sideslip supported")
    if (min(reference_area, reference_chord, density) <= 0
            or not np.isfinite([reference_area, reference_chord, density]).all()):
        raise ValueError("Positive SI reference quantities required")
    start = 0
    for item, group_wake in zip(declared, group_wakes):
        stop = start + len(group_wake)
        # The v2 contract indexes wake triangles, not their nodes: each group
        # owns a contiguous slice of the concatenated surface wake array.
        item["wake_faces"] = list(range(start, stop))
        start = stop
    wake_faces = np.vstack([np.asarray(w, dtype=np.int64) for w in group_wakes])
    surface = dict(
        schema="dpanel.half_shed_groups_surface.v2",
        coordinates=xyz.tolist(), faces=wall.tolist(),
        component_ids=tags.tolist(),
        wake_faces=wake_faces.tolist(),
        cut_edges=cuts, cut_kind=cut_kind, cut_group=cut_group,
        reference_area=float(reference_area), reference_chord=float(reference_chord),
        moment_reference=ref.tolist(), freestream=flow.tolist(),
        density=float(density),
        wake_groups=declared,
        audit=dict(
            symmetry_plane_vertex_count=int(np.sum(xyz[:, 1] == 0)),
            symmetry_seam_edges=int(len(seam)),
            max_plane_snap=float(plane_correction),
            half_wall_panels=int(len(wall)), half_wake_panels=int(len(wake_faces)),
            public_wall_positive_mirrored_volumes=volumes,
            groups=len(declared),
            exposed_te_edges=int(sum(len(item["shed_edges"]) for item in declared)),
            wake_wall_contact_edges=len(cuts) - int(sum(len(item["shed_edges"]) for item in declared)),
            no_artificial_symmetry_wall=True,
            physical_accuracy_accepted=False,
        ),
    )
    spec = dict(schema="dpanel.half_shed_groups.v2",
                source=dict(reference_area=float(reference_area),
                            reference_chord=float(reference_chord),
                            reference_point=ref.tolist()),
                groups=declared)
    return surface, spec