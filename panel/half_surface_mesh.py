"""Build a half-native panel fixture from an already meshed positive-y CAD fluid domain.

This NumPy-only layer classifies the real exterior/wake topology; it DOES NOT
clip triangles from an independently triangulated full aircraft. Gmsh must
provide conforming wall and wake edges. Only the half physical wall and wake
are exported. The reflecting left side never becomes physical panel data.
"""
from __future__ import annotations

import numpy as np


def half_contact_station_map(stations, intervals, *, tolerance):
    """Unlike the full-span map, root circulation is FREE, tip is zero.

    A fuselage contacting the wake at the root shares its root DOF with the
    whole contact interval. Otherwise the root is an ordinary independent
    circulation station. This respects mirror symmetry without a fake root TE.
    """
    y = np.asarray(stations, dtype=np.float64)
    contacts = np.asarray(intervals, dtype=np.float64).reshape(-1, 2)
    if (y.ndim != 1 or len(y) < 2 or abs(y[0]) > tolerance
            or np.any(np.diff(y) <= 0) or not np.isfinite(y).all()):
        raise ValueError("Half stations must start at y=0 and increase")
    groups = np.arange(len(y))
    for left, right in contacts:
        if (left < -tolerance or right >= y[-1]-tolerance
                or right <= left or not np.any(abs(y-left) <= tolerance)
                or not np.any(abs(y-right) <= tolerance)):
            raise ValueError("Contact intervals must terminate at sampled interior half stations")
        selected = np.flatnonzero((y >= left-tolerance) & (y <= right+tolerance))
        groups[selected] = groups[selected[0]]
    labels = np.full(len(y), -1, dtype=np.int64)
    _, labels[:-1] = np.unique(groups[:-1], return_inverse=True)
    return labels


def _orient_open_half_wall(x, faces, tol):
    """Orient right wall using edge propagation and positive mirrored volume.

    An open boundary is allowed only on y=0. A temporary flat cut-plane CAD
    cap would be a nonphysical wall and is explicitly rejected.
    """
    faces = np.array(faces, dtype=np.int64, copy=True)
    edges = {}
    for i, face in enumerate(faces):
        for a, b in zip(face, np.roll(face, -1)):
            a, b = int(a), int(b)
            key = tuple(sorted((a, b)))
            edges.setdefault(key, []).append((i, a < b))
    bad = [key for key, sides in edges.items() if len(sides) > 2]
    if bad:
        raise ValueError(f"Non-manifold right wall edge {bad[0]}")
    boundary = np.asarray([e for e, v in edges.items() if len(v) == 1], dtype=np.int64).reshape(-1, 2)
    if not len(boundary):
        raise ValueError("Expected an open symmetry-plane seam, not a capped half CAD solid")
    if np.any(abs(x[boundary, 1]) > tol):
        raise ValueError("Physical right wall has an open edge away from symmetry plane")
    # A true half-surface cut has closed seam loops; a detached slit or
    # broken Gmsh curve produces endpoints of degree != 2.
    boundary_degrees = np.bincount(boundary.ravel(), minlength=len(x))
    if np.any(boundary_degrees[boundary_degrees > 0] != 2):
        raise ValueError("Half symmetry seam has dangling or branching boundary vertices")
    if np.any(np.all(abs(x[faces, 1]) <= tol, axis=1)):
        raise ValueError("Artificial symmetry cap must not be included in physical wall")
    if np.any((x[faces, 1].min(axis=1) < -tol)):
        raise ValueError("Wall triangles cross into left half")
    neighbors = [[] for _ in faces]
    for entries in edges.values():
        if len(entries) == 2:
            (a, ai), (b, bi) = entries
            # Equal directed orientation on shared edge requires a flip.
            neighbors[a].append((b, ai == bi))
            neighbors[b].append((a, ai == bi))
    sign = np.zeros(len(faces), dtype=np.int8)
    components = []
    for root in range(len(faces)):
        if sign[root]:
            continue
        sign[root] = 1
        members, pending = [], [root]
        while pending:
            i = pending.pop()
            members.append(i)
            for j, same in neighbors[i]:
                expected = -sign[i] if same else sign[i]
                if sign[j] == 0:
                    sign[j] = expected
                    pending.append(j)
                elif sign[j] != expected:
                    raise ValueError("Inconsistent CAD wall winding across an edge")
        components.append(np.asarray(members, dtype=np.int64))
    faces[sign < 0] = faces[sign < 0][:, [0, 2, 1]]
    volumes = []
    for ids in components:
        tri = x[faces[ids]]
        contribution = np.einsum(
            "ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))
        # Reflection doubles volume; the y=0 cap contributes zero to
        # the divergence theorem's x.n integral.
        volume = 2 * contribution.sum()/6
        if abs(volume) <= 1e-13:
            raise ValueError("Mirrored closed physical wall has zero signed volume")
        if volume < 0:
            faces[ids] = faces[ids][:, [0, 2, 1]]
            volume = -volume
        volumes.append(float(volume))
    return faces, boundary, volumes


def build_half_surface(points, wall_faces, wall_tags, wake_faces, *,
                       te_polyline, stations, contact_intervals,
                       reference_area, reference_chord, reference_point,
                       freestream, density, wake_end_x):
    """Build dpanel.half_planar_cut.v1 without full-mesh remeshing.

    points, wall and wake triangles must come from ONE y>=0 OCC fragment
    and ONE conforming Gmsh surface mesh. No wall clipping, distance welding,
    or reconstruction of buried faces occurs here.
    """
    xyz = np.asarray(points, dtype=np.float64)
    wall = np.asarray(wall_faces, dtype=np.int64)
    tags = np.asarray(wall_tags, dtype=np.int64)
    wake = np.asarray(wake_faces, dtype=np.int64)
    te = np.asarray(te_polyline, dtype=np.float64)
    y = np.asarray(stations, dtype=np.float64)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or
            wall.ndim != 2 or wall.shape[1] != 3 or len(wall) == 0 or
            wake.ndim != 2 or wake.shape[1] != 3 or len(wake) == 0 or
            len(tags) != len(wall) or
            te.ndim != 2 or te.shape[1] != 3 or len(te) < 2):
        raise ValueError("Invalid half CAD wall/wake/TE mesh shape")
    if (not np.isfinite(xyz).all() or not np.isfinite(te).all() or
            np.any(np.diff(te[:, 1]) <= 0) or
            not np.allclose(te[:, 2], te[0, 2], rtol=0, atol=1e-12)):
        raise ValueError("Nonfinite or non-planar declared TE")
    if (min(wall.min(), wake.min()) < 0 or
            max(wall.max(), wake.max()) >= len(xyz)):
        raise ValueError("Half mesh indices out of bounds")
    # Only coalesce identical Gmsh NODE TAGs via the caller's indexed mesh.
    # Never weld two different nodes merely because they are geometrically near.
    used = np.unique(np.r_[wall.ravel(), wake.ravel()])
    remap = np.full(len(xyz), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    xyz = xyz[used].copy()
    wall, wake = remap[wall], remap[wake]
    scale = max(float(np.ptp(xyz, axis=0).max()), float(reference_chord), 1.)
    tol = 1e-8*scale
    # Mark the exact OCC half-plane as topology: only epsilon corrections
    # below CAD tolerance are allowed, not arbitrary mesh clipping.
    if np.any(xyz[:, 1] < -tol):
        raise ValueError("OCC half domain contains negative-y nodes")
    ynear = abs(xyz[:, 1]) <= tol
    plane_correction = float(np.max(abs(xyz[ynear, 1]))) if np.any(ynear) else 0.
    xyz[ynear, 1] = 0.
    if not np.any(ynear):
        raise ValueError("Half CAD wall has no symmetry seam nodes")
    wall, seam, volumes = _orient_open_half_wall(xyz, wall, tol)
    wall_xyz = xyz[wall]
    areas_twice = np.linalg.norm(np.cross(
        wall_xyz[:, 1]-wall_xyz[:, 0],
        wall_xyz[:, 2]-wall_xyz[:, 0]), axis=1)
    if np.any(areas_twice <= 0):
        raise ValueError("Degenerate wall triangle after exact symmetry-plane snapping")
    wake_tri = xyz[wake]
    av = np.cross(wake_tri[:, 1]-wake_tri[:, 0],
                  wake_tri[:, 2]-wake_tri[:, 0])
    invert = av[:, 2] < 0.
    wake[invert] = wake[invert][:, [0, 2, 1]]
    wake_tri = xyz[wake]
    av = np.cross(wake_tri[:, 1]-wake_tri[:, 0],
                  wake_tri[:, 2]-wake_tri[:, 0])
    if (np.any(av[:, 2] <= 0) or np.ptp(wake_tri[:, :, 2]) > tol
            or np.max(abs(av[:, :2])) > 1e-9*scale*scale
            or abs(float(wake_tri[0, 0, 2] - te[0, 2])) > tol):
        raise ValueError("OCC half wake must be planar and +z oriented")

    wall_edges, wake_edges = {}, {}
    for i, face in enumerate(wall):
        for a, b in zip(face, np.roll(face, -1)):
            wall_edges.setdefault(tuple(sorted((int(a), int(b)))), []).append(i)
    for i, face in enumerate(wake):
        for a, b in zip(face, np.roll(face, -1)):
            wake_edges.setdefault(tuple(sorted((int(a), int(b)))), []).append(i)
    if any(len(v) > 2 for v in wake_edges.values()):
        raise ValueError("Non-manifold wake edge")
    boundary_wake = [key for key, inc in wake_edges.items() if len(inc) == 1]
    if not boundary_wake:
        raise ValueError("No physical wake boundary in half CAD mesh")

    # The far wake boundary is extended using the same two triangles per edge
    # as the original build_surface.py. Contact/cut edges are read from the
    # *conforming shared node IDs*, not matched by approximate coordinates.
    far_x = float(wake_tri[:, :, 0].max())
    if wake_end_x <= far_x+tol:
        raise ValueError("Wake extension must end farther downstream")
    outflow = [edge for edge in boundary_wake
               if np.all(abs(xyz[np.asarray(edge), 0]-far_x) <= tol)]
    if not outflow:
        raise ValueError("No downstream wake boundary edge")
    far_nodes = sorted(set(int(v) for edge in outflow for v in edge))
    mapping = {}
    additions = []
    for node in far_nodes:
        mapping[node] = len(xyz)+len(additions)
        p = xyz[node].copy()
        p[0] = float(wake_end_x)
        additions.append(p)
    extension = []
    for a, b in outflow:
        if xyz[a, 1] > xyz[b, 1]:
            a, b = b, a
        extension.extend(((a, mapping[a], mapping[b]),
                          (a, mapping[b], b)))
    xyz = np.vstack((xyz, additions))
    wake = np.vstack((wake, np.asarray(extension, dtype=np.int64)))
    new_av = np.cross(xyz[wake][:, 1]-xyz[wake][:, 0],
                      xyz[wake][:, 2]-xyz[wake][:, 0])
    if np.any(new_av[:, 2] <= 0):
        raise ValueError("Extended half wake triangles must remain +z oriented")

    cuts, cut_kind, te_edges, upper, lower = [], [], [], [], []
    for edge in boundary_wake:
        adjacent = wall_edges.get(edge)
        if adjacent is None:
            continue
        if len(adjacent) != 2:
            raise ValueError("Physical wake/wall edge must have exactly two wall banks")
        v = xyz[np.asarray(edge)]
        z = xyz[wall[adjacent]].mean(axis=1)[:, 2] - te[0, 2]
        if not (z.min() < 0 and z.max() > 0):
            raise ValueError("Wake/wall cut must have distinct upper and lower banks")
        edge_order = tuple(sorted(edge, key=lambda index: xyz[index, 1]))
        diff = abs(v[:, 0] - np.interp(v[:, 1], te[:, 1], te[:, 0]))
        on_te = bool(np.max(diff) <= tol and
                     np.all(abs(v[:, 2]-te[0, 2]) <= tol))
        cuts.append(list(edge_order))
        cut_kind.append("trailing_edge" if on_te else "smooth_wall_contact")
        if on_te:
            te_edges.append(list(edge_order))
            upper.append(adjacent[int(np.argmax(z))])
            lower.append(adjacent[int(np.argmin(z))])
    if not te_edges:
        raise ValueError("No exposed physical TE edge shared with the half wake")
    if len(set(map(tuple, cuts))) != len(cuts):
        raise ValueError("Duplicate wake/wall contact cut edges")

    if (len(y) < 2 or abs(y[0]) > tol or np.any(np.diff(y) <= 0)
            or abs(y[-1]-te[-1, 1]) > tol):
        raise ValueError("Half circulation stations must span y=0 to the declared tip")
    dof = half_contact_station_map(y, contact_intervals, tolerance=tol)
    centers = xyz[wake].mean(axis=1)
    wi = np.searchsorted(y, centers[:, 1], side="right") - 1
    if np.any(wi < 0) or np.any(wi >= len(y)-1):
        raise ValueError("Wake leaves the circulation stations")
    if (np.any(xyz[wake][:, :, 1].min(axis=1) < y[wi]-tol) or
            np.any(xyz[wake][:, :, 1].max(axis=1) > y[wi+1]+tol)):
        raise ValueError("Wake triangles cross a circulation strip")
    for e in te_edges:
        ends = xyz[e]
        if np.any(np.diff(ends[:, 1]) <= 0):
            raise ValueError("Half TE edges must run outboard")
    # Ensure no artificial symmetry plane cap is retained; mirror image
    # closure and real edge incidence are checked by prepare_half_topology.
    ref = np.asarray(reference_point, dtype=float)
    flow = np.asarray(freestream, dtype=float)
    if (ref.shape != (3,) or flow.shape != (3,)
            or abs(ref[1]) > tol or abs(flow[1]) > 1e-12*np.linalg.norm(flow)):
        raise ValueError("Only symmetric reference and zero sideslip supported")
    if (min(reference_area, reference_chord, density) <= 0
            or not np.isfinite([reference_area, reference_chord, density]).all()):
        raise ValueError("Positive SI reference quantities required")

    return dict(
        schema="dpanel.half_planar_cut.v1",
        coordinates=xyz.tolist(), faces=wall.tolist(),
        wake_faces=wake.tolist(), component_ids=tags.tolist(),
        trailing_edges=te_edges, upper_te_faces=upper, lower_te_faces=lower,
        cut_edges=cuts, cut_kind=cut_kind,
        gamma_stations=y.tolist(), gamma_station_dof=dof.tolist(),
        reference_area=float(reference_area), reference_chord=float(reference_chord),
        moment_reference=ref.tolist(), freestream=flow.tolist(),
        density=float(density),
        audit=dict(
            symmetry_plane_vertex_count=int(np.sum(xyz[:, 1] == 0)),
            symmetry_seam_edges=int(len(seam)),
            max_plane_snap=float(plane_correction),
            half_wall_panels=int(len(wall)), half_wake_panels=int(len(wake)),
            public_wall_positive_mirrored_volumes=volumes,
            exposed_te_edges=int(len(te_edges)),
            wake_wall_contact_edges=len(cuts)-len(te_edges),
            wall_cut_kinds=cut_kind,
            no_artificial_symmetry_wall=True,
            physical_accuracy_accepted=False,
        ),
    )
