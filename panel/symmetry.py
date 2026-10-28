"""JAX-free symmetry diagnostics for `dpanel.planar_cut.v1` surfaces.

The half-model reduction R A E needs an *exactly* mirror-invariant discrete
problem: node pairing, face connectivity, wake connectivity, trailing-edge
bank pairing and circulation stations must all map onto themselves under
y -> -y. Independent triangulation of a symmetric CAD body satisfies the
node layer at best; it does not satisfy the face or TE-bank layers. This
module reports each layer separately so a "half model blocked" verdict can
be attributed instead of guessed.

No JAX, no physics, no CAD: pure NumPy over the saved surface record.
"""
import numpy as np


def scale_of(points):
    points = np.asarray(points, dtype=np.float64)
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def mirror_map(points, atol):
    """Nearest mirror partner of every point under y -> -y.

    Returns (pair, residual). A *strict* map needs residual <= atol on every
    row, a bijection and an involution; each is reported separately because
    they fail for different mesh defects.
    """
    points = np.asarray(points, dtype=np.float64)
    target = points * np.array([1.0, -1.0, 1.0])
    distance = np.linalg.norm(points[None, :, :] - target[:, None, :], axis=2)
    pair = np.argmin(distance, axis=1)
    residual = distance[np.arange(len(points)), pair]
    return pair, residual


def _map_stats(pair, residual, atol):
    index = np.arange(len(pair))
    unpaired = int(np.sum(residual > atol))
    bijective = len(set(pair.tolist())) == len(pair)
    involution = bool(np.all(pair[pair] == index))
    fixed = int(np.sum(pair == index))
    return dict(
        count=int(len(pair)),
        paired=int(len(pair) - unpaired),
        unpaired=unpaired,
        bijective=bool(bijective),
        involution=involution,
        fixed=fixed,
        median_residual=float(np.median(residual)),
        max_residual=float(residual.max()) if len(residual) else 0.0,
        tolerance=atol,
    )


def face_map(faces, node_pair, coordinates=None, atol=None):
    """Mirror each face through the node map and verify the geometry.

    Set membership alone is not a mirror test. If the vertex map is not a
    bijection (independent triangulation of a symmetric body gives a nearest
    map in which several vertices share a partner), applying it to a triangle
    can land on an unrelated but *existing* face: on `cases/panel_cut/
single_wing.json` this yields 376/484 "hits" of which 0 are geometrically
mirrored. The geometric check is therefore mandatory; it is the same mistake
shape as testing a mirrored upper-TE face against a set of lower-TE faces.

    Returns (dict) with face_pair (-1 where no verified partner), the winding
    verdict (1 reversed = correct for a reflection, 2 preserved, 3 rotated),
    the raw set-membership hit count and the number of those hits that survive
    the geometric check.
    """
    faces = np.asarray(faces, dtype=np.int64)
    lookup = {}
    for index, tri in enumerate(faces):
        lookup[frozenset(int(v) for v in tri)] = index
    face_pair = np.full(len(faces), -1, dtype=np.int64)
    winding = np.zeros(len(faces), dtype=np.int64)  # 1 reversed, 2 preserved, 3 rotated
    check = coordinates is not None and atol is not None
    center = np.asarray(coordinates, dtype=np.float64)[faces].mean(axis=1) if check else None
    set_hits = 0
    spurious = 0
    worst_spurious = 0.0
    for index, tri in enumerate(faces):
        mapped = [int(node_pair[int(v)]) for v in tri]
        other = lookup.get(frozenset(mapped))
        if other is None:
            continue
        set_hits += 1
        if check:
            delta = float(np.linalg.norm(center[index] - center[other] * np.array([1.0, -1.0, 1.0])))
            if delta > atol:
                spurious += 1
                worst_spurious = max(worst_spurious, delta)
                continue
        face_pair[index] = other
        reference = [int(v) for v in faces[other]]
        if mapped == list(reversed(reference)):
            winding[index] = 1
        elif mapped == reference:
            winding[index] = 2
        else:
            winding[index] = 3
    return dict(pair=face_pair, winding=winding, set_hits=set_hits,
                spurious=spurious, worst_spurious_distance=worst_spurious,
                geometric_check=bool(check))


def geometry_classes(coordinates, faces, node_pair, atol):
    """Separate real plane panels, straddling panels and pseudo fixed points.

    A panel entirely inside the symmetry plane would need a fixed-point
    unknown (the reducer does not support wall fixed points). A panel that
    straddles the plane has no mirror image at all. A panel whose *centroid* is
    near y=0 while its vertices are not in the plane is merely a nearest-map
    artefact and must not be counted as a plane cap.
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    y = coordinates[faces][:, :, 1]
    centroid = np.abs(y.mean(axis=1))
    plane = np.all(np.abs(y) <= atol, axis=1)
    straddle = (y.min(axis=1) < -atol) & (y.max(axis=1) > atol)
    return dict(
        plane_faces=int(np.sum(plane)),
        straddle_faces=int(np.sum(straddle)),
        centroid_near_plane=int(np.sum(centroid <= atol)),
        plane_indices=np.flatnonzero(plane).tolist()[:32],
        straddle_indices=np.flatnonzero(straddle).tolist()[:32],
    )


def te_bank_gate(surface, node_pair, tol=1e-9, atol=None):
    """Mirror gate for the trailing-edge *bank pairs*, the XZ-specific part.

    Each TE edge owns two wall faces: `upper_te_faces[i]` (larger z) and
    `lower_te_faces[i]` (smaller z), i.e. the suction and pressure bank at that
    spanwise station. A y -> -y reflection keeps z, so the mirror of pair i must
    be a pair i' with

        mirror(upper[i]) == upper[i']   and   mirror(lower[i]) == lower[i'],

    plus mirrored edge endpoints. Comparing `mirror(upper[i])` against the set of
    all *lower* faces is the wrong test: it demands that the reflection swap the
    z banks, which no XZ-symmetric mesh can satisfy, and it silently counts
    coincidental matches.
    """
    edges = np.asarray(surface["trailing_edges"], dtype=np.int64)
    upper = np.asarray(surface["upper_te_faces"], dtype=np.int64)
    lower = np.asarray(surface["lower_te_faces"], dtype=np.int64)
    if atol is None:
        atol = tol * max(scale_of(surface["coordinates"]), 1e-12)
    mapped = face_map(surface["faces"], node_pair, surface["coordinates"], atol)
    face_pair = mapped["pair"]
    edge_lookup = {}
    for index, edge in enumerate(edges):
        edge_lookup[frozenset(int(v) for v in edge)] = index
    paired = 0
    mismatches = []
    unpaired_edges = []
    for index in range(len(edges)):
        mapped_edge = frozenset(int(node_pair[int(v)]) for v in edges[index])
        other = edge_lookup.get(mapped_edge)
        if other is None:
            unpaired_edges.append(int(index))
            continue
        mapped_upper = int(face_pair[upper[index]])
        mapped_lower = int(face_pair[lower[index]])
        if mapped_upper < 0 or mapped_lower < 0:
            mismatches.append(dict(
                index=int(index), mirror_index=int(other),
                reason="no geometrically verified mirror face for a TE bank panel",
                upper=int(upper[index]), upper_mirror=mapped_upper, upper_expected=int(upper[other]),
                lower=int(lower[index]), lower_mirror=mapped_lower, lower_expected=int(lower[other]),
            ))
            continue
        if mapped_upper == int(upper[other]) and mapped_lower == int(lower[other]):
            paired += 1
        else:
            mismatches.append(dict(
                index=int(index), mirror_index=int(other),
                upper=int(upper[index]), upper_mirror=mapped_upper, upper_expected=int(upper[other]),
                lower=int(lower[index]), lower_mirror=mapped_lower, lower_expected=int(lower[other]),
            ))
    return dict(
        te_edges=int(len(edges)),
        te_edges_geometrically_mirrored=int(len(edges) - len(unpaired_edges)),
        te_edges_paired=paired,
        te_edges_unpaired=len(unpaired_edges),
        bank_mismatches=len(mismatches),
        unpaired_edges=unpaired_edges[:32],
        mismatch_examples=mismatches[:8],
        tolerance=atol,
        face_map_set_hits=mapped["set_hits"],
        face_map_spurious= mapped["spurious"],
    )


def analyze_surface(surface, tol=1e-9, station_tol=1e-6):
    """Full layered symmetry report for one `dpanel.planar_cut.v1` record."""
    coordinates = np.asarray(surface["coordinates"], dtype=np.float64)
    faces = np.asarray(surface["faces"], dtype=np.int64)
    wake = np.asarray(surface["wake_faces"], dtype=np.int64)
    stations = np.asarray(surface["gamma_stations"], dtype=np.float64)
    atol = tol * max(scale_of(coordinates), 1e-12)
    node_pair, node_residual = mirror_map(coordinates, atol)
    report = dict(
        schema=surface.get("schema"),
        tolerance=atol,
        nodes=_map_stats(node_pair, node_residual, atol),
        geometry=geometry_classes(coordinates, faces, node_pair, atol),
    )
    report["nodes"]["usable_map"] = bool(
        report["nodes"]["unpaired"] == 0 and report["nodes"]["bijective"] and report["nodes"]["involution"])
    mapped = face_map(faces, node_pair, coordinates, atol)
    face_pair, winding = mapped["pair"], mapped["winding"]
    face_unpaired = int(np.sum(face_pair < 0))
    report["faces"] = dict(
        count=int(len(faces)),
        paired=int(len(faces) - face_unpaired),
        unpaired=face_unpaired,
        bijective=(face_unpaired == 0 and len(set(face_pair.tolist())) == len(faces)),
        involution=bool(np.all(face_pair[face_pair] == np.arange(len(faces)))),
        winding_reversed=int(np.sum(winding == 1)),
        winding_preserved=int(np.sum(winding == 2)),
        winding_inconsistent=int(np.sum(winding == 3)),
        self_faces=int(np.sum(face_pair == np.arange(len(faces)))),
        vertex_set_hits=int(mapped["set_hits"]),
        spurious_vertex_set_hits=int(mapped["spurious"]),
        worst_spurious_distance=float(mapped["worst_spurious_distance"]),
        geometric_check=bool(mapped["geometric_check"]),
    )
    mapped_wake = face_map(wake, node_pair, coordinates, atol)
    wake_pair = mapped_wake["pair"]
    wake_unpaired = int(np.sum(wake_pair < 0))
    report["wake"] = dict(
        count=int(len(wake)),
        paired=int(len(wake) - wake_unpaired),
        unpaired=wake_unpaired,
        involution=bool(np.all(wake_pair[wake_pair] == np.arange(len(wake)))),
        self_faces=int(np.sum(wake_pair == np.arange(len(wake)))),
        vertex_set_hits=int(mapped_wake["set_hits"]),
        spurious_vertex_set_hits=int(mapped_wake["spurious"]),
    )
    report["trailing_edge"] = te_bank_gate(surface, node_pair, tol=tol, atol=atol)
    station_residual = np.abs(stations[:, None] + stations[None, :])
    station_pair = np.argmin(station_residual, axis=1)
    stol = station_tol * max(float(np.ptp(stations)), 1e-12)
    report["stations"] = _map_stats(station_pair, station_residual[np.arange(len(stations)), station_pair], stol)
    layers = dict(
        nodes=report["nodes"]["unpaired"] == 0 and report["nodes"]["bijective"] and report["nodes"]["involution"],
        faces=report["faces"]["unpaired"] == 0 and report["faces"]["involution"]
              and report["faces"]["winding_reversed"] == report["faces"]["count"],
        wake=report["wake"]["unpaired"] == 0 and report["wake"]["involution"],
        trailing_edge=report["trailing_edge"]["te_edges_paired"] == report["trailing_edge"]["te_edges"]
                      and report["trailing_edge"]["bank_mismatches"] == 0,
        stations=report["stations"]["unpaired"] == 0 and report["stations"]["involution"],
        no_plane_faces=report["geometry"]["plane_faces"] == 0,
        no_straddle_faces=report["geometry"]["straddle_faces"] == 0,
    )
    report["layers"] = {k: bool(v) for k, v in layers.items()}
    report["status"] = "ok" if all(layers.values()) else "blocked"
    report["blocking_layers"] = [k for k, v in layers.items() if not v]
    return report
