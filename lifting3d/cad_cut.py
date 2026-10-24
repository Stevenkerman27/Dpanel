"""Cut a conforming P2 tetrahedral mesh along an internal planar wake."""

import numpy as np


TET_EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
FACE_CORNERS = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))


class CutTopologyError(ValueError):
    """A cut failure carrying diagnostics accumulated before the failure."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def split_planar_cut(points, tets, wake_tri6, tip_y):
    """Split negative-bank cut nodes; tip_y is +/- scalar or [min_y, max_y].

    Returns connectivity, upper/lower wake faces and adjacent tet indices,
    `(original, duplicate)` cut pairs, original-node mapping, tip-node IDs,
    and diagnostics. Gamma values are not part of this geometry operation.
    """
    points = np.asarray(points, dtype=float)
    raw_tets = np.asarray(tets)
    raw_wake_tri6 = np.asarray(wake_tri6)
    tip_values = np.asarray(tip_y, dtype=float)
    diagnostics = {
        "status": "checking",
        "method": "cut-node incident-tet-star connectivity across non-wake faces",
        "global_tet_centroid_side_classification_used": False,
        "tip_y": tip_values.tolist(),
        "node_stars": [],
        "errors": [],
    }

    if not np.issubdtype(raw_tets.dtype, np.integer):
        diagnostics["errors"].append("tet10 connectivity must use integer node indices")
        raise CutTopologyError(diagnostics["errors"][-1], diagnostics)
    if not np.issubdtype(raw_wake_tri6.dtype, np.integer):
        diagnostics["errors"].append("wake tri6 connectivity must use integer node indices")
        raise CutTopologyError(diagnostics["errors"][-1], diagnostics)
    tets = raw_tets.astype(np.int64, copy=False)
    wake_tri6 = raw_wake_tri6.astype(np.int64, copy=False)

    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        diagnostics["errors"].append("points must be finite with shape (n,3)")
        raise CutTopologyError(diagnostics["errors"][-1], diagnostics)
    if tets.ndim != 2 or tets.shape[1] != 10 or len(tets) == 0:
        diagnostics["errors"].append("tets must be nonempty tet10 connectivity with shape (m,10)")
        raise CutTopologyError(diagnostics["errors"][-1], diagnostics)
    if wake_tri6.ndim != 2 or wake_tri6.shape[1] != 6 or len(wake_tri6) == 0:
        diagnostics["errors"].append("wake_tri6 must be nonempty tri6 connectivity with shape (f,6)")
        raise CutTopologyError(diagnostics["errors"][-1], diagnostics)
    if tip_values.ndim == 0:
        if not np.isfinite(tip_values) or tip_values <= 0.:
            diagnostics["errors"].append("scalar tip_y must be finite and positive")
            raise CutTopologyError(diagnostics["errors"][-1], diagnostics)
        tip_limits = np.array([-float(tip_values), float(tip_values)])
    elif (tip_values.shape == (2,) and np.isfinite(tip_values).all()
          and tip_values[0] < tip_values[1]):
        tip_limits = tip_values
    else:
        diagnostics["errors"].append(
            "tip_y must be a positive scalar or finite increasing [min_y, max_y]")
        raise CutTopologyError(diagnostics["errors"][-1], diagnostics)
    if np.min(tets) < 0 or np.max(tets) >= len(points):
        diagnostics["errors"].append("tet10 connectivity contains an out-of-range node index")
        raise CutTopologyError(diagnostics["errors"][-1], diagnostics)
    if np.min(wake_tri6) < 0 or np.max(wake_tri6) >= len(points):
        diagnostics["errors"].append("wake tri6 connectivity contains an out-of-range node index")
        raise CutTopologyError(diagnostics["errors"][-1], diagnostics)
    if any(len(np.unique(row)) != 10 for row in tets):
        diagnostics["errors"].append("each tet10 must contain ten distinct node indices")
        raise CutTopologyError(diagnostics["errors"][-1], diagnostics)
    if any(len(np.unique(row)) != 6 for row in wake_tri6):
        diagnostics["errors"].append("each wake tri6 must contain six distinct node indices")
        raise CutTopologyError(diagnostics["errors"][-1], diagnostics)

    coordinate_scale = max(1., float(np.ptp(points, axis=0).max()))
    tolerance = 1.e-10 * coordinate_scale
    diagnostics["coordinate_tolerance"] = tolerance

    first_face = points[wake_tri6[0, :3]]
    plane_normal = np.cross(first_face[1]-first_face[0], first_face[2]-first_face[0])
    normal_length = float(np.linalg.norm(plane_normal))
    if normal_length <= tolerance**2:
        diagnostics["errors"].append("first wake triangle has degenerate corner geometry")
        raise CutTopologyError(diagnostics["errors"][-1], diagnostics)
    plane_normal /= normal_length
    if plane_normal[2] < -1.e-12:
        plane_normal *= -1.
    elif abs(plane_normal[2]) <= 1.e-12:
        dominant_axis = int(np.argmax(np.abs(plane_normal)))
        if plane_normal[dominant_axis] < 0.:
            plane_normal *= -1.
    plane_origin = first_face[0]
    wake_coordinates = points[wake_tri6.reshape(-1)]
    wake_plane_distance = (wake_coordinates-plane_origin) @ plane_normal
    if np.max(np.abs(wake_plane_distance)) > tolerance:
        diagnostics["errors"].append(
            "wake_tri6 nodes do not lie on one planar wake surface within tolerance")
        raise CutTopologyError(diagnostics["errors"][-1], diagnostics)
    diagnostics["plane_origin"] = plane_origin.tolist()
    diagnostics["plane_normal_positive_side"] = plane_normal.tolist()
    diagnostics["max_wake_plane_distance"] = float(np.max(np.abs(wake_plane_distance)))

    edge_midpoint = {}
    face_records = {}
    for tet_index, tet in enumerate(tets):
        for edge_index, (first, second) in enumerate(TET_EDGES):
            edge_key = tuple(sorted((int(tet[first]), int(tet[second]))))
            midpoint_node = int(tet[4+edge_index])
            if edge_key in edge_midpoint and edge_midpoint[edge_key] != midpoint_node:
                diagnostics["errors"].append(
                    f"tet10 cells disagree on the midpoint node for edge {edge_key}")
                raise CutTopologyError(diagnostics["errors"][-1], diagnostics)
            edge_midpoint[edge_key] = midpoint_node
        for omitted_corner, local_corners in enumerate(FACE_CORNERS):
            face_corner_nodes = [int(tet[index]) for index in local_corners]
            face_key = tuple(sorted(face_corner_nodes))
            face_nodes = list(face_corner_nodes)
            for edge_first, edge_second in (
                (local_corners[0], local_corners[1]),
                (local_corners[1], local_corners[2]),
                (local_corners[2], local_corners[0]),
            ):
                edge_key = tuple(sorted((int(tet[edge_first]), int(tet[edge_second]))))
                face_nodes.append(edge_midpoint[edge_key])
            face_records.setdefault(face_key, []).append(
                (tet_index, int(tet[omitted_corner]), tuple(face_nodes)))

    if any(len(adjacent) > 2 for adjacent in face_records.values()):
        diagnostics["errors"].append("tet mesh has a non-manifold corner face")
        raise CutTopologyError(diagnostics["errors"][-1], diagnostics)

    wake_face_records = {}
    for face_index, face in enumerate(wake_tri6):
        face_key = tuple(sorted(int(node) for node in face[:3]))
        if face_key in wake_face_records:
            diagnostics["errors"].append(f"duplicate wake corner face {face_key}")
            raise CutTopologyError(diagnostics["errors"][-1], diagnostics)
        adjacent = face_records.get(face_key, [])
        if len(adjacent) != 2:
            diagnostics["errors"].append(
                f"wake face {face_index} must have exactly two adjacent tetrahedra; found {len(adjacent)}")
            raise CutTopologyError(diagnostics["errors"][-1], diagnostics)
        expected_mids = set(int(node) for node in face[3:])
        for tet_index, opposite_node, tet_face_nodes in adjacent:
            if set(tet_face_nodes) != set(int(node) for node in face):
                diagnostics["errors"].append(
                    f"wake face {face_index} tri6 nodes disagree with adjacent tet10 {tet_index}")
                raise CutTopologyError(diagnostics["errors"][-1], diagnostics)
            if set(tet_face_nodes[3:]) != expected_mids:
                diagnostics["errors"].append(
                    f"wake face {face_index} midside nodes disagree with adjacent tet10 {tet_index}")
                raise CutTopologyError(diagnostics["errors"][-1], diagnostics)
        wake_face_records[face_key] = (face_index, adjacent)

    cut_nodes = np.unique(wake_tri6)
    node_y = points[cut_nodes, 1]
    tip_nodes = (np.isclose(node_y, tip_limits[0], rtol=0., atol=tolerance)
                 | np.isclose(node_y, tip_limits[1], rtol=0., atol=tolerance))
    beyond_tip = (node_y < tip_limits[0]-tolerance) | (node_y > tip_limits[1]+tolerance)
    if np.any(beyond_tip):
        bad_nodes = cut_nodes[beyond_tip].tolist()
        diagnostics["errors"].append(
            f"wake nodes lie beyond tip_y={tip_limits.tolist()}: {bad_nodes[:12]}")
        raise CutTopologyError(diagnostics["errors"][-1], diagnostics)
    non_tip_nodes = cut_nodes[~tip_nodes]
    diagnostics["cut_node_count"] = int(len(cut_nodes))
    diagnostics["tip_node_count"] = int(tip_nodes.sum())
    tip_node_ids = cut_nodes[tip_nodes]

    node_tets = {int(node): [] for node in cut_nodes}
    for tet_index, tet in enumerate(tets):
        for node in np.intersect1d(tet, cut_nodes, assume_unique=False):
            node_tets[int(node)].append(tet_index)

    node_adjacency = {int(node): {} for node in cut_nodes}
    node_seed_tets = {int(node): {"positive": set(), "negative": set()} for node in cut_nodes}
    wake_face_keys = set(wake_face_records)
    for face_key, adjacent in face_records.items():
        if len(adjacent) != 2:
            continue
        first_tet, first_opposite, first_face_nodes = adjacent[0]
        second_tet, second_opposite, second_face_nodes = adjacent[1]
        face_nodes = set(first_face_nodes)
        if face_key in wake_face_keys:
            for tet_index, opposite_node in (
                (first_tet, first_opposite), (second_tet, second_opposite)
            ):
                signed_distance = float((points[opposite_node]-plane_origin) @ plane_normal)
                if abs(signed_distance) <= tolerance:
                    diagnostics["errors"].append(
                        f"wake-adjacent tet {tet_index} has an opposite corner on the wake plane")
                    raise CutTopologyError(diagnostics["errors"][-1], diagnostics)
                bank = "positive" if signed_distance > 0. else "negative"
                for node in face_nodes.intersection(node_seed_tets):
                    node_seed_tets[node][bank].add(tet_index)
        else:
            for node in face_nodes.intersection(node_adjacency):
                node_adjacency[node].setdefault(first_tet, set()).add(second_tet)
                node_adjacency[node].setdefault(second_tet, set()).add(first_tet)

    tet_bank = {}
    for node in non_tip_nodes:
        node = int(node)
        incident = set(node_tets[node])
        unseen = set(incident)
        components = []
        while unseen:
            seed = unseen.pop()
            component = {seed}
            stack = [seed]
            while stack:
                current = stack.pop()
                for neighbor in node_adjacency[node].get(current, ()):
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)
            components.append(component)

        component_records = []
        for component in components:
            labels = set()
            for bank, seeds in node_seed_tets[node].items():
                if component.intersection(seeds):
                    labels.add(bank)
            component_records.append({
                "tets": sorted(component),
                "banks": sorted(labels),
            })
        component_records.sort(key=lambda row: (row["banks"], row["tets"]))
        component_banks = [row["banks"] for row in component_records]
        node_record = {
            "node": node,
            "incident_tet_count": len(incident),
            "components": component_records,
            "component_tet_counts": sorted(len(component) for component in components),
            "component_banks": component_banks,
            "positive_seed_tets": sorted(node_seed_tets[node]["positive"]),
            "negative_seed_tets": sorted(node_seed_tets[node]["negative"]),
            "bank_count": len(components),
        }
        diagnostics["node_stars"].append(node_record)
        if (len(components) != 2
                or component_banks != [["negative"], ["positive"]]):
            diagnostics["errors"].append(
                f"cut-node {node} star must have exactly two banks seeded once each; "
                f"found {node_record}")
            raise CutTopologyError(diagnostics["errors"][-1], diagnostics)

        positive_component = set(next(row["tets"] for row in component_records
                                      if row["banks"] == ["positive"]))
        negative_component = set(next(row["tets"] for row in component_records
                                      if row["banks"] == ["negative"]))
        for tet_index in positive_component:
            previous = tet_bank.setdefault(tet_index, "positive")
            if previous != "positive":
                diagnostics["errors"].append(
                    f"tet {tet_index} receives inconsistent bank labels across cut-node stars")
                raise CutTopologyError(diagnostics["errors"][-1], diagnostics)
        for tet_index in negative_component:
            previous = tet_bank.setdefault(tet_index, "negative")
            if previous != "negative":
                diagnostics["errors"].append(
                    f"tet {tet_index} receives inconsistent bank labels across cut-node stars")
                raise CutTopologyError(diagnostics["errors"][-1], diagnostics)

    negative_nodes = non_tip_nodes.astype(np.int64, copy=True)
    duplicate_nodes = np.arange(len(points), len(points)+len(negative_nodes), dtype=np.int64)
    duplicate_for_node = dict(zip(negative_nodes.tolist(), duplicate_nodes.tolist()))
    output_points = np.vstack([points, points[negative_nodes]])
    output_tets = tets.copy()
    for tet_index, bank in tet_bank.items():
        if bank != "negative":
            continue
        for local_node, node in enumerate(output_tets[tet_index]):
            if int(node) in duplicate_for_node:
                output_tets[tet_index, local_node] = duplicate_for_node[int(node)]

    cut_pairs = np.column_stack([negative_nodes, duplicate_nodes]).astype(np.int64)
    cut_upper_nodes = cut_pairs[:, 0].copy()
    cut_lower_nodes = cut_pairs[:, 1].copy()
    wake_upper_tet = np.empty(len(wake_tri6), dtype=np.int64)
    wake_lower_tet = np.empty(len(wake_tri6), dtype=np.int64)
    for face_index, face in enumerate(wake_tri6):
        adjacent = wake_face_records[tuple(sorted(int(node) for node in face[:3]))][1]
        for tet_index, opposite_node, _ in adjacent:
            signed_distance = float((points[opposite_node]-plane_origin) @ plane_normal)
            bank = "positive" if signed_distance > 0. else "negative"
            if bank == "positive":
                wake_upper_tet[face_index] = tet_index
            else:
                wake_lower_tet[face_index] = tet_index
    wake_upper = wake_tri6.copy()
    wake_lower = wake_tri6.copy()
    for local_node, node in enumerate(wake_tri6.reshape(-1)):
        if int(node) in duplicate_for_node:
            wake_lower.reshape(-1)[local_node] = duplicate_for_node[int(node)]

    original_node = np.r_[np.arange(len(points), dtype=np.int64), negative_nodes]
    diagnostics["status"] = "passed"
    diagnostics["bank_tet_count"] = {
        "positive": int(sum(bank == "positive" for bank in tet_bank.values())),
        "negative": int(sum(bank == "negative" for bank in tet_bank.values())),
        "unlabeled_bulk": int(len(tets)-len(tet_bank)),
    }
    diagnostics["split_node_count"] = int(len(negative_nodes))
    diagnostics["tip_nodes_preserved"] = True
    diagnostics["tip_gamma_constraint"] = "tip nodes retain one shared ID; circulation basis must prescribe Gamma=0 there"

    return {
        "points": output_points,
        "tets": output_tets,
        "wake_upper": wake_upper,
        "wake_lower": wake_lower,
        "wake_upper_tet": wake_upper_tet,
        "wake_lower_tet": wake_lower_tet,
        "cut_pairs": cut_pairs,
        "cut_upper_nodes": cut_upper_nodes,
        "cut_lower_nodes": cut_lower_nodes,
        "tip_nodes": tip_node_ids.copy(),
        "original_node": original_node,
        "diagnostics": diagnostics,
    }
