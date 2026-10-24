"""CAD-face ownership from retained OCC source-solid copies."""
import numpy as np


def copy_source_solids(gmsh, sources_by_role):
    copies = {}
    for role, solids in sources_by_role.items():
        copies[role] = gmsh.model.occ.copy(list(solids))
    gmsh.model.occ.synchronize()
    return copies


def topology_entities(gmsh, volumes):
    entities = set((int(dim), int(tag)) for dim, tag in volumes)
    frontier = list(entities)
    while frontier:
        boundary = gmsh.model.getBoundary(frontier, combined=False, oriented=False)
        new_entities = set((int(dim), int(tag)) for dim, tag in boundary) - entities
        entities.update(new_entities)
        frontier = list(new_entities)
    return entities


def sample_surface_points(gmsh, surface_tag, sample_count=5):
    lower, upper = gmsh.model.getParametrizationBounds(2, surface_tag)
    u = np.linspace(lower[0], upper[0], 9)
    v = np.linspace(lower[1], upper[1], 9)
    candidates = []
    for ui in u[1:-1]:
        for vi in v[1:-1]:
            if gmsh.model.isInside(2, surface_tag, [float(ui), float(vi)], parametric=True):
                xyz = np.asarray(gmsh.model.getValue(2, surface_tag, [float(ui), float(vi)]))
                if not any(np.linalg.norm(xyz - old) < 1e-13 for old in candidates):
                    candidates.append(xyz)
    if len(candidates) < 3:
        return np.asarray(candidates, dtype=float).reshape((-1, 3))
    indices = np.linspace(0, len(candidates) - 1, min(sample_count, len(candidates))).round().astype(int)
    return np.asarray([candidates[index] for index in indices])


def closest_trimmed_source_distance(gmsh, point, source_faces):
    distances = []
    closest_tags = []
    for source_tag in source_faces:
        closest, uv = gmsh.model.getClosestPoint(2, int(source_tag), point.tolist())
        if len(uv) != 2:
            continue
        if gmsh.model.isInside(2, int(source_tag), uv, parametric=True):
            distance = float(np.linalg.norm(np.asarray(closest) - point))
            distances.append(distance)
            closest_tags.append(int(source_tag))
    if not distances:
        return float("inf"), None
    index = int(np.argmin(distances))
    return distances[index], closest_tags[index]


def classify_wall_surfaces(gmsh, fluid_volumes, source_copies_by_role,
                           geometric_tolerance, sample_count=5,
                           ambiguity_rule="body_precedence"):
    assert 3 <= sample_count <= 5
    assert geometric_tolerance > 0.0
    assert ambiguity_rule in ["body_precedence", "wing_precedence", "reject"]
    role_faces = {}
    for role, copied_solids in source_copies_by_role.items():
        role_faces[role] = sorted({
            int(tag)
            for dim, tag in gmsh.model.getBoundary(
                copied_solids, combined=False, oriented=False)
            if dim == 2
        })

    fluid_faces = sorted({
        int(tag)
        for dim, tag in gmsh.model.getBoundary(
            list(fluid_volumes), combined=False, oriented=False)
        if dim == 2
    })
    surface_tags = {}
    diagnostics = []
    wing_only = len(role_faces.get("body", [])) == 0

    for surface_tag in fluid_faces:
        samples = sample_surface_points(gmsh, surface_tag, sample_count)
        role_residuals = {}
        role_source_faces = {}
        for role, source_faces in role_faces.items():
            sample_distances = []
            nearest_sources = []
            for point in samples:
                distance, source_tag = closest_trimmed_source_distance(
                    gmsh, point, source_faces)
                sample_distances.append(distance)
                nearest_sources.append(source_tag)
            role_residuals[role] = (
                float(max(sample_distances)) if sample_distances else float("inf"))
            role_source_faces[role] = nearest_sources

        matched_roles = [
            role for role, residual in role_residuals.items()
            if len(samples) >= 3 and residual <= geometric_tolerance
        ]
        ambiguous = len(matched_roles) > 1
        selected_role = None
        if len(matched_roles) == 1:
            selected_role = matched_roles[0]
        elif ambiguous:
            if ambiguity_rule == "body_precedence":
                selected_role = "body" if "body" in matched_roles else sorted(matched_roles)[0]
            elif ambiguity_rule == "wing_precedence":
                selected_role = "wing" if "wing" in matched_roles else sorted(matched_roles)[0]

        if selected_role == "body":
            physical_tag = 20
        elif selected_role == "wing":
            physical_tag = 10 if wing_only else 30
        else:
            physical_tag = None

        if physical_tag is not None:
            surface_tags[surface_tag] = physical_tag
        diagnostics.append({
            "surface_tag": surface_tag,
            "physical_tag": physical_tag,
            "sample_count": int(len(samples)),
            "sample_xyz": samples.tolist(),
            "matched_roles": matched_roles,
            "selected_role": selected_role,
            "ambiguous": ambiguous,
            "ambiguity_rule": ambiguity_rule if ambiguous else None,
            "role_max_sample_residual": role_residuals,
            "nearest_source_face_by_sample": role_source_faces,
            "unmatched_reason": (
                "insufficient_trim_interior_samples" if len(samples) < 3 else
                "no_source_geometry_within_tolerance" if not matched_roles else None
            ),
        })
    return surface_tags, diagnostics


def remove_source_copies_before_mesh(gmsh, fluid_volumes, source_copies_by_role):
    fluid_volumes = [(int(dim), int(tag)) for dim, tag in fluid_volumes]
    assert fluid_volumes and all(dim == 3 for dim, _ in fluid_volumes)
    fluid_entities_before = topology_entities(gmsh, fluid_volumes)
    fluid_faces_before = {
        (int(dim), int(tag))
        for dim, tag in gmsh.model.getBoundary(
            fluid_volumes, combined=False, oriented=False)
        if dim == 2
    }
    fluid_mass_before = sum(gmsh.model.occ.getMass(3, tag) for _, tag in fluid_volumes)
    source_solids = [
        (int(dim), int(tag))
        for solids in source_copies_by_role.values()
        for dim, tag in solids
    ]
    source_entities = topology_entities(gmsh, source_solids)
    shared_entities = source_entities.intersection(fluid_entities_before)
    removable_entities = source_entities - shared_entities

    for dim in [3, 2, 1, 0]:
        remove_now = sorted(tag for entity_dim, tag in removable_entities if entity_dim == dim)
        if remove_now:
            gmsh.model.occ.remove([(dim, tag) for tag in remove_now], recursive=False)
    gmsh.model.occ.synchronize()

    model_entities = set(
        (int(dim), int(tag)) for dim, tag in gmsh.model.getEntities())
    fluid_entities_after = topology_entities(gmsh, fluid_volumes)
    fluid_faces_after = {
        (int(dim), int(tag))
        for dim, tag in gmsh.model.getBoundary(
            fluid_volumes, combined=False, oriented=False)
        if dim == 2
    }
    fluid_mass_after = sum(gmsh.model.occ.getMass(3, tag) for _, tag in fluid_volumes)
    assert set(fluid_volumes).issubset(model_entities)
    assert fluid_entities_after == fluid_entities_before
    assert fluid_faces_after == fluid_faces_before
    assert abs(fluid_mass_after - fluid_mass_before) <= 1e-11 * max(1.0, abs(fluid_mass_before))
    assert all(entity not in model_entities for entity in removable_entities)
    return {
        "source_copy_entity_count": len(source_entities),
        "removed_source_copy_entity_count": len(removable_entities),
        "protected_shared_entity_count": len(shared_entities),
        "fluid_volume_count": len(fluid_volumes),
        "fluid_mass_before": float(fluid_mass_before),
        "fluid_mass_after": float(fluid_mass_after),
        "fluid_mass_abs_change": float(abs(fluid_mass_after - fluid_mass_before)),
        "fluid_boundary_face_count_before": len(fluid_faces_before),
        "fluid_boundary_face_count_after": len(fluid_faces_after),
        "fluid_boundary_unchanged": fluid_faces_after == fluid_faces_before,
    }
