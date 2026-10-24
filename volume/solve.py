import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits


GAMMA = 1.4


def element_geometry(points, tets):
    xyz = points[tets]
    matrices = np.concatenate((np.ones((len(tets), 4, 1)), xyz), axis=2)
    determinants = np.linalg.det(matrices)
    if np.any(determinants <= 1.0e-14):
        raise ValueError("tetrahedra must have positive, nonzero signed volume")
    inverse = np.linalg.inv(matrices)
    gradients = np.transpose(inverse[:, 1:, :], (0, 2, 1))
    volumes = determinants / 6.0
    return gradients, volumes


def assemble(points, tets, gradients, volumes, phi, mach, with_jacobian):
    element_velocity = np.einsum("eij,ei->ej", gradients, phi[tets])
    q2 = np.einsum("ei,ei->e", element_velocity, element_velocity)
    beta = 1.0 + 0.2 * mach**2 * (1.0 - q2)
    if np.any(beta <= 0.0):
        raise ValueError("iterate is not admissible: beta must be positive")
    local_mach2 = mach**2 * q2 / beta
    if np.any(local_mach2 >= 1.0):
        raise ValueError("iterate is not strictly subsonic: local Mach must be below one")
    density = beta ** (1.0 / (GAMMA - 1.0))
    flux = density[:, None] * element_velocity
    local_residual = volumes[:, None] * np.einsum("eij,ej->ei", gradients, flux)
    residual = np.zeros(len(points))
    np.add.at(residual, tets.ravel(), local_residual.ravel())
    if not with_jacobian:
        return residual, None, element_velocity, beta, local_mach2
    constitutive = density[:, None, None] * (
        np.eye(3)[None, :, :]
        - (mach**2 / beta)[:, None, None]
        * element_velocity[:, :, None]
        * element_velocity[:, None, :]
    )
    local_jacobian = volumes[:, None, None] * np.einsum(
        "eia,eab,ejb->eij", gradients, constitutive, gradients
    )
    rows = np.broadcast_to(tets[:, :, None], local_jacobian.shape).ravel()
    cols = np.broadcast_to(tets[:, None, :], local_jacobian.shape).ravel()
    jacobian = coo_matrix((local_jacobian.ravel(), (rows, cols)), shape=(len(points), len(points))).tocsr()
    return residual, jacobian, element_velocity, beta, local_mach2


def energy(volumes, element_velocity, mach):
    q2 = np.einsum("ei,ei->e", element_velocity, element_velocity)
    c = 0.2 * mach**2
    if c == 0.0:
        primitive = 0.5 * q2
    else:
        log_ratio = np.log1p(-c * q2 / (1.0 + c))
        primitive = -(1.0 + c) ** 3.5 * np.expm1(3.5 * log_ratio) / (7.0 * c)
    return float(np.dot(volumes, primitive))


def pressure_coefficient(q2, mach):
    if mach == 0.0 or mach**2 == 0.0:
        return 1.0 - q2
    mach2 = mach**2
    logarithm = np.log1p(0.2 * mach2 * (1.0 - q2))
    return 2.0 * np.expm1((GAMMA / (GAMMA - 1.0)) * logarithm) / (GAMMA * mach2)


def farfield_values(points, direction, boundary, radius):
    if boundary == "uniform":
        return points @ direction
    distance = np.linalg.norm(points, axis=1)
    if np.any(distance == 0.0):
        raise ValueError("sphere-exact boundary is undefined at the sphere center")
    return (distance + 0.5 * radius**3 / distance**2) * (points @ direction) / distance


def load_mesh(path):
    raw = Path(path).read_bytes()
    with np.load(path) as archive:
        points = np.asarray(archive["points"], dtype=float)
        tets = np.asarray(archive["tets"], dtype=np.int64)
        triangles = np.asarray(archive["boundary_triangles"], dtype=np.int64).copy()
        tags = np.asarray(archive["boundary_tags"], dtype=np.int64)
        boundary_tet = np.asarray(archive["boundary_tet"], dtype=np.int64) if "boundary_tet" in archive else None
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("points must have shape (n, 3) and finite coordinates")
    if tets.ndim != 2 or tets.shape[1] != 4 or len(tets) == 0:
        raise ValueError("tets must have shape (m, 4)")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or tags.shape != (len(triangles),):
        raise ValueError("boundary_triangles and boundary_tags have incompatible shapes")
    if np.any(tets < 0) or np.any(tets >= len(points)) or np.any(triangles < 0) or np.any(triangles >= len(points)):
        raise ValueError("mesh contains an out-of-range point index")
    gradients, volumes = element_geometry(points, tets)
    face_map = {}
    for e, tet in enumerate(tets):
        for omitted in range(4):
            face = tuple(sorted(np.delete(tet, omitted).tolist()))
            face_map.setdefault(face, []).append(e)
    if any(len(adjacent) > 2 for adjacent in face_map.values()):
        raise ValueError("non-manifold tetrahedron face has more than two adjacent cells")
    exterior_faces = {face for face, adjacent in face_map.items() if len(adjacent) == 1}
    boundary_faces = [tuple(sorted(triangle.tolist())) for triangle in triangles]
    if len(set(boundary_faces)) != len(boundary_faces) or set(boundary_faces) != exterior_faces:
        raise ValueError("boundary triangles must list every exterior tetrahedron face exactly once")
    inferred = []
    reoriented_by_tag = {}
    for index, triangle in enumerate(triangles):
        key = tuple(sorted(triangle.tolist()))
        adjacent = face_map.get(key, [])
        if len(adjacent) != 1:
            raise ValueError(f"boundary triangle {index} must have exactly one adjacent tetrahedron")
        e = adjacent[0]
        inferred.append(e)
        normal = np.cross(points[triangle[1]] - points[triangle[0]], points[triangle[2]] - points[triangle[0]])
        opposite = next(node for node in tets[e] if node not in triangle)
        if np.dot(normal, points[opposite] - points[triangle[0]]) > 0.0:
            triangles[index, 1], triangles[index, 2] = triangles[index, 2], triangles[index, 1]
            reoriented_by_tag[str(int(tags[index]))] = reoriented_by_tag.get(str(int(tags[index])), 0) + 1
    boundary_tet_mismatch_count = 0
    if boundary_tet is not None:
        if boundary_tet.shape != (len(triangles),):
            boundary_tet_mismatch_count = len(triangles)
        else:
            boundary_tet_mismatch_count = int(np.count_nonzero(boundary_tet != inferred))
    if 100 not in tags:
        raise ValueError("boundary tag 100 is required for the farfield Dirichlet condition")
    if not np.all(np.isin(tags, [10, 20, 30, 100])):
        raise ValueError("boundary_tags must be one of 10, 20, 30, 100")
    face_rows = []
    for face, adjacent in face_map.items():
        if len(adjacent) == 2:
            face_rows.append((face, adjacent[0], adjacent[1]))
    cell_rows = [row[1] for row in face_rows] + [row[2] for row in face_rows]
    cell_cols = [row[2] for row in face_rows] + [row[1] for row in face_rows]
    cell_adjacency = coo_matrix((np.ones(len(cell_rows)), (cell_rows, cell_cols)), shape=(len(tets), len(tets))).tocsr()
    component_count = connected_components(cell_adjacency, directed=False, return_labels=False)
    if component_count != 1:
        raise ValueError(f"fluid tetrahedral mesh must be connected; found {component_count} components")
    return points, tets, triangles, tags, np.asarray(inferred), gradients, volumes, face_rows, hashlib.sha256(raw).hexdigest(), reoriented_by_tag, boundary_tet_mismatch_count


def main():
    parser = argparse.ArgumentParser(description="3-D subsonic full-potential P1 tetrahedral FEM reference solver")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--mach", type=float, required=True)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--boundary", choices=["uniform", "sphere-exact"], default="uniform")
    parser.add_argument("--sref", type=float, default=1.0)
    parser.add_argument("--ncpu", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = {"status": "failed", "input": vars(args), "history": [], "algorithm": "P1 continuous Galerkin; Laplace start; exact-Jacobian damped Newton; sparse direct"}
    solver_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["solver_sha256"] = solver_hash
    result["source_sha256"] = solver_hash
    try:
        mesh_hash = hashlib.sha256(Path(args.mesh).read_bytes()).hexdigest()
        result["mesh_sha256"] = mesh_hash
        result["meshcodehash"] = hashlib.sha256(bytes.fromhex(mesh_hash) + bytes.fromhex(solver_hash)).hexdigest()
        if not (0.0 <= args.mach < 1.0) or args.sref <= 0.0 or args.ncpu < 1:
            raise ValueError("require 0 <= mach < 1, sref > 0, and ncpu >= 1")
        metadata = Path(args.mesh).with_name("mesh.json")
        mesh_metadata = None
        if metadata.exists():
            mesh_metadata = json.loads(metadata.read_text(encoding="utf-8"))
        if args.boundary == "sphere-exact":
            if args.mach != 0.0:
                raise ValueError("sphere-exact is only valid for mach=0")
            if mesh_metadata is None or mesh_metadata.get("shape") != "sphere":
                raise ValueError("sphere-exact requires adjacent mesh.json with shape='sphere'")
            radius_value = mesh_metadata.get("sphere_radius", mesh_metadata.get("radius", mesh_metadata.get("reference_radius")))
            if radius_value is None:
                raise ValueError("sphere-exact requires an explicit sphere radius in mesh.json")
            sphere_radius = float(radius_value)
            if not np.isfinite(sphere_radius) or sphere_radius <= 0.0:
                raise ValueError("sphere-exact requires a finite positive sphere radius in mesh.json")
        else:
            sphere_radius = 1.0
        points, tets, triangles, tags, boundary_tet, gradients, volumes, internal_faces, mesh_hash, reoriented_by_tag, boundary_tet_mismatch_count = load_mesh(args.mesh)
        result.update({"mesh_sha256": mesh_hash, "nodes": int(len(points)), "elements": int(len(tets)), "dofs_total": int(len(points))})
        result["mesh_orientation_corrections_by_tag"] = reoriented_by_tag
        result["boundary_tet_input_mismatch_count"] = boundary_tet_mismatch_count
        result["mesh_shape"] = mesh_metadata.get("shape", "unknown") if mesh_metadata else "unknown"
        alpha = math.radians(args.alpha)
        sideslip = math.radians(args.beta)
        direction = np.array([math.cos(alpha) * math.cos(sideslip), math.sin(sideslip), math.sin(alpha) * math.cos(sideslip)])
        far_nodes = np.unique(triangles[tags == 100])
        fixed = np.zeros(len(points), dtype=bool)
        fixed[far_nodes] = True
        free = np.flatnonzero(~fixed)
        result["dofs_free"] = int(len(free))
        result["dofs_dirichlet"] = int(len(far_nodes))
        boundary_phi = farfield_values(points[far_nodes], direction, args.boundary, sphere_radius)
        phi = np.zeros(len(points))
        phi[far_nodes] = boundary_phi
        with threadpool_limits(limits=args.ncpu):
            _, laplace, _, _, _ = assemble(points, tets, gradients, volumes, phi, 0.0, True)
            phi[free] = spsolve(laplace[free][:, free], -laplace[free][:, far_nodes] @ boundary_phi)
            initial = assemble(points, tets, gradients, volumes, phi, args.mach, False)
            current_norm = float(np.linalg.norm(initial[0][free]))
            current_energy = energy(volumes, initial[2], args.mach)
            result["history"].append({"iteration": 0, "method": "Laplace initial field", "residual_norm": current_norm, "energy": current_energy, "step": 0.0})
            converged = current_norm <= 1.0e-10
            for iteration in range(1, 51):
                if converged:
                    break
                residual, jacobian, velocity, beta_field, local_mach2 = assemble(points, tets, gradients, volumes, phi, args.mach, True)
                delta = np.zeros(len(points))
                delta[free] = spsolve(jacobian[free][:, free], -residual[free])
                accepted = False
                step = 1.0
                for _ in range(40):
                    trial_phi = phi + step * delta
                    trial_phi[far_nodes] = boundary_phi
                    try:
                        trial = assemble(points, tets, gradients, volumes, trial_phi, args.mach, False)
                        trial_norm = float(np.linalg.norm(trial[0][free]))
                    except ValueError:
                        trial_norm = math.inf
                    if trial_norm < current_norm:
                        phi = trial_phi
                        accepted = True
                        break
                    step *= 0.5
                if not accepted:
                    raise RuntimeError(f"Newton line search failed at iteration {iteration}; no admissible strict residual decrease")
                current_norm = trial_norm
                current_energy = energy(volumes, trial[2], args.mach)
                result["history"].append({"iteration": iteration, "method": "exact-Jacobian Newton", "residual_norm": current_norm, "energy": current_energy, "step": step})
                converged = current_norm <= 1.0e-10
            if not converged:
                raise RuntimeError("Newton did not reach residual tolerance 1e-10 within 50 iterations")
        residual, _, velocity, beta_field, local_mach2 = assemble(points, tets, gradients, volumes, phi, args.mach, False)
        surface_velocity = velocity[boundary_tet]
        surface_speed2 = np.einsum("bi,bi->b", surface_velocity, surface_velocity)
        surface_beta = 1.0 + 0.2 * args.mach**2 * (1.0 - surface_speed2)
        if np.any(surface_beta <= 0.0):
            raise ValueError("surface raw gradient violates beta > 0")
        cp = pressure_coefficient(surface_speed2, args.mach)
        surface_density = surface_beta ** (1.0 / (GAMMA - 1.0))
        triangle_xyz = points[triangles]
        area_vectors = 0.5 * np.cross(triangle_xyz[:, 1] - triangle_xyz[:, 0], triangle_xyz[:, 2] - triangle_xyz[:, 0])
        areas = np.linalg.norm(area_vectors, axis=1)
        normals = area_vectors / areas[:, None]
        forces = {}
        for tag in [10, 20, 30]:
            selected = tags == tag
            force = np.sum(cp[selected, None] * area_vectors[selected], axis=0)
            forces[str(tag)] = {"area": float(np.sum(areas[selected])), "force": force.tolist(), "force_over_sref": (force / args.sref).tolist(), "cp_area_mean": float(np.average(cp[selected], weights=areas[selected])) if np.any(selected) else None}
        z_projected = np.array([0.0, 0.0, 1.0]) - direction[2] * direction
        if np.linalg.norm(z_projected) < 1.0e-12:
            z_projected = np.array([0.0, 1.0, 0.0]) - direction[1] * direction
        lift_direction = z_projected / np.linalg.norm(z_projected)
        total_force = sum((np.asarray(forces[str(tag)]["force"]) for tag in [10, 20, 30]), np.zeros(3))
        result["loads"] = {"by_tag": forces, "force": total_force.tolist(), "drag_over_sref": float(np.dot(total_force, direction) / args.sref), "lift_over_sref": float(np.dot(total_force, lift_direction) / args.sref), "lift_direction": lift_direction.tolist(), "pressure_force_convention": "Cp times outward-from-fluid area vector; normalized by freestream dynamic pressure and sref"}
        result["convergence"] = {"converged": True, "final_free_residual_norm": float(np.linalg.norm(residual[free])), "residual_tolerance": 1.0e-10, "iterations": len(result["history"]) - 1}
        face_jumps = []
        for face, left, right in internal_faces:
            xyz = points[list(face)]
            area_vector = 0.5 * np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])
            normal = area_vector / np.linalg.norm(area_vector)
            q2_lr = np.array([velocity[left] @ velocity[left], velocity[right] @ velocity[right]])
            beta_lr = 1.0 + 0.2 * args.mach**2 * (1.0 - q2_lr)
            flux_lr = beta_lr[:, None] ** (1.0 / (GAMMA - 1.0)) * velocity[[left, right]]
            face_jumps.append((abs(float(np.dot(flux_lr[0] - flux_lr[1], normal))), float(np.linalg.norm(area_vector))))
        jump_values = np.array([item[0] for item in face_jumps])
        jump_areas = np.array([item[1] for item in face_jumps])
        boundary_normal_velocity = np.einsum("bi,bi->b", surface_velocity, normals)
        boundary_mass_flux = surface_density * boundary_normal_velocity
        boundary_metrics = {}
        for tag in [10, 20, 30, 100]:
            selected = tags == tag
            boundary_metrics[str(tag)] = {
                "area_weighted_rms_raw_normal_velocity": float(np.sqrt(np.average(boundary_normal_velocity[selected] ** 2, weights=areas[selected]))) if np.any(selected) else None,
                "max_abs_raw_normal_velocity": float(np.max(np.abs(boundary_normal_velocity[selected]))) if np.any(selected) else None,
                "area_weighted_rms_raw_normal_mass_flux": float(np.sqrt(np.average(boundary_mass_flux[selected] ** 2, weights=areas[selected]))) if np.any(selected) else None,
                "max_abs_raw_normal_mass_flux": float(np.max(np.abs(boundary_mass_flux[selected]))) if np.any(selected) else None,
            }
        integrated_mass_flux = {str(tag): float(np.sum(areas[tags == tag] * boundary_mass_flux[tags == tag])) for tag in [10, 20, 30, 100]}
        wall_mass_flux = sum(integrated_mass_flux[str(tag)] for tag in [10, 20, 30])
        result["diagnostics"] = {
            "internal_face_mass_flux_jump_area_weighted_rms": float(np.sqrt(np.average(jump_values**2, weights=jump_areas))) if len(jump_values) else None,
            "internal_face_mass_flux_jump_max_abs": float(np.max(jump_values)) if len(jump_values) else None,
            "raw_boundary_tet_gradient_normal_metrics": boundary_metrics,
            "raw_boundary_integrated_mass_flux_by_tag": integrated_mass_flux,
            "raw_wall_integrated_mass_flux": wall_mass_flux,
            "raw_outer_integrated_mass_flux": integrated_mass_flux["100"],
            "raw_total_boundary_integrated_mass_flux": wall_mass_flux + integrated_mass_flux["100"],
            "weak_nodal_balance": {
                "free_residual_l2": float(np.linalg.norm(residual[free])),
                "free_residual_sum": float(np.sum(residual[free])),
                "dirichlet_nodal_reaction_sum": float(np.sum(residual[far_nodes])),
                "all_nodal_residual_sum": float(np.sum(residual)),
            },
            "max_local_mach": float(np.sqrt(np.max(local_mach2))),
            "min_beta": float(np.min(beta_field)),
        }
        sphere_selected = tags == 10
        if np.any(sphere_selected) and args.boundary == "sphere-exact" and args.mach == 0.0 and result["mesh_shape"] == "sphere":
            centers = triangle_xyz[sphere_selected].mean(axis=1)
            radii = np.linalg.norm(centers, axis=1)
            radial_direction = centers / radii[:, None]
            exact_cp = 1.0 - 2.25 * (1.0 - (radial_direction @ direction) ** 2)
            error = cp[sphere_selected] - exact_cp
            result["sphere_reference"] = {"area_weighted_cp_rms_error": float(np.sqrt(np.average(error**2, weights=areas[sphere_selected]))), "max_abs_cp_error": float(np.max(np.abs(error))), "analytic_lift_over_sref": 0.0, "analytic_drag_over_sref": 0.0, "lift_error_over_sref": result["loads"]["lift_over_sref"], "drag_error_over_sref": result["loads"]["drag_over_sref"], "mesh_centroid_radius_min": float(np.min(radii)), "mesh_centroid_radius_max": float(np.max(radii)), "farfield_truncation_error_note": "exact sphere potential prescribed on tag 100; Cp reference is evaluated at each face centroid angular direction; this is not a general farfield condition"}
        np.savez_compressed(output / "field.npz", points=points, tets=tets, Phi_total=phi, element_gradient=velocity, element_local_mach=np.sqrt(local_mach2), element_beta=beta_field, boundary_triangles=triangles, boundary_tags=tags, boundary_tet=boundary_tet, surface_velocity=surface_velocity, surface_density=surface_density, surface_mass_flux=surface_density[:, None] * surface_velocity, surface_cp=cp, surface_area=areas, surface_normal=normals)
        result.update({"status": "converged", "wall_seconds": time.perf_counter() - started})
    except Exception as error:
        result.update({"error": f"{type(error).__name__}: {error}", "wall_seconds": time.perf_counter() - started})
        print(result["error"], file=sys.stderr)
    finally:
        (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    if result["status"] != "converged":
        return 1
    print(json.dumps({"status": result["status"], "iterations": result["convergence"]["iterations"], "final_residual": result["convergence"]["final_free_residual_norm"], "out": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
