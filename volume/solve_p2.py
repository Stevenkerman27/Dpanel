import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits

import p2


def main():
    parser = argparse.ArgumentParser(description="3-D curved P2 subsonic full-potential tetrahedral FEM reference")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--mach", type=float, required=True)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--boundary", choices=["uniform", "sphere-exact"], default="uniform")
    parser.add_argument("--sref", type=float, default=1.0)
    parser.add_argument("--ncpu", type=int, default=8)
    parser.add_argument("--quad-order", type=int, choices=[3, 4], default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source_hash = hashlib.sha256(Path(__file__).read_bytes() + Path(p2.__file__).read_bytes()).hexdigest()
    result = {
        "status": "failed",
        "input": vars(args),
        "history": [],
        "algorithm": "isoparametric P2 continuous Galerkin; P2 Laplace start; exact-Jacobian damped Newton; sparse direct",
        "source_sha256": source_hash,
    }
    try:
        mesh_hash = hashlib.sha256(Path(args.mesh).read_bytes()).hexdigest()
        result["mesh_sha256"] = mesh_hash
        result["meshcodehash"] = hashlib.sha256(bytes.fromhex(mesh_hash) + bytes.fromhex(source_hash)).hexdigest()
        if (
            not np.isfinite(args.mach)
            or not (0.0 <= args.mach < 1.0)
            or not np.isfinite(args.alpha)
            or not np.isfinite(args.beta)
            or not np.isfinite(args.sref)
            or args.sref <= 0.0
            or args.ncpu < 1
        ):
            raise ValueError("require finite alpha/beta, 0 <= mach < 1, sref > 0, and ncpu >= 1")
        if args.quad_order not in [3, 4]:
            raise ValueError("quadrature order must be 3 or 4")
        metadata_path = Path(args.mesh).with_name("mesh.json")
        mesh_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
        sphere_radius = None
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
                raise ValueError("sphere-exact requires a finite positive sphere radius")
        points, tets, triangles, tags, boundary_tet, mesh_hash, reoriented_by_tag, boundary_tet_mismatch = p2.load_mesh(args.mesh)
        result.update(
            {
                "mesh_sha256": mesh_hash,
                "mesh_shape": mesh_metadata.get("shape", "unknown") if mesh_metadata else "unknown",
                "nodes": int(len(points)),
                "elements": int(len(tets)),
                "dofs_total": int(len(points)),
                "quadrature_order": args.quad_order,
                "quadrature_volume_points_per_element": len(p2.tetra_quadrature(args.quad_order)[0]),
                "quadrature_surface_points_per_face": len(p2.triangle_quadrature(args.quad_order)[0]),
                "mesh_orientation_corrections_by_tag": reoriented_by_tag,
                "boundary_tet_input_mismatch_count": boundary_tet_mismatch,
            }
        )
        direction_alpha = math.radians(args.alpha)
        direction_beta = math.radians(args.beta)
        freestream = np.array(
            [
                math.cos(direction_alpha) * math.cos(direction_beta),
                math.sin(direction_beta),
                math.sin(direction_alpha) * math.cos(direction_beta),
            ]
        )
        far_nodes = np.unique(triangles[tags == 100])
        fixed = np.zeros(len(points), dtype=bool)
        fixed[far_nodes] = True
        free = np.flatnonzero(~fixed)
        result["dofs_free"] = int(len(free))
        result["dofs_dirichlet"] = int(len(far_nodes))
        if args.boundary == "uniform":
            boundary_values = points[far_nodes] @ freestream
        else:
            far_xyz = points[far_nodes]
            radius = np.linalg.norm(far_xyz, axis=1)
            if np.any(radius == 0.0):
                raise ValueError("sphere-exact is undefined at the sphere center")
            boundary_values = (radius + 0.5 * sphere_radius**3 / radius**2) * (far_xyz @ freestream) / radius
        phi = np.zeros(len(points))
        phi[far_nodes] = boundary_values
        with threadpool_limits(limits=args.ncpu):
            _, laplace, _, _, _ = p2.evaluate(points, tets, phi, 0.0, args.quad_order, True)
            if len(free):
                phi[free] = spsolve(laplace[free][:, free], -laplace[free][:, far_nodes] @ boundary_values)
                if not np.all(np.isfinite(phi[free])):
                    raise RuntimeError("P2 Laplace initialization returned nonfinite values")
            initial = p2.evaluate(points, tets, phi, args.mach, args.quad_order, False)
            current_norm = float(np.linalg.norm(initial[0][free]))
            current_energy = initial[4]
            result["history"].append(
                {"iteration": 0, "method": "P2 Laplace initial field", "residual_norm": current_norm, "energy": current_energy, "step": 0.0}
            )
            converged = current_norm <= 1.0e-10
            for iteration in range(1, 51):
                if converged:
                    break
                residual, jacobian, _, _, _ = p2.evaluate(points, tets, phi, args.mach, args.quad_order, True)
                delta = np.zeros(len(points))
                if len(free):
                    delta[free] = spsolve(jacobian[free][:, free], -residual[free])
                    if not np.all(np.isfinite(delta[free])):
                        raise RuntimeError("P2 Newton linear solve returned nonfinite values")
                accepted = False
                step = 1.0
                for _ in range(40):
                    trial_phi = phi + step * delta
                    trial_phi[far_nodes] = boundary_values
                    try:
                        trial = p2.evaluate(points, tets, trial_phi, args.mach, args.quad_order, False)
                        trial_norm = float(np.linalg.norm(trial[0][free]))
                    except ValueError:
                        trial_norm = math.inf
                    if trial_norm < current_norm:
                        phi = trial_phi
                        accepted = True
                        break
                    step *= 0.5
                if not accepted:
                    raise RuntimeError(f"P2 Newton line search failed at iteration {iteration}")
                current_norm = trial_norm
                current_energy = trial[4]
                result["history"].append(
                    {"iteration": iteration, "method": "exact-Jacobian P2 Newton", "residual_norm": current_norm, "energy": current_energy, "step": step}
                )
                converged = current_norm <= 1.0e-10
            if not converged:
                raise RuntimeError("P2 Newton did not reach residual tolerance 1e-10 within 50 iterations")
        residual, _, element_max_mach2, element_min_beta, final_energy = p2.evaluate(
            points, tets, phi, args.mach, args.quad_order, False
        )
        surface = p2.evaluate_surface(points, tets, triangles, tags, boundary_tet, phi, args.mach, args.quad_order)
        integrated_mass_by_tag = {
            str(tag): float(np.sum(surface["integrated_mass_flux"][tags == tag])) for tag in [10, 20, 30, 100]
        }
        wall_mass = sum(integrated_mass_by_tag[str(tag)] for tag in [10, 20, 30])
        raw_boundary_mass_closure = float(sum(integrated_mass_by_tag.values()))
        raw_boundary_mass_scale = float(sum(abs(value) for value in integrated_mass_by_tag.values()))
        internal_flux_jump = p2.internal_face_flux_jump(points, tets, phi, args.mach, args.quad_order)
        tag_loads = {}
        for tag in [10, 20, 30]:
            face_selection = tags == tag
            force = np.sum(
                surface["cp"][face_selection, :, None]
                * surface["area_vector"][face_selection]
                * p2.triangle_quadrature(args.quad_order)[1][None, :, None],
                axis=(0, 1),
            )
            face_area = float(np.sum(surface["area_weight"][face_selection]))
            cp_values = surface["cp"][face_selection]
            cp_weights = surface["area_weight"][face_selection]
            tag_loads[str(tag)] = {
                "area": face_area,
                "force": force.tolist(),
                "force_over_sref": (force / args.sref).tolist(),
                "cp_area_mean": float(np.sum(cp_values * cp_weights) / face_area) if face_area > 0.0 else None,
            }
        total_force = sum((np.asarray(tag_loads[str(tag)]["force"]) for tag in [10, 20, 30]), np.zeros(3))
        lift_axis = np.array([0.0, 0.0, 1.0]) - freestream[2] * freestream
        if np.linalg.norm(lift_axis) < 1.0e-12:
            lift_axis = np.array([0.0, 1.0, 0.0]) - freestream[1] * freestream
        lift_axis /= np.linalg.norm(lift_axis)
        result["loads"] = {
            "by_tag": tag_loads,
            "force": total_force.tolist(),
            "drag_over_sref": float(total_force @ freestream / args.sref),
            "lift_over_sref": float(total_force @ lift_axis / args.sref),
            "lift_direction": lift_axis.tolist(),
        }
        raw_boundary_metrics = {}
        for tag in [10, 20, 30, 100]:
            selected = tags == tag
            area_weight = surface["area_weight"][selected]
            area_sum = float(np.sum(area_weight))
            if area_sum == 0.0:
                raw_boundary_metrics[str(tag)] = None
                continue
            vn = surface["normal_velocity"][selected]
            mass_vn = surface["normal_mass_flux"][selected]
            raw_boundary_metrics[str(tag)] = {
                "area_weighted_rms_normal_velocity": float(np.sqrt(np.sum(area_weight * vn**2) / area_sum)),
                "max_abs_normal_velocity": float(np.max(np.abs(vn))),
                "area_weighted_rms_normal_mass_flux": float(np.sqrt(np.sum(area_weight * mass_vn**2) / area_sum)),
                "max_abs_normal_mass_flux": float(np.max(np.abs(mass_vn))),
                "integrated_mass_flux": integrated_mass_by_tag[str(tag)],
            }
        result["convergence"] = {
            "converged": True,
            "final_free_residual_norm": float(np.linalg.norm(residual[free])),
            "residual_tolerance": 1.0e-10,
            "iterations": len(result["history"]) - 1,
            "final_energy": final_energy,
        }
        result["diagnostics"] = {
            "raw_boundary_tet10_gradient_normal_metrics": raw_boundary_metrics,
            "raw_wall_integrated_mass_flux": wall_mass,
            "raw_outer_integrated_mass_flux": integrated_mass_by_tag["100"],
            "raw_total_boundary_integrated_mass_flux": raw_boundary_mass_closure,
            "raw_boundary_mass_closure_abs": abs(raw_boundary_mass_closure),
            "raw_boundary_mass_closure_relative_to_sum_abs_tags": (
                abs(raw_boundary_mass_closure) / raw_boundary_mass_scale if raw_boundary_mass_scale > 0.0 else 0.0
            ),
            "raw_boundary_mass_gate": {
                "residual": raw_boundary_mass_closure,
                "normalized_residual": (
                    abs(raw_boundary_mass_closure) / raw_boundary_mass_scale if raw_boundary_mass_scale > 0.0 else 0.0
                ),
                "threshold": None,
                "status": "measured_no_acceptance_threshold",
            },
            "weak_nodal_balance": {
                "free_residual_l2": float(np.linalg.norm(residual[free])),
                "free_residual_sum": float(np.sum(residual[free])),
                "dirichlet_nodal_reaction_sum": float(np.sum(residual[far_nodes])),
                "all_nodal_residual_sum": float(np.sum(residual)),
            },
            "internal_face_flux_jump": internal_flux_jump,
            "max_volume_quadrature_local_mach": float(np.sqrt(np.max(element_max_mach2))),
            "min_volume_quadrature_beta": float(np.min(element_min_beta)),
            "max_surface_quadrature_local_mach": float(np.max(surface["max_local_mach"])),
        }
        sphere_faces = tags == 10
        if args.mach == 0.0 and result["mesh_shape"] == "sphere" and np.any(sphere_faces):
            face_coordinates = points[triangles[sphere_faces]]
            triangle_shape_values, _ = p2.triangle_shape(p2.triangle_quadrature(args.quad_order)[0])
            surface_coordinates = np.einsum("qi,bik->bqk", triangle_shape_values, face_coordinates, optimize=True)
            radial = surface_coordinates / np.linalg.norm(surface_coordinates, axis=2)[:, :, None]
            exact_cp = 1.0 - 2.25 * (1.0 - np.einsum("bqk,k->bq", radial, freestream) ** 2)
            cp_error = surface["cp"][sphere_faces] - exact_cp
            cp_weight = surface["area_weight"][sphere_faces]
            result["sphere_reference"] = {
                "area_weighted_cp_rms_error": float(np.sqrt(np.sum(cp_error**2 * cp_weight) / np.sum(cp_weight))),
                "max_abs_cp_error": float(np.max(np.abs(cp_error))),
                "analytic_lift_over_sref": 0.0,
                "analytic_drag_over_sref": 0.0,
                "lift_error_over_sref": result["loads"]["lift_over_sref"],
                "drag_error_over_sref": result["loads"]["drag_over_sref"],
                "farfield_boundary_condition": args.boundary,
                "farfield_boundary_note": "uniform and sphere-exact Dirichlet data compared on the same R=5 mesh; this does not measure domain-radius truncation sensitivity",
            }
        result["quadrature_sensitivity_note"] = "compare otherwise identical runs with --quad-order 3 and 4"
        np.savez_compressed(
            output / "field.npz",
            points=points,
            tets=tets,
            Phi_total=phi,
            element_max_local_mach=np.sqrt(element_max_mach2),
            element_min_beta=element_min_beta,
            boundary_triangles=triangles,
            boundary_tags=tags,
            boundary_tet=boundary_tet,
            surface_velocity=surface["velocity"],
            surface_density=surface["density"],
            surface_cp=surface["cp"],
            surface_normal=surface["normal"],
            surface_area_weight=surface["area_weight"],
            surface_normal_velocity=surface["normal_velocity"],
            surface_normal_mass_flux=surface["normal_mass_flux"],
        )
        result.update({"status": "converged", "wall_seconds": time.perf_counter() - started})
    except Exception as error:
        result.update({"error": f"{type(error).__name__}: {error}", "wall_seconds": time.perf_counter() - started})
        print(result["error"], file=sys.stderr)
    finally:
        (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    if result["status"] != "converged":
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "iterations": result["convergence"]["iterations"],
                "final_residual": result["convergence"]["final_free_residual_norm"],
                "out": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
