"""Curved P2 potential FEM with an analytic circulation basis and a nodal TE condition."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse import bmat, coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits

from p2 import shape_functions

parser = argparse.ArgumentParser()
parser.add_argument("--mesh", type=Path, required=True)
parser.add_argument("--mach", type=float, default=0.)
parser.add_argument("--alpha", type=float, default=5.)
parser.add_argument("--boundary", choices=["uniform", "exact"], default="uniform")
parser.add_argument("--center", type=float, nargs=2)
parser.add_argument("--quadrature", type=int, default=4)
parser.add_argument("--maxiter", type=int, default=100)
parser.add_argument("--ncpu", type=int, default=8)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
started = time.perf_counter()
args.out.mkdir(parents=True, exist_ok=True)


def record_failure(exception_type, exception, traceback):
    failure = dict(status="failed_exception", converged=False, exception_type=exception_type.__name__,
                   message=str(exception), elapsed_seconds=time.perf_counter()-started,
                   solver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   mesh_path=str(args.mesh), mach=args.mach, alpha_deg=args.alpha)
    if args.mesh.is_file():
        failure["mesh_sha256"] = hashlib.sha256(args.mesh.read_bytes()).hexdigest()
    (args.out/"failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
    sys.__excepthook__(exception_type, exception, traceback)


sys.excepthook = record_failure
assert 0. <= args.mach < 1. and args.quadrature >= 3 and args.maxiter > 0 and args.ncpu > 0
threadpool_limits(args.ncpu)
mesh = np.load(args.mesh)
metadata = json.loads(args.mesh.with_name("mesh.json").read_text())
points, triangles, edges, tags, adjacent = [mesh[key] for key in
    ["points", "triangles", "edges", "edge_tags", "edge_triangle"]]
assert triangles.shape[1] == 6 and edges.shape[1] == 3
physical_chord = metadata["chord"]
assert np.isfinite(physical_chord) and physical_chord > 0.
normalization_origin = np.array(metadata.get("normalization_origin", metadata["farfield_center"]))
physical_points = points.copy()
points = (points-normalization_origin)/physical_chord
chord = 1.
alpha = np.deg2rad(args.alpha)
flow = np.array([np.cos(alpha), np.sin(alpha)])
center_physical = np.array(args.center) if args.center is not None else np.array(metadata["farfield_center"])
assert np.isfinite(center_physical).all(), "Circulation center coordinates must be finite"
center = (center_physical-normalization_origin)/physical_chord
contour = (mesh["contour"]-normalization_origin)/physical_chord
relative = (contour[:, 0]-center[0])+1j*(contour[:, 1]-center[1])
winding = np.angle(np.roll(relative, -1)/relative).sum()
assert abs(abs(winding)-2.*np.pi) < 1e-8, "Circulation center must lie inside the body"
gauss, weights_1d = leggauss(args.quadrature)
unit = (gauss+1.)/2.
weights_1d = weights_1d/2.
xi, eta_base = np.meshgrid(unit, unit, indexing="ij")
reference = np.column_stack([xi.ravel(), ((1.-xi)*eta_base).ravel()])
quadrature_weights = ((weights_1d[:, None]*weights_1d[None])*(1.-xi)).ravel()
# Pair the rule with its vertex-1/2 permutation so mirrored elements integrate identically.
reference = np.vstack([reference, reference[:, ::-1]])
quadrature_weights = np.tile(quadrature_weights/2., 2)
shape, reference_gradient = shape_functions(reference)
element_points = points[triangles]
jacobian = np.einsum("eia,qib->eqab", element_points, reference_gradient)
determinant = np.linalg.det(jacobian)
assert np.all(determinant > 0.), "Curved element has a nonpositive quadrature Jacobian"
gradient = np.einsum("qib,eqba->eqia", reference_gradient, np.linalg.inv(jacobian))
integration_weight = determinant*quadrature_weights
quadrature_points = np.einsum("qi,eia->eqa", shape, element_points)
delta = quadrature_points-center
harmonic = np.stack([-delta[:, :, 1], delta[:, :, 0]], axis=2)/np.sum(delta**2, axis=2)[:, :, None]
index_rows = np.broadcast_to(triangles[:, :, None], (len(triangles), 6, 6)).ravel()
index_cols = np.broadcast_to(triangles[:, None, :], (len(triangles), 6, 6)).ravel()
fixed = np.unique(edges[tags == 100])
free = np.setdiff1d(np.arange(len(points)), fixed)
boundary_base = points[fixed]@flow
boundary_k = np.zeros(len(fixed))
if args.boundary == "exact":
    assert args.mach == 0., "Exact boundary requires Mach=0"
    assert metadata["shape"] in ["joukowski", "trefftz"], (
        "Exact boundary is unavailable for this geometry; rotated exact mapping is unsupported")
    c = complex(-metadata["epsilon"], metadata["eta"])
    a = abs(1.-c)
    mapping_scale = metadata.get("coordinate_scale", 1.)
    mapping_translation = np.array(metadata.get("coordinate_translation", [0., 0.]))
    mapping_points = (physical_points[fixed]-mapping_translation)/mapping_scale
    z = mapping_points[:, 0]+1j*mapping_points[:, 1]
    exponent = metadata.get("mapping_exponent", 2.)
    if exponent == 2.:
        discriminant = np.sqrt(z*z-4.)
        roots1, roots2 = (z+discriminant)/2., (z-discriminant)/2.
        zeta = np.where(abs(roots1-c) > abs(roots2-c), roots1, roots2)
    else:
        ratio = ((z-exponent)/(z+exponent))**(1./exponent)
        zeta = (1.+ratio)/(1.-ratio)
    w = zeta-c
    boundary_base = mapping_scale/physical_chord*np.real(np.exp(-1j*alpha)*w+a**2*np.exp(1j*alpha)/w)
    boundary_k = np.angle((w/(1.-c))/((points[fixed, 0]-center[0])+1j*(points[fixed, 1]-center[1])))
reference_vertices = np.array([[0., 0.], [1., 0.], [0., 1.]])
te_node = int(mesh["trailing_node"])
te_edges = mesh["trailing_edges"]
trailing_edge_point = points[te_node]
if "leading_edge" in metadata:
    leading_edge_point = (np.asarray(metadata["leading_edge"])-normalization_origin)/physical_chord
else:
    contour_distance_from_te = np.linalg.norm(contour-contour[0], axis=1)
    leading_edge_point = contour[np.argmax(contour_distance_from_te)]
chord_direction = trailing_edge_point-leading_edge_point
chord_direction /= np.linalg.norm(chord_direction)
kutta_row = np.zeros(len(points))
for sign, edge_index in zip([1., -1.], te_edges):
    element = adjacent[edge_index]
    local_vertex = int(np.flatnonzero(triangles[element, :3] == te_node)[0])
    _, te_reference_gradient = shape_functions(reference_vertices[local_vertex][None])
    te_jacobian = element_points[element].T@te_reference_gradient[0]
    te_gradient = te_reference_gradient[0]@np.linalg.inv(te_jacobian)
    kutta_row[triangles[element]] += sign*(te_gradient@chord_direction)
# The analytic harmonic field is single valued at TE, so its two traces cancel here.
kutta_free = csr_matrix(kutta_row[free][None])
kutta_boundary_coefficient = float(kutta_row[fixed]@boundary_k)
kutta_rhs = -float(kutta_row[fixed]@boundary_base)
rho = np.ones(integration_weight.shape)
history = []
status = "iteration_limit"
for iteration in range(args.maxiter):
    local_matrix = np.einsum("eq,eqia,eqja->eij", integration_weight*rho, gradient, gradient)
    matrix = coo_matrix((local_matrix.ravel(), (index_rows, index_cols)),
                        shape=(len(points), len(points))).tocsr()
    local_harmonic = np.einsum("eq,eqia,eqa->ei", integration_weight*rho, gradient, harmonic)
    harmonic_column = np.zeros(len(points))
    np.add.at(harmonic_column, triangles.ravel(), local_harmonic.ravel())
    circulation_column = harmonic_column[free]+matrix[free][:, fixed]@boundary_k
    coupled = bmat([[matrix[free][:, free], csr_matrix(circulation_column[:, None])],
                    [kutta_free, csr_matrix([[kutta_boundary_coefficient]])]], format="csc")
    rhs = np.concatenate([-matrix[free][:, fixed]@boundary_base, [kutta_rhs]])
    solution = spsolve(coupled, rhs)
    assert np.isfinite(solution).all(), "Nonfinite coupled circulation solution"
    k = float(solution[-1])
    psi = np.zeros(len(points))
    psi[free] = solution[:-1]
    psi[fixed] = boundary_base+k*boundary_k
    velocity = np.einsum("eqia,ei->eqa", gradient, psi[triangles])+k*harmonic
    speed2 = np.sum(velocity**2, axis=2)
    beta = 1.+.2*args.mach**2*(1.-speed2)
    if beta.min() <= 0.:
        status = "nonpositive_isentropic_state"
        break
    mach2 = args.mach**2*speed2/beta
    if mach2.max() >= 1.:
        status = "local_sonic_state_outside_elliptic_solver"
        history.append(dict(iteration=iteration, max_local_mach=float(np.sqrt(mach2.max()))))
        break
    rho = beta**2.5
    local_residual = np.einsum("eq,eqia,eqa->ei", integration_weight*rho, gradient, velocity)
    residual = np.zeros(len(points))
    np.add.at(residual, triangles.ravel(), local_residual.ravel())
    residual_norm = float(np.linalg.norm(residual[free]))
    kutta_residual = float(kutta_row@psi)
    history.append(dict(iteration=iteration, free_nodal_mass_residual_norm=residual_norm,
                        kutta_residual=kutta_residual, circulation_ccw=2.*np.pi*k*physical_chord,
                        max_local_mach=float(np.sqrt(mach2.max())),
                        sum_all_nodal_residual=float(residual.sum())))
    if residual_norm <= 1e-9 and abs(kutta_residual) <= 1e-9:
        status = "converged_subsonic"
        break

result = dict(status=status, converged=status == "converged_subsonic",
              method="Curved P2 Galerkin with analytic circulation basis and nodal trailing-edge Kutta",
              conservation="Nodal weak mass balance; raw element face fluxes are not constrained to agree",
              scope="2D single sharp-TE airfoil; entirely subsonic; no shock or viscous/free wake model",
              mach=args.mach, alpha_deg=args.alpha, boundary=args.boundary,
              nodes=len(points), elements=len(triangles), free_nodes=len(free),
              circulation_center=center_physical.tolist(), circulation_ccw=2.*np.pi*k*physical_chord,
              normalization_origin=normalization_origin.tolist(), normalization_length=physical_chord,
              cl_circulation=-4.*np.pi*k/chord, kutta_residual=float(kutta_row@psi),
              quadrature_order=args.quadrature, ncpu_requested=args.ncpu, history=history,
              mesh_sha256=hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
              solver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              p2_sha256=hashlib.sha256(Path(__file__).with_name("p2.py").read_bytes()).hexdigest(),
              protocol_sha256=hashlib.sha256(Path(__file__).with_name("PROTOCOL.md").read_bytes()).hexdigest())
if result["converged"]:
    selected = np.arange(len(edges))
    is_wall = tags == 10
    boundary_elements = adjacent[selected]
    local_left = np.array([np.flatnonzero(triangles[e, :3] == edges[b, 0])[0]
                           for b, e in zip(selected, boundary_elements)])
    local_right = np.array([np.flatnonzero(triangles[e, :3] == edges[b, 1])[0]
                            for b, e in zip(selected, boundary_elements)])
    edge_reference = ((1.-unit)[None, :, None]*reference_vertices[local_left, None]
                      +unit[None, :, None]*reference_vertices[local_right, None])
    boundary_shape, boundary_reference_gradient = shape_functions(edge_reference.reshape(-1, 2))
    boundary_shape = boundary_shape.reshape(len(selected), len(unit), 6)
    boundary_reference_gradient = boundary_reference_gradient.reshape(len(selected), len(unit), 6, 2)
    boundary_jacobian = np.einsum("eia,eqib->eqab", element_points[boundary_elements], boundary_reference_gradient)
    boundary_gradient = np.einsum("eqib,eqba->eqia", boundary_reference_gradient,
                                   np.linalg.inv(boundary_jacobian))
    boundary_point = np.einsum("eqi,eia->eqa", boundary_shape, element_points[boundary_elements])
    edge_derivative = np.column_stack([-3.+4.*unit, -1.+4.*unit, 4.-8.*unit])
    tangent = np.einsum("qi,eia->eqa", edge_derivative, points[edges[selected]])
    normal_ds = np.stack([tangent[:, :, 1], -tangent[:, :, 0]], axis=2)
    toward_cell = element_points[boundary_elements, :3].mean(axis=1)-boundary_point.mean(axis=1)
    flip = np.sum(normal_ds.mean(axis=1)*toward_cell, axis=1) > 0.
    normal_ds[flip] *= -1.
    boundary_weight = np.linalg.norm(tangent, axis=2)*weights_1d
    normal = normal_ds/np.linalg.norm(normal_ds, axis=2)[:, :, None]
    boundary_delta = boundary_point-center
    boundary_h = (np.stack([-boundary_delta[:, :, 1], boundary_delta[:, :, 0]], axis=2)
                  /np.sum(boundary_delta**2, axis=2)[:, :, None])
    boundary_velocity = (np.einsum("eqia,ei->eqa", boundary_gradient, psi[triangles[boundary_elements]])
                         +k*boundary_h)
    boundary_speed2 = np.sum(boundary_velocity**2, axis=2)
    boundary_beta = 1.+.2*args.mach**2*(1.-boundary_speed2)
    assert np.all(boundary_beta > 0.)
    boundary_mach2 = args.mach**2*boundary_speed2/boundary_beta
    cp = 1.-boundary_speed2 if args.mach == 0. else (
        2.*np.expm1(3.5*np.log1p(.2*args.mach**2*(1.-boundary_speed2)))/(1.4*args.mach**2))
    boundary_vn = np.sum(boundary_velocity*normal, axis=2)
    boundary_rho = boundary_beta**2.5
    force = np.einsum("eq,eqa,q->a", cp[is_wall], normal_ds[is_wall], weights_1d)/chord
    mass_flux = boundary_rho*boundary_vn*boundary_weight
    result.update(cl_pressure=float(force@np.array([-np.sin(alpha), np.cos(alpha)])),
                  cd_pressure=float(force@flow), pressure_force_coefficient=force.tolist(),
                  wall_normal_rms=float(np.sqrt(np.average(boundary_vn[is_wall]**2, weights=boundary_weight[is_wall]))),
                  wall_normal_max=float(np.max(abs(boundary_vn[is_wall]))),
                  wall_raw_mass_flux=float(np.sum(mass_flux[is_wall])*physical_chord),
                  outer_raw_mass_flux=float(np.sum(mass_flux[~is_wall])*physical_chord),
                  total_raw_boundary_mass_flux=float(np.sum(mass_flux)*physical_chord),
                  wall_mass_flux_per_chord=float(np.sum(mass_flux[is_wall])),
                  outer_mass_flux_per_chord=float(np.sum(mass_flux[~is_wall])),
                  weak_farfield_reaction_sum=float(residual[fixed].sum()*physical_chord),
                  mass_flux_units="rho_inf*U_inf*physical_length; per_chord values are dimensionless",
                  max_local_mach=float(np.sqrt(max(mach2.max(), boundary_mach2.max()))),
                  cp_min=float(cp.min()), cp_max=float(cp.max()),
                  min_quadrature_jacobian=float(determinant.min()))
    if boundary_mach2.max() >= 1.:
        result.update(status="surface_sonic_state_outside_elliptic_solver", converged=False)
    exact_cp = np.full(cp.shape, np.nan)
    if metadata["shape"] in ["joukowski", "trefftz"] and args.mach == 0.:
        c = complex(-metadata["epsilon"], metadata["eta"])
        a = abs(1.-c)
        mapping_scale = metadata.get("coordinate_scale", 1.)
        mapping_translation = np.array(metadata.get("coordinate_translation", [0., 0.]))
        mapping_points = (boundary_point*physical_chord+normalization_origin-mapping_translation)/mapping_scale
        z = mapping_points[:, :, 0]+1j*mapping_points[:, :, 1]
        exponent = metadata.get("mapping_exponent", 2.)
        if exponent == 2.:
            discriminant = np.sqrt(z*z-4.)
            roots1, roots2 = (z+discriminant)/2., (z-discriminant)/2.
            zeta = np.where(abs(roots1-c) > abs(roots2-c), roots1, roots2)
            mapping_derivative = 1.-1./zeta**2
        else:
            ratio = ((z-exponent)/(z+exponent))**(1./exponent)
            zeta = (1.+ratio)/(1.-ratio)
            mapping_derivative = (4.*exponent**2*ratio**(exponent-1.)
                                  /((zeta+1.)**2*(1.-ratio**exponent)**2))
        w = zeta-c
        k_exact = 2.*a*np.sin(np.angle(1.-c)-alpha)
        exact_velocity = ((np.exp(-1j*alpha)-a*a*np.exp(1j*alpha)/w**2-1j*k_exact/w)
                          /mapping_derivative)
        exact_cp = 1.-abs(exact_velocity)**2
        result.update(cl_exact=-4.*np.pi*k_exact*mapping_scale/physical_chord,
                      circulation_ccw_exact=2.*np.pi*k_exact*mapping_scale,
                      cp_rmse_exact=float(np.sqrt(np.average((cp[is_wall]-exact_cp[is_wall])**2,
                                                              weights=boundary_weight[is_wall]))),
                      cp_max_error_exact=float(np.max(abs(cp[is_wall]-exact_cp[is_wall]))))
    # Raw gradients on the two traces of an internal edge are independently evaluated.
    face_map = {}
    for element, triangle in enumerate(triangles):
        for left, right in [(0, 1), (1, 2), (2, 0)]:
            face_map.setdefault(tuple(sorted(triangle[[left, right]])), []).append((element, left, right))
    interior = [sides for sides in face_map.values() if len(sides) == 2]
    trace_elements = np.array([[side[0] for side in sides] for sides in interior])
    trace_reference = np.array([[(reference_vertices[side[1]]+reference_vertices[side[2]])/2.
                                  for side in sides] for sides in interior])
    trace_shape, trace_dref = shape_functions(trace_reference.reshape(-1, 2))
    trace_points = element_points[trace_elements.ravel()]
    trace_jacobian = np.einsum("eia,eib->eab", trace_points, trace_dref)
    trace_grad = np.einsum("eib,eba->eia", trace_dref, np.linalg.inv(trace_jacobian))
    trace_position = np.einsum("ei,eia->ea", trace_shape, trace_points)
    trace_delta = trace_position-center
    trace_h = np.column_stack([-trace_delta[:, 1], trace_delta[:, 0]])/np.sum(trace_delta**2, axis=1)[:, None]
    trace_velocity = np.einsum("eia,ei->ea", trace_grad, psi[triangles[trace_elements.ravel()]])+k*trace_h
    trace_beta = 1.+.2*args.mach**2*(1.-np.sum(trace_velocity**2, axis=1))
    assert np.all(trace_beta > 0.)
    trace_flux = (trace_beta[:, None]**2.5*trace_velocity).reshape(-1, 2, 2)
    endpoint_pairs = np.array([[triangles[sides[0][0], sides[0][1]],
                                triangles[sides[0][0], sides[0][2]]] for sides in interior])
    edge_vector = points[endpoint_pairs[:, 1]]-points[endpoint_pairs[:, 0]]
    edge_length = np.linalg.norm(edge_vector, axis=1)
    edge_normal = np.column_stack([edge_vector[:, 1], -edge_vector[:, 0]])/edge_length[:, None]
    flux_jump = np.sum((trace_flux[:, 0]-trace_flux[:, 1])*edge_normal, axis=1)
    result.update(internal_midpoint_mass_flux_jump_rms=float(np.sqrt(np.average(flux_jump**2, weights=edge_length))),
                  internal_midpoint_mass_flux_jump_max=float(np.max(abs(flux_jump))),
                  internal_flux_jump_definition="Independent two-sided midpoint traces; chord-length weighted, not shared FV flux")
    np.savez_compressed(args.out/"field.npz", points=physical_points, triangles=triangles,
                        psi=psi*physical_chord, k=k*physical_chord,
                        velocity=velocity, rho=rho, nodal_residual=residual,
                        surface_point=boundary_point[is_wall]*physical_chord+normalization_origin,
                        surface_velocity=boundary_velocity[is_wall],
                        surface_normal=normal[is_wall], surface_weight=boundary_weight[is_wall]*physical_chord,
                        cp=cp[is_wall], exact_cp=exact_cp[is_wall],
                        boundary_point=boundary_point*physical_chord+normalization_origin,
                        boundary_tags=tags, boundary_mass_flux=mass_flux*physical_chord,
                        interior_mass_flux_jump=flux_jump)
result["elapsed_seconds"] = time.perf_counter()-started
(args.out/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
for source_name in ["solve.py", "p2.py", "PROTOCOL.md"]:
    (args.out/(source_name+".snapshot")).write_bytes(Path(__file__).with_name(source_name).read_bytes())
print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2), flush=True)
if not result["converged"]:
    raise SystemExit(2)
