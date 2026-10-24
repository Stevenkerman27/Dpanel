"""Curved P2 potential FEM with an analytic circulation basis and a nodal TE condition."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse import bmat, coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits


def shape_functions(reference):
    barycentric = np.column_stack([1.-reference.sum(axis=1), reference])
    barycentric_gradient = np.array([[-1., -1.], [1., 0.], [0., 1.]])
    values = np.column_stack([barycentric*(2.*barycentric-1.),
                              4.*barycentric[:, 0]*barycentric[:, 1],
                              4.*barycentric[:, 1]*barycentric[:, 2],
                              4.*barycentric[:, 2]*barycentric[:, 0]])
    gradients = np.empty((len(reference), 6, 2))
    gradients[:, :3] = (4.*barycentric-1.)[:, :, None]*barycentric_gradient
    for edge, (left, right) in enumerate([(0, 1), (1, 2), (2, 0)]):
        gradients[:, 3+edge] = 4.*(barycentric[:, left, None]*barycentric_gradient[right]
                                         +barycentric[:, right, None]*barycentric_gradient[left])
    return values, gradients


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
assert 0. <= args.mach < 1. and args.quadrature >= 3
threadpool_limits(args.ncpu)
started = time.perf_counter()
args.out.mkdir(parents=True, exist_ok=True)
mesh = np.load(args.mesh)
metadata = json.loads(args.mesh.with_name("mesh.json").read_text())
points, triangles, edges, tags, adjacent = [mesh[key] for key in
    ["points", "triangles", "edges", "edge_tags", "edge_triangle"]]
assert triangles.shape[1] == 6 and edges.shape[1] == 3
chord = metadata["chord"]
alpha = np.deg2rad(args.alpha)
flow = np.array([np.cos(alpha), np.sin(alpha)])
center = np.array(args.center) if args.center is not None else np.array(metadata["farfield_center"])
contour = mesh["contour"]
relative = (contour[:, 0]-center[0])+1j*(contour[:, 1]-center[1])
winding = np.angle(np.roll(relative, -1)/relative).sum()
assert abs(abs(winding)-2.*np.pi) < 1e-8, "Circulation center must lie inside the body"
gauss, weights_1d = leggauss(args.quadrature)
unit = (gauss+1.)/2.
weights_1d = weights_1d/2.
xi, eta_base = np.meshgrid(unit, unit, indexing="ij")
reference = np.column_stack([xi.ravel(), ((1.-xi)*eta_base).ravel()])
quadrature_weights = ((weights_1d[:, None]*weights_1d[None])*(1.-xi)).ravel()
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
    assert metadata["shape"] == "joukowski" and args.mach == 0.
    c = complex(-metadata["epsilon"], metadata["eta"])
    a = abs(1.-c)
    z = points[fixed, 0]+1j*points[fixed, 1]
    discriminant = np.sqrt(z*z-4.)
    roots1, roots2 = (z+discriminant)/2., (z-discriminant)/2.
    zeta = np.where(abs(roots1-c) > abs(roots2-c), roots1, roots2)
    w = zeta-c
    boundary_base = np.real(np.exp(-1j*alpha)*w+a**2*np.exp(1j*alpha)/w)
    boundary_k = np.angle((w/(1.-c))/((points[fixed, 0]-center[0])+1j*(points[fixed, 1]-center[1])))
reference_vertices = np.array([[0., 0.], [1., 0.], [0., 1.]])
te_node = int(mesh["trailing_node"])
te_edges = mesh["trailing_edges"]
kutta_row = np.zeros(len(points))
for sign, edge_index in zip([1., -1.], te_edges):
    element = adjacent[edge_index]
    local_vertex = int(np.flatnonzero(triangles[element, :3] == te_node)[0])
    _, te_reference_gradient = shape_functions(reference_vertices[local_vertex][None])
    te_jacobian = element_points[element].T@te_reference_gradient[0]
    te_gradient = te_reference_gradient[0]@np.linalg.inv(te_jacobian)
    kutta_row[triangles[element]] += sign*te_gradient[:, 0]
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
                        kutta_residual=kutta_residual, circulation_ccw=2.*np.pi*k,
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
              circulation_center=center.tolist(), circulation_ccw=2.*np.pi*k,
              cl_circulation=-4.*np.pi*k/chord, kutta_residual=float(kutta_row@psi),
              quadrature_order=args.quadrature, ncpu_requested=args.ncpu, history=history,
              mesh_sha256=hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
              solver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              protocol_sha256=hashlib.sha256(Path(__file__).with_name("PROTOCOL.md").read_bytes()).hexdigest())
if result["converged"]:
    wall = np.flatnonzero(tags == 10)
    wall_elements = adjacent[wall]
    local_left = np.array([np.flatnonzero(triangles[e, :3] == edges[b, 0])[0]
                           for b, e in zip(wall, wall_elements)])
    local_right = np.array([np.flatnonzero(triangles[e, :3] == edges[b, 1])[0]
                            for b, e in zip(wall, wall_elements)])
    edge_reference = ((1.-unit)[None, :, None]*reference_vertices[local_left, None]
                      +unit[None, :, None]*reference_vertices[local_right, None])
    boundary_shape, boundary_reference_gradient = shape_functions(edge_reference.reshape(-1, 2))
    boundary_shape = boundary_shape.reshape(len(wall), len(unit), 6)
    boundary_reference_gradient = boundary_reference_gradient.reshape(len(wall), len(unit), 6, 2)
    boundary_jacobian = np.einsum("eia,eqib->eqab", element_points[wall_elements], boundary_reference_gradient)
    boundary_gradient = np.einsum("eqib,eqba->eqia", boundary_reference_gradient,
                                   np.linalg.inv(boundary_jacobian))
    boundary_point = np.einsum("eqi,eia->eqa", boundary_shape, element_points[wall_elements])
    edge_derivative = np.column_stack([-3.+4.*unit, -1.+4.*unit, 4.-8.*unit])
    tangent = np.einsum("qi,eia->eqa", edge_derivative, points[edges[wall]])
    normal_ds = np.stack([tangent[:, :, 1], -tangent[:, :, 0]], axis=2)
    toward_cell = element_points[wall_elements, :3].mean(axis=1)-boundary_point.mean(axis=1)
    flip = np.sum(normal_ds.mean(axis=1)*toward_cell, axis=1) > 0.
    normal_ds[flip] *= -1.
    boundary_weight = np.linalg.norm(tangent, axis=2)*weights_1d
    normal = normal_ds/np.linalg.norm(normal_ds, axis=2)[:, :, None]
    boundary_delta = boundary_point-center
    boundary_h = (np.stack([-boundary_delta[:, :, 1], boundary_delta[:, :, 0]], axis=2)
                  /np.sum(boundary_delta**2, axis=2)[:, :, None])
    boundary_velocity = (np.einsum("eqia,ei->eqa", boundary_gradient, psi[triangles[wall_elements]])
                         +k*boundary_h)
    boundary_speed2 = np.sum(boundary_velocity**2, axis=2)
    boundary_beta = 1.+.2*args.mach**2*(1.-boundary_speed2)
    assert np.all(boundary_beta > 0.)
    boundary_mach2 = args.mach**2*boundary_speed2/boundary_beta
    cp = 1.-boundary_speed2 if args.mach == 0. else (
        2.*np.expm1(3.5*np.log1p(.2*args.mach**2*(1.-boundary_speed2)))/(1.4*args.mach**2))
    boundary_vn = np.sum(boundary_velocity*normal, axis=2)
    boundary_rho = boundary_beta**2.5
    force = np.einsum("eq,eqa,q->a", cp, normal_ds, weights_1d)/chord
    result.update(cl_pressure=float(force@np.array([-np.sin(alpha), np.cos(alpha)])),
                  cd_pressure=float(force@flow), pressure_force_coefficient=force.tolist(),
                  wall_normal_rms=float(np.sqrt(np.average(boundary_vn**2, weights=boundary_weight))),
                  wall_normal_max=float(np.max(abs(boundary_vn))),
                  wall_raw_mass_flux=float(np.sum(boundary_rho*boundary_vn*boundary_weight)),
                  max_local_mach=float(np.sqrt(max(mach2.max(), boundary_mach2.max()))),
                  cp_min=float(cp.min()), cp_max=float(cp.max()),
                  min_quadrature_jacobian=float(determinant.min()))
    if boundary_mach2.max() >= 1.:
        result.update(status="surface_sonic_state_outside_elliptic_solver", converged=False)
    exact_cp = np.full(cp.shape, np.nan)
    if metadata["shape"] == "joukowski" and args.mach == 0.:
        c = complex(-metadata["epsilon"], metadata["eta"])
        a = abs(1.-c)
        z = boundary_point[:, :, 0]+1j*boundary_point[:, :, 1]
        discriminant = np.sqrt(z*z-4.)
        roots1, roots2 = (z+discriminant)/2., (z-discriminant)/2.
        zeta = np.where(abs(roots1-c) > abs(roots2-c), roots1, roots2)
        w = zeta-c
        k_exact = 2.*a*np.sin(np.angle(1.-c)-alpha)
        exact_velocity = ((np.exp(-1j*alpha)-a*a*np.exp(1j*alpha)/w**2-1j*k_exact/w)
                          /(1.-1./zeta**2))
        exact_cp = 1.-abs(exact_velocity)**2
        result.update(cl_exact=-4.*np.pi*k_exact/chord, circulation_ccw_exact=2.*np.pi*k_exact,
                      cp_rmse_exact=float(np.sqrt(np.average((cp-exact_cp)**2, weights=boundary_weight))),
                      cp_max_error_exact=float(np.max(abs(cp-exact_cp))))
    np.savez_compressed(args.out/"field.npz", points=points, triangles=triangles, psi=psi, k=k,
                        velocity=velocity, rho=rho, nodal_residual=residual,
                        surface_point=boundary_point, surface_velocity=boundary_velocity,
                        surface_normal=normal, surface_weight=boundary_weight, cp=cp, exact_cp=exact_cp)
result["elapsed_seconds"] = time.perf_counter()-started
(args.out/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2), flush=True)
if not result["converged"]:
    raise SystemExit(2)
