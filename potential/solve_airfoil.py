"""Conservative subsonic FV with an explicit circulation unknown on a conformal grid."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import bmat, coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits

parser = argparse.ArgumentParser()
parser.add_argument("--nr", type=int, default=64)
parser.add_argument("--nt", type=int, default=256)
parser.add_argument("--radius", type=float, default=20.)
parser.add_argument("--mach", type=float, default=0.)
parser.add_argument("--alpha", type=float, default=5.)
parser.add_argument("--epsilon", type=float, default=.08)
parser.add_argument("--eta", type=float, default=0.)
parser.add_argument("--boundary", choices=["exact", "uniform"], default="exact")
parser.add_argument("--maxiter", type=int, default=100)
parser.add_argument("--ncpu", type=int, default=8)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
assert args.nr >= 8 and args.nt >= 32 and args.radius > 1. and 0. <= args.mach < 1.
assert args.epsilon > 0.
threadpool_limits(args.ncpu)
started = time.perf_counter()
args.out.mkdir(parents=True, exist_ok=True)
nr, nt = args.nr, args.nt
gamma = 1.4
alpha = np.deg2rad(args.alpha)
c = complex(-args.epsilon, args.eta)
a = abs(1.-c)
theta0 = np.angle(1.-c)
rf = np.geomspace(1., args.radius, nr+1)
rc = (rf[:-1]+rf[1:])/2.
dr = np.diff(rf)
dt = 2.*np.pi/nt
t = (np.arange(nt)+.5)*dt
theta = theta0+t
w = a*rc[:, None]*np.exp(1j*theta)
zeta = c+w
mapping_derivative = 1.-1./zeta**2
metric = abs(mapping_derivative)
volume = (.5*(rf[1:]**2-rf[:-1]**2)[:, None]*dt*a**2*metric**2)
wall_w = a*np.exp(1j*theta)
wall_zeta = c+wall_w
wall_z = wall_zeta+1./wall_zeta
wall_map = 1.-1./wall_zeta**2
wall_metric = abs(wall_map)
wall_dzdt = wall_map*1j*wall_w
wall_ds = abs(wall_dzdt)*dt
wall_normal_ds = np.column_stack([wall_dzdt.imag, -wall_dzdt.real])*dt
dense_zeta = c+a*np.exp(1j*(theta0+np.linspace(0., 2.*np.pi, 65537)))
dense_z = dense_zeta+1./dense_zeta
chord = float(dense_z.real.max()-dense_z.real.min())
outer_w = a*args.radius*np.exp(1j*theta)
outer_zeta = c+outer_w
outer_z = outer_zeta+1./outer_zeta
if args.boundary == "exact":
    boundary = np.real(np.exp(-1j*alpha)*outer_w+a**2*np.exp(1j*alpha)/outer_w)
else:
    boundary = np.real(np.exp(-1j*alpha)*outer_z)
index = np.arange(nr*nt).reshape(nr, nt)
left_r, right_r = index[:-1].ravel(), index[1:].ravel()
left_t, right_t = index.ravel(), np.roll(index, -1, axis=1).ravel()
left = np.concatenate([left_r, left_t])
right = np.concatenate([right_r, right_t])
outer = index[-1]
radial_metric = rf[1:-1]*dt/np.diff(rc)
angular_metric = dr/(rc*dt)
outer_metric = rf[-1]*dt/(rf[-1]-rc[-1])
offset = rc[:3]-1.
vandermonde = np.column_stack([np.ones(3), offset, offset**2])
wall_value_weight = np.linalg.solve(vandermonde.T, [1., 0., 0.])
wall_derivative_weight = np.linalg.solve(vandermonde.T, [0., 1., 0.])
kutta_columns = np.concatenate([index[:3, 0], index[:3, -1]])
kutta_values = np.concatenate([wall_value_weight/dt, -wall_value_weight/dt])
kutta_row = coo_matrix((kutta_values, (np.zeros(6, dtype=int), kutta_columns)),
                       shape=(1, nr*nt)).tocsr()
rho = np.ones((nr, nt))
history = []
status = "iteration_limit"
previous_psi = None
previous_k = None
for iteration in range(args.maxiter):
    coefficient_r = ((rho[:-1]+rho[1:])/2.*radial_metric[:, None]).ravel()
    coefficient_t = ((rho+np.roll(rho, -1, axis=1))/2.*angular_metric[:, None]).ravel()
    coefficient = np.concatenate([coefficient_r, coefficient_t])
    outer_coefficient = rho[-1]*outer_metric
    diagonal = np.bincount(np.concatenate([left, right]),
                           weights=np.concatenate([coefficient, coefficient]), minlength=nr*nt)
    diagonal[outer] += outer_coefficient
    rows = np.concatenate([np.arange(nr*nt), left, right])
    cols = np.concatenate([np.arange(nr*nt), right, left])
    values = np.concatenate([diagonal, -coefficient, -coefficient])
    matrix = coo_matrix((values, (rows, cols)), shape=(nr*nt, nr*nt)).tocsr()
    # Every oriented angular face sees dPhi = dpsi + k*dt, including the cut.
    circulation_column = np.bincount(np.concatenate([left_t, right_t]),
                                     weights=np.concatenate([-coefficient_t*dt, coefficient_t*dt]),
                                     minlength=nr*nt)
    coupled = bmat([[matrix, csr_matrix(circulation_column[:, None])],
                    [kutta_row, csr_matrix([[1.]])]], format="csc")
    rhs = np.zeros(nr*nt+1)
    rhs[outer] = outer_coefficient*boundary
    solution = spsolve(coupled, rhs)
    assert np.isfinite(solution).all(), "Nonfinite sparse solution"
    psi, k = solution[:-1].reshape(nr, nt), float(solution[-1])
    vr_mapped = np.gradient(psi, rc, axis=0, edge_order=2)/a
    vt_mapped = ((np.roll(psi, -1, axis=1)-np.roll(psi, 1, axis=1))/(2.*dt)+k)/(a*rc[:, None])
    q2 = (vr_mapped**2+vt_mapped**2)/metric**2
    wall_psi = wall_value_weight@psi[:3]
    wall_vr = (wall_derivative_weight@psi[:3])/(a*wall_metric)
    wall_vt = ((np.roll(wall_psi, -1)-np.roll(wall_psi, 1))/(2.*dt)+k)/(a*wall_metric)
    wall_q2 = wall_vr**2+wall_vt**2
    beta = 1.+.5*(gamma-1.)*args.mach**2*(1.-q2)
    wall_beta = 1.+.5*(gamma-1.)*args.mach**2*(1.-wall_q2)
    kutta_residual = float((wall_psi[0]-wall_psi[-1])/dt+k)
    if min(beta.min(), wall_beta.min()) <= 0.:
        status = "nonpositive_isentropic_state"
        history.append(dict(iteration=iteration, kutta_residual=kutta_residual))
        break
    next_rho = beta**(1./(gamma-1.))
    max_mach = float(np.sqrt(max((args.mach**2*q2/beta).max(),
                                 (args.mach**2*wall_q2/wall_beta).max())))
    if max_mach >= 1.:
        status = "local_sonic_state_outside_elliptic_pilot"
        history.append(dict(iteration=iteration, max_local_mach=max_mach,
                            kutta_residual=kutta_residual))
        break
    # Check the original nonlinear equation with updated density and oriented fluxes.
    radial_flux = ((next_rho[:-1]+next_rho[1:])/2.*radial_metric[:, None]*(psi[1:]-psi[:-1]))
    angular_flux = ((next_rho+np.roll(next_rho, -1, axis=1))/2.*angular_metric[:, None]
                    *(np.roll(psi, -1, axis=1)-psi+k*dt))
    net = angular_flux-np.roll(angular_flux, 1, axis=1)
    net[:-1] += radial_flux
    net[1:] -= radial_flux
    outer_flux = next_rho[-1]*outer_metric*(boundary-psi[-1])
    net[-1] += outer_flux
    divergence = net/volume
    residual_rms = float(np.sqrt(np.average(divergence**2, weights=volume)))
    residual_max = float(np.max(abs(divergence)))
    update = None if previous_psi is None else float(np.max(abs(psi-previous_psi)))
    history.append(dict(iteration=iteration, residual_divergence_rms=residual_rms,
                        residual_divergence_max=residual_max, kutta_residual=kutta_residual,
                        normalized_net_mass_flux=float(outer_flux.sum()/chord),
                        max_local_mach=max_mach, circulation_ccw=2.*np.pi*k,
                        last_potential_update=update,
                        last_circulation_update=None if previous_k is None else 2.*np.pi*(k-previous_k)))
    rho = next_rho
    if residual_rms <= 1e-9 and residual_max <= 1e-7 and abs(kutta_residual) <= 1e-10:
        status = "converged_subsonic"
        break
    previous_psi, previous_k = psi.copy(), k

accepted = status == "converged_subsonic"
k_exact = 2.*a*np.sin(theta0-alpha)
exact_complex_velocity = ((np.exp(-1j*alpha)-a**2*np.exp(1j*alpha)/wall_w**2
                           -1j*k_exact/wall_w)/wall_map)
exact_cp = 1.-abs(exact_complex_velocity)**2
result = dict(method="Conservative conformally mapped FV; circulation unknown; discrete cusp Kutta",
              scope="2D Joukowski topology; entirely subsonic; no general mesh, shock or free wake",
              status=status, converged=accepted, mach=args.mach, alpha_deg=args.alpha,
              epsilon=args.epsilon, eta=args.eta, gamma=gamma, nr=nr, nt=nt,
              far_radius=args.radius, boundary=args.boundary, chord=chord, unknowns=nr*nt+1,
              ncpu_requested=args.ncpu, circulation_ccw=2.*np.pi*k,
              circulation_ccw_exact_incompressible=2.*np.pi*k_exact,
              cl_circulation=-4.*np.pi*k/chord,
              cl_exact_incompressible=-4.*np.pi*k_exact/chord,
              potential_jump=2.*np.pi*k, kutta_residual=kutta_residual,
              artificial_dissipation="none; local sonic iterates rejected",
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              protocol_sha256=hashlib.sha256(Path(__file__).with_name("LIFTING_PROTOCOL.md").read_bytes()).hexdigest(),
              elapsed_seconds=time.perf_counter()-started, history=history)
if accepted:
    cp = 1.-wall_q2 if args.mach == 0. else (
        2.*np.expm1(gamma/(gamma-1.)*np.log(wall_beta))/(gamma*args.mach**2))
    force = -np.sum(cp[:, None]*wall_normal_ds, axis=0)/chord
    cl = float(force@np.array([-np.sin(alpha), np.cos(alpha)]))
    cd = float(force@np.array([np.cos(alpha), np.sin(alpha)]))
    result.update(cl_pressure=cl, cd_pressure=cd, pressure_force_coefficient=force.tolist(),
                  cp_min=float(cp.min()), cp_max=float(cp.max()),
                  reconstructed_wall_normal_rms=float(np.sqrt(np.average(wall_vr**2, weights=wall_ds))),
                  reconstructed_wall_normal_max=float(np.max(abs(wall_vr))),
                  max_local_mach=max_mach,
                  wall_normal_definition="Independent 3-layer quadratic derivative, retained in Cp; no TE points removed")
    if args.mach == 0.:
        result.update(cp_rmse_exact=float(np.sqrt(np.average((cp-exact_cp)**2, weights=wall_ds))),
                      cp_max_error_exact=float(np.max(abs(cp-exact_cp))),
                      cl_error_exact=abs(cl-result["cl_exact_incompressible"]),
                      cl_pressure_circulation_difference=cl-result["cl_circulation"])
    np.savez_compressed(args.out/"field.npz", rf=rf, rc=rc, theta=theta, psi=psi, k=k,
                        rho=rho, vr_mapped=vr_mapped, vt_mapped=vt_mapped, volume=volume,
                        wall_psi=wall_psi, wall_vr=wall_vr, wall_vt=wall_vt,
                        wall_z=wall_z, wall_ds=wall_ds, cp=cp, exact_cp=exact_cp,
                        boundary=boundary, radial_flux=radial_flux, angular_flux=angular_flux,
                        outer_flux=outer_flux)
(args.out/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2), flush=True)
if not accepted:
    raise SystemExit(2)
