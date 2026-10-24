"""Subsonic conservative FV in log-radius coordinates with matched cusp derivatives."""
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
parser.add_argument("--surface-order", type=int, choices=[2, 4], default=4)
parser.add_argument("--wall-degree", type=int, choices=[2, 3, 4], default=2)
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
alpha = np.deg2rad(args.alpha)
gamma = 1.4
c = complex(-args.epsilon, args.eta)
a = abs(1.-c)
theta0 = np.angle(1.-c)
sf = np.linspace(0., np.log(args.radius), nr+1)
sc = (sf[:-1]+sf[1:])/2.
ds = sf[1]-sf[0]
dt = 2.*np.pi/nt
theta = theta0+(np.arange(nt)+.5)*dt
w = a*np.exp(sc[:, None]+1j*theta)
zeta = c+w
physical_metric = abs(w*(1.-1./zeta**2))
volume = physical_metric**2*ds*dt
wall_w = a*np.exp(1j*theta)
wall_zeta = c+wall_w
wall_z = wall_zeta+1./wall_zeta
wall_map = 1.-1./wall_zeta**2
wall_metric = abs(wall_w*wall_map)
wall_dzdt = wall_map*1j*wall_w
wall_ds = abs(wall_dzdt)*dt
normal_ds = np.column_stack([wall_dzdt.imag, -wall_dzdt.real])*dt
dense_zeta = c+a*np.exp(1j*(theta0+np.linspace(0., 2.*np.pi, 65537)))
dense_z = dense_zeta+1./dense_zeta
chord = float(np.ptp(dense_z.real))
outer_w = a*args.radius*np.exp(1j*theta)
outer_zeta = c+outer_w
if args.boundary == "exact":
    boundary = np.real(np.exp(-1j*alpha)*outer_w+a**2*np.exp(1j*alpha)/outer_w)
else:
    boundary = np.real(np.exp(-1j*alpha)*(outer_zeta+1./outer_zeta))
index = np.arange(nr*nt).reshape(nr, nt)
left_r, right_r = index[:-1].ravel(), index[1:].ravel()
left_t, right_t = index.ravel(), np.roll(index, -1, axis=1).ravel()
left = np.concatenate([left_r, left_t])
right = np.concatenate([right_r, right_t])
outer = index[-1]
layers = args.wall_degree+1
vandermonde = np.vander(sc[:layers], layers, increasing=True)
value_rhs = np.zeros(layers)
value_rhs[0] = 1.
derivative_rhs = np.zeros(layers)
derivative_rhs[1] = 1.
value_weight = np.linalg.solve(vandermonde.T, value_rhs)
derivative_weight = np.linalg.solve(vandermonde.T, derivative_rhs)
theta_weight = np.zeros(nt)
theta_weight[0], theta_weight[-1] = 1./dt, -1./dt
if args.surface_order == 4:
    theta_weight[0], theta_weight[-1] = 9./(8.*dt), -9./(8.*dt)
    theta_weight[1], theta_weight[-2] = -1./(24.*dt), 1./(24.*dt)
active = np.flatnonzero(theta_weight)
kutta_values = (value_weight[:, None]*theta_weight[active]).ravel()
kutta_columns = index[:layers, active].ravel()
kutta_row = coo_matrix((kutta_values, (np.zeros(len(kutta_values), dtype=int), kutta_columns)),
                       shape=(1, nr*nt)).tocsr()
rho = np.ones((nr, nt))
history = []
status = "iteration_limit"
for iteration in range(args.maxiter):
    coefficient_r = ((rho[:-1]+rho[1:])/2.*dt/ds).ravel()
    coefficient_t = ((rho+np.roll(rho, -1, axis=1))/2.*ds/dt).ravel()
    coefficient = np.concatenate([coefficient_r, coefficient_t])
    outer_coefficient = rho[-1]*2.*dt/ds
    diagonal = np.bincount(np.concatenate([left, right]),
                           weights=np.concatenate([coefficient, coefficient]), minlength=nr*nt)
    diagonal[outer] += outer_coefficient
    rows = np.concatenate([np.arange(nr*nt), left, right])
    cols = np.concatenate([np.arange(nr*nt), right, left])
    values = np.concatenate([diagonal, -coefficient, -coefficient])
    matrix = coo_matrix((values, (rows, cols)), shape=(nr*nt, nr*nt)).tocsr()
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
    phi_s = np.gradient(psi, ds, axis=0, edge_order=2)
    phi_t = (np.roll(psi, -1, axis=1)-np.roll(psi, 1, axis=1))/(2.*dt)+k
    wall_psi = value_weight@psi[:layers]
    wall_phi_s = derivative_weight@psi[:layers]
    wall_phi_t = (np.roll(wall_psi, -1)-np.roll(wall_psi, 1))/(2.*dt)+k
    if args.surface_order == 4:
        phi_t = (np.roll(psi, 2, axis=1)-8.*np.roll(psi, 1, axis=1)
                 +8.*np.roll(psi, -1, axis=1)-np.roll(psi, -2, axis=1))/(12.*dt)+k
        wall_phi_t = (np.roll(wall_psi, 2)-8.*np.roll(wall_psi, 1)
                      +8.*np.roll(wall_psi, -1)-np.roll(wall_psi, -2))/(12.*dt)+k
    q2 = (phi_s**2+phi_t**2)/physical_metric**2
    wall_vr, wall_vt = wall_phi_s/wall_metric, wall_phi_t/wall_metric
    wall_q2 = wall_vr**2+wall_vt**2
    beta = 1.+.5*(gamma-1.)*args.mach**2*(1.-q2)
    wall_beta = 1.+.5*(gamma-1.)*args.mach**2*(1.-wall_q2)
    kutta_residual = float(theta_weight@wall_psi+k)
    if min(beta.min(), wall_beta.min()) <= 0.:
        status = "nonpositive_isentropic_state"
        history.append(dict(iteration=iteration, kutta_residual=kutta_residual))
        break
    next_rho = beta**(1./(gamma-1.))
    max_mach = float(np.sqrt(max((args.mach**2*q2/beta).max(),
                                 (args.mach**2*wall_q2/wall_beta).max())))
    if max_mach >= 1.:
        status = "local_sonic_state_outside_elliptic_pilot"
        history.append(dict(iteration=iteration, kutta_residual=kutta_residual, max_local_mach=max_mach))
        break
    radial_flux = (next_rho[:-1]+next_rho[1:])/2.*dt/ds*(psi[1:]-psi[:-1])
    angular_flux = ((next_rho+np.roll(next_rho, -1, axis=1))/2.*ds/dt
                    *(np.roll(psi, -1, axis=1)-psi+k*dt))
    net = angular_flux-np.roll(angular_flux, 1, axis=1)
    net[:-1] += radial_flux
    net[1:] -= radial_flux
    outer_flux = next_rho[-1]*2.*dt/ds*(boundary-psi[-1])
    net[-1] += outer_flux
    divergence = net/volume
    residual_rms = float(np.sqrt(np.average(divergence**2, weights=volume)))
    residual_max = float(np.max(abs(divergence)))
    rho = next_rho
    history.append(dict(iteration=iteration, residual_divergence_rms=residual_rms,
                        residual_divergence_max=residual_max, kutta_residual=kutta_residual,
                        normalized_net_mass_flux=float(outer_flux.sum()/chord), max_local_mach=max_mach,
                        circulation_ccw=2.*np.pi*k,
                        linear_relative_residual=float(np.linalg.norm(coupled@solution-rhs)/np.linalg.norm(rhs))))
    if residual_rms <= 1e-9 and residual_max <= 1e-7 and abs(kutta_residual) <= 1e-10:
        status = "converged_subsonic"
        break

accepted = status == "converged_subsonic"
k_exact = 2.*a*np.sin(theta0-alpha)
exact_velocity = ((np.exp(-1j*alpha)-a**2*np.exp(1j*alpha)/wall_w**2-1j*k_exact/wall_w)/wall_map)
exact_cp = 1.-abs(exact_velocity)**2
result = dict(method="Conservative log-radius FV with matched-order cusp derivative",
              scope="2D Joukowski; entirely subsonic; not a general geometry or shock solver",
              status=status, converged=accepted, nr=nr, nt=nt, unknowns=nr*nt+1,
              mach=args.mach, alpha_deg=args.alpha, epsilon=args.epsilon, eta=args.eta,
              far_radius=args.radius, boundary=args.boundary, chord=chord,
              surface_order=args.surface_order, wall_degree=args.wall_degree, ncpu_requested=args.ncpu,
              circulation_ccw=2.*np.pi*k, circulation_ccw_exact=2.*np.pi*k_exact,
              cl_circulation=-4.*np.pi*k/chord, cl_exact=-4.*np.pi*k_exact/chord,
              kutta_residual=kutta_residual, artificial_dissipation="none",
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              protocol_sha256=hashlib.sha256(Path(__file__).with_name("LOG_PROTOCOL.md").read_bytes()).hexdigest(),
              elapsed_seconds=time.perf_counter()-started, history=history)
if accepted:
    cp = 1.-wall_q2 if args.mach == 0. else (
        2.*np.expm1(gamma/(gamma-1.)*np.log(wall_beta))/(gamma*args.mach**2))
    force = -np.sum(cp[:, None]*normal_ds, axis=0)/chord
    result.update(cl_pressure=float(force@np.array([-np.sin(alpha), np.cos(alpha)])),
                  cd_pressure=float(force@np.array([np.cos(alpha), np.sin(alpha)])),
                  pressure_force_coefficient=force.tolist(), cp_min=float(cp.min()), cp_max=float(cp.max()),
                  reconstructed_wall_normal_rms=float(np.sqrt(np.average(wall_vr**2, weights=wall_ds))),
                  reconstructed_wall_normal_max=float(np.max(abs(wall_vr))), max_local_mach=max_mach,
                  wall_normal_definition="Independent polynomial derivative in s, retained in Cp without projection")
    if args.mach == 0.:
        result.update(cp_rmse_exact=float(np.sqrt(np.average((cp-exact_cp)**2, weights=wall_ds))),
                      cp_max_error_exact=float(np.max(abs(cp-exact_cp))))
    exact_psi = np.real(np.exp(-1j*alpha)*w+a**2*np.exp(1j*alpha)/w)
    exact_sample_vr = (derivative_weight@exact_psi[:layers])/wall_metric
    if args.mach == 0. and args.boundary == "exact":
        result.update(exact_sample_wall_normal_max=float(np.max(abs(exact_sample_vr))),
                      solution_contribution_wall_normal_max=float(np.max(abs(wall_vr-exact_sample_vr))))
    np.savez_compressed(args.out/"field.npz", sf=sf, sc=sc, theta=theta, psi=psi, k=k, rho=rho,
                        volume=volume, physical_metric=physical_metric, phi_s=phi_s, phi_t=phi_t,
                        wall_psi=wall_psi, wall_vr=wall_vr, wall_vt=wall_vt, wall_z=wall_z,
                        wall_ds=wall_ds, cp=cp, exact_cp=exact_cp, boundary=boundary,
                        radial_flux=radial_flux, angular_flux=angular_flux, outer_flux=outer_flux)
(args.out/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2), flush=True)
if not accepted:
    raise SystemExit(2)
