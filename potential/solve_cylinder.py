"""Conservative, entirely subsonic full-potential pilot on a polar finite-volume grid."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits

parser = argparse.ArgumentParser()
parser.add_argument("--nr", type=int, default=64)
parser.add_argument("--nt", type=int, default=128)
parser.add_argument("--radius", type=float, default=20.)
parser.add_argument("--mach", type=float, default=.3)
parser.add_argument("--maxiter", type=int, default=100)
parser.add_argument("--ncpu", type=int, default=8)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
assert args.nr >= 8 and args.nt >= 16 and args.radius > 1. and 0. <= args.mach < 1.
threadpool_limits(args.ncpu)
started = time.perf_counter()
args.out.mkdir(parents=True, exist_ok=True)
nr, nt = args.nr, args.nt
gamma = 1.4
rf = np.geomspace(1., args.radius, nr+1)
rc = (rf[:-1]+rf[1:])/2.
dr = np.diff(rf)
dtheta = 2.*np.pi/nt
theta = (np.arange(nt)+.5)*dtheta
volume = .5*(rf[1:]**2-rf[:-1]**2)[:,None]*np.full((1,nt),dtheta)
index = np.arange(nr*nt).reshape(nr,nt)
left_r, right_r = index[:-1].ravel(), index[1:].ravel()
left_t, right_t = index.ravel(), np.roll(index,-1,axis=1).ravel()
left, right = np.concatenate([left_r,left_t]), np.concatenate([right_r,right_t])
outer = index[-1]
boundary = args.radius*np.cos(theta)
radial_metric = rf[1:-1]*dtheta/np.diff(rc)
angular_metric = dr/(rc*dtheta)
outer_metric = rf[-1]*dtheta/(rf[-1]-rc[-1])
offset = rc[:3]-1.
vandermonde = np.column_stack([np.ones(3),offset,offset**2])
wall_value_weight = np.linalg.solve(vandermonde.T,[1.,0.,0.])
wall_derivative_weight = np.linalg.solve(vandermonde.T,[0.,1.,0.])
phi = rc[:,None]*np.cos(theta)
history = []
status = "iteration_limit"
last_update = None
cp_rmse = None
for iteration in range(args.maxiter+1):
    vr = np.gradient(phi,rc,axis=0,edge_order=2)
    vt = (np.roll(phi,-1,axis=1)-np.roll(phi,1,axis=1))/(2.*dtheta*rc[:,None])
    q2 = vr**2+vt**2
    phi_wall = wall_value_weight@phi[:3]
    wall_vr = wall_derivative_weight@phi[:3]
    wall_vt = (np.roll(phi_wall,-1)-np.roll(phi_wall,1))/(2.*dtheta)
    wall_q2 = wall_vr**2+wall_vt**2
    beta = 1.+.5*(gamma-1.)*args.mach**2*(1.-q2)
    wall_beta = 1.+.5*(gamma-1.)*args.mach**2*(1.-wall_q2)
    if min(beta.min(),wall_beta.min()) <= 0.:
        status = "nonpositive_isentropic_state"
        break
    rho = beta**(1./(gamma-1.))
    mach2 = args.mach**2*q2/beta
    wall_mach2 = args.mach**2*wall_q2/wall_beta
    max_mach = float(np.sqrt(max(mach2.max(),wall_mach2.max())))
    if max_mach >= 1.:
        status = "local_sonic_state_outside_elliptic_pilot"
        history.append(dict(iteration=iteration,max_local_mach=max_mach))
        break
    # Each interior face has one coefficient shared by its two adjacent cells.
    coeff_r = ((rho[:-1]+rho[1:])/2.*radial_metric[:,None]).ravel()
    coeff_t = ((rho+np.roll(rho,-1,axis=1))/2.*angular_metric[:,None]).ravel()
    coefficient = np.concatenate([coeff_r,coeff_t])
    outer_coefficient = rho[-1]*outer_metric
    diagonal = np.bincount(np.concatenate([left,right]),
                           weights=np.concatenate([coefficient,coefficient]),minlength=nr*nt)
    diagonal[outer] += outer_coefficient
    rows = np.concatenate([np.arange(nr*nt),left,right])
    cols = np.concatenate([np.arange(nr*nt),right,left])
    values = np.concatenate([diagonal,-coefficient,-coefficient])
    matrix = coo_matrix((values,(rows,cols)),shape=(nr*nt,nr*nt)).tocsr()
    rhs = np.zeros(nr*nt)
    rhs[outer] = outer_coefficient*boundary
    balance = matrix@phi.ravel()-rhs
    divergence = balance.reshape(nr,nt)/volume
    residual_rms = float(np.sqrt(np.average(divergence**2,weights=volume)))
    residual_max = float(np.max(abs(divergence)))
    outer_flux = float(np.sum(outer_coefficient*(boundary-phi[-1])))
    history.append(dict(iteration=iteration,residual_divergence_rms=residual_rms,
                        residual_divergence_max=residual_max,last_potential_update=last_update,
                        normalized_net_mass_flux=outer_flux/(2.*np.pi),max_local_mach=max_mach))
    if residual_rms <= 1e-9 and residual_max <= 1e-7:
        status = "converged_subsonic"
        break
    if iteration == args.maxiter:
        break
    next_phi = spsolve(matrix,rhs).reshape(nr,nt)
    assert np.isfinite(next_phi).all(), "Nonfinite sparse linear solution"
    last_update = float(np.max(abs(next_phi-phi)))
    phi = next_phi

accepted = status == "converged_subsonic"
result = dict(method="Conservative polar FV; centered shared-face density; Picard sparse direct solve",
              scope="2D closed cylinder; entirely subsonic; no circulation, wake, viscosity or shock scheme",
              status=status, converged=accepted, mach=args.mach, gamma=gamma,nr=nr,nt=nt,
              unknowns=nr*nt,far_radius=args.radius,ncpu_requested=args.ncpu,
              normalization="Cylinder radius=1, Uinf=1, rhoinf=1; full total potential",
              boundary="Inner zero mass flux; outer total potential R*cos(theta)",
              artificial_dissipation="none; local sonic states explicitly rejected",
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              elapsed_seconds=time.perf_counter()-started,history=history)
if accepted:
    if args.mach == 0.:
        cp = 1.-wall_q2
    else:
        cp = 2.*np.expm1(gamma/(gamma-1.)*np.log(wall_beta))/(gamma*args.mach**2)
    force = -np.array([np.sum(cp*np.cos(theta)),np.sum(cp*np.sin(theta))])*dtheta/2.
    result.update(cp_min=float(cp.min()),cp_max=float(cp.max()),pressure_force_coefficient=force.tolist(),
                  reconstructed_wall_normal_rms=float(np.sqrt(np.mean(wall_vr**2))),
                  reconstructed_wall_normal_max=float(np.max(abs(wall_vr))),
                  wall_normal_definition="Independent quadratic derivative from first 3 cell centers; retained in Cp",
                  flux_boundary_normal_velocity=0.,
                  max_local_mach=max_mach)
    if args.mach == 0.:
        confinement = args.radius**2/(args.radius**2+1.)
        exact_cp = 1.-4.*confinement**2*np.sin(theta)**2
        exact_phi = confinement*(rc[:,None]+1./rc[:,None])*np.cos(theta)
        cp_rmse = float(np.sqrt(np.mean((cp-exact_cp)**2)))
        result.update(cp_rmse_finite_annulus_exact=cp_rmse,
                      potential_rmse_exact=float(np.sqrt(np.average((phi-exact_phi)**2,weights=volume))),
                      finite_annulus_factor=confinement)
    np.savez_compressed(args.out/"field.npz",rf=rf,rc=rc,theta=theta,phi=phi,rho=rho,
                        vr=vr,vt=vt,wall_phi=phi_wall,wall_vr=wall_vr,wall_vt=wall_vt,cp=cp)
(args.out/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
print(json.dumps({key:value for key,value in result.items() if key != "history"},indent=2),flush=True)
if not accepted:
    raise SystemExit(2)
