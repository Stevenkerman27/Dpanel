"""Independent line quadrature and FV balance checks; no additional solver runs."""
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

root = Path(__file__).resolve().parent
point, weight = leggauss(96)
first = np.array([.2,-.1])
last = np.array([1.1,.4])
edge = last-first
length = np.linalg.norm(edge)
tangent = edge/length
normal = np.array([tangent[1],-tangent[0]])
targets = np.array([[.5,.7],[.4,-.6],[1.8,.8]])
delta = targets-first
s, h = delta@tangent, delta@normal
vt = -np.log((s*s+h*h)/((s-length)**2+h*h))/(4.*np.pi)
vn = -np.arctan2(h*length,h*h+s*(s-length))/(2.*np.pi)
analytic = vt[:,None]*tangent+vn[:,None]*normal
quad_points = first+(point[:,None]+1.)/2.*edge
r = targets[:,None]-quad_points[None]
quadrature = -np.sum(weight[None,:,None]*length/2.*r/np.sum(r*r,axis=2)[:,:,None],axis=1)/(2.*np.pi)
error = float(np.max(abs(analytic-quadrature)))
records = [dict(test="2D_line_gradient_vs_independent_quadrature",error=error,passed=error < 1e-12)]
study = root/"results"/"cylinder_v1"
summary = json.loads((study/"summary.json").read_text())
for row in summary["full_potential"]:
    if not row["converged"]:
        continue
    field = np.load(study/row["case"]/"field.npz")
    rc,rf,theta,phi,rho = [field[key] for key in ["rc","rf","theta","phi","rho"]]
    dt = 2.*np.pi/len(theta)
    radial_flux = ((rho[:-1]+rho[1:])/2.*rf[1:-1,None]*dt
                   *(phi[1:]-phi[:-1])/np.diff(rc)[:,None])
    angular_flux = ((rho+np.roll(rho,-1,axis=1))/2.*np.diff(rf)[:,None]/rc[:,None]
                    *(np.roll(phi,-1,axis=1)-phi)/dt)
    net = angular_flux-np.roll(angular_flux,1,axis=1)
    net[:-1] += radial_flux
    net[1:] -= radial_flux
    outer_flux = rho[-1]*rf[-1]*dt*(rf[-1]*np.cos(theta)-phi[-1])/(rf[-1]-rc[-1])
    net[-1] += outer_flux
    volume = .5*(rf[1:]**2-rf[:-1]**2)[:,None]*np.full((1,len(theta)),dt)
    rms = float(np.sqrt(np.average((net/volume)**2,weights=volume)))
    peak = float(np.max(abs(net/volume)))
    discrepancy = float(abs(net.sum()-outer_flux.sum()))
    final = row["history"][-1]
    # Matrix assembly and oriented face summation are independently evaluated.
    records.append(dict(test="FV_oriented_face_balance_"+row["case"],rms=rms,maximum=peak,
                        global_cancellation_error=discrepancy,
                        rms_difference_from_matrix=abs(rms-final["residual_divergence_rms"]),
                        passed=bool(rms < 1.1e-9 and peak < 1.1e-7 and discrepancy < 1e-10)))
    if row["mach"] > 0.:
        q2 = field["vr"]**2+field["vt"]**2
        sound2 = rho**(row["gamma"]-1.)/row["mach"]**2
        enthalpy = sound2/(row["gamma"]-1.)+q2/2.
        reference = 1./((row["gamma"]-1.)*row["mach"]**2)+.5
        relative = float(np.max(abs(enthalpy-reference))/reference)
        records.append(dict(test="isentropic_enthalpy_"+row["case"],relative_error=relative,
                            passed=relative < 1e-12))
(root/"model_tests.json").write_text(json.dumps(records,indent=2,allow_nan=False),encoding="utf-8")
assert all(row["passed"] for row in records)
print(f"{len(records)} independent quadrature, face-flux and enthalpy checks passed.")
