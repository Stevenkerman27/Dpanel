"""Independent transformed-gradient, polynomial and stored shared-flux checks."""
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
out = root/"results"/"log_v1"
records = []
theta = np.array([.3, .9, 1.7, 4.2, 5.7])
s = np.array([.1, .3, .6, 1.2, .04])
delta = 1e-6
c = complex(-.08, .04)
a = abs(1.-c)
alpha = np.deg2rad(5.)
k = 2.*a*np.sin(np.angle(1.-c)-alpha)
w = a*np.exp(s+1j*theta)
zeta = c+w
dzds = (1.-1./zeta**2)*w
complex_velocity = (np.exp(-1j*alpha)-a**2*np.exp(1j*alpha)/w**2-1j*k/w)/(1.-1./zeta**2)
wsp, wsm = w*np.exp(delta), w*np.exp(-delta)
wtp, wtm = w*np.exp(1j*delta), w*np.exp(-1j*delta)
ps = np.real(np.exp(-1j*alpha)*(wsp-wsm)+a**2*np.exp(1j*alpha)*(1./wsp-1./wsm))/(2.*delta)
pt = np.real(np.exp(-1j*alpha)*(wtp-wtm)+a**2*np.exp(1j*alpha)*(1./wtp-1./wtm))/(2.*delta)+k
error = float(np.max(abs((ps**2+pt**2)/abs(dzds)**2-abs(complex_velocity)**2)))
records.append(dict(test="log_gradient_vs_analytic_complex_velocity", max_q2_error=error, passed=error < 1e-8))
zsp, zsm = c+wsp+1./(c+wsp), c+wsm+1./(c+wsm)
ztp, ztm = c+wtp+1./(c+wtp), c+wtm+1./(c+wtm)
zs, zt = (zsp-zsm)/(2.*delta), (ztp-ztm)/(2.*delta)
jacobian = zs.real*zt.imag-zs.imag*zt.real
error = float(np.max(abs(jacobian/abs(dzds)**2-1.)))
records.append(dict(test="physical_area_Jacobian_by_coordinate_differences",
                    relative_error=error, passed=error < 1e-8))
dt = .01
t_samples = np.array([-.015, -.005, .005, .015])
weights = np.array([1., -27., 27., -1.])/(24.*dt)
for degree in range(5):
    derivative = float(weights@t_samples**degree)
    expected = 1. if degree == 1 else 0.
    error = abs(derivative-expected)
    records.append(dict(test=f"cusp_fourth_order_polynomial_degree_{degree}", error=error, passed=error < 1e-12))
summary = json.loads((out/"summary.json").read_text())
for row in summary["records"]:
    if not row["converged"]:
        continue
    f = np.load(out/row["case"]/"field.npz")
    psi, rho, sc, sf = [f[key] for key in ["psi", "rho", "sc", "sf"]]
    ds = sf[1]-sf[0]
    dt = 2.*np.pi/row["nt"]
    angular_gradient = np.diff(np.column_stack([psi, psi[:, 0]]), axis=1)/dt+float(f["k"])
    radial_gradient = np.diff(psi, axis=0)/ds
    fr = (rho[:-1]+rho[1:])/2.*radial_gradient*dt
    ft = (rho+np.roll(rho, -1, axis=1))/2.*angular_gradient*ds
    fo = rho[-1]*(f["boundary"]-psi[-1])/(ds/2.)*dt
    net = ft-np.roll(ft, 1, axis=1)
    net += np.vstack([fr, fo[None]])-np.vstack([np.zeros((1, row["nt"])), fr])
    rms = float(np.sqrt(np.average((net/f["volume"])**2, weights=f["volume"])))
    peak = float(np.max(abs(net/f["volume"])))
    difference = float(max(np.max(abs(fr-f["radial_flux"])), np.max(abs(ft-f["angular_flux"]))))
    records.append(dict(test="independent_log_shared_flux_"+row["case"], rms=rms, maximum=peak,
                        saved_flux_difference=difference,
                        passed=bool(rms < 1.1e-9 and peak < 1.1e-7 and difference < 1e-10)))
(root/"log_tests.json").write_text(json.dumps(records, indent=2, allow_nan=False), encoding="utf-8")
print(f"{sum(row['passed'] for row in records)}/{len(records)} log-coordinate checks passed.")
assert all(row["passed"] for row in records)
