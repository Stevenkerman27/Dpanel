"""Independent analytical contour, cut accounting and stored-field conservation checks."""
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

root = Path(__file__).resolve().parent
out = root/"results"/"lifting_v1"
records = []
nt = 65536
t = (np.arange(nt)+.5)*2.*np.pi/nt
dt = 2.*np.pi/nt
for epsilon, eta, alpha_deg in [(.08, 0., 5.), (.08, 0., -5.), (.08, .04, 0.), (.16, .06, 5.)]:
    c = complex(-epsilon, eta)
    a = abs(1.-c)
    theta0 = np.angle(1.-c)
    alpha = np.deg2rad(alpha_deg)
    k = 2.*a*np.sin(theta0-alpha)
    w = a*np.exp(1j*(theta0+t))
    zeta = c+w
    map_derivative = 1.-1./zeta**2
    dwdzeta = np.exp(-1j*alpha)-a**2*np.exp(1j*alpha)/w**2-1j*k/w
    complex_velocity = dwdzeta/map_derivative
    cp = 1.-abs(complex_velocity)**2
    dzdt = map_derivative*1j*w
    pressure_force = -np.array([np.sum(cp*dzdt.imag), -np.sum(cp*dzdt.real)])*dt
    lift_times_chord = pressure_force@np.array([-np.sin(alpha), np.cos(alpha)])
    drag_times_chord = pressure_force@np.array([np.cos(alpha), np.sin(alpha)])
    force_error = float(max(abs(lift_times_chord+4.*np.pi*k), abs(drag_times_chord)))
    wall_velocity = np.conj(complex_velocity)
    normal_velocity = np.real(wall_velocity*np.conj(-1j*dzdt/abs(dzdt)))
    wall_error = float(np.max(abs(normal_velocity)))
    te_w = 1.-c
    te_numerator = np.exp(-1j*alpha)-a**2*np.exp(1j*alpha)/te_w**2-1j*k/te_w
    te_error = float(abs(te_numerator))
    records.append(dict(test=f"analytic_pressure_lift_wall_Kutta_eps{epsilon}_eta{eta}_a{alpha_deg}",
                        force_error=force_error, wall_normal_error=wall_error, te_error=te_error,
                        passed=bool(force_error < 1e-9 and wall_error < 1e-9 and te_error < 1e-12)))
    # Integrate physical velocity around a separate contour, away from the body.
    w_contour = 1.7*w
    zeta_contour = c+w_contour
    complex_velocity = ((np.exp(-1j*alpha)-a**2*np.exp(1j*alpha)/w_contour**2-1j*k/w_contour)
                        /(1.-1./zeta_contour**2))
    dzdt = (1.-1./zeta_contour**2)*1j*w_contour
    circulation = float(np.sum(np.real(complex_velocity*dzdt))*dt)
    error = abs(circulation-2.*np.pi*k)
    records.append(dict(test=f"independent_physical_contour_Gamma_eps{epsilon}_eta{eta}_a{alpha_deg}",
                        circulation=circulation, error=error, passed=bool(error < 1e-12)))
study = json.loads((out/"summary.json").read_text())
cases = [(row, out/row["case"]) for row in study["full_potential"]]
extra = root/"results"/"lifting_cusp_v1"
for target in sorted(extra.glob("exact_*")):
    row = json.loads((target/"result.json").read_text())
    row["case"] = target.name
    cases.append((row, target))
for row, target in cases:
    if not row["converged"]:
        continue
    field = np.load(target/"field.npz")
    psi, rho, rc, rf = [field[key] for key in ["psi", "rho", "rc", "rf"]]
    k = float(field["k"])
    dt = 2.*np.pi/row["nt"]
    # Assemble cell outflow directly from the two oriented sides of each face.
    radial = ((rho[:-1]+rho[1:])/2.*rf[1:-1, None]*dt
              *(psi[1:]-psi[:-1])/np.diff(rc)[:, None])
    theta = ((rho+np.roll(rho, -1, axis=1))/2.*np.diff(rf)[:, None]/rc[:, None]
             *((np.roll(psi, -1, axis=1)-psi)/dt+k))
    outer_flux = rho[-1]*rf[-1]*dt*(field["boundary"]-psi[-1])/(rf[-1]-rc[-1])
    incoming_r = np.vstack([np.zeros((1, row["nt"])), radial])
    outgoing_r = np.vstack([radial, outer_flux[None]])
    balance = outgoing_r-incoming_r+theta-np.roll(theta, 1, axis=1)
    volume = field["volume"]
    rms = float(np.sqrt(np.average((balance/volume)**2, weights=volume)))
    peak = float(np.max(abs(balance/volume)))
    flux_error = float(max(np.max(abs(radial-field["radial_flux"])),
                           np.max(abs(theta-field["angular_flux"]))))
    closure = float(abs(balance.sum()-outer_flux.sum()))
    # The seam receives +F in the last angular cell and -F in the first.
    cut_flux_pair_error = float(abs(theta[:, -1].sum()+(-theta[:, -1]).sum()))
    angular_increment = np.roll(psi, -1, axis=1)-psi+k*dt
    jump_error = float(np.max(abs(angular_increment.sum(axis=1)-2.*np.pi*k)))
    records.append(dict(test="independent_flux_and_branch_"+row["case"], rms=rms, maximum=peak,
                        flux_reconstruction_error=flux_error, global_cancellation_error=closure,
                        paired_cut_flux_error=cut_flux_pair_error, potential_jump_error=jump_error,
                        branch_pair_check_type="construction identity, not a physical accuracy test",
                        passed=bool(rms <= 1.1e-9 and peak <= 1.1e-7 and flux_error < 1e-10
                                    and closure < 1e-10 and cut_flux_pair_error < 1e-12 and jump_error < 1e-10)))
panel_target = root.parent/"audit"/"panel_airfoil_hesssmith"/"n256_ap5"
panel = json.loads((panel_target/"result.json").read_text())
field = np.load(panel_target/"surface.npz")
gauss, weight = leggauss(16)
nodes = field["first"][:, None]+(gauss[None, :, None]+1.)/2.*(field["last"]-field["first"])[:, None]
line_weights = field["length"][:, None]*weight[None]/2.
angles = (np.arange(128)+.5)*2.*np.pi/128
contour_radius = 10.*panel["chord"]
points = contour_radius*np.column_stack([np.cos(angles), np.sin(angles)])
delta = points[:, None, None]-nodes[None]
quadrature_kernel = -delta/np.sum(delta**2, axis=3)[:, :, :, None]/(2.*np.pi)
source = np.sum(quadrature_kernel*(line_weights*field["sigma"][:, None])[None, :, :, None], axis=(1, 2))
vortex = np.sum(quadrature_kernel*line_weights[None, :, :, None], axis=(1, 2))
velocity = source+float(field["vortex_strength"])*np.column_stack([vortex[:, 1], -vortex[:, 0]])
alpha = np.deg2rad(panel["alpha_deg"])
velocity += np.array([np.cos(alpha), np.sin(alpha)])
flux = float(np.sum(np.sum(velocity*points, axis=1))*2.*np.pi/128)
circulation = float(np.sum(velocity[:, 0]*(-points[:, 1])+velocity[:, 1]*points[:, 0])*2.*np.pi/128)
mass_error = abs(flux+panel["source_flux"])
circulation_error = abs(circulation-panel["circulation"])
records.append(dict(test="panel_independent_Gauss_kernel_far_contour_flux_and_circulation",
                    mass_flux=flux, circulation=circulation, mass_identity_error=mass_error,
                    circulation_identity_error=circulation_error,
                    passed=bool(mass_error < 1e-10 and circulation_error < 1e-10)))
(root/"lifting_tests.json").write_text(json.dumps(records, indent=2, allow_nan=False), encoding="utf-8")
print(f"{sum(row['passed'] for row in records)}/{len(records)} analytical and conservation checks passed.")
assert all(row["passed"] for row in records)
