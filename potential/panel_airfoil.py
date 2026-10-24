"""Independent 2D Hess-Smith source/vortex panel reference for a Joukowski airfoil."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits


parser = argparse.ArgumentParser()
parser.add_argument("--panels", type=int, default=128)
parser.add_argument("--alpha", type=float, default=5.0, help="angle of attack in degrees")
parser.add_argument("--epsilon", type=float, default=0.08)
parser.add_argument("--eta", type=float, default=0.0)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if args.panels < 8 or args.panels % 2:
    parser.error("--panels must be an even integer of at least 8")
if not np.isfinite([args.alpha, args.epsilon, args.eta]).all() or args.epsilon <= 0:
    parser.error("--alpha and --eta must be finite; --epsilon must be finite and positive")

threadpool_limits(8)
started = time.perf_counter()
args.out.mkdir(parents=True, exist_ok=True)

alpha = np.deg2rad(args.alpha)
center_circle = -args.epsilon + 1j * args.eta
radius = abs(1.0 - center_circle)
theta0 = np.angle(1.0 - center_circle)
parameter = np.linspace(0.0, 2.0 * np.pi, args.panels + 1)
circle = center_circle + radius * np.exp(1j * (theta0 + parameter))
vertices_complex = circle + 1.0 / circle
vertices = np.column_stack([vertices_complex.real, vertices_complex.imag])
parameter_dense = np.linspace(0.0, 2.0 * np.pi, max(65537, args.panels * 256 + 1))
circle_dense = center_circle + radius * np.exp(1j * (theta0 + parameter_dense))
surface_dense = circle_dense + 1.0 / circle_dense
chord_dense = float(surface_dense.real.max() - surface_dense.real.min())
first, last = vertices[:-1], vertices[1:]
edge = last - first
length = np.linalg.norm(edge, axis=1)
tangent = edge / length[:, None]
normal = np.column_stack([tangent[:, 1], -tangent[:, 0]])
center = (first + last) / 2.0
perimeter = length.sum()
polygon_chord = vertices[:, 0].max() - vertices[:, 0].min()
chord = chord_dense
if not np.isfinite(vertices).all() or np.any(length <= 1e-12) or chord <= 0:
    raise ValueError("Joukowski mapping produced degenerate panels or a nonpositive chord")

# Exact constant-panel line-integral kernels; the normal trace is exterior -1/2.
delta = center[:, None, :] - first[None, :, :]
s = np.einsum("ijk,jk->ij", delta, tangent)
h = np.einsum("ijk,jk->ij", delta, normal)
source_t = -np.log((s * s + h * h) / ((s - length[None, :]) ** 2 + h * h)) / (4.0 * np.pi)
source_n = -np.arctan2(h * length[None, :], h * h + s * (s - length[None, :])) / (2.0 * np.pi)
np.fill_diagonal(source_t, 0.0)
np.fill_diagonal(source_n, -0.5)
source_velocity = source_t[:, :, None] * tangent[None, :, :] + source_n[:, :, None] * normal[None, :, :]
vortex_velocity = np.stack([source_velocity[:, :, 1], -source_velocity[:, :, 0]], axis=2)
flow = np.array([np.cos(alpha), np.sin(alpha)])

# Solve n impermeability equations and one equal-sign trailing-edge Kutta equation.
matrix = np.zeros((args.panels + 1, args.panels + 1))
rhs = np.zeros(args.panels + 1)
matrix[:args.panels, :args.panels] = np.einsum("ijk,ik->ij", source_velocity, normal)
matrix[:args.panels, -1] = np.einsum("ijk,ik->i", vortex_velocity, normal)
rhs[:args.panels] = -(normal @ flow)
matrix[-1, :args.panels] = np.einsum("ijk,ik->j", source_velocity[[0, -1]], tangent[[0, -1]])
matrix[-1, -1] = np.einsum("ijk,ik->", vortex_velocity[[0, -1]], tangent[[0, -1]])
rhs[-1] = -(flow @ (tangent[0] + tangent[-1]))
solution = solve(matrix, rhs, assume_a="gen")
sigma, vortex_strength = solution[:-1], solution[-1]

velocity = flow + np.einsum("ijk,j->ik", source_velocity, sigma) + vortex_strength * vortex_velocity.sum(axis=1)
cp = 1.0 - np.einsum("ij,ij->i", velocity, velocity)
normal_residual = np.einsum("ij,ij->i", velocity, normal)
kutta_residual = float(velocity[[0, -1]][0] @ tangent[0] + velocity[[0, -1]][1] @ tangent[-1])
source_flux = float(sigma @ length)
force = -np.sum(cp[:, None] * normal * length[:, None], axis=0) / chord
lift_direction = np.array([-np.sin(alpha), np.cos(alpha)])
cl_pressure = float(force @ lift_direction)
cd_pressure = float(force @ flow)
circulation = float(vortex_strength * perimeter)
cl_circulation = float(-2.0 * circulation / chord)

# Evaluate the exact mapped-cylinder flow at each mapped parameter midpoint.
mid_parameter = (parameter[:-1] + parameter[1:]) / 2.0
zeta_mid = center_circle + radius * np.exp(1j * (theta0 + mid_parameter))
w_mid = zeta_mid - center_circle
z_mid = zeta_mid + 1.0 / zeta_mid
dz_dzeta = 1.0 - 1.0 / (zeta_mid * zeta_mid)
gamma_exact = float(4.0 * np.pi * radius * np.sin(theta0 - alpha))
k_exact = gamma_exact / (2.0 * np.pi)
complex_velocity_exact = (
    np.exp(-1j * alpha)
    - radius * radius * np.exp(1j * alpha) / (w_mid * w_mid)
    - 1j * k_exact / w_mid
) / dz_dzeta
cp_exact = 1.0 - np.abs(complex_velocity_exact) ** 2

result = dict(
    method="2D Hess-Smith constant source panels plus one constant surface vortex sheet",
    scope="Incompressible inviscid potential flow; Joukowski airfoil; exterior Neumann collocation",
    panels=args.panels,
    alpha_deg=args.alpha,
    epsilon=args.epsilon,
    eta=args.eta,
    circulation_convention="positive counterclockwise",
    vortex_strength=float(vortex_strength),
    circulation=float(circulation),
    circulation_exact=float(gamma_exact),
    circulation_error=float(circulation - gamma_exact),
    circulation_relative_error=(
        float(abs(circulation - gamma_exact) / abs(gamma_exact))
        if abs(gamma_exact) > 1e-12
        else None
    ),
    chord=float(chord),
    polygon_chord=float(polygon_chord),
    chord_dense=float(chord_dense),
    cl_pressure=cl_pressure,
    cd_pressure=cd_pressure,
    cl_circulation=cl_circulation,
    cl_exact=float(-2.0 * gamma_exact / chord_dense),
    cl_pressure_exact_error=float(cl_pressure - (-2.0 * gamma_exact / chord_dense)),
    cl_circulation_exact_error=float(cl_circulation - (-2.0 * gamma_exact / chord_dense)),
    cp_rmse=float(np.sqrt(np.average((cp - cp_exact) ** 2, weights=length))),
    cp_exact_reference="analytic mapped-cylinder velocity at mapped parameter midpoints",
    cp_polygon_midpoint_note="Panel-center geometry differs from mapped-arc midpoint; pressure error includes this polygon sampling difference.",
    normal_residual_rms=float(np.sqrt(np.average(normal_residual ** 2, weights=length))),
    normal_residual_max=float(np.max(np.abs(normal_residual))),
    kutta_residual=float(kutta_residual),
    source_flux=float(source_flux),
    normalized_source_flux=float(source_flux / perimeter),
    condition_1=float(np.linalg.cond(matrix, 1)),
    elapsed_seconds=float(time.perf_counter() - started),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
)
np.savez_compressed(
    args.out / "surface.npz",
    vertices=vertices,
    first=first,
    last=last,
    center=center,
    tangent=tangent,
    normal=normal,
    length=length,
    sigma=sigma,
    vortex_strength=vortex_strength,
    velocity=velocity,
    cp=cp,
    cp_exact=cp_exact,
    normal_residual=normal_residual,
    exact_velocity=np.column_stack([complex_velocity_exact.real, -complex_velocity_exact.imag]),
    exact_surface_x=z_mid.real,
    exact_surface_y=z_mid.imag,
)
(args.out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps(result, indent=2))
