"""Independent Prandtl lifting-line reference and saved-solver comparison."""
import hashlib
import json
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

root = Path(__file__).resolve().parent
airfoil_root = root.parent / "airfoil2d"
result_root = root / "results/lifting_line_v2"
protocol = root / "LL_PROTOCOL.md"
solver_source = root / "solve.py"
mesh_metadata_path = root / "results/invariance_v1/base/mesh.json"
assert not result_root.exists(), "Preserve prior lifting_line_v1 evidence"
result_root.mkdir(parents=True)
threadpool_limiter = threadpool_limits(limits=8)
started = time.perf_counter()
u_inf = 1.
alpha_l0 = 0.
b_ar4 = 4.
c_ar4 = 1.
area_ar4 = b_ar4*c_ar4
aspect_ratio_ar4 = b_ar4**2/area_ar4
alpha_values = [-5., 0., 5.]
orders = [16, 32, 64]
reference_2d_path = airfoil_root / "results/general_v1/20_n12_02_a5_m0/result.json"
reference_2d = json.loads(reference_2d_path.read_text())
assert reference_2d["mach"] == 0. and reference_2d["alpha_deg"] == 5.
assert abs(reference_2d["cl_pressure"]-0.5990206726228917) < 1.e-14
a0_naca = reference_2d["cl_pressure"]/np.deg2rad(5.)
a0_models = {"thin_airfoil_2pi_per_rad": 2.*np.pi,
             "naca0012_measured_slope_per_rad": a0_naca}

patch_records = []
gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(256)
theta_quad = (gauss_nodes+1.)*np.pi/2.
theta_weights = gauss_weights*np.pi/2.
elliptic_root_chord = 4.*b_ar4/(np.pi*aspect_ratio_ar4)
elliptic_area = np.pi*b_ar4*elliptic_root_chord/4.
elliptic_ar = b_ar4**2/elliptic_area
patch_alpha = np.deg2rad(5.)
for model_name, a0 in a0_models.items():
    for order in orders:
        theta = np.arange(1, order+1)*np.pi/(order+1)
        mode = np.arange(1, order+1)
        sine = np.sin(theta[:, None]*mode[None, :])
        elliptic_chord = elliptic_root_chord*np.sin(theta)
        matrix = (4.*b_ar4/(a0*elliptic_chord[:, None])*sine
                  +mode[None, :]*sine/np.sin(theta)[:, None])
        rhs = np.full(order, patch_alpha-alpha_l0)
        exact_coefficients = np.zeros(order)
        exact_coefficients[0] = patch_alpha/(1.+4.*b_ar4/(a0*elliptic_root_chord))
        equation_residual = matrix@exact_coefficients-rhs
        numeric_coefficients = np.linalg.solve(matrix, rhs)
        cl_exact_patch = np.pi*elliptic_ar*exact_coefficients[0]
        cdi_exact_patch = np.pi*elliptic_ar*np.sum(mode*exact_coefficients**2)
        gamma_quad = 2.*b_ar4*u_inf*(np.sin(theta_quad[:, None]*mode[None, :])
                                     @exact_coefficients)
        induced_angle_quad = (np.sin(theta_quad[:, None]*mode[None, :])
                              @(mode*exact_coefficients))/np.sin(theta_quad)
        cl_quad = (2./(u_inf*elliptic_area)
                   *np.sum(theta_weights*gamma_quad*(b_ar4/2.)*np.sin(theta_quad)))
        cdi_quad = (2./(u_inf**2*elliptic_area)
                    *np.sum(theta_weights*gamma_quad*(u_inf*induced_angle_quad)
                            *(b_ar4/2.)*np.sin(theta_quad)))
        patch_records.append(dict(
            model=model_name, a0_per_rad=float(a0), order=order,
            elliptic_root_chord=float(elliptic_root_chord), elliptic_area=float(elliptic_area),
            aspect_ratio=float(elliptic_ar), alpha_deg=5.,
            exact_A1=float(exact_coefficients[0]),
            numeric_A1=float(numeric_coefficients[0]),
            max_exact_higher_mode=float(np.max(abs(exact_coefficients[1:]))) if order > 1 else 0.,
            max_numeric_higher_mode=float(np.max(abs(numeric_coefficients[1:]))) if order > 1 else 0.,
            max_exact_equation_residual=float(np.max(abs(equation_residual))),
            max_numeric_equation_residual=float(np.max(abs(matrix@numeric_coefficients-rhs))),
            cl_modal=float(cl_exact_patch), cl_theta_integral=float(cl_quad),
            cdi_modal=float(cdi_exact_patch), cdi_theta_integral=float(cdi_quad),
            cl_relative_error=float(abs(cl_quad-cl_exact_patch)/abs(cl_exact_patch)),
            cdi_relative_error=float(abs(cdi_quad-cdi_exact_patch)/abs(cdi_exact_patch)),
            passed=bool(np.max(abs(equation_residual)) < 1.e-12
                        and abs(numeric_coefficients[0]-exact_coefficients[0]) < 1.e-12
                        and np.max(abs(numeric_coefficients[1:])) < 1.e-12
                        and abs(cl_quad-cl_exact_patch)/abs(cl_exact_patch) < 1.e-12
                        and abs(cdi_quad-cdi_exact_patch)/abs(cdi_exact_patch) < 1.e-12)))

rectangular_records = []
rectangular_coefficients = {}
for model_name, a0 in a0_models.items():
    for order in orders:
        theta = np.arange(1, order+1)*np.pi/(order+1)
        mode = np.arange(1, order+1)
        sine = np.sin(theta[:, None]*mode[None, :])
        matrix = (4.*b_ar4/(a0*c_ar4)*sine
                  +mode[None, :]*sine/np.sin(theta)[:, None])
        for alpha_deg in alpha_values:
            alpha = np.deg2rad(alpha_deg)
            coefficients = np.linalg.solve(matrix, np.full(order, alpha-alpha_l0))
            cl = np.pi*aspect_ratio_ar4*coefficients[0]
            cdi = np.pi*aspect_ratio_ar4*np.sum(mode*coefficients**2)
            gamma_integral = (np.sin(theta_quad[:, None]*mode[None, :])
                              @coefficients)*(2.*b_ar4*u_inf)
            induced_angle_integral = ((np.sin(theta_quad[:, None]*mode[None, :])
                                      @(mode*coefficients))/np.sin(theta_quad))
            cl_from_integral = (2./(u_inf*area_ar4)
                                *np.sum(theta_weights*gamma_integral*(b_ar4/2.)
                                        *np.sin(theta_quad)))
            cdi_from_integral = (2./(u_inf**2*area_ar4)
                                 *np.sum(theta_weights*gamma_integral*(u_inf*induced_angle_integral)
                                         *(b_ar4/2.)*np.sin(theta_quad)))
            key = (model_name, order, alpha_deg)
            rectangular_coefficients[key] = coefficients
            rectangular_records.append(dict(
                model=model_name, a0_per_rad=float(a0), order=order,
                aspect_ratio=aspect_ratio_ar4, chord=c_ar4, span=b_ar4,
                alpha_deg=alpha_deg, alpha_rad=float(alpha), alpha_l0_rad=alpha_l0,
                coefficients=coefficients.tolist(), cl=float(cl), cdi=float(cdi),
                cl_theta_integral=float(cl_from_integral),
                cdi_theta_integral=float(cdi_from_integral),
                max_linear_residual=float(np.max(abs(matrix@coefficients-alpha))),
                cl_integral_difference=float(cl_from_integral-cl),
                cdi_integral_difference=float(cdi_from_integral-cdi)))

convergence_records = []
for model_name in a0_models:
    for alpha_deg in [-5., 5.]:
        n32 = next(row for row in rectangular_records
                   if row["model"] == model_name and row["order"] == 32
                   and row["alpha_deg"] == alpha_deg)
        n64 = next(row for row in rectangular_records
                   if row["model"] == model_name and row["order"] == 64
                   and row["alpha_deg"] == alpha_deg)
        cl_relative_change = abs(n32["cl"]-n64["cl"])/abs(n64["cl"])
        cdi_relative_change = abs(n32["cdi"]-n64["cdi"])/abs(n64["cdi"])
        convergence_records.append(dict(
            model=model_name, alpha_deg=alpha_deg,
            cl_N32=n32["cl"], cl_N64=n64["cl"],
            cl_relative_change=float(cl_relative_change),
            cdi_N32=n32["cdi"], cdi_N64=n64["cdi"],
            cdi_relative_change=float(cdi_relative_change),
            passed=bool(cl_relative_change < 1.e-3 and cdi_relative_change < 1.e-3)))

symmetry_records = []
for model_name in a0_models:
    for order in orders:
        negative = next(row for row in rectangular_records
                        if row["model"] == model_name and row["order"] == order
                        and row["alpha_deg"] == -5.)
        zero = next(row for row in rectangular_records
                    if row["model"] == model_name and row["order"] == order
                    and row["alpha_deg"] == 0.)
        positive = next(row for row in rectangular_records
                        if row["model"] == model_name and row["order"] == order
                        and row["alpha_deg"] == 5.)
        symmetry_records.append(dict(
            model=model_name, order=order,
            cl_zero=zero["cl"], cl_antisymmetry_error=positive["cl"]+negative["cl"],
            max_zero_coefficient=float(np.max(abs(
                rectangular_coefficients[(model_name, order, 0.)]))),
            passed=bool(abs(zero["cl"]) < 1.e-12
                        and abs(positive["cl"]+negative["cl"]) < 1.e-12
                        and np.max(abs(rectangular_coefficients[(model_name, order, 0.)])) < 1.e-12)))

high_ar_records = []
for aspect_ratio in [8., 20.]:
    span = aspect_ratio
    area = span
    for model_name, a0 in a0_models.items():
        for alpha_deg in alpha_values:
            theta = np.arange(1, 65)*np.pi/65.
            mode = np.arange(1, 65)
            sine = np.sin(theta[:, None]*mode[None, :])
            matrix = (4.*span/(a0*1.)*sine
                      +mode[None, :]*sine/np.sin(theta)[:, None])
            coefficients = np.linalg.solve(matrix, np.full(64, np.deg2rad(alpha_deg)))
            high_ar_records.append(dict(
                aspect_ratio=aspect_ratio, span=span, chord=1., area=area,
                model=model_name, a0_per_rad=float(a0), order=64, alpha_deg=alpha_deg,
                coefficients=coefficients.tolist(),
                cl=float(np.pi*aspect_ratio*coefficients[0]),
                cdi=float(np.pi*aspect_ratio*np.sum(mode*coefficients**2))))

mesh_metadata = json.loads(mesh_metadata_path.read_text())
three_d_cases = [
    ("alpha_p5", root/"results/invariance_v1/base/solve/result.json"),
    ("alpha_0", root/"results/invariance_v1/zero/solve/result.json"),
    ("alpha_m5", root/"results/invariance_v1/negative/solve/result.json"),
]
three_d_records = []
three_d_gamma_plot = {}
three_d_mesh_hashes = []
three_d_solver_hashes = []
three_d_p2_hashes = []
three_d_protocol_hashes = []
for label, result_path in three_d_cases:
    saved = json.loads(result_path.read_text())
    stations = np.asarray(saved["gamma_stations"])
    gamma_potential = np.asarray(saved["potential_jump"])
    chord = float(mesh_metadata["chord"])
    span = float(mesh_metadata["span"])
    area = float(mesh_metadata["reference_area"])
    gamma_nondimensional = gamma_potential/(u_inf*chord)
    cl_jump_integral = 2.*np.trapezoid(gamma_potential, stations)/(u_inf*area)
    theta_existing = np.arccos(np.clip(2.*stations/span, -1., 1.))
    interior = (stations > -span/2.+1.e-10) & (stations < span/2.-1.e-10)
    projection_order = 12
    projection_mode = np.arange(1, projection_order+1)
    projection_matrix = np.sin(theta_existing[interior, None]*projection_mode[None, :])
    projection_coefficients = np.linalg.lstsq(
        projection_matrix,
        gamma_potential[interior]/(2.*span*u_inf),
        rcond=None)[0]
    projected_cdi = (np.pi*(span**2/area)
                     *np.sum(projection_mode*projection_coefficients**2))
    predicted_at_stations = {}
    gamma_errors = {}
    ll_coefficients = {}
    for model_name, a0 in a0_models.items():
        model_row = next(row for row in rectangular_records
                         if row["model"] == model_name and row["order"] == 64
                         and row["alpha_deg"] == saved["alpha_deg"])
        coefficients = np.asarray(model_row["coefficients"])
        gamma_ll = 2.*span*u_inf*(np.sin(theta_existing[:, None]
                                          *np.arange(1, 65)[None, :])@coefficients)
        gamma_ll_nondimensional = gamma_ll/(u_inf*chord)
        predicted_at_stations[model_name] = gamma_ll_nondimensional.tolist()
        gamma_errors[model_name] = dict(
            rmse=float(np.sqrt(np.mean((gamma_ll_nondimensional-gamma_nondimensional)**2))),
            max_abs=float(np.max(abs(gamma_ll_nondimensional-gamma_nondimensional))))
        ll_coefficients[model_name] = dict(cl=model_row["cl"], cdi=model_row["cdi"])
    three_d_records.append(dict(
        case=label, source=str(result_path.relative_to(root)),
        source_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
        mesh_sha256=saved["mesh_sha256"], alpha_deg=saved["alpha_deg"],
        solver_sha256=saved["solver_sha256"], p2_sha256=saved["p2_sha256"],
        protocol_sha256=saved["protocol_sha256"],
        mach=saved["mach"], span=span, chord=chord, reference_area=area,
        cl_jump_saved=saved.get("cl_jump"), cl_jump_from_saved_distribution=float(cl_jump_integral),
        cl_pressure=saved.get("cl_pressure"), cd_pressure=saved.get("cd_pressure"),
        wall_normal_rms=saved.get("wall_normal_rms"),
        gamma_stations=stations.tolist(), gamma_over_u_chord=gamma_nondimensional.tolist(),
        lifting_line_at_stations=predicted_at_stations, gamma_errors=gamma_errors,
        lifting_line_coefficients=ll_coefficients,
        projected_3d_gamma_fourier_order=projection_order,
        projected_3d_gamma_coefficients=projection_coefficients.tolist(),
        projected_3d_gamma_cdi=float(projected_cdi),
        projected_gamma_rmse=float(np.sqrt(np.mean(
            (projection_matrix@projection_coefficients
             -gamma_potential[interior]/(2.*span*u_inf))**2)))))
    three_d_mesh_hashes.append(saved["mesh_sha256"])
    three_d_solver_hashes.append(saved["solver_sha256"])
    three_d_p2_hashes.append(saved["p2_sha256"])
    three_d_protocol_hashes.append(saved["protocol_sha256"])
    if saved["alpha_deg"] == 5.:
        three_d_gamma_plot["three_d_saved"] = (stations/(span/2.), gamma_nondimensional)

high_resolution_path = root/"results/paired_v1/wing015_curv32_a5/result.json"
high_resolution = json.loads(high_resolution_path.read_text())
high_resolution_mesh_metadata_path = root/"meshes/wing015_curv32/mesh.json"
high_resolution_mesh_metadata = json.loads(high_resolution_mesh_metadata_path.read_text())
high_resolution_stations = np.asarray(high_resolution["gamma_stations"])
high_resolution_gamma = np.asarray(high_resolution["potential_jump"])/(u_inf*1.)
three_d_gamma_plot["three_d_curv32"] = (high_resolution_stations/2., high_resolution_gamma)

theta_plot = np.linspace(0., np.pi, 401)
for model_name in a0_models:
    record = next(row for row in rectangular_records
                  if row["model"] == model_name and row["order"] == 64
                  and row["alpha_deg"] == 5.)
    coefficients = np.asarray(record["coefficients"])
    gamma = 2.*b_ar4*u_inf*(np.sin(theta_plot[:, None]
                                  *np.arange(1, 65)[None, :])@coefficients)
    y_normalized = np.cos(theta_plot)
    three_d_gamma_plot[model_name] = (y_normalized, gamma/(u_inf*c_ar4))

figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
for label, (y_normalized, gamma_normalized) in three_d_gamma_plot.items():
    axis.plot(y_normalized, gamma_normalized, label=label)
axis.set_xlabel("2y / b")
axis.set_ylabel("Gamma / (U c)")
axis.set_title("AR=4, alpha=5 deg: saved 3D circulation and Prandtl lifting-line")
axis.grid(True, alpha=.3)
axis.legend()
figure.savefig(result_root/"gamma_ar4_alpha5.png", dpi=180)
plt.close(figure)

gamma_csv_data = np.column_stack([
    np.asarray(three_d_gamma_plot["three_d_saved"][0]),
    np.asarray(three_d_gamma_plot["three_d_saved"][1]),
    np.interp(np.asarray(three_d_gamma_plot["three_d_saved"][0]),
              three_d_gamma_plot["thin_airfoil_2pi_per_rad"][0],
              three_d_gamma_plot["thin_airfoil_2pi_per_rad"][1]),
    np.interp(np.asarray(three_d_gamma_plot["three_d_saved"][0]),
              three_d_gamma_plot["naca0012_measured_slope_per_rad"][0],
              three_d_gamma_plot["naca0012_measured_slope_per_rad"][1])])
np.savetxt(result_root/"gamma_ar4_alpha5.csv", gamma_csv_data, delimiter=",",
           header="2y_over_b,gamma3d_over_Uc,gamma_LL_2pi_over_Uc,gamma_LL_NACA0012_over_Uc",
           comments="")

assert len(set(three_d_mesh_hashes)) == 1, "Invariance alpha cases do not share one mesh"
assert len(set(three_d_solver_hashes)) == 1, "Invariance alpha cases mix solver versions"
assert len(set(three_d_p2_hashes)) == 1, "Invariance alpha cases mix P2 versions"
assert len(set(three_d_protocol_hashes)) == 1, "Invariance alpha cases mix protocols"
assert abs(mesh_metadata["span"]-4.) < 1.e-14
assert abs(mesh_metadata["chord"]-1.) < 1.e-14
assert abs(mesh_metadata["reference_area"]-4.) < 1.e-14
assert abs(high_resolution_mesh_metadata["span"]-4.) < 1.e-14
assert abs(high_resolution_mesh_metadata["chord"]-1.) < 1.e-14
assert abs(high_resolution_mesh_metadata["thickness"]-0.12) < 1.e-14
summary = dict(
    protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    python_executable=os.path.abspath(os.sys.executable),
    python_executable_sha256=hashlib.sha256(Path(os.sys.executable).read_bytes()).hexdigest(),
    current_lifting3d_solver_sha256=hashlib.sha256(solver_source.read_bytes()).hexdigest(),
    two_d_reference_path=str(reference_2d_path.relative_to(root.parent)),
    two_d_reference_sha256=hashlib.sha256(reference_2d_path.read_bytes()).hexdigest(),
    two_d_cl_alpha5=float(reference_2d["cl_pressure"]),
    alpha5_rad=float(np.deg2rad(5.)), a0_models={key: float(value) for key, value in a0_models.items()},
    thread_count_requested=8, threadpools=threadpool_info(),
    aspect_ratio_ar4=aspect_ratio_ar4, elliptic_patch=patch_records,
    rectangular=rectangular_records, convergence=convergence_records,
    symmetry=symmetry_records, high_aspect_ratio=high_ar_records,
    three_d_mesh_metadata_path=str(mesh_metadata_path.relative_to(root)),
    three_d_mesh_metadata_sha256=hashlib.sha256(mesh_metadata_path.read_bytes()).hexdigest(),
    three_d_source_mesh_sha256=three_d_mesh_hashes[0],
    three_d_source_solver_sha256=three_d_solver_hashes[0],
    three_d_source_p2_sha256=three_d_p2_hashes[0],
    three_d_source_protocol_sha256=three_d_protocol_hashes[0],
    three_d_comparison=three_d_records,
    high_resolution_3d_source=str(high_resolution_path.relative_to(root)),
    high_resolution_3d_sha256=hashlib.sha256(high_resolution_path.read_bytes()).hexdigest(),
    high_resolution_mesh_metadata_path=str(high_resolution_mesh_metadata_path.relative_to(root)),
    high_resolution_mesh_metadata_sha256=hashlib.sha256(
        high_resolution_mesh_metadata_path.read_bytes()).hexdigest(),
    high_resolution_geometry=dict(
        shape=high_resolution_mesh_metadata["shape"],
        thickness=high_resolution_mesh_metadata["thickness"],
        mesh_size=high_resolution_mesh_metadata["size"],
        span=high_resolution_mesh_metadata["span"],
        chord=high_resolution_mesh_metadata["chord"]),
    elapsed_seconds=time.perf_counter()-started)
summary["elliptic_patch_passed"] = all(row["passed"] for row in patch_records)
summary["rectangular_convergence_passed"] = all(row["passed"] for row in convergence_records)
summary["symmetry_passed"] = all(row["passed"] for row in symmetry_records)
(result_root/"summary.json").write_text(
    json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
print("elliptic patch", summary["elliptic_patch_passed"],
      "rectangular N32/N64", summary["rectangular_convergence_passed"],
      "symmetry", summary["symmetry_passed"], flush=True)
print("a0 models", summary["a0_models"], flush=True)
print("3D mesh shared", len(set(three_d_mesh_hashes)) == 1,
      "elapsed_seconds", summary["elapsed_seconds"], flush=True)
