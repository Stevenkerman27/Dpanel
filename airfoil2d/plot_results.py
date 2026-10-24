"""Read the saved FEM and log-FV studies and write comparison figures."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent.parent
general_dir = root / "airfoil2d/results/general_v1"
invariance_path = root / "airfoil2d/results/invariance_v1/summary.json"
log_dir = root / "potential/results/log_v1"
log_tests_path = root / "potential/log_tests.json"
general = json.loads((general_dir / "summary.json").read_text(encoding="utf-8"))
invariance = json.loads(invariance_path.read_text(encoding="utf-8"))
log_summary = json.loads((log_dir / "summary.json").read_text(encoding="utf-8"))
log_tests = json.loads(log_tests_path.read_text(encoding="utf-8"))
fem = general["results"]
log_records = log_summary["records"]

fem_refinement = [fem[name] for name in
                  ["02_j08_a5_m0", "05_j04_a5_m0", "08_j02_a5_m0"]]
log_refinement = [row for row in log_records
                  if row["boundary"] == "exact" and row["mach"] == 0.0
                  and row["alpha_deg"] == 5.0 and row["epsilon"] == 0.08
                  and row["eta"] == 0.0 and row["surface_order"] == 4
                  and row["wall_degree"] == 2 and row["nt"] == 4 * row["nr"]]
log_refinement.sort(key=lambda row: row["nr"])
log_ref = next(row for row in log_records
               if row["boundary"] == "exact" and row["mach"] == 0.0
               and row["alpha_deg"] == 5.0 and row["epsilon"] == 0.08
               and row["eta"] == 0.0 and row["nr"] == 128 and row["nt"] == 512
               and row["surface_order"] == 4 and row["wall_degree"] == 2)
fem_ref = fem["08_j02_a5_m0"]

figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
axes[0].plot([row["elements"] for row in fem_refinement],
             [row["cl_pressure"] for row in fem_refinement], "o-", label="P2 FEM pressure")
axes[0].plot([row["elements"] for row in fem_refinement],
             [row["cl_circulation"] for row in fem_refinement], "s--", label="P2 FEM circulation")
axes[0].plot([row["unknowns"] for row in log_refinement],
             [row["cl_pressure"] for row in log_refinement], "o-", label="log-FV pressure")
axes[0].plot([row["unknowns"] for row in log_refinement],
             [row["cl_circulation"] for row in log_refinement], "s--", label="log-FV circulation")
axes[0].axhline(log_ref["cl_exact"], color="black", linewidth=1, linestyle=":", label="analytic CL")
axes[0].set_xscale("log")
axes[0].set_xlabel("离散规模（FEM: 单元数；FV: 未知量数）")
axes[0].set_ylabel("升力系数 CL")
axes[0].grid(True, which="both", alpha=.25)
axes[0].legend(fontsize=8)
axes[0].set_title("Joukowski, alpha=5 deg, M=0")

axes[1].plot([row["elements"] for row in fem_refinement],
             [row["cd_pressure"] for row in fem_refinement], "o-", label="P2 FEM")
axes[1].plot([row["unknowns"] for row in log_refinement],
             [row["cd_pressure"] for row in log_refinement], "s-", label="log-FV")
axes[1].axhline(0., color="black", linewidth=1, linestyle=":")
axes[1].set_xscale("log")
axes[1].set_xlabel("离散规模（FEM: 单元数；FV: 未知量数）")
axes[1].set_ylabel("压力阻力系数 CD")
axes[1].grid(True, which="both", alpha=.25)
axes[1].legend()
axes[1].set_title("压力积分阻力")
figure.savefig(general_dir / "coefficients.png", dpi=180)
plt.close(figure)

fem_field = np.load(general_dir / "08_j02_a5_m0/field.npz")
log_field = np.load(log_dir / log_ref["case"] / "field.npz")
fem_point = fem_field["surface_point"].reshape(-1, 2)
fem_cp = fem_field["cp"].ravel()
fem_exact_cp = fem_field["exact_cp"].ravel()
fem_xc = (fem_point[:, 0] - fem_point[:, 0].min()) / fem_ref["normalization_length"]
fem_order = np.argsort(fem_xc)
log_x = log_field["wall_z"].real
log_xc = (log_x - log_x.min()) / log_ref["chord"]
log_order = np.argsort(log_xc)

figure, axis = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
axis.scatter(log_xc[log_order], log_field["exact_cp"][log_order], color="black",
             s=8, marker=".", label="解析 Cp")
axis.scatter(fem_xc[fem_order], fem_cp[fem_order], s=13, marker="o",
             alpha=.75, label=f"P2 FEM, {fem_ref['elements']} elements")
axis.scatter(log_xc[log_order], log_field["cp"][log_order], s=8, marker=".",
             alpha=.65, label=f"log-FV, {log_ref['nr']}x{log_ref['nt']}")
axis.scatter(fem_xc[fem_order], fem_exact_cp[fem_order], s=8, marker="x",
             alpha=.4, label="FEM quadrature exact Cp")
axis.set_xlim(0., 1.)
axis.set_ylim(2.5, -3.)
axis.set_xlabel("x/c")
axis.set_ylabel("Cp")
axis.set_title("Joukowski, alpha=5 deg, M=0, exact far-field boundary")
axis.grid(True, alpha=.25)
axis.legend(fontsize=8)
figure.savefig(general_dir / "cp_comparison.png", dpi=180)
plt.close(figure)

figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
axes[0].loglog([row["elements"] for row in fem_refinement],
               [row["cp_rmse_exact"] for row in fem_refinement], "o-", label="P2 FEM")
axes[0].loglog([row["unknowns"] for row in log_refinement],
               [row["cp_rmse_exact"] for row in log_refinement], "s-", label="log-FV")
axes[0].set_xlabel("離散規模（FEM: 單元數；FV: 未知量數）")
axes[0].set_ylabel("加權物面 Cp RMSE")
axes[0].grid(True, which="both", alpha=.25)
axes[0].legend()
axes[0].set_title("相對解析物面的壓力誤差")
axes[1].loglog([row["elements"] for row in fem_refinement],
               [row["wall_normal_max"] for row in fem_refinement], "o-", label="P2 FEM")
axes[1].loglog([row["unknowns"] for row in log_refinement],
               [row["reconstructed_wall_normal_max"] for row in log_refinement], "s-", label="log-FV")
axes[1].set_xlabel("離散規模（FEM: 單元數；FV: 未知量數）")
axes[1].set_ylabel("壁面法向速度最大值 / U_inf")
axes[1].grid(True, which="both", alpha=.25)
axes[1].legend()
axes[1].set_title("獨立重構法向速度；方法定义见报告")
figure.savefig(general_dir / "wall_error_convergence.png", dpi=180)
plt.close(figure)

log_iteration = next(row for row in log_records
                     if row["boundary"] == "uniform" and row["mach"] == 0.2
                     and row["alpha_deg"] == 5.0 and row["nr"] == 128
                     and row["nt"] == 512 and row["far_radius"] == 20.0
                     and row["surface_order"] == 4 and row["wall_degree"] == 2)
history = log_iteration["history"]
iteration = [row["iteration"] for row in history]
residual_rms = [row.get("residual_divergence_rms", np.nan) for row in history]
residual_max = [row.get("residual_divergence_max", np.nan) for row in history]

figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
axes[0].semilogy(iteration, residual_rms, "o-", label="log-FV divergence RMS")
axes[0].semilogy(iteration, residual_max, "s-", label="log-FV divergence max")
axes[0].set_xlabel("非线性迭代")
axes[0].set_ylabel("密度加权控制体残差")
axes[0].grid(True, which="both", alpha=.25)
axes[0].legend()
axes[0].set_title("uniform boundary, M=0.2, 128x512")
axes[1].loglog([row["nr"] * row["nt"] + 1 for row in log_refinement],
               [row["history"][-1]["residual_divergence_rms"] for row in log_refinement],
               "o-", label="最终 divergence RMS")
axes[1].loglog([row["nr"] * row["nt"] + 1 for row in log_refinement],
               [row["history"][-1]["residual_divergence_max"] for row in log_refinement],
               "s-", label="最终 divergence max")
axes[1].set_xlabel("未知量数")
axes[1].set_ylabel("最终密度加权控制体残差")
axes[1].grid(True, which="both", alpha=.25)
axes[1].legend()
axes[1].set_title("exact boundary, M=0, alpha=5 deg")
figure.savefig(log_dir / "log_convergence.png", dpi=180)
plt.close(figure)

print("general checks:", general["passed"], "/", general["total"])
print("invariance checks:", invariance["passed"], "/", invariance["total"])
print("log cases:", sum(row["converged"] for row in log_records), "/", len(log_records), "converged")
print("log numerical tests:", sum(row["passed"] for row in log_tests), "/", len(log_tests), "passed")
print("figures:", general_dir / "coefficients.png", general_dir / "cp_comparison.png",
      general_dir / "wall_error_convergence.png", log_dir / "log_convergence.png")
