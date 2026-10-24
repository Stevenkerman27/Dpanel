"""Generate measured tables and standalone plots for the potential-model decision."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
out = root/"results"/"cylinder_v1"
study = json.loads((out/"summary.json").read_text())
rows = study["full_potential"]
text = ["# 全势原型实测", "", "预设范围与门禁见 PROTOCOL.md；本表不代表三维或跨声速验证。", "",
        "| Case | 未知量 | 非线性步数 | 时间/s | 散度残差 RMS | 壁面法向重构 RMS | Cp 最小值 | 状态 |",
        "|---|---:|---:|---:|---:|---:|---:|---|"]
for row in rows:
    if row["converged"]:
        final = row["history"][-1]
        text.append(f"| {row['case']} | {row['unknowns']} | {len(row['history'])-1} | "
                    f"{row['elapsed_seconds']:.4g} | {final['residual_divergence_rms']:.3g} | "
                    f"{row['reconstructed_wall_normal_rms']:.4g} | {row['cp_min']:.6g} | "
                    f"{row['passed_pilot_gates']} |")
    else:
        text.append(f"| {row['case']} | {row['unknowns']} | {len(row['history'])-1} | "
                    f"{row['elapsed_seconds']:.4g} | - | - | - | {row['status']} |")
text += ["", "## M=0 解析解", "",
         "| Nr x Nt | Cp RMSE（有限环域解析解） | 加密误差比 |",
         "|---|---:|---:|"]
for row in rows:
    if row["mach"] == 0.:
        ratio = row.get("cp_refinement_ratio")
        ratio_text = f"{ratio:.4g}" if ratio is not None else "-"
        text.append(f"| {row['nr']} x {row['nt']} | {row['cp_rmse_finite_annulus_exact']:.6g} | {ratio_text} |")
text += ["", "## 面元参考", "",
         "以下为无限域圆周均匀多边形的特殊对称配点结果，存在特殊精确性，不能泛化到任意几何。",
         "计时含首次线性代数初始化等波动；未知量、远场边界和几何均不同，不作公平性能排名。", "",
         "| 面数 | Cp RMSE | 法向 RMS | 时间/s |", "|---|---:|---:|---:|"]
for row in study["panel"]:
    text.append(f"| {row['panels']} | {row['cp_rmse']:.3g} | {row['normal_rms']:.3g} | "
                f"{row['elapsed_seconds']:.4g} |")
base = np.load(out/"M0.3_r20_128x256"/"field.npz")
text += ["", "## 远场敏感性", "",
         "保持近壁径向步长近似一致，周向点数均为 256，与 R=20 比较：", "",
         "| R | Cp 差 RMS | Cp 最大差 |", "|---|---:|---:|"]
for case in ["M0.3_r10_98x256","M0.3_r40_158x256"]:
    row = next(item for item in rows if item["case"] == case)
    other = np.load(out/case/"field.npz")
    difference = base["cp"]-other["cp"]
    text.append(f"| {row['far_radius']:g} | {np.sqrt(np.mean(difference**2)):.6g} | "
                f"{np.max(abs(difference)):.6g} |")
text += ["", "M>0 没有独立可压缩解析/实验参考，本轮仅验证方程一致性、守恒、低 Mach 极限及敏感性。",
         "壁面离散零通量与独立重构梯度的法向误差分开报告，后者保留在压力公式中。",
         "局部声速超限是当前椭圆型算法的适用范围检查，不是全势方程的通用求解能力上限。"]
(root/"RESULTS.md").write_text("\n".join(text)+"\n",encoding="utf-8")
fig,ax = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for mach in [0.,.1,.2,.3]:
    field = np.load(out/f"M{mach:g}_r20_128x256"/"field.npz")
    ax[0,0].plot(np.rad2deg(field["theta"]),field["cp"],label=f"M={mach:g}")
ax[0,0].set(xlabel="Theta (deg)",ylabel="Cp",title="Full potential: compressibility changes the model")
ax[0,0].legend()
zero = [row for row in rows if row["mach"] == 0.]
ax[0,1].loglog([row["nr"] for row in zero],
               [row["cp_rmse_finite_annulus_exact"] for row in zero],"o-",label="Cp RMSE")
ax[0,1].loglog([row["nr"] for row in zero],
               [row["reconstructed_wall_normal_rms"] for row in zero],"s-",label="Independent wall normal RMS")
ax[0,1].set(xlabel="Radial cells (Ntheta = 2 Nr)",ylabel="Error",title="M=0: refinement against finite-domain exact solution")
ax[0,1].legend()
for mach in [.1,.2,.3]:
    row = next(item for item in rows if item["mach"] == mach and item["nr"] == 128 and item["far_radius"] == 20.)
    h = row["history"]
    ax[1,0].semilogy([step["iteration"] for step in h],
                     [step["residual_divergence_rms"] for step in h],label=f"M={mach:g}")
ax[1,0].axhline(1e-9,color="gray",ls=":")
ax[1,0].set(xlabel="Picard iteration",ylabel="Nonlinear mass-divergence RMS",title="Residual uses updated density")
ax[1,0].legend()
for radius,nr in [(10,98),(20,128),(40,158)]:
    field = np.load(out/f"M0.3_r{radius}_{nr}x256"/"field.npz")
    ax[1,1].plot(np.rad2deg(field["theta"]),field["cp"],label=f"R={radius}")
ax[1,1].set(xlabel="Theta (deg)",ylabel="Cp",title="M=0.3: far-field location is a measurable error source")
ax[1,1].legend()
for axis in ax.flat:
    axis.grid(True,alpha=.2)
fig.savefig(root/"results"/"pilot.png",dpi=180)
fig.savefig(root/"results"/"pilot.pdf")
print(root/"RESULTS.md")
