"""Publish measured lifting, reconstruction and independent panel evidence."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
out = root/"results"/"lifting_v1"
rows = json.loads((out/"summary.json").read_text())["full_potential"]
cusp = json.loads((root/"results"/"lifting_cusp_v1"/"summary.json").read_text())["records"]
tests = json.loads((root/"lifting_tests.json").read_text())
main = [row for row in rows if row["boundary"] == "exact" and row["mach"] == 0.
        and row["eta"] == 0. and row["alpha_deg"] == 5.]
fine = main[-1]
field = np.load(out/fine["case"]/"field.npz")
panel_root = root.parent/"audit"/"panel_airfoil_hesssmith"
panels = [json.loads((panel_root/f"n{n}_ap5"/"result.json").read_text()) for n in [32, 64, 128, 256]]
panel_field = np.load(panel_root/"n256_ap5"/"surface.npz")
far = [row for row in rows if row["boundary"] == "uniform" and row["mach"] == .2 and row["nt"] == 512]
far = sorted(far, key=lambda row: row["far_radius"])
base_far = next(row for row in far if row["far_radius"] == 20.)
base_cp = np.load(out/base_far["case"]/"field.npz")["cp"]
far_metrics = []
for row in far:
    other = np.load(out/row["case"]/"field.npz")
    difference = other["cp"]-base_cp
    far_metrics.append(dict(radius=row["far_radius"], cl=row["cl_pressure"], cd=row["cd_pressure"],
                            max_cp_difference_R20=float(np.max(abs(difference))),
                            rms_cp_difference_R20=float(np.sqrt(np.average(difference**2, weights=field["wall_ds"]))),
                            cl_difference_R20=row["cl_pressure"]-base_far["cl_pressure"]))
(out/"farfield_metrics.json").write_text(json.dumps(far_metrics, indent=2), encoding="utf-8")
text = ["# 二维升力全势验证与误差审计", "",
        "## Material Passport", "",
        "- ID：lifting-v1-2026-10-23；类型：确定性数值实验。",
        "- 状态：主序列 15 组亚声速通过指定门禁，1 组范围退出；追加 4 组局部误差诊断；独立面元 12 例。",
        "- 来源：本地独立实现、解析推导和保存的场数据；不使用 WBCAero 私有内核，也不声称已验证外部可压缩数据。",
        "- 范围：二维 Joukowski 解析映射，附着无黏无旋流；不认证任意体网格、三维翼身、自由尾迹或激波。",
        "", "## 结论", "",
        f"本阶段建立了环量未知量、尖后缘 Kutta、势函数分支切割和压力力的一致验证。"
        f"对称翼型 M=0、攻角 5 度、128x512 时，压力 CL={fine['cl_pressure']:.10f}，"
        f"解析 CL={fine['cl_exact_incompressible']:.10f}，"
        f"相对误差 {100*fine['cl_error_exact']/abs(fine['cl_exact_incompressible']):.4f}%；"
        f"数值环量给 CL={fine['cl_circulation']:.10f}。"
        "正负攻角反号与弯度零攻角升力均符合解析参考。",
        "", "该结果支持继续保守全势路线，但局部压力/梯度仍需独立审查。"
        "不能把本阶段的整体通过写成原项目厚翼身连接奇异问题已经解决。",
        "", "## 方法为何可解释", "",
        "`Phi=psi+k*t` 中 psi 是周期速度势部分，不是流函数。k 作为额外未知量与所有势未知量联立求解，"
        "没有将解析环量灌入数值解。逆时针环量 `Gamma=2*pi*k`，本符号约定下正升力要求负环量。",
        "", "Kutta 方程使映射平面尖后缘切向势导数为零，消去映射导数零点带来的速度奇异。"
        "物面零通量由 FV 边界施加；另由前三层单元独立重构法向速度，并保留在 Cp 中。"
        "接缝质量通量共享且势跳跃为 Gamma；它是多值势的分支切割，没有尾迹涡片输运或自由尾迹自由度。",
        "", "映射后质量通量中的二维保角度量抵消；计算物理速度时仍显式使用映射导数。"
        "散度残差的控制体积使用单元中心映射 Jacobian 与计算平面环扇体积，是物理面积的离散近似。"
        "没有压力裁剪、任意核半径、法向速度投影或额外耗散。方案见 LIFTING_PROTOCOL.md，"
        "独立数学审查见 LIFTING_REVIEW.md。",
        "", "## M=0 网格收敛", "",
        "| Nr x Nt | Cp 弧长加权 RMSE | Cp 最大误差 | 法向 RMS | 法向最大值 | 压力 CL | CD | 时间/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
for row in main:
    text.append(f"| {row['nr']} x {row['nt']} | {row['cp_rmse_exact']:.6g} | "
                f"{row['cp_max_error_exact']:.6g} | {row['reconstructed_wall_normal_rms']:.6g} | "
                f"{row['reconstructed_wall_normal_max']:.6g} | {row['cl_pressure']:.9f} | "
                f"{row['cd_pressure']:.4g} | {row['elapsed_seconds']:.4g} |")
text += ["", "Cp 加密误差比为 "
         +", ".join(f"{row['cp_refinement_ratio']:.5g}" for row in main if "cp_refinement_ratio" in row)
         +"；它们在首次运行前规定的 0.8 以下。"
         "最细主序列环量相对误差约 0.00687%，压力与数值环量 CL 的相对差约 0.00504%。",
         "", "## 尖后缘局部误差", "",
         "整体指标通过后，仍保留约 0.215 Uinf 的局部法向误差。追加诊断方案 CUSP_PROTOCOL.md"
         "在这批追加计算前写定，不改写 v1 的原始压力或通过状态。",
         "", "| Nr x Nt | 二次外推 vn 最大值 | 三次外推 vn 最大值 | 四次外推 vn 最大值 |",
         "|---|---:|---:|---:|"]
for nr, nt in [(128, 512), (256, 512), (512, 512), (128, 1024), (256, 1024)]:
    selected = [next(row for row in cusp if row.get("nr") == nr and row.get("nt") == nt
                     and row["wall_polynomial_degree"] == degree) for degree in [2, 3, 4]]
    text.append(f"| {nr} x {nt} | "+" | ".join(f"{row['wall_normal_max']:.6g}" for row in selected)+" |")
text += ["", "只把 Nt 从 512 加到 1024、Nr 保持 128，二次外推最大误差从 0.215 增至 0.429。"
         "样点离尖后缘更近，映射尺度更小，径向导数误差受到更强放大。"
         "保持 Nt=512，把 Nr 从 128 加到 256、512，则该误差约缩小为原来的 1/4、1/16。"
         "所以需要与几何奇点相匹配的径向/周向分辨率，不能只靠增加表面点数。",
         "", "在 128x512，二次外推对解析势样本本身也产生 0.224913 的最大法向误差，"
         "数值势相对解析样本对该导数的最大贡献为 0.010267，说明当前主要问题确实来自径向外推截断误差。"
         "但提高外推阶数后，数值势解误差会逐渐成为主要部分。",
         "", "尤其 256x512 的三次外推法向最大值只有 4.59e-5，看起来很好；"
         "其解析样本截断误差约 0.002529，而数值解贡献约 0.002575，两者恰好抵消。"
         "四次外推最大值反而约 0.002436。这是误差抵消证据，不能据此选择最小数字、宣称已修复局部边界。"
         "二/三/四次结果与分项误差全部保存于 results/lifting_cusp_v1/summary.json。",
         "", "512x512 原二次重构 Cp RMSE=8.24e-6，法向 RMS=1.95e-4、最大值=0.01397，"
         "压力 CL=0.58818812。其散度最大残差已约 9.02e-8，接近本原型 1e-7 门禁；"
         "进一步加密还需关注小单元中势差相减和线性求解误差，不能把无限加密当作鲁棒方案。",
         "", "## 非零 Mach 与远场", "",
         "| Case | 压力 CL | 环量 CL | CD | 最大局部 Mach | 密度求解次数 | 时间/s |",
         "|---|---:|---:|---:|---:|---:|---:|"]
for row in rows:
    if row["boundary"] == "uniform":
        if row["converged"]:
            text.append(f"| {row['case']} | {row['cl_pressure']:.8f} | {row['cl_circulation']:.8f} | "
                        f"{row['cd_pressure']:.4g} | {row['max_local_mach']:.6g} | "
                        f"{len(row['history'])} | {row['elapsed_seconds']:.4g} |")
        else:
            text.append(f"| {row['case']} | - | - | - | {row['history'][-1].get('max_local_mach', 0.):.6g} | "
                        f"{len(row['history'])} | 范围退出 |")
text += ["", "exact 外边界的 M=0.001 与 M=0 最大 Cp 差为 "
         +f"{next(row['low_mach_max_cp_difference'] for row in rows if row['mach'] == .001):.6g}。"
         "非零 Mach 当前仅验证非线性质量残差、低马赫一致性和敏感性；没有独立可压缩精度认证。"
         "M=0.6 在不可压初始解阶段得到局部 Mach=1.20564，退出的是中心椭圆算法，未求得跨声速解。",
         "", "R 是生成圆坐标中的半径比，不是以弦长归一化的远场距离。R=20 对应生成圆半径约 5.37 个弦长，"
         "R=40 约 10.74 个弦长；物理远场为映射曲线。"
         "M=0.2 从 R=20 到 R=40，压力 CL 从 0.59861732 变为 0.60103055，变化约 0.40%。"
         "uniform 边界的截断误差明显大于主序列 exact 边界的不可压离散误差。",
         "", "| R | 与 R=20 的最大 Cp 差 | 与 R=20 的 CL 差 |",
         "|---|---:|---:|"]
for row in far_metrics:
    text.append(f"| {row['radius']:g} | {row['max_cp_difference_R20']:.6g} | {row['cl_difference_R20']:.6g} |")
text += ["", "## 独立二维面元参考", "",
         "Luna 独立实现了常源加单一常涡片强度的 Hess-Smith 参考，共 32/64/128/256 面、"
         "-5/0/5 度 12 例。此实现采用直线 polygon 面心配点，与 FV 解析曲面几何、外边界和未知量数不同。"
         "下表不是两类方法的一般性能排名。",
         "", "| 面数 | 压力 CL | 环量 CL | Cp RMSE | 净源通量 | 矩阵 1-范数条件数 |",
         "|---|---:|---:|---:|---:|---:|"]
for row in panels:
    text.append(f"| {row['panels']} | {row['cl_pressure']:.8f} | {row['cl_circulation']:.8f} | "
                f"{row['cp_rmse']:.6g} | {row['source_flux']:.6g} | {row['condition_1']:.6g} |")
text += ["", "256 面的法向残差最大值约 6.24e-14，Kutta 残差约 1.22e-15，"
         "但压力 CL 误差仍约 5.70%，净源通量为 -0.00249。"
         "独立 Gauss 线积分在远场轮廓上测得对应的净质量流出和环量，"
         "确认源通量不是仅由面心后处理定义出来的量。"
         "加密改善精度，同时条件数增长；本轮没有用质量约束或人为正则化改写它。"
         "该参考尚未通过原项目的全局无源门禁，不能作为已完成的升力面元产品。",
         "", "此前三维升力原型的负升力异常没有因这次二维正负号验证而被定位或修复。"
         "本阶段只给出了可独立审计的二维参照链条。",
         "", "## 验证与下一步", "",
         f"{len(tests)} 项解析压力力、独立物理轮廓环量、壁面法向、共享通量重算与分支记账检查全部通过。"
         "分支配对与势跳跃部分是构造恒等式，不是独立物理精度证明；"
         "代码和方案哈希、全部命令、每步残差与失败退出状态都保存在相应 JSON/log。",
         "", "下一步先改进有推导依据的尖后缘局部重构与网格分辨率匹配，并用解析样本拆分误差认证；"
         "随后才为弱跨声速阶段实现熵条件明确的保守方向通量，并引入独立公开 Cp/激波位置参考。"
         "当前不把高阶外推的偶然抵消或简单压力修正包装成跨声速稳定化。",
         "", "三维仍需单独验证一般体网格生成、厚翼身交界拓扑、非正交面通量、分部件压力力与总成本。"
         "继续保留低速面元对照，暂停 WBCAero 大主函数整体反编译。",
         "", "## 复现", "", "```powershell",
         "& C:/Users/zyx20/anaconda3/python.exe potential/run_lifting_study.py",
         "& C:/Users/zyx20/anaconda3/python.exe potential/audit_lifting_cusp.py",
         "& C:/Users/zyx20/anaconda3/python.exe potential/panel_airfoil.py --panels 256 --alpha 5 --out audit/panel_airfoil_hesssmith/n256_ap5",
         "& C:/Users/zyx20/anaconda3/python.exe potential/test_lifting.py",
         "& C:/Users/zyx20/anaconda3/python.exe potential/report_lifting.py", "```", "",
         "请求 NCPU=8，并限制已知线程库；稀疏 SuperLU 不保证八线程。时间是当前实现的单次内部计时，"
         "不含 Python 启动、场文件压缩和网格生成工具成本；不是工程速度承诺。"]
(root/"LIFTING_RESULTS.md").write_text("\n".join(text)+"\n", encoding="utf-8")

fig, ax = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
x0 = field["wall_z"].real.min()
chord = fine["chord"]
x = (field["wall_z"].real-x0)/chord
y = field["wall_z"].imag/chord
ax[0, 0].plot(x, y, color="black")
ax[0, 0].set(xlabel="x / chord", ylabel="y / chord", title="Analytic Joukowski geometry")
ax[0, 0].set_aspect("equal", adjustable="datalim")
ax[0, 1].plot(x, field["exact_cp"], color="black", lw=1.8, label="Exact M=0")
ax[0, 1].plot(x, field["cp"], "--", color="#007c91", label="FV 128 x 512")
panel_x = (panel_field["center"][:, 0]-x0)/chord
ax[0, 1].plot(panel_x, panel_field["cp"], color="#bc472f", alpha=.8, label="Panel 256")
ax[0, 1].set(xlabel="x / chord", ylabel="Cp", title="Same airfoil and angle; different discretizations")
ax[0, 1].invert_yaxis()
ax[0, 1].legend(fontsize=8)
ax[0, 2].loglog([row["nr"] for row in main], [row["cp_rmse_exact"] for row in main],
                "o-", color="#007c91", label="Arc-weighted Cp RMSE")
ax[0, 2].loglog([row["nr"] for row in main], [row["cp_max_error_exact"] for row in main],
                "s-", color="#bc472f", label="Max Cp error")
ax[0, 2].set(xlabel="Nr (Nt = 4 Nr)", ylabel="Error", title="FV: exact boundary isolates discretization")
ax[0, 2].legend(fontsize=8)
for degree, marker in [(2, "o"), (3, "s"), (4, "^")]:
    selected = sorted([row for row in cusp if row.get("nt") == 512
                       and row["wall_polynomial_degree"] == degree], key=lambda row: row["nr"])
    ax[1, 0].loglog([row["nr"] for row in selected], [row["wall_normal_max"] for row in selected],
                    marker+"-", label=f"Degree {degree}")
ax[1, 0].set(xlabel="Nr (Nt = 512 fixed)", ylabel="Max independent wall normal / Uinf",
             title="Cubic dip includes error cancellation")
ax[1, 0].legend(fontsize=8)
compressible = next(row for row in rows if row["boundary"] == "uniform" and row["mach"] == .3)
history = compressible["history"]
ax[1, 1].semilogy(np.arange(len(history)), [item["residual_divergence_rms"] for item in history],
                  "o-", label="Updated-density RMS")
ax[1, 1].semilogy(np.arange(len(history)), [item["residual_divergence_max"] for item in history],
                  "s-", label="Maximum")
ax[1, 1].axhline(1e-9, color="gray", ls=":", lw=1.)
ax[1, 1].set(xlabel="Density solve index", ylabel="Mass-divergence residual",
             title="M=0.3: convergence is not accuracy validation")
ax[1, 1].legend(fontsize=8)
ax[1, 2].plot([row["radius"] for row in far_metrics], [row["cl"] for row in far_metrics], "o-", color="#007c91")
ax[1, 2].set(xlabel="Computational far radius R", ylabel="Pressure CL",
             title="M=0.2: uniform far-field truncation")
for axis in ax.flat:
    axis.grid(True, alpha=.2)
fig.savefig(root/"results"/"lifting.png", dpi=180)
fig.savefig(root/"results"/"lifting.pdf")
print(root/"LIFTING_RESULTS.md")
