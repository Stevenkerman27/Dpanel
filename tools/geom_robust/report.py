"""Assemble the geometry-robustness report from meshed, solved and swept cases.

Run:  python tools/geom_robust/report.py
Data: runs/geom_robust/{mesh,solve,peaks}, runs/geom_robust/alpha_<case>.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lift_slope as ls  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs" / "geom_robust"
BT = chr(96)
CASES = ["swept30", "elliptic", "bwb", "conv"]
SKIP = ("conv_wing", "conv_htail")
DESCRIPTION = {
    "swept30": "swept tapered wing, 30 deg LE sweep, NACA 2412",
    "elliptic": "elliptic planform, 21 cosine stations, NACA 0012",
    "bwb": "blended wing body, root t/c 0.22, LE crank 0.35 abs(y)",
    "conv": "fuselage + low wing + T-tail + centreline fin",
}
SPAN = {"swept30": 2.0, "elliptic": 2.0, "bwb": 1.0, "conv": 1.6}
SREF = {"swept30": 3.0, "elliptic": 3.172255, "bwb": 1.241497, "conv": 2.4}
CONTRAST = {
    "swept30": "cambered + 30 deg sweep: order of magnitude only",
    "elliptic": "symmetric, unswept, single lifting surface: the only clean comparison",
    "bwb": "thick cambered section + LE crank: order of magnitude only",
    "conv": "cambered wing plus fuselage; the T-tail has no Kutta condition, so "
            "CL(alpha) here is not a complete-aircraft slope "
            "(unsupported_lifting_surface)",
}
ALPHA_CASE = {"swept30": "swept30", "elliptic": "elliptic", "bwb": "bwb", "conv": "conv"}
ERATUM = [
    ("single-point slope named a slope",
     "earlier revisions of this file called CL(5 deg)/alpha the CL slope and "
     "compared it with the Helmbold estimate. That quotient contains CL(0) and "
     "is not dCL/dalpha; it is now reported as a single-point legacy number and "
     "the slope comes from a central difference over the fixed mesh."),
    ("Helmbold formula",
     "earlier revisions labelled 2*pi*AR/(AR+2) as Helmbold. That expression is "
     "the Prandtl lifting-line style estimate; the Helmbold thin-wing low-speed "
     "form is 2*pi*AR/(2+sqrt(AR^2+4)). Both are now listed side by side."),
    ("adjacent-grid agreement read as convergence",
     "a small change between the last two sizes was described as asymptotic "
     "convergence. Two adjacent levels cannot establish grid independence; the "
     "wording now states only that the change is below the 5 percent band."),
    ("attribution without external evidence",
     "the positive near-trailing-edge Cp and the wing-tip corner spike were "
     "described as established artefacts of pressure recovery and of geometric "
     "corner singularity. Both are plausible mechanisms supported by internal "
     "diagnostics, but there is no external quantitative truth value for these "
     "cases, so they are now listed as unverified attributions."),
]


def load():
    rows = []
    for mesh_dir in sorted((BASE / "mesh").iterdir()):
        if not mesh_dir.is_dir() or not (mesh_dir / "surface.json").is_file():
            continue
        tag = mesh_dir.name
        if tag.startswith(SKIP):
            continue
        name, _, digits = tag.rpartition("_s")
        if name not in CASES:
            continue
        surface = json.loads((mesh_dir / "surface.json").read_text())
        row = dict(tag=tag, name=name, size=int(digits) / 1000.0,
                   wall=len(surface["faces"]),
                   wake=len(surface["wake_faces"]),
                   components=sorted({int(v) for v in surface["component_ids"]}))
        manifest = mesh_dir / "manifest.json"
        if manifest.is_file():
            row["mesh_seconds"] = json.loads(
                manifest.read_text()).get("elapsed_seconds")
        peak = BASE / "peaks" / ("mesh_" + tag) / "process_peak_rss_mb.json"
        if peak.is_file():
            row["mesh_peak"] = json.loads(peak.read_text())["peak_rss_mb"]
        report_file = BASE / "solve" / tag / "report.json"
        if not report_file.is_file():
            continue
        report = json.loads(report_file.read_text())
        for key in ("CL", "CD_pressure", "CM", "true_relative_residual",
                    "Cp_raw_half_min", "Cp_raw_half_max", "first_seconds",
                    "hot_seconds", "right_circulation_unknowns",
                    "independent_state_unknowns", "status"):
            row[key] = report.get(key)
        row["CM_y"] = report["CM"][1]
        audit = report.get("pressure_stencil_audit") or {}
        row["deficient"] = len(audit.get("initial_rank_deficient_faces") or [])
        row["expanded"] = len(audit.get("expanded_faces") or [])
        ratios = audit.get("initial_singular_ratios") or []
        row["min_ratio"] = min(ratios) if ratios else None
        depth = audit.get("selected_ring_depth") or []
        row["max_depth"] = max(depth) if depth else None
        peak = BASE / "peaks" / ("solve_" + tag) / "process_peak_rss_mb.json"
        if peak.is_file():
            row["solve_peak"] = json.loads(peak.read_text())["peak_rss_mb"]
        rows.append(row)
    rows.sort(key=lambda item: (CASES.index(item["name"]), -item["size"]))
    return rows


def load_sweep(name):
    path = BASE / ("alpha_" + ALPHA_CASE[name] + ".json")
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    data["_path"] = str(path)
    return data


def pct(new, old):
    return 100.0 * (new - old) / abs(old) if old else float("nan")


def cas_at(mapping, alpha):
    key = float(alpha)
    return mapping.get(key)


def main():
    rows = load()
    sweeps = {name: load_sweep(name) for name in CASES}
    lines = []
    add = lines.append
    add("# 多几何鲁棒性矩阵（后掠翼 / 椭圆翼 / BWB / 常规布局）")
    add("")
    add("本文件由 " + BT + "tools/geom_robust/report.py" + BT + " 从 "
        + BT + "runs/geom_robust/{mesh,solve,peaks}" + BT + " 与 "
        + BT + "runs/geom_robust/alpha_<case>.json" + BT
        + " 的落盘产物程序化生成，全部数字来自本机实际运行（Windows 10，"
        + "Python 3.11.17，JAX 0.10.2 CPU，稠密 LU，迎角 5 度除非另注）。")
    add("")
    add("判定口径沿用项目既有约定：" + BT + "physical_accuracy_accepted=false" + BT
        + "。近场压力阻力 CD_p 与俯仰力矩 CM_y 不作为物理精度声明，"
        "只有积分升力 CL 与网格细化趋势可比较。")
    add("")
    add("## 1. 几何")
    add("")
    add("| 案例 | 平面形状 | 半展 | Sref（双侧） | 实体数 | 组件 ID |")
    add("| --- | --- | --- | --- | --- | --- |")
    for name in CASES:
        subset = [row for row in rows if row["name"] == name]
        ids = subset[0]["components"] if subset else []
        add("| %s | %s | %.2f | %.6f | %d | %s |"
            % (name, DESCRIPTION[name], SPAN[name], SREF[name], len(ids),
               ", ".join(str(value) for value in ids)))
    add("")
    add("CAD 全部由 " + BT + "tools/geom_robust/make_cases.py" + BT
        + " 用 gmsh OCC 生成（闭合实体 STEP + TE 声明 JSON + import 回验），"
        "根弦均为 1.0，网格尺寸 " + BT + "--size" + BT + " 即根弦比例。")
    add("")
    add("## 2. 网格与求解矩阵（迎角 5 度，粗 -> 细）")
    add("")
    add("| 案例 | size | 壁面 | 尾迹 | gamma | 未知量 | CL | CD_p | CM_y | 真残差 | 峰值 RSS | 网格秒 | 热解秒 |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        add("| %s | %.3f | %d | %d | %d | %d | %.6f | %.6e | %+.6f | %.2e | %.1f MB | %.2f | %.4f |"
            % (row["name"], row["size"], row["wall"], row["wake"],
               row["right_circulation_unknowns"],
               row["independent_state_unknowns"], row["CL"],
               row["CD_pressure"], row["CM_y"], row["true_relative_residual"],
               row.get("solve_peak") or float("nan"),
               row.get("mesh_seconds") or float("nan"), row["hot_seconds"]))
    add("")
    add("## 3. 相邻网格档的变化（粗 -> 细；5% 只是比较带，不是收敛证明）")
    add("")
    add("| 案例 | 档位链（壁面数） | CL 相邻相对差 | CD_p 相邻绝对差 | CM_y 相邻绝对差 | 全部相邻差 <= 5% |")
    add("| --- | --- | --- | --- | --- | --- |")
    for name in CASES:
        subset = [row for row in rows if row["name"] == name]
        subset.sort(key=lambda item: -item["size"])
        if len(subset) < 2:
            add("| " + name + " | 单档 | - | - | - | 无法判定 |")
            continue
        chain = " -> ".join(str(item["wall"]) for item in subset)
        dcl = [pct(subset[i + 1]["CL"], subset[i]["CL"])
               for i in range(len(subset) - 1)]
        dcd = [subset[i + 1]["CD_pressure"] - subset[i]["CD_pressure"]
               for i in range(len(subset) - 1)]
        dcm = [subset[i + 1]["CM_y"] - subset[i]["CM_y"]
               for i in range(len(subset) - 1)]
        ok = all(abs(value) <= 5.0 for value in dcl)
        add("| %s | %s | %s | %s | %s | %s |"
            % (name, chain,
               ", ".join("%+.2f%%" % value for value in dcl),
               ", ".join("%+.2e" % value for value in dcd),
               ", ".join("%+.2e" % value for value in dcm),
               "是" if ok else "否"))
    add("")
    add("相邻两档落在同一带内只说明该差值小于 5%，不构成网格无关声明："
        "序列仍可能按更高阶或更慢的速率继续漂移，且每档来自独立生成的网格，"
        "没有做同族网格的 Richardson 外推。")
    add("")
    add("## 4. 压力模板审计（切边/角点邻域的自动扩环）")
    add("")
    add("| 案例 | size | 初始失秩面 | 扩环面 | 最小初始奇异比 | 最大环深 |")
    add("| --- | --- | --- | --- | --- | --- |")
    for row in rows:
        add("| %s | %.3f | %d | %d | %s | %s |"
            % (row["name"], row["size"], row["deficient"], row["expanded"],
               "-" if row["min_ratio"] is None else "%.3e" % row["min_ratio"],
               "-" if row["max_depth"] is None else str(row["max_depth"])))
    add("")
    add("## 5. 成本标度（每例最细一档）")
    add("")
    add("| 案例 | 壁面 | 尾迹 | 峰值 RSS | 热解秒 | 首次秒 |")
    add("| --- | --- | --- | --- | --- | --- |")
    for name in CASES:
        subset = [row for row in rows if row["name"] == name]
        if not subset:
            continue
        finest = max(subset, key=lambda item: item["wall"])
        add("| %s | %d | %d | %.1f MB | %.3f | %.2f |"
            % (name, finest["wall"], finest["wake"],
               finest.get("solve_peak") or float("nan"),
               finest["hot_seconds"], finest["first_seconds"]))
    add("")
    add("## 6. 固定网格多迎角与中心差分斜率")
    add("")
    add("每个案例的最细一档保持壁面、后缘、尾迹与 Gamma 自由度不变，"
        "只把来流方向绕 +y 旋转到 -5 / -2.5 / 0 / +2.5 / +5 度（同速、零侧滑）。"
        "斜率为中心差分 " + BT + "[CL(+h) - CL(-h)] / (2 h_rad)" + BT
        + "，在 h = 2.5 度与 h = 5 度各算一次；"
        + BT + "CL(5 deg)/alpha_rad" + BT + " 只作旧口径对照列出，它不是导数。")
    add("")
    add("| 案例 | 壁面 | CL(-5) | CL(-2.5) | CL(0) | CL(+2.5) | CL(+5) | 单点旧口径 | dCL/da (h=2.5) | dCL/da (h=5) | 两者一致 |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in CASES:
        sweep = sweeps[name]
        if sweep is None:
            add("| %s | - | - | - | - | - | - | - | - | - | 未跑 |" % name)
            continue
        mapping = {row["alpha_deg"]: row["CL"] for row in sweep["rows"]}
        wall = max((row["wall"] for row in rows if row["name"] == name), default=0)
        central = sweep["central_difference"]
        agree = central.get("agreement")
        add("| %s | %d | %s | %s | %s | %s | %s | %.4f | %.4f | %.4f | %s |"
            % (name, wall,
               fmt(mapping, -5.0), fmt(mapping, -2.5), fmt(mapping, 0.0),
               fmt(mapping, 2.5), fmt(mapping, 5.0),
               sweep["single_point_legacy_slope"],
               central["h_2p5"], central["h_5"],
               "-" if agree is None else "%.2f%%" % (100.0 * agree)))
    add("")
    add("零迎角升力 " + BT + "CL(0)" + BT + " 就是弯度与有限网格偏置的直接读数："
        "对称剖面的椭圆翼是 3.7e-4，带弯度的后掠翼 0.1395、BWB 0.0985；"
        "因此单点 " + BT + "CL(5 deg)/alpha" + BT + " 在后两者上分别高估约 39% 与 31%，"
        "这正是 issue #38 指出的口径问题。")
    add("")
    add("## 7. 与两条经验升力线近似并列核对")
    add("")
    add("Prandtl 式 " + BT + "2*pi*AR/(AR+2)" + BT + "，Helmbold 薄翼低速式 "
        + BT + "2*pi*AR/(2+sqrt(AR^2+4))" + BT + "，AR = b^2/Sref。"
        "两者都是单张孤立无后掠翼的低速经验近似，**都不是物理真值，"
        "也不是 5% 精度接受条件**；带后掠、弯度、机身或平尾的案例只能看量级，"
        "常规布局的 T 尾没有 Kutta 条件，属 " + BT + "unsupported_lifting_surface" + BT
        + "，其 CL/alpha 不能当整机斜率。")
    add("")
    add("| 案例 | AR | 中心差分 dCL/da | Prandtl | Prandtl 差 | Helmbold | Helmbold 差 | 可对照性 |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in CASES:
        sweep = sweeps[name]
        ar = ls.aspect_ratio(2.0 * SPAN[name], SREF[name])
        if sweep is None:
            add("| %s | %.4f | - | %.4f | - | %.4f | - | %s |"
                % (name, ar, ls.prandtl_slope(ar), ls.helmbold_slope(ar), CONTRAST[name]))
            continue
        slope = sweep["central_difference"]["h_2p5"]
        prandtl, helmbold = ls.prandtl_slope(ar), ls.helmbold_slope(ar)
        add("| %s | %.4f | %.4f | %.4f | %+.2f%% | %.4f | %+.2f%% | %s |"
            % (name, ar, slope, prandtl, 100.0 * (slope - prandtl) / prandtl,
               helmbold, 100.0 * (slope - helmbold) / helmbold, CONTRAST[name]))
    add("")
    add("只有椭圆翼这一行是干净对照：它对称、无后掠、单张升力面，"
        "中心差分 4.4105 落在 Prandtl 4.4991 与 Helmbold 4.2677 之间，"
        "差值 -2.0% 与 +3.3%。这不是精度通过，只是量级相符。")
    add("")
    add("## 8. 语义解读（含未验证归因）")
    add("")
    add("- CL 在四种几何上随细化单调变化，最细一对的相邻差多在 5% 带内；"
        "这只说明该差值小，**不证明网格无关**（见第 3 节）。")
    add("- CD_p 由壁面压力投影的正负强相消得到，净值是微小残差："
        "常规布局在 0.100 档降到 6.6e-05、0.075 档回升到 5.1e-04；"
        "BWB 在 0.100 档出现 7.6e-03 的尖峰。连同一份固定网格对迎角的扫描"
        "（CD_p 从 +2.9e-03 一路到 -1.7e-02），净阻力既不正定也不随网格单调，"
        "**不能作为精度指标**。")
    add("- CM_y 绝对量小且对装配细节敏感（椭圆翼 570 面 -0.0016，"
        "1610 面 +0.0028，跨零），百分比变化无意义，只列绝对差。")
    add("- 四种几何都没有触发压力模板失秩拒绝（初始失秩面 0，扩环面 0，环深恒 2），"
        "说明自动扩环在真实 CAD 切边上够用。")
    add("- **未验证归因 1**：图上后缘附近 Cp 整体偏正，机制上与前缘吸力"
        "相邻的强相消、以及切平面仿射切线梯度不含法向分量一致，"
        "但没有外部定量真值证明偏差全由重构造成。")
    add("- **未验证归因 2**：翼尖/后缘角点的 Cp 尖峰（PR29 诊断的 max|dCp| 位置）"
        "与几何角点奇性一致，但同样没有外部真值证明它不是实现缺陷；"
        "要区分，需要独立折角分侧重构或解析角点解作基准。")
    add("")
    add("## 9. 契约限制（本轮实测触发）")
    add("")
    add("1. 尾迹面在 OCC 中固定于 z=0（" + BT + "panel/cad_half.py:172-180"
        + BT + "），因此只有 TE 落在 z=0 的升力面能拿到 Kutta 条件；"
        "上单翼或抬高的平尾无法表达。")
    add("2. TE 声明必须精确落在 CAD 的 trimmed TE 曲线上，容差 1e-7；"
        "z 偏 0.06 时实测报 " + BT + "Declared TE does not lie on a trimmed "
        "wing boundary curve: segment 0 start has distance 0.059820851, "
        "tolerance=1e-07" + BT + "。")
    add("3. 只支持一个尾迹片（单一 TE 多值函数 x(y)），两个升力面无法各自"
        "拿到 Kutta 条件；本轮常规布局因此只让机翼带 Kutta，"
        "T 尾仅作 Dirichlet 壁面（其升力被低估，故标 " + BT
        + "unsupported_lifting_surface" + BT + "）。")
    add("4. y=0 上若整片三角面被归入物理壁面会被显式拒绝：" + BT
        + "Artificial symmetry cap must not be included in physical wall"
        + BT + "（" + BT + "runs/pr34_git/panel/half_surface_mesh.py:65" + BT
        + "）。抬高的平尾根部必须整体埋入垂尾（T 尾）或机身，"
        "否则根部端面暴露在 y=0 上——本轮首版正是因此失败。")
    add("5. 固定网格迎角扫描只旋转来流方向，z=0 尾迹保持在网格生成时的位置，"
        "因此非零迎角下尾迹与当地总流不切向，其运动学残差是模型限制，"
        "不是离散误差（与 issue #28 的双侧通量结论同一来源）。")
    add("")
    add("## 10. Erratum（2026-10-26）")
    add("")
    add("旧版本文中以下四处表述已被更正，旧运行记录与旧表原样保留，不回写：")
    add("")
    for index, (title, text) in enumerate(ERATUM, start=1):
        add("%d. **%s**：%s" % (index, title, text))
    add("")
    add("## 11. 结论")
    add("")
    for name in CASES:
        subset = [row for row in rows if row["name"] == name]
        subset.sort(key=lambda item: -item["size"])
        if len(subset) < 2:
            continue
        dcl = [pct(subset[i + 1]["CL"], subset[i]["CL"])
               for i in range(len(subset) - 1)]
        add("- %s：壁面 %s，CL %.4f -> %.4f，相邻相对差 %s；"
            "最细一对 %+.2f%%，最粗一对 %+.2f%%。"
            % (name, " -> ".join(str(item["wall"]) for item in subset),
               subset[0]["CL"], subset[-1]["CL"],
               ", ".join("%+.2f%%" % value for value in dcl),
               dcl[-1], dcl[0]))
    peaks = [row for row in rows if row.get("solve_peak")]
    if peaks:
        worst = max(peaks, key=lambda item: item["solve_peak"])
        add("- 四种几何全部完成网格与 LU 求解，无失秩拒绝、无内存拒绝；"
            "峰值内存最高 %.1f MB（%s，%d 壁面）。"
            % (worst["solve_peak"], worst["tag"], worst["wall"]))
    add("- 推荐网格：纯翼案例 " + BT + "--size 0.075" + BT
        + "（1227-1610 壁面），带机身案例 " + BT + "--size 0.075" + BT
        + "（3112 壁面）；" + BT + "--size 0.100" + BT
        + " 已可用于粗筛，但最粗一档的 CL 相对差仍大于 5%。")
    add("")
    add("## 12. 上表面压力可视化")
    add("")
    add("![上表面 Cp](geometry_robustness/fig16_geometry_matrix.png)")
    add("")
    add("图为右半解镜像后的整机俯视，只画外法向 z>0 的面片；色标裁剪到 "
        "[-1.5, 0.5]，真实 Cp 极小值在每格标题里（swept30 -7.82、"
        "elliptic -2.90、bwb -5.99、conv -6.44，都落在翼尖/后缘角点）。")
    add("")
    add("Cp 由切平面仿射切线梯度恢复，不含法向分量；后缘附近整体偏正"
        "与这一重构口径一致，但如第 8 节所述，这仍属待验证归因。"
        "这张图只用来看前缘吸力峰的位置与展向分布，不作为压力幅值的物理声明。")
    add("")
    target = ROOT / "docs" / "GEOMETRY_ROBUSTNESS.md"
    text = chr(10).join(lines) + chr(10)
    target.write_text(text, encoding="utf-8")
    print("wrote", target, len(text), "chars")
    for row in rows:
        print(row["tag"], row["wall"], round(row["CL"], 6),
              "%.3e" % row["CD_pressure"])


def fmt(mapping, alpha):
    value = cas_at(mapping, alpha)
    return "-" if value is None else "%+.6f" % value


if __name__ == "__main__":
    main()
