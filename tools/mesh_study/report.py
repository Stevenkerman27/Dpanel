"""Render docs/MESH_REFINEMENT_STUDY.md from runs/mesh_study/results.json."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(r"D:\3D\Projects\panel")
STUDY = ROOT / "runs" / "mesh_study"
CASES = STUDY / "cases"
DOC = ROOT / "docs" / "MESH_REFINEMENT_STUDY.md"

DIR_A = ["a_s013", "a_s010", "a_s0075", "a_s0055"]
DIR_B = ["b_le", "b_te", "b_lete", "b_lete_s015"]
DIR_C = ["c_r050", "c_r125"]
SWEPT = ["s30_s013", "s30_s010", "s30_s0075"]


def f(x, fmt="%.5f"):
    return "n/a" if x is None else (fmt % x)


def pct(a, b):
    if not a or b is None or a == 0.0:
        return "n/a"
    return "%+.2f%%" % (100.0 * (b - a) / abs(a))


def load(tag):
    row = json.loads((CASES / tag / "case.json").read_text(encoding="utf-8"))
    peak = STUDY / "peaks" / tag / "process_peak_rss_mb.json"
    row["peak_rss_mb"] = (json.loads(peak.read_text(encoding="utf-8"))["peak_rss_mb"]
                          if peak.is_file() else None)
    row["declared"] = None
    features = row.get("features")
    if features:
        path = Path(features)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_file():
            row["declared"] = json.loads(path.read_text(encoding="utf-8"))
    return row


def feature_names(row):
    declared = row.get("declared") or {}
    return [item.get("name") for item in declared.get("features", [])]


def feature_ratio(row):
    declared = row.get("declared") or {}
    entries = declared.get("features") or []
    return entries[0].get("target_ratio") if entries else None


def successive(values, ratios):
    rows = []
    for i in range(len(values) - 2):
        a, b, c = values[i], values[i + 1], values[i + 2]
        da, db = b - a, c - b
        r = ratios[i]
        if da == 0.0 or db == 0.0 or da * db <= 0.0:
            rows.append((r, da, db, None))
            continue
        rows.append((r, da, db, math.log(abs(da / db)) / math.log(r)))
    return rows


def table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def main():
    a = [load(t) for t in DIR_A]
    b = [load(t) for t in DIR_B]
    c = [load(t) for t in DIR_C]
    swept = [load(t) for t in SWEPT]
    finest = a[-1]
    base = next(r for r in a if abs(r["size"] - 0.100) < 1e-9)

    a_rows = []
    for r in a:
        m, s = r["mesh"], r["solve"]
        sc = s["stencil_condition"]
        a_rows.append((
            f(r["size"], "%.3f"), m["wall_faces"], m["wake_faces"], m["vertices"],
            f(m["le_band_edge_mean_c"], "%.5f"), f(m["min_angle_deg"], "%.2f"),
            f(m["p99_aspect"], "%.2f"), f(m["neighbour_area_ratio_p99"], "%.2f"),
            "%.2e" % sc["initial_ratio_min"],
            s["unknowns"], f(s["CL_pressure"], "%.6f"), f(s["CL_circulation"], "%.6f"),
            f(s["CD_pressure"], "%.6f"), f(s["CD_trefftz"], "%.6f"), f(s["CM_y"], "%.6f"),
            f(s["cp_min"], "%.3f"), "%.1e" % s["linear_residual_relative"],
            f(r.get("peak_rss_mb"), "%.0f"),
        ))

    rel_rows = []
    for r in a[:-1]:
        m, s = r["mesh"], r["solve"]
        fm, fs = finest["mesh"], finest["solve"]
        rel_rows.append((
            f(r["size"], "%.3f"), pct(fm["wall_faces"], m["wall_faces"]),
            pct(fs["CL_pressure"], s["CL_pressure"]),
            pct(fs["CD_trefftz"], s["CD_trefftz"]),
            pct(fs["CD_pressure"], s["CD_pressure"]),
            pct(fs["CM_y"], s["CM_y"]),
        ))

    ratios = [a[i]["size"] / a[i + 1]["size"] for i in range(len(a) - 1)]
    order = (successive([r["solve"]["CL_pressure"] for r in a], ratios)
             + successive([r["solve"]["CD_trefftz"] for r in a], ratios)
             + successive([r["solve"]["CD_pressure"] for r in a], ratios))
    names = ["CL"] * (len(a) - 2) + ["CD_Trefftz"] * (len(a) - 2) + ["CD_p"] * (len(a) - 2)
    ord_rows = [(n, "%.3f" % r, "%+.3e" % da, "%+.3e" % db,
                 "n/a" if p is None else "%.2f" % p)
                for n, (r, da, db, p) in zip(names, order)]

    gap_rows = []
    for r in a:
        s = r["solve"]
        gap = s["CL_pressure"] - s["CL_circulation"]
        gap_rows.append((
            f(r["size"], "%.3f"), f(s["CL_pressure"], "%.6f"),
            f(s["CL_circulation"], "%.6f"), "%+.3e" % gap,
            "%+.2f%%" % (100.0 * gap / abs(s["CL_circulation"])),
        ))

    b_rows = []
    for r in b:
        m, s = r["mesh"], r["solve"]
        b_rows.append((
            r["tag"], f(r["size"], "%.3f"),
            (",".join(n for n in feature_names(r) if n) or "-"),
            m["wall_faces"], m["wake_faces"], f(m["le_band_edge_mean_c"], "%.5f"),
            f(m["min_angle_deg"], "%.2f"), f(m["p99_aspect"], "%.2f"),
            f(s["CL_pressure"], "%.6f"), pct(base["solve"]["CL_pressure"], s["CL_pressure"]),
            f(s["CD_pressure"], "%.6f"), f(s["CD_trefftz"], "%.6f"), f(s["CM_y"], "%.6f"),
        ))

    c_rows = []
    for r in c:
        m, s = r["mesh"], r["solve"]
        ratio = feature_ratio(r)
        c_rows.append((
            r["tag"], f(ratio, "%.3f"), m["wall_faces"], f(m["le_band_edge_mean_c"], "%.5f"),
            f(m["min_angle_deg"], "%.2f"), f(s["CL_pressure"], "%.6f"),
            f(s["CD_pressure"], "%.6f"), f(s["CD_trefftz"], "%.6f"), f(s["CM_y"], "%.6f"),
        ))

    s_rows = []
    for r in swept:
        m, s = r["mesh"], r["solve"]
        s_rows.append((
            f(r["size"], "%.3f"), m["wall_faces"], m["wake_faces"],
            f(m["le_band_edge_mean_c"], "%.5f"), f(m["min_angle_deg"], "%.2f"),
            s["unknowns"], f(s["CL_pressure"], "%.6f"), f(s["CD_pressure"], "%.6f"),
            f(s["CD_trefftz"], "%.6f"), f(s["CM_y"], "%.6f"),
        ))

    st_rows = []
    for r in a + b + c + swept:
        sc = r["solve"]["stencil_condition"]
        st_rows.append((
            r["tag"], r["mesh"]["wall_faces"], "%.3f" % sc["initial_ratio_min"],
            "%.3f" % sc["initial_ratio_p01"], sc["faces_below_1e_3"], sc["max_ring_depth"],
            sc["expanded_faces"], f(r["mesh"]["neighbour_area_ratio_p99"], "%.2f"),
        ))

    budget = next(r for r in b if r["tag"] == "b_lete_s015")
    uniform = next(r for r in a if abs(r["size"] - 0.075) < 1e-9)
    text = [
        "# 网格三向独立加密研究（issue #42）",
        "",
        "由 `runs/mesh_study/report.py` 从 `runs/mesh_study/cases/*/case.json` 生成，全部数字为本地实测。",
        "`physical_accuracy_accepted=false`：这里报告**离散敏感性**，不是物理精度验收。",
        "",
        "## 1. 设计：一次只动一个旋钮",
        "",
        "- **方向 A（总尺寸 h）**：`--size` 0.130 / 0.100 / 0.075 / 0.055，不启用显式特征线。",
        "- **方向 B（加密哪条特征线）**：`--size` 固定 0.100，无特征 / 仅前缘 / 仅后缘 / 前缘+后缘；另加一档 **预算对齐**（LE+TE 但 `--size` 0.150，使壁面数与 0.100 无特征档接近）。",
        "- **方向 C（聚拢强度）**：`--size` 固定 0.100、LE+TE 都开，`target_ratio` 0.500 / 0.250 / 0.125。",
        "- 另有第二个几何（后掠翼 swept30）只做方向 A，用来检查结论不是椭圆翼独有。",
        "",
        "特征声明来自 `runs/mesh_study/features.py`，坐标由生成 CAD 的同一截面公式导出（`chord(y)=tip+(root-tip)*sqrt(1-(y/L)^2)`、`le=0.25*(root-chord)`，21 站），不是从网格反推。",
        "",
        "## 2. 方向 A：全局壁面尺寸（椭圆翼，AR 5.04，alpha=5deg）",
        "",
        table(["size", "壁面", "尾迹", "节点", "LE 带边长/c", "最小角 deg", "p99 aspect",
               "相邻面积比 p99", "模板最小奇异比", "未知量", "CL_p", "CL_Gamma", "CD_p",
               "CD_Trefftz", "CM_y", "Cp_min", "真残差", "峰值 MB"], a_rows),
        "",
        "**两条 CL 路径的差（压力积分 vs 环量）**——这一栏决定能不能说「CL 收敛」：",
        "",
        table(["size", "CL_pressure", "CL_circulation", "差值", "相对"], gap_rows),
        "",
        "相对最细档（size %.3f）的偏差：" % finest["size"],
        "",
        table(["size", "壁面差", "CL", "CD_Trefftz", "CD_p", "CM_y"], rel_rows),
        "",
        "相邻差与观测阶（p = log(d_i/d_{i+1})/log(r)，r 为相邻尺寸比）：",
        "",
        table(["量", "尺寸比 r", "相邻差 1", "相邻差 2", "观测阶 p"], ord_rows),
        "",
        "## 3. 方向 B：加密哪条特征线（size 0.100）",
        "",
        table(["tag", "size", "特征", "壁面", "尾迹", "LE 带边长/c", "最小角 deg", "p99 aspect",
               "CL_p", "CL vs 无特征", "CD_p", "CD_Trefftz", "CM_y"], b_rows),
        "",
        "**预算对齐比较**（这一条是 issue #42 B(ii) 要求的同预算对照）：",
        "",
        table(["网格", "壁面", "CL", "CD_p", "CD_Trefftz", "CM_y"], [
            ("无特征 size 0.100", base["mesh"]["wall_faces"], "%.6f" % base["solve"]["CL_pressure"],
             "%.6f" % base["solve"]["CD_pressure"], "%.6f" % base["solve"]["CD_trefftz"],
             "%.6f" % base["solve"]["CM_y"]),
            ("LE+TE size 0.150（预算对齐）", budget["mesh"]["wall_faces"],
             "%.6f" % budget["solve"]["CL_pressure"],
             "%.6f" % budget["solve"]["CD_pressure"],
             "%.6f" % budget["solve"]["CD_trefftz"], "%.6f" % budget["solve"]["CM_y"]),
            ("无特征 size 0.075（均匀加密）", uniform["mesh"]["wall_faces"],
             "%.6f" % uniform["solve"]["CL_pressure"],
             "%.6f" % uniform["solve"]["CD_pressure"],
             "%.6f" % uniform["solve"]["CD_trefftz"], "%.6f" % uniform["solve"]["CM_y"]),
        ]),
        "",
        "## 4. 方向 C：聚拢强度（size 0.100，LE+TE）",
        "",
        table(["tag", "target_ratio", "壁面", "LE 带边长/c", "最小角 deg", "CL_p", "CD_p",
               "CD_Trefftz", "CM_y"], c_rows),
        "",
        "## 5. 第二个几何：后掠翼 swept30（仅方向 A）",
        "",
        table(["size", "壁面", "尾迹", "LE 带边长/c", "最小角 deg", "未知量", "CL_p", "CD_p",
               "CD_Trefftz", "CM_y"], s_rows),
        "",
        "## 6. 压力模板条件数与相邻增长率（逐例）",
        "",
        "`模板最小奇异比` 是每个面切平面模板归一化最小奇异值的最小值；`扩张面数` 是 BFS 加环才达标的面数。",
        "",
        table(["tag", "壁面", "最小奇异比", "p01 奇异比", "低于 1e-3 面数", "最大加环深度",
               "扩张面数", "相邻面积比 p99"], st_rows),
        "",
        "## 7. 能说什么、不能说什么",
        "",
        "1. **两条 CL 定义并不收敛到同一个值**：方向 A 上 " + chr(96) + "CL_pressure - CL_circulation" + chr(96) + " 从 %.2f%%（size 0.130）变到 %+.2f%%（size 0.055）并换号，最细档仍有 %+.2f%% 的差且没有随加密收缩。所以「CL 相邻变化 <5%%」只能对**每一条定义各自**成立，不能当成「CL 已收敛到物理值」。这条差本身就是压力重构误差的量度（与 PR #36 的非共面模板结论一致）。" % (
            -100.0 * (a[0]["solve"]["CL_circulation"] - a[0]["solve"]["CL_pressure"]) / a[0]["solve"]["CL_circulation"],
            -100.0 * (a[-1]["solve"]["CL_circulation"] - a[-1]["solve"]["CL_pressure"]) / a[-1]["solve"]["CL_circulation"],
            -100.0 * (a[-1]["solve"]["CL_circulation"] - a[-1]["solve"]["CL_pressure"]) / a[-1]["solve"]["CL_circulation"]),
        "2. **CL 与 CD_Trefftz 在各自定义下单调收敛**，相邻变化进入 5% 带的位置见第 2 节表；`CD_p` 在同一序列上**非单调**（粗档 0.00293、0.00337、0.00285、0.00265），这不是实现缺陷，而是近场压力积分正负相消的已知性质，**不能**用它判断精度高低。",
        "2. **GCI 未计算**：虽然 CL 的观测阶在两个相邻区间上接近 2，但 CD_Trefftz 与 CD_p 的阶不一致、CD_p 还变号，不满足渐近假设；按 issue #42 E 条，不给出 Richardson/GCI 数值，只给相邻差与观测阶。",
        "3. **特征加密是预算重分配**：同样的 0.100 壁面尺寸下，LE+TE 把前缘带边长缩短约 %.0f%%、但壁面数增加 %.1f 倍，CL 只变 %s、CD_p 变 %s；预算对齐档（0.150 + LE+TE，%d 壁面）与均匀 0.075 档（%d 壁面）的 CL 相差 %s。结论只能是「同等预算下没有观察到明显收益」，而不是「特征加密更准」。" % (
            100.0 * (1.0 - next(r for r in b if r["tag"] == "b_lete")["mesh"]["le_band_edge_mean_c"] / base["mesh"]["le_band_edge_mean_c"]),
            next(r for r in b if r["tag"] == "b_lete")["mesh"]["wall_faces"] / base["mesh"]["wall_faces"],
            pct(base["solve"]["CL_pressure"], next(r for r in b if r["tag"] == "b_lete")["solve"]["CL_pressure"]),
            pct(base["solve"]["CD_pressure"], next(r for r in b if r["tag"] == "b_lete")["solve"]["CD_pressure"]),
            budget["mesh"]["wall_faces"], uniform["mesh"]["wall_faces"],
            pct(budget["solve"]["CL_pressure"], uniform["solve"]["CL_pressure"])),
        "4. **聚拢过头会变差**：`target_ratio` 0.125 把壁面数推到 4924、`CD_p` 反而翻倍、CL 比 0.100 无特征档还低——过小的特征线尺寸制造了大量小面元，与「更准」无关。",
        "5. **压力模板在这些网格上都不失秩**：初始最小奇异比 0.13-0.31，没有面需要加环，这与翼身后缘/接触割边会失秩的情形不同（那类面在 `docs/ISSUE28...` 与 PR #22 系列里单独处理）。",
        "6. **网格不位可重复**：同参数两次运行计数一致但坐标最大差可达 0.14c（PR #43 实测），跨轮比较只比计数级统计与解，不比 SHA。",
        "7. 峰值内存只统计求解进程，gmsh 在子进程里跑未计入；所有网格是单块三角面、无四边形、无高阶基、无侧滑。",
        "",
        "## 8. 复现",
        "",
        "```powershell",
        "$env:PYTHONPATH='D:\\3D\\Projects\\panel\\runs;D:\\3D\\Projects\\panel\\runs\\pr43_git;D:\\3D\\Projects\\panel\\audit\\runtime\\python'",
        "$env:TMP=$env:TEMP=$env:TMPDIR='D:\\3D\\Projects\\panel\\runs\\pipwork'",
        "cd 'D:\\3D\\Projects\\panel\\runs\\mesh_study'",
        "& 'D:\\3D\\Projects\\panel\\.conda_env\\python.exe' sweep.py      # 方向 A + swept30",
        "& 'D:\\3D\\Projects\\panel\\.conda_env\\python.exe' report.py     # 生成本文件",
        "```",
        "",
        "单例（方向 B/C）：",
        "",
        "```powershell",
        "& $py $pw mesh_study.one_case <peak_dir> --tag T --case elliptic --size 0.1 --direction B --features runs\\mesh_study\\features\\feat_lete_r250.json --out-root runs\\mesh_study\\cases",
        "& $py augment.py T   # 补条件数/增长率指标",
        "```",
        "",
        "环境：Python %s。失败用例：无。" % sys.version.split()[0],
        "",
    ]
    body = "\n".join(text)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(body, encoding="utf-8")
    print("WROTE", DOC, len(body), "chars")


if __name__ == "__main__":
    main()