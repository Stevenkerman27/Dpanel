"""Generate docs/issue49_vspaero/README.md from the staged evidence JSON.

Numbers are read from the committed files themselves, so the tables in the README cannot
drift away from the data they summarise. Run after evidence_stage.py.
"""
from __future__ import annotations

import json
from pathlib import Path

DEST = Path(__file__).resolve().parents[2] / "docs" / "issue49_vspaero"

PANEL_ORDER = ["api_panel_24x036", "api_panel_34x052", "api_panel_44x076",
               "api_panel_52x120", "api_panel_74x170"]
VLM_ORDER = ["api_vlm_24x036", "api_vlm_34x052", "api_vlm_44x076",
             "api_vlm_52x120", "api_vlm_74x170"]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def alpha_row(rows, alpha):
    for row in rows:
        if abs(row["AoA"] - alpha) < 1e-9:
            return row
    return None


def fmt(value, digits=6):
    return "%.*f" % (digits, value)


def main() -> int:
    levels = load(DEST / "levels.json")
    dpanel = load(DEST / "dpanel.json")
    ladders = {p.parent.name: load(p) for p in sorted(DEST.glob("ladders/*/ladder.json"))}

    out = []
    add = out.append

    add("# issue #49 外部对照证据：VSPAERO 原始输出")
    add("")
    add("本目录是 issue #49 回执 <https://github.com/Stevenkerman27/Dpanel/issues/49#issuecomment-6435519407> 要求的")
    add("「VSPAERO 原始输出归档」：气动系数、截面载荷、Cp 分布、收敛日志与配置文件都按源样保留，**没有为对齐 DPanel")
    add("改动过 `Sref`、`Γ`、`Cp` 或任何参考量**。")
    add("")
    add("`sha256.json` 覆盖本目录全部原始与数值文件（仓库相对路径，README 自身除外），校验方法见第 5 节。")
    add("")
    add("## 1. 版本与模式矩阵")
    add("")
    add("| 表 | OpenVSP / 网格 | 求解器 | 模式 | GeomSet / ThinGeomSet |")
    add("| --- | --- | --- | --- | --- |")
    add("| `levels.json`（`raw/l1`–`raw/l4`） | 3.52.1 API（DEP-OPT `create_wing`） | 3.52.1 `vspaero.exe` | VLM 薄壳 | `-1 / 3` |")
    add("| `ladders/api_panel_*` | 3.55.0 API（`vspapi`） | 3.55.0 `vspaero.exe -omp N` | panel 厚面元 | `3 / -1` |")
    add("| `ladders/api_vlm_*` | 3.55.0 API（`vspapi`） | 3.55.0 `vspaero.exe -omp N` | VLM 薄壳 | `-1 / 3` |")
    add("| `dpanel/*.json` | DPanel `panel.cad_half`（Gmsh） | DPanel JAX 半模（commit `535f907`） | 厚面元（线性涡环 + 尾迹） | — |")
    add("")
    add("**模式由几何集成员关系决定，不由 `SET_ALL` 决定**：`GeomSet = SET_FIRST_USER(3)` + `ThinGeomSet = SET_NONE(-1)`")
    add("剖出厚面元（湿面积/平面面积 ≈ 1.02），反过来剖出 VLM 薄片（≈ 0.50）。判据看 stdout 的 `Model type:` 行与")
    add("`Number Of Vortex Loop Equations`。")
    add("")
    add("## 2. 收敛阶梯")
    add("")
    add("所有表里 `CLtot` 是 VSPAERO 的面元表面积分升力，`CLwtot` 是尾迹诱导（Kutta–Joukowski）通道的升力；两者差")
    add("0.3%–3%，都在原始 polar 里（`ladders/*/trap.polar` 的 48 列）。")
    add("")
    add("### 2.1 VSPAERO VLM，3.52 阶梯（`levels.json`，5° 为单点复跑）")
    add("")
    add("| 档 | 面板 | CLtot(0°) | CLtot(5°) | CLwtot(5°) | CDi(5°) | CDtot(5°) | CMytot(5°) | 日志 |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for entry in levels:
        rows = entry["polar"]
        r0, r5 = alpha_row(rows, 0), alpha_row(rows, 5)
        add("| %s | %d | %s | %s | %s | %s | %s | %s | `raw/%s/stdout.log` |"
            % (entry["level"], entry["mesh"]["panels"], fmt(r0["CLtot"]), fmt(r5["CLtot"]),
               fmt(r5["CLwtot"]), fmt(r5["CDi"]), fmt(r5["CDtot"]), fmt(r5["CMytot"], 4),
               entry["level"]))
    add("")
    add("### 2.2 VSPAERO panel 厚面元，3.55 阶梯（同一 mesher，作主要对照）")
    add("")
    add("| 档 | 面板 | 湿/平面面积 | CLtot(0°) | CLtot(5°) | CLwtot(5°) | CDi(5°) | CDtot(5°) | CMytot(5°) | lod ΣCl·dSpan | 秒 | `-omp` |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for tag in PANEL_ORDER:
        d = ladders[tag]
        rows = d["polar"]
        r0, r5 = alpha_row(rows, 0), alpha_row(rows, 5)
        add("| %s | %d | %.3f | %s | %s | %s | %s | %s | %s | %s | %.1f | %s |"
            % (tag.replace("api_panel_", ""), d["mesh"]["panels"],
               d["mesh"]["area_over_planform"], fmt(r0["CLtot"]), fmt(r5["CLtot"]),
               fmt(r5["CLwtot"]), fmt(r5["CDi"]), fmt(r5["CDtot"]), fmt(r5["CMytot"], 4),
               fmt(d["lod_cl_integral"]), d["seconds"], d.get("ncpu", "?")))
    add("")
    add("### 2.3 VSPAERO VLM，3.55 阶梯（同一 mesher，作 2.2 的对照）")
    add("")
    add("| 档 | 面板 | 湿/平面面积 | CLtot(0°) | CLtot(5°) | CLwtot(5°) | CDi(5°) | CMytot(5°) | 秒 |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for tag in VLM_ORDER:
        d = ladders[tag]
        rows = d["polar"]
        r0, r5 = alpha_row(rows, 0), alpha_row(rows, 5)
        add("| %s | %d | %.3f | %s | %s | %s | %s | %s | %.1f |"
            % (tag.replace("api_vlm_", ""), d["mesh"]["panels"],
               d["mesh"]["area_over_planform"], fmt(r0["CLtot"]), fmt(r5["CLtot"]),
               fmt(r5["CLwtot"]), fmt(r5["CDi"]), fmt(r5["CMytot"], 4), d["seconds"]))
    add("")
    add("### 2.4 DPanel，五档（同一几何）")
    add("")
    add("| 档 | 壁面 | 尾迹 | γ dof | CL_p(0°) | CL_p(5°) | CL_Γ(5°) | CD_Trefftz(5°) | CM_y(5°) |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for entry in dpanel:
        rows = entry["rows"]
        r0 = [r for r in rows if abs(r["alpha_deg"]) < 1e-9][0]
        r5 = [r for r in rows if abs(r["alpha_deg"] - 5.0) < 1e-9][0]
        add("| %s | %d | %d | %d | %s | %s | %s | %s | %s |"
            % (entry["tag"], entry["wall_panels"], entry["wake_panels"],
               entry["gamma_dof"], fmt(r0["CL_pressure"]), fmt(r5["CL_pressure"]),
               fmt(r5["CL_circulation"]), fmt(r5["CD_trefftz"]), fmt(r5["CM_y"], 4)))
    add("")
    add("DPanel 的 `CL_Γ` 用物理展向积分（PR #51 修正后的 `dy`，不是尾缘弧长 `ds`）；`CD_Trefftz` 已按真 `Sref = 0.30`")
    add("归一化（不是 `span × MAC = 0.3111`）。")
    add("")
    add("## 3. 清单逐条对应（回执 6435519407）")
    add("")
    add("| 要求 | 位置 |")
    add("| --- | --- |")
    add("| OpenVSP/VSPAERO 具体版本 | 第 1 节；`raw/lX/run.txt` 与 `ladders/*/stdout.log` 的 banner |")
    add("| 厚体 panel 还是薄 VLM、GeomSet/ThinGeomSet | 第 1 节；每档 stdout 的 `Model type:` 行 |")
    add("| CAD/TE/翼型/后缘厚度 SHA | `docs/VSP_TRAPEZOID_CONVERGENCE.md` §2；`cases/vsp/trap.vspgeom`、`runs/vsp_conv/cad_trap/{wing.step,trailing_edge.json}` |")
    add("| 网格与 wake 长/迭代/松弛 | 每档 `trap.vspaero`（`WakeIters`、`NumWakeNodes`、`WakeRelax`、`FarDist`）；`raw/lX/trap.vspgeom` |")
    add("| α/β/Mach/Re | `trap.vspaero`（`AoA`、`Beta`、`Mach`、`ReCref = 1e7`）与 polar 行 |")
    add("| Sref/bref/cref/参考点 | `Sref = 0.30`、`Bref = 2.0`、`Cref = 0.15`（3.52 档）/ `0.20`（3.55 档）；`X_cg = Y_cg = Z_cg = 0`；DPanel 的 `reference_chord` 是 MAC `0.15555556`，`moment_reference = [0.0388889, 0, 0]` |")
    add("| symmetry | 全部 `Symmetry = 1`（半模，y=0 对称面） |")
    add("| CL/CDi/CDtot/CM 原始值 | `levels.json`（VLM 3.52）、`ladders/*/ladder.json` 与 `ladders/*/trap.polar`（48 列全表） |")
    add("| 截面 Cp 或 ΔCp | `sections.json` 与 `raw/lX/trap.tec.dat`（`VARIABLES=x,y,z,Cp`，`VARLOCATION=([1-3]=NODAL,[4]=CELLCENTERED)`） |")
    add("| 每档壁面 DOF / 网格级别 | 第 2 节各表的面板数；DPanel 的 `gamma_dof` 与 `wall_panels`/`wake_panels` 见 2.4 |")
    add("| run exit 与日志 | `raw/lX/run.txt`（`exit=0` + 耗时 + 命令行）与 `stdout.log`；`ladders/*/stdout.log` |")
    add("")
    add("## 4. 使用这些数字的硬约束")
    add("")
    add("1. **panel 模式判断升力用 `CLwtot` / `trap.lod` 的条带载荷**。用户原始网格（`LECluster = 0.25`）上 `CLi` 会给出")
    add("   O(10¹)–O(10⁴) 的坏值（前缘聚类后出现尺寸突变的相邻排，逐条贴体涡条的 KJ 环量求和有 O(1/h) 奇异项）；")
    add("   `LECluster = 0.8` 的 3.55 网格上 `CLi` 正常（`CLi/CLwtot` ≈ 1）。")
    add("2. **`CDo` 是 VSPAERO 的加性经验项**（α=0 时 0.0205–0.0228，几乎与网格无关），所以 `CDtot`、`L/D`、`E` 不能当")
    add("   无粘阻力用；与 DPanel 无粘压力阻力比较只能看 `CDi` 或 `trap.lod` 的 `Cd`。")
    add("3. **绝对值不能跨 mesher 版本引用**：同一 `.vspaero`、同一 exe，3.52 剖分的 6222 面板给 CL(5°) = 0.6544，")
    add("   3.55 剖分的 6344 面板给 0.7126（+8.9%），而几何差 <0.1%。收敛阶梯必须由同一个 mesher 生成。")
    add("4. **`NCPU` 字段无效**，线程数只能通过命令行 `-omp N`（1/8/32 线程 = 66.1 / 14.2 / 11.0 s，结果位级相同）。")
    add("5. **`trap.lod` 的 357 行是 3 个迎角 × 119 站**（文件无 AoA 列，按迎角顺序分块，每块 `ΣdSpan = 1.0`）；")
    add("   整表求和会得到 3 倍计数。")
    add("6. DPanel 展向环量用 PR #51 的物理展向积分口径：`elliptic_1028` 0.375065、`elliptic_1610` 0.375919；")
    add("   旧值 0.3944/0.3948 含约 5% 的 `ds/dy` 误差。")
    add("")
    add("## 5. 复现")
    add("")
    add("```powershell")
    add("# 0) 所有脚本都在 tools/vsp_evidence/（已入库）；数据目录 runs/vsp_conv 不入库")
    add("Invoke-Expression (Get-Content tools/vsp_evidence/evidence_rerun.ps1.txt -Raw)")
    add("# 2) 重建数值表")
    add(".conda_env/python.exe tools/vsp_evidence/evidence_build.py")
    add("# 3) 重新落盘原始文件 + 重建 sha256.json + 本 README 的表")
    add(".conda_env/python.exe tools/vsp_evidence/evidence_stage.py")
    add(".conda_env/python.exe tools/vsp_evidence/evidence_readme.py")
    add("# 4) 校验哈希")
    add(".conda_env/python.exe tools/vsp_evidence/verify_sha.py")
    add("```")
    add("")
    add("3.55 的 panel/VLM 阶梯走 `vspapi`（见 `docs/VSPAPI.md`）：")
    add("")
    add("```powershell")
    add("C:/Users/zyx20/anaconda3/envs/vsppytools/python.exe -m vspapi sweep <spec.json> --mode panel --ncpu 8")
    add("```")
    add("")
    add("## 6. 边界")
    add("")
    add("- VSPAERO 是独立势流对照，**不是实验真值**；本目录不声明物理精度验收，`physical_accuracy_accepted = false` 不变。")
    add("- 入库的是 3.55 阶梯（888–12996 面板 panel、420–6422 面 VLM）与 3.52 的 VLM 阶梯（240–12480 面）；")
    add("  用户原始 `cases/vsp/trap.vspgeom`（6249 面）的读数见 `docs/VSPAERO_PANEL_POLAR_DIAGNOSIS.md`。")
    add("- 粗档 `.vspgeom` 未入库（每档 0.1–6 MB），可用 `ladders/*/ladder.json` 里的 `tess_w`/`sect_tess_u` 一条命令重剖。")
    add("")
    text = "\n".join(out)
    (DEST / "README.md").write_text(text, encoding="utf-8")
    print("wrote", DEST / "README.md", len(text), "chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())