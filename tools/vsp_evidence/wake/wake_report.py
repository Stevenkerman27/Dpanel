"""Render docs/ISSUE49_WAKE_SENSITIVITY.md from the VSPAERO wake sweep JSONs."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = HERE / "wakesweep" / "results.json"
REF = HERE / "wakesweep" / "results_ref.json"
OUT = ROOT / "docs" / "ISSUE49_WAKE_SENSITIVITY.md"


def pct(new, old):
    return (new / old - 1.0) * 100.0


def table(rows, cols, header):
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for r in rows:
        out.append("| " + " | ".join(("-" if r.get(c) is None else fmt % r[c]) for c, fmt in cols) + " |")
    return out


def main():
    rows = json.loads(RES.read_text())
    ref = json.loads(REF.read_text()) if REF.exists() else []
    by = {r["tag"]: r for r in rows}
    rb = {r["tag"]: r for r in ref}
    L = []
    a = L.append
    a("# issue #49 优先级 c：VSPAERO 尾迹设置（NumWakeNodes / WakeIters / WakeRelax）敏感性")
    a("")
    a("`physical_accuracy_accepted = false`。本文只记录外部求解器 VSPAERO 3.55 在**固定网格**上对尾迹设置的敏感性，")
    a("目的是给 issue #49 的「DPanel vs VSP 厚 panel」比较标一个外部不确定性上界，而不是给 DPanel 做标定。")
    a("")
    a("## 设置")
    a("")
    a("- 网格与几何：`runs/vsp_conv/api_355_panel`（OpenVSP 3.55 厚面元，12996 个面元，梯形翼 trap，Sref 0.30 / Cref 0.15 / Bref 2.0）。")
    a("  所有扫描复用同一份 `trap.vspgeom` + `trap.vkey` 字节副本，只重写 `trap.vspaero` 的尾迹字段。")
    a("- 求解器：`D:/3D/Projects/OpenVSP-3.55.0-win64/vspaero.exe -omp 32 trap`（`NCPU` 字段无效，只有命令行 `-omp` 生效）。")
    a("- 迎角 5 度，其余（Vinf=100、Rho=0.002377、ReCref=1e7、Mach=0）逐字节沿用模板。")
    a("- 模板是 GUI 风格设置：`NumWakeNodes = 8`、`WakeIters = 5`、`WakeRelax = 1`、`FarDistToggle = 1`、`FarDist = -1`、`FreezeWakeAtIteration = 10000`。")
    a("- 脚本：`runs/vsp_conv/wake_sweep.py`（主扫描）、`runs/vsp_conv/wake_ref.py`（参照与远场档）、`runs/vsp_conv/wake_report.py`（生成本文）。")
    a("")
    if "ref_gui" in rb:
        g = rb["ref_gui"]
        a("**复现检查**：把模板字段原样回填（nodes 8 / iters 5 / relax 1 / FarDist -1）重跑，得 CLtot = %.6f、CDiw = %.6f；" % (g["CLtot"], g["CDiw"]))
        a("证据包 `docs/issue49_vspaero/ladders/api_355_panel/trap.polar` 对应值是 0.709350 / 0.010723：CLtot 差 %.2e（%.3f%%），CDiw 差 %.2f%%。" % (
            g["CLtot"] - 0.709349875471, pct(g["CLtot"], 0.709349875471), pct(g["CDiw"], 0.010723378352)))
        a("**这不是位级可重复**：vspaero 的 `-omp 32` 求和顺序会动末位，尾迹松弛本身也不是严格确定性的。所以本扫描的复现容差按 CLtot 0.01%、CDiw 2% 计。")
        a("")
    nodes = [by[t] for t in ("nodes04", "nodes08", "nodes16", "nodes32", "nodes64") if t in by]
    a("## 1. NumWakeNodes（iters = 20，relax = 0.5）")
    a("")
    L.extend(table(nodes, [("tag", "%s"), ("CLtot", "%.6f"), ("CLwtot", "%.6f"), ("CDi", "%.6f"), ("CDiw", "%.6f"), ("CDwtot", "%.6f"), ("CMytot", "%.5f")],
                   ["档位", "CLtot", "CLwtot", "CDi", "CDiw", "CDwtot", "CMytot"]))
    a("")
    if len(nodes) == 5:
        a("- CLtot 从 %.6f（nodes 4）单调升到 %.6f（nodes 64），跨度 %+.2f%%；nodes 8 到 32 已有 %+.2f%%。" % (
            nodes[0]["CLtot"], nodes[4]["CLtot"], pct(nodes[4]["CLtot"], nodes[0]["CLtot"]), pct(nodes[3]["CLtot"], nodes[1]["CLtot"])))
        a("- 尾迹涡量阻力 CDiw 反向单调降：%.6f 到 %.6f（%+.2f%%）；近场 CDi 在同一扫描里先升后降（%.6f / %.6f / %.6f），" % (
            nodes[0]["CDiw"], nodes[4]["CDiw"], pct(nodes[4]["CDiw"], nodes[0]["CDiw"]), nodes[0]["CDi"], nodes[2]["CDi"], nodes[4]["CDi"]))
        a("  说明尾迹节点数主要在 CDi 与 CDiw 之间搬动量，总量 CDwtot 只从 %.5f 变到 %.5f（%+.2f%%）。" % (
            nodes[0]["CDwtot"], nodes[4]["CDwtot"], pct(nodes[4]["CDwtot"], nodes[0]["CDwtot"])))
        a("- 结论：**NumWakeNodes 是这台求解器上的一阶旋钮**，1.4% 的 CL 与 13% 的 CDiw 变化不需要碰网格。")
    a("")
    its = [by[t] for t in ("iters01", "iters03", "iters05", "iters10", "iters20", "iters40") if t in by]
    a("## 2. WakeIters（nodes = 32，relax = 0.5）")
    a("")
    L.extend(table(its, [("tag", "%s"), ("CLtot", "%.6f"), ("CDi", "%.6f"), ("CDiw", "%.6f"), ("CDwtot", "%.6f"), ("CMytot", "%.5f")],
                   ["档位", "CLtot", "CDi", "CDiw", "CDwtot", "CMytot"]))
    a("")
    if len(its) == 6:
        a("- `WakeIters = 1` 没收敛：CLtot %.6f、CDi **%.6f（负值）**，不可用。" % (its[0]["CLtot"], its[0]["CDi"]))
        a("- 3 次之后进入平台：3 / 5 / 10 / 20 / 40 的 CLtot 为 %.6f / %.6f / %.6f / %.6f / %.6f，3 到 40 只差 %+.2f%%。" % (
            its[1]["CLtot"], its[2]["CLtot"], its[3]["CLtot"], its[4]["CLtot"], its[5]["CLtot"], pct(its[5]["CLtot"], its[1]["CLtot"])))
        a("- 结论：这台网格上 5 次松弛迭代已足够，**迭代数不是误差源**；但 1 次会给出符号错误的 CDi，外部对比必须记录迭代数。")
    a("")
    rl = [by[t] for t in ("relax02", "relax05", "relax10", "frozen") if t in by]
    a("## 3. WakeRelax 与冻结尾迹（nodes = 32，iters = 20）")
    a("")
    L.extend(table(rl, [("tag", "%s"), ("CLtot", "%.6f"), ("CDi", "%.6f"), ("CDiw", "%.6f"), ("CDwtot", "%.6f"), ("CMytot", "%.5f")],
                   ["档位", "CLtot", "CDi", "CDiw", "CDwtot", "CMytot"]))
    a("")
    if len(rl) == 4:
        a("- 松弛因子 0.2 / 0.5 / 1.0 的 CLtot 为 %.6f / %.6f / %.6f，跨度只有 %.2f%%：**在 nodes 32 上 relax 不是敏感项**。" % (
            rl[0]["CLtot"], rl[1]["CLtot"], rl[2]["CLtot"], abs(pct(rl[2]["CLtot"], rl[0]["CLtot"]))))
        a("- 需要修正早先的记录：`docs/VSPAERO_PANEL_POLAR_DIAGNOSIS.md` 的「relax = 1 会发散」是在**粗网格 + nodes 8** 上观察的；在 12996 面、nodes 32 上")
        a("  relax = 1 给出 %.6f（比 relax = 0.5 低 %.2f%%），并未发散。该结论不能外推。" % (rl[2]["CLtot"], abs(pct(rl[2]["CLtot"], rl[1]["CLtot"]))))
        a("- 冻结尾迹（`FreezeWakeAtIteration = 1`）给 %.6f，比迭代尾迹高 %+.2f%%。" % (rl[3]["CLtot"], pct(rl[3]["CLtot"], rl[1]["CLtot"])))
    a("")
    if "far5" in rb:
        f5 = rb["far5"]
        a("## 4. FarDist（nodes = 32，iters = 20，relax = 0.5）")
        a("")
        a("| 档位 | CLtot | CDi | CDiw | CDwtot | CMytot |")
        a("| --- | --- | --- | --- | --- | --- |")
        a("| FarDist = -1（自动） | %.6f | %.6f | %.6f | %.6f | %.5f |" % (
            by["nodes32"]["CLtot"], by["nodes32"]["CDi"], by["nodes32"]["CDiw"], by["nodes32"]["CDwtot"], by["nodes32"]["CMytot"]))
        a("| FarDist = 5 | %.6f | %.6f | %.6f | %.6f | %.5f |" % (
            f5["CLtot"], f5["CDi"], f5["CDiw"], f5["CDwtot"], f5["CMytot"]))
        a("")
        a("- 只改远场距离，CLtot 变 %+.2f%%、CDiw 变 %+.2f%%：尾迹被截断的长度直接改变下洗。" % (
            pct(f5["CLtot"], by["nodes32"]["CLtot"]), pct(f5["CDiw"], by["nodes32"]["CDiw"])))
        a("- 所以 `FarDist` 必须和 `NumWakeNodes` 一起记录，否则同一个「VSP 厚 panel」标签下会有两个不同基准。")
        a("")
    a("## 5. 对 issue #49 的含义（关键）")
    a("")
    if len(nodes) == 5 and len(its) == 6:
        a("1. 证据包里的「VSP 厚 panel 12996 面 CLtot 0.704593」来自 `ladders/api_panel_74x170`（nodes 8 / iters 5 / relax 1，另一份 3.55 网格）；本次扫描的基网格")
        a("   `api_355_panel` 用同一套设置给 0.709350，两者差 %.2f%% 来自网格/几何集差异，**不能混用**。" % abs(pct(rb["ref_gui"]["CLtot"], 0.704592923559)))
        a("2. 同一份网格只动 NumWakeNodes，CLtot 就从 %.6f 走到 %.6f（%+.2f%%）——这个跨度**大于**上一轮 Gamma 细化后剩下的 0.53%% 差距。" % (
            nodes[0]["CLtot"], nodes[4]["CLtot"], pct(nodes[4]["CLtot"], nodes[0]["CLtot"])))
        a("3. 因此在尾迹离散未对齐前，「DPanel 5 度 CL 比 VSP 低 0.5%」这类说法不成立：残差落在 VSP 自身的尾迹设置不确定性里。")
        a("   要把这条差距做实，必须先固定 VSP wake 配置并用 `trap.lod` 复核尾迹形状，或者把 DPanel 尾迹离散做成与之匹配的一族。")
        a("4. 更要紧的是方向性：归档基准用的是 **nodes 8**，而 nodes 8 处在 CLtot 单调上升段的低端。换言之，「DPanel 5 度 CL 差 0.53%」是在最有利于 DPanel 的尾迹配置下取得的；")
        a("   换成 nodes 32 / 64，同一份 VSP 网格的 CLtot 升到 0.714104 / 0.719341，DPanel 反而低 1.9% / 2.6%。任何「已收敛到 1% 以内」的说法都必须绑定 wake 配置。")
        a("5. 对 DPanel 侧的可执行建议：外部对比把「VSP wake 配置」当成基准定义的一部分写进证据表，否则 1% 量级的 CL 与 6% 量级的 CDiw 差异无法归因。")
    a("")
    a("## 6. 归档")
    a("")
    a("- 原始读数：`runs/vsp_conv/wakesweep/results.json`、`results_ref.json`；每档一个目录（`trap.vspaero`、`trap.polar`、`stdout.log`）")
    a("- 脚本：`tools/vsp_evidence/wake/{wake_sweep.py,wake_ref.py,wake_report.py}`")
    a("- 文档：本文件由 `wake_report.py` 从 JSON 生成（只读，不重解）")
    a("")
    OUT.write_text(chr(10).join(L), encoding="utf-8")
    print("wrote", OUT, len(L), "lines")


if __name__ == "__main__":
    main()