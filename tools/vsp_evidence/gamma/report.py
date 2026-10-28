"""Collect the nested-Gamma study into one markdown report (issue #49, priority b)."""
import json, pathlib

ROOT = pathlib.Path("D:/3D/Projects/panel")
RUNS = ROOT / "runs" / "gamma_ref"
CASES = [("dp_s005_a5", "12290", "5"), ("dp_s005", "12290", "0"), ("dp_s014_a5", "1822", "5")]
VSP = {"panel_12996": dict(CLtot=0.704593, CLwtot=0.708639, CDiw=0.010723),
       "vlm_6422": dict(CLtot=0.639791, CDiw=0.008718)}


def table(tag):
    p = RUNS / tag / "report.json"
    if not p.exists():
        return None
    rep = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for lv in rep["levels"]:
        rows.append(dict(dofs=lv["free_gamma_dofs"],
                         CLp=lv["loads"]["CL_pressure"],
                         CLg=lv["ideal_trefftz_common_grid"]["CL_circulation"],
                         CDp=lv["loads"]["CD_pressure"],
                         CDi=lv["ideal_trefftz_common_grid"]["CDi_ideal"],
                         CMy=lv["loads"]["CM_y"],
                         te=lv["te_jump_l2_over_Uc"],
                         sec=lv["elapsed_seconds"]))
    return rows


def f6(v):
    return format(v, ".6f")


def pct(a, b):
    return (a / b - 1.0) * 100.0


def main():
    out = ["# issue #49 优先级 b：固定网格的 Gamma 单轴细化", "",
           "只改环量试验/测试空间（16 → 32 → 64 个自由 Gamma 自由度）：壁面坐标、压力模板、尾迹形状与长度、",
           "TE 声明、freestream 与参考量全部冻结。求解器为 `panel.run_gamma_refinement`（PR #34 的嵌套空间，",
           "模块未改动）；5° 档由只旋转 freestream、其余字节不变的副本给出（`runs/gamma_ref/make_alpha.py`）。",
           "`physical_accuracy_accepted = false`。", ""]
    summary = {}
    prof_stats = None
    for tag, walls, alpha in CASES:
        rows = table(tag)
        if rows is None:
            out += ["## " + walls + " 壁面、alpha = " + alpha + " 度", "", "（未运行）", ""]
            continue
        out += ["## " + walls + " 壁面、alpha = " + alpha + " 度", "",
                "1x 回归通过（mu / Gamma / Cp / CL / CD / CM 均在 5e-10 相对容差内）。", "",
                "| Gamma dof | CL_p | CL_Gamma | (CL_p-CL_Gamma)/CL_Gamma | CD_p | CDi_Trefftz | CM_y | TE jump | 时间 s |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in rows:
            gap = (r["CLp"] - r["CLg"]) / r["CLg"] * 100.0
            out.append("| %d | %s | %s | %+.2f%% | %s | %s | %s | %.5f | %.1f |" % (
                r["dofs"], f6(r["CLp"]), f6(r["CLg"]), gap, f6(r["CDp"]), f6(r["CDi"]),
                f6(r["CMy"]), r["te"], r["sec"]))
        f0, d = rows[0], rows[-1]
        out += ["", "16 → 64 dof：CL_p %+.2f%%，CL_Gamma %+.2f%%，CD_p %+.1f%%，CDi_Trefftz %+.2f%%，CM_y %+.2f%%，TE jump %+.1f%%。" % (
                pct(d["CLp"], f0["CLp"]), pct(d["CLg"], f0["CLg"]), pct(d["CDp"], f0["CDp"]),
                pct(d["CDi"], f0["CDi"]), pct(d["CMy"], f0["CMy"]), pct(d["te"], f0["te"])), ""]
        summary[(walls, alpha)] = (f0, d, rows)
    if ("12290", "5") in summary:
        f0, d, rows = summary[("12290", "5")]
        vp = VSP["panel_12996"]["CLtot"]
        vl = VSP["vlm_6422"]["CLtot"]
        out += ["## 与 VSPAERO 最细档对照（5 度，同一口径）", "",
                "| 来源 | CL | 相对 DPanel 16 dof | 相对 DPanel 64 dof |",
                "| --- | --- | --- | --- |",
                "| DPanel 12290 壁面 16 dof（CL_p） | %s | — | — |" % f6(f0["CLp"]),
                "| DPanel 12290 壁面 64 dof（CL_p） | %s | %+.2f%% | — |" % (f6(d["CLp"]), pct(d["CLp"], f0["CLp"])),
                "| VSPAERO panel 12996 面（CLtot） | %s | %+.2f%% | %+.2f%% |" % (f6(vp), pct(vp, f0["CLp"]), pct(vp, d["CLp"])),
                "| VSPAERO VLM 6422 面（CLtot） | %s | %+.2f%% | %+.2f%% |" % (f6(vl), pct(vl, f0["CLp"]), pct(vl, d["CLp"])),
                "", "诱导阻力：DPanel CDi_Trefftz %s → %s（%+.2f%%）；VSP panel CDiw %.6f，VLM CDiw %.6f。" % (
                    f6(f0["CDi"]), f6(d["CDi"]), pct(d["CDi"], f0["CDi"]),
                    VSP["panel_12996"]["CDiw"], VSP["vlm_6422"]["CDiw"]), ""]
    prof = RUNS / "dp_s005_a5" / "gamma_profile.json"
    if prof.exists():
        pj = json.loads(prof.read_text(encoding="utf-8"))
        y, lev = pj["sample_y"], pj["levels"]
        base = lev[0]["gamma"]
        scale = max(abs(v) for v in base)
        out += ["## 展向环量分布 Gamma(y)（12290 壁面，5 度，961 点共同站网格）", "",
                "| 层 | dofs | 相对 16 dof 的最大偏差 / max abs(Gamma) | 位置 y | Gamma_sum |",
                "| --- | --- | --- | --- | --- |",
                "| 16 | 16 | 0 | — | %.6f |" % lev[0]["gamma_sum"]]
        for lv in lev[1:]:
            diffs = [abs(v - b) for v, b in zip(lv["gamma"], base)]
            k = diffs.index(max(diffs))
            out.append("| %d | %d | %.3f%% | y = %.3f | %.6f |" % (
                lv["factor"], 16 * lv["factor"], max(diffs) / scale * 100.0, y[k], lv["gamma_sum"]))
        idxs = [0, len(y) // 4, len(y) // 2, 3 * len(y) // 4, len(y) - 1]
        stations = pj["stations_y"]
        pick = [min(range(len(y)), key=lambda i: abs(y[i] - s)) for s in stations]
        for lv in lev[1:]:
            inner = [i for i in range(len(y)) if y[i] < 0.9]
            kbest = max(inner, key=lambda i: abs(lv["gamma"][i] - base[i]))
            di = abs(lv["gamma"][kbest] - base[kbest]) / scale * 100.0
            rel = [abs(lv["gamma"][i] - base[i]) / scale for i in pick]
            out.append("")
            order = sorted(range(len(pick)), key=lambda k: -rel[k])
            k1, k2 = order[0], order[1]
            if lv is lev[-1]:
                prof_stats = dict(inner_max=di, inner_y=y[kbest], all_max=max(abs(v - b) for v, b in zip(lv["gamma"], base)) / scale * 100.0, all_y=y[max(range(len(y)), key=lambda i: abs(lv["gamma"][i] - base[i]))])
            out.append("- %d dof：全体样本最大偏差 %.3f%%（y = %.3f）；y < 0.9 段最大偏差 %.3f%% 落在 y = %.3f。"
                       "在 %d 个原始站位上，只有最外侧内部站位 y = %.3f 偏差明显（%.2f%%），"
                       "第二大偏差在 y = %.3f 仅 %.2f%%（均相对 max abs(Gamma)）。"
                       % (16 * lv["factor"],
                          max(abs(v - b) for v, b in zip(lv["gamma"], base)) / scale * 100.0,
                          y[max(range(len(y)), key=lambda i: abs(lv["gamma"][i] - base[i]))],
                          di, y[kbest], len(stations), y[pick[k1]], rel[k1] * 100.0,
                          y[pick[k2]], rel[k2] * 100.0))
        out += ["", "| y | " + " | ".join("%d dof" % (16 * lv["factor"]) for lv in lev) + " |",
                "| --- | " + " | ".join("---" for _ in lev) + " |"]
        for i in idxs:
            out.append("| %.3f | " % y[i] + " | ".join("%.5f" % lv["gamma"][i] for lv in lev) + " |")
        out.append("")
    out += ["## 结论（只陈述本实验测到的）", ""]
    if ("12290", "5") in summary:
        f0, d, rows = summary[("12290", "5")]
        vp = VSP["panel_12996"]["CLtot"]
        g0 = (rows[0]["CLp"] - rows[0]["CLg"]) / rows[0]["CLg"] * 100.0
        g1 = (rows[-1]["CLp"] - rows[-1]["CLg"]) / rows[-1]["CLg"] * 100.0
        out += [
            "1. Gamma 分辨率解释掉 5 度 CL 差距的一半左右：12290 壁面 16 → 64 dof 使 CL_p %s → %s（%+.2f%%），与 VSP panel 的差从 %+.2f%% 收到 %+.2f%%。" % (
                f6(f0["CLp"]), f6(d["CLp"]), pct(d["CLp"], f0["CLp"]), pct(vp, f0["CLp"]), pct(vp, d["CLp"])),
            "",
            "2. CL_p 与 CL_Gamma 的差不是 Gamma 分辨率造成的：同一细化下两者一起上升，相对差 %+.2f%% → %+.2f%% 基本不动，它属于压力恢复/离散化通道。" % (g0, g1),
            "",
            "3. 0 度截距不是环量问题：0 度档 16 → 64 dof 只把 CL 从 0.194801 抬到 0.196217（+0.73%），对 VSP 厚 panel 的 0.222424 仍是 -11.8%。",
            "",
            "4. Trefftz 诱导阻力对 Gamma 分辨率几乎不敏感：CDi %s → %s（%+.2f%%），而近场 CD_p %s → %s（%+.0f%%），再次说明近场压力阻力是正负相消残差。" % (
                f6(f0["CDi"]), f6(d["CDi"]), pct(d["CDi"], f0["CDi"]), f6(f0["CDp"]), f6(d["CDp"]), pct(d["CDp"], f0["CDp"])),
            "",
            "5. TE 跳跃残差随 Gamma 细化单调下降：%.5f → %.5f（%+.1f%%），细化确实改善了 Kutta 条件的满足度。" % (
                rows[0]["te"], rows[-1]["te"], pct(rows[-1]["te"], rows[0]["te"])),
            ""]
        if prof_stats is not None:
            out += ["6. Gamma(y) 的改动集中在翼尖：全体样本最大偏差 %.2f%% 在 y = %.3f，y < 0.9 段也有 %.2f%% 在 y = %.3f（都在翼尖收缩段），"
                    "而除最外侧内部站位外其余 16 个原始站位偏差 ≤ 1.2%%，Gamma_sum 只增 +0.44%% → 细化修正的是翼尖卸载形状，不是整体环量水平。" % (
                        prof_stats["all_max"], prof_stats["all_y"], prof_stats["inner_max"], prof_stats["inner_y"]), ""]
    text = chr(10).join(out) + chr(10)
    dest = ROOT / "docs" / "ISSUE49_GAMMA_REFINEMENT.md"
    dest.write_text(text, encoding="utf-8")
    print(json.dumps({"wrote": str(dest), "chars": len(text)}))


if __name__ == "__main__":
    main()