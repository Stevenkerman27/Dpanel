"""Compare DPanel (half, thick surface) and VSPAERO (VLM, thin sheet) on the
trapezoid wing of cases/vsp, and emit summary.json plus a markdown table.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "runs" / "vsp_conv"
MAC = 0.15555555555555556
HALF_SPAN = 1.0
SREF = 0.30
VINF = 20.0
RHO = 1.225
DPANEL = [("s020", 0.20), ("s014", 0.14), ("s010", 0.10), ("s007", 0.07),
          ("s005", 0.05)]
VLM = ["l1", "l2", "l3", "l4"]


def chord_of(y):
    return 0.2 - 0.1 * np.asarray(y)


def read_lod(path):
    lines = Path(path).read_text(errors="ignore").splitlines()
    head = next(i for i, line in enumerate(lines) if line.strip().startswith("Iter"))
    cols = lines[head].split()
    rows = [dict(zip(cols, (float(v) for v in line.split())))
            for line in lines[head + 1:] if len(line.split()) == len(cols)]
    return rows


def dpanel_levels():
    levels = []
    for tag, size in DPANEL:
        data = json.loads((WORK / ("dp_" + tag) / "case.json")
                          .read_text(encoding="utf-8"))
        alpha = np.asarray([r["alpha_deg"] for r in data["rows"]])
        clp = np.asarray([r["CL_pressure"] for r in data["rows"]])
        clg = np.asarray([r["CL_circulation"] for r in data["rows"]])
        cdp = np.asarray([r["CD_pressure"] for r in data["rows"]])
        cdt = np.asarray([r["CD_trefftz"] for r in data["rows"]])
        cmy = np.asarray([r["CM_y"] for r in data["rows"]])
        order = np.argsort(alpha)
        # Each mesh row carries the half-model circulation at the spanwise
        # stations of surface.json; the mirror image carries the opposite
        # sign, so local lift per unit span is rho*U*Gamma and
        # cl = 2*Gamma/(U*c) for both halves of the physical wing.
        surface = json.loads((WORK / ("dp_" + tag) / "mesh" / "surface.json")
                             .read_text(encoding="utf-8"))
        stations = np.asarray(surface["gamma_stations"])
        gammas = np.asarray([r["gamma_stations"] for r in data["rows"]])[order]
        cl_section = 2.0 * gammas / (VINF * chord_of(stations))[None, :]
        levels.append(dict(tag=tag, size=size, walls=data["wall_panels"],
                           unknowns=data["wall_panels"] + data["gamma_dof"],
                           wake=data["wake_panels"], alpha=alpha[order],
                           cl=clp[order], cl_circ=clg[order], cdp=cdp[order],
                           cdt=cdt[order], cmy=cmy[order],
                           span_y=stations, span_cl=cl_section, gamma=gammas,
                           gamma_stations=stations, seconds=data["solve_seconds_total"]))
    return levels


def read_polar(path):
    lines = Path(path).read_text(errors="ignore").splitlines()
    head = next(i for i, line in enumerate(lines) if line.strip().startswith("Beta "))
    cols = lines[head].split()
    rows = [dict(zip(cols, (float(v) for v in line.split())))
            for line in lines[head + 1:] if len(line.split()) == len(cols)]
    return rows


def vspgeom_panels(path):
    lines = Path(path).read_text(errors="ignore").splitlines()
    return int(lines[2].split()[1])


def vlm_levels():
    levels = []
    for tag in VLM:
        work = WORK / ("vlm_" + tag)
        lod = work / "tec" / "trap.lod"
        rows = read_lod(lod) if lod.exists() else []
        polar = {round(r["AoA"], 6): r for r in read_polar(work / "trap.polar")}
        alpha = np.asarray(sorted(polar))
        panels = vspgeom_panels(work / "trap.vspgeom")
        y = np.asarray([r["Yavg"] for r in rows], float)
        cl_span = np.asarray([r["Cl"] for r in rows], float)
        chord = np.asarray([r["Chord"] for r in rows], float)
        levels.append(dict(tag=tag, panels=panels,
                           strips=int(len(rows)), alpha=alpha,
                           cl=np.asarray([polar[a]["CLtot"] for a in alpha]),
                           cli=np.asarray([polar.get(a, {}).get("CLi",
                                                                polar[a]["CLtot"])
                                           for a in alpha]),
                           cdi=np.asarray([polar[a]["CDi"] for a in alpha]),
                           cdtot=np.asarray([polar[a]["CDtot"] for a in alpha]),
                           cmy=np.asarray([polar[a]["CMytot"] for a in alpha]),
                           span_y=y, span_cl=cl_span,
                           gamma=0.5 * VINF * chord * cl_span,
                           seconds=0.0))
    return levels


def central(alpha, values, at=5.0):
    index = int(np.argmin(abs(alpha - at)))
    return float((values[index + 1] - values[index - 1])
                 / (alpha[index + 1] - alpha[index - 1]))


def fit_slope(alpha, values):
    design = np.column_stack((alpha, np.ones_like(alpha)))
    slope, intercept = np.linalg.lstsq(design, values, rcond=None)[0]
    return float(slope), float(intercept)


def pad(rows):
    """Stack unequal-length spanwise rows on a NaN-padded rectangle."""
    width = max(len(row) for row in rows)
    out = np.full((len(rows), width), np.nan)
    for index, row in enumerate(rows):
        out[index, :len(row)] = row
    return out


def relative(value, reference):
    return float((value - reference) / reference)


def main():
    dp = dpanel_levels()
    vlm = vlm_levels()
    at5 = {level["tag"]: int(np.argmin(abs(level["alpha"] - 5.0))) for level in dp}

    dp_table = []
    for level in dp:
        i = at5[level["tag"]]
        slope, intercept = fit_slope(level["alpha"], level["cl"])
        dp_table.append(dict(
            tag=level["tag"], size=level["size"], walls=level["walls"],
            wake=level["wake"], seconds=level["seconds"],
            cl0=float(level["cl"][0]), cl5=float(level["cl"][i]),
            cl5_circ=float(level["cl_circ"][i]),
            cdp5=float(level["cdp"][i]), cdt5=float(level["cdt"][i]),
            cmy5=float(level["cmy"][i]),
            dclda_central=central(level["alpha"], level["cl"]),
            dclda_fit=slope, cl_intercept=intercept,
            alpha=[float(v) for v in level["alpha"]],
            cl_curve=[float(v) for v in level["cl"]],
            cmy_curve=[float(v) for v in level["cmy"]],
            cdt_curve=[float(v) for v in level["cdt"]],
            cdp_curve=[float(v) for v in level["cdp"]]))

    vlm_table = []
    for level in vlm:
        i = int(np.argmin(abs(level["alpha"] - 5.0)))
        slope, intercept = fit_slope(level["alpha"], level["cl"])
        vlm_table.append(dict(
            tag=level["tag"], panels=level["panels"], strips=level["strips"],
            seconds=level["seconds"],
            cl0=float(level["cl"][0]), cl5=float(level["cl"][i]),
            cdi5=float(level["cdi"][i]), cdtot5=float(level["cdtot"][i]),
            cmy5=float(level["cmy"][i]),
            dclda_central=central(level["alpha"], level["cl"]),
            dclda_fit=slope, cl_intercept=intercept,
            alpha=[float(v) for v in level["alpha"]],
            cl_curve=[float(v) for v in level["cl"]],
            cmy_curve=[float(v) for v in level["cmy"]],
            cdi_curve=[float(v) for v in level["cdi"]]))

    finest_dp, finest_vlm = dp_table[-1], vlm_table[-1]
    for row in dp_table:
        row["cl5_rel_to_finest"] = relative(row["cl5"], finest_dp["cl5"])
        row["cdt5_rel_to_finest"] = relative(row["cdt5"], finest_dp["cdt5"])
        row["cdp5_rel_to_finest"] = relative(row["cdp5"], finest_dp["cdp5"])
    for row in vlm_table:
        row["cl5_rel_to_finest"] = relative(row["cl5"], finest_vlm["cl5"])
        row["cdi5_rel_to_finest"] = relative(row["cdi5"], finest_vlm["cdi5"])

    boundary = dict(
        cl5_dpanel_minus_vlm=float(finest_dp["cl5"] - finest_vlm["cl5"]),
        cl5_ratio=float(finest_dp["cl5"] / finest_vlm["cl5"]),
        cdt5_vs_cdi5_ratio=float(finest_dp["cdt5"] / finest_vlm["cdi5"]),
        cd_dpanel_pressure_over_trefftz=float(finest_dp["cdp5"] / finest_dp["cdt5"]),
        span_dof_dpanel=int(len(dp[-1]["span_y"])),
        strips_vlm=int(finest_vlm["strips"]),
        tip_cl_dpanel=float(dp[-1]["span_cl"][at5["s005"], -1]),
        tip_cl_vlm=float(vlm[-1]["span_cl"][-1]),
        mid_cl_dpanel=float(np.interp(0.5, dp[-1]["span_y"],
                                      dp[-1]["span_cl"][at5["s005"]])),
        mid_cl_vlm=float(np.interp(0.5, vlm[-1]["span_y"], vlm[-1]["span_cl"])),
    )

    payload = dict(reference=dict(sref=SREF, mac=MAC, bref=2.0, vinf=VINF, rho=RHO,
                                  alpha_deg=[0.0, 2.5, 5.0, 7.5, 10.0],
                                  note="full wing Sref; both codes same geometry"),
                   dpanel=dp_table, vspaero=vlm_table, boundary=boundary)
    (WORK / "summary.json").write_text(json.dumps(payload, indent=1),
                                       encoding="utf-8")
    # span loading is compared at 5 deg only
    pick = 5.0
    np.savez(WORK / "span_loading.npz",
             dp_y=np.asarray([l["span_y"] for l in dp]),
             dp_cl=np.asarray([l["span_cl"][int(np.argmin(abs(l["alpha"] - pick)))]
                               for l in dp]),
             dp_gamma=np.asarray([l["gamma"][int(np.argmin(abs(l["alpha"] - pick)))]
                                  for l in dp]),
             **{name: pad([level[key] for level in vlm])
                for name, key in (("vlm_y", "span_y"), ("vlm_cl", "span_cl"),
                                  ("vlm_gamma", "gamma"))})

    lines = ["| mesh | panels | CL(5) | CL dev vs finest | CD(5) | CD dev vs finest |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in dp_table:
        lines.append("| DPanel size %.2f | %d | %.6f | %+.2f%% | %.6f (Trefftz) | %+.2f%% |"
                     % (row["size"], row["walls"], row["cl5"],
                        100 * row["cl5_rel_to_finest"], row["cdt5"],
                        100 * row["cdt5_rel_to_finest"]))
    for row in vlm_table:
        lines.append("| VSPAERO %s | %d | %.6f | %+.2f%% | %.6f (CDi) | %+.2f%% |"
                     % (row["tag"], row["panels"], row["cl5"],
                        100 * row["cl5_rel_to_finest"], row["cdi5"],
                        100 * row["cdi5_rel_to_finest"]))
    (WORK / "summary_table.md").write_text("\n".join(lines) + "\n",
                                           encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps(boundary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
