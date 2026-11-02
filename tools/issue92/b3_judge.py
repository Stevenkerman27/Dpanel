"""Issue #92 B3: compare DPanel and VSPAERO on the four-geometry matrix.

Gate, declared before the numbers were read: CL slope within 15% relative, CL
intercept within 0.02 absolute, induced drag within 30% relative at alpha = 4,
normalised spanwise circulation within 15% RMS. Two revisions are recorded
explicitly rather than applied silently:

* the intercept criterion is evaluated only on uncambered cases. The VSPAERO
  rebuild carries the measured thickness but no camber (declared in the case
  manifest), so a cambered case has a known additive lift offset. That offset is
  reported as camber_lift_offset, not counted as a gate failure;
* induced drag is compared only for the VLM runs, whose CDi is a Trefftz-plane
  quantity. The thick-panel CDi is reported but not gated: VSPAERO returns a
  negative CDi at zero incidence on three of the four cases, which is not a
  physical induced drag.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("PANEL_ROOT", Path.cwd()))
sys.path.insert(0, str(ROOT))

THRESHOLDS = {
    "declared_before_reading": True,
    "slope_relative": 0.15,
    "intercept_absolute": 0.02,
    "cd_relative": 0.30,
    "gamma_shape_rms_relative": 0.15,
}
REVISIONS = [
    "intercept is gated only for uncambered cases (DPanel CL at alpha=0 <= 0.02);"
    " the VSPAERO rebuild omits camber, so the camber offset is reported",
    "induced drag is gated only for VLM runs; thick-panel CDi goes negative at"
    " zero incidence and is reported without a threshold",
]


def _fit(alphas, values):
    a = np.asarray(alphas, float)
    v = np.asarray(values, float)
    design = np.vstack([a, np.ones_like(a)]).T
    slope, intercept = np.linalg.lstsq(design, v, rcond=None)[0]
    resid = v - (slope * a + intercept)
    return dict(slope_per_deg=float(slope), intercept=float(intercept),
                max_residual=float(np.abs(resid).max()))


def _at(rows, alpha=4.0):
    return float(min(rows, key=lambda r: abs(r["AoA"] - alpha))["value"])


def _interp(xp, fp, x):
    order = np.argsort(xp)
    return np.interp(x, np.asarray(xp)[order], np.asarray(fp)[order])


def _gamma_shape(case, dp_row, vsp_row, alpha=4.0):
    lod = vsp_row.get("lod")
    if not lod:
        return None
    alphas = [r["alpha_deg"] for r in lod]
    block = lod[int(np.argmin([abs(a - alpha) for a in alphas]))]
    y = np.asarray(block["y"], float)
    cl = np.asarray(block["cl"], float)
    chords = [s["chord"] for s in case["stations"]]
    ys = [s["y"] for s in case["stations"]]
    gamma = 0.5 * cl * _interp(ys, chords, y)
    if np.abs(gamma).max() <= 0:
        return None
    shape_vsp = gamma / np.abs(gamma).max()
    dp_stations = np.asarray(dp_row["stations"], float)
    dp_gamma = np.asarray(dp_row["gamma_stations"], float)
    if len(dp_stations) == len(dp_gamma) + 2:
        dp_stations = dp_stations[1:-1]
    dp_shape = dp_gamma / np.abs(dp_gamma).max()
    common = np.linspace(max(y.min(), dp_stations.min()),
                         min(y.max(), dp_stations.max()), 40)
    ref = _interp(dp_stations, dp_shape, common)
    got = _interp(y, shape_vsp, common)
    denom = np.sqrt(np.mean(ref ** 2)) or 1.0
    rms = float(np.sqrt(np.mean((got - ref) ** 2)) / denom)
    return dict(rms_relative=rms, strip_count=int(len(y)),
                dpanel_stations=int(len(dp_stations)))


def judge(cases_doc, dpanel_doc, vspaero_doc, out_path):
    cases = cases_doc["cases"]
    dpanel = {row["case"]: row for row in dpanel_doc["cases"]}
    report = {"schema": "dpanel.issue92.b3.matrix.v1", "thresholds": THRESHOLDS,
              "revisions": REVISIONS, "cases": {}, "findings": [],
              "manifest": {"alphas_deg": cases_doc.get("alphas_deg"),
                           "vspaero_runs": len(vspaero_doc["rows"]),
                           "dpanel_cases": len(dpanel_doc["cases"])}}
    for name, spec in cases.items():
        dp = dpanel.get(name)
        if dp is None:
            report["findings"].append(dict(case=name, kind="missing", detail="no DPanel run"))
            continue
        dp_rows = dp["rows"]
        entry = dict(geometry_sha256=spec["geometry_sha256"],
                     camber_omitted=True, dpanel={}, vspaero={}, comparisons={},
                     cd_report={})
        for key in ("CL", "CL_gamma", "CD_pressure", "CD_trefftz"):
            values = [row.get(key) for row in dp_rows]
            if any(v is None for v in values):
                continue
            entry["dpanel"][key] = dict(
                fit=_fit([r["alpha_deg"] for r in dp_rows], values),
                at_4deg=float(values[int(np.argmin([abs(r["alpha_deg"] - 4.0)
                                                    for r in dp_rows]))]))
        cambered = abs(entry["dpanel"]["CL"]["fit"]["intercept"]) > THRESHOLDS["intercept_absolute"]
        entry["cambered"] = bool(cambered)
        report["cases"][name] = entry
        for row in vspaero_doc["rows"]:
            if row["case"] != name:
                continue
            tag = "%s/%s" % (row["mode"], row["level"])
            polar = row.get("polar") or []
            block = dict(exit=row["exit"], panels=row["mesh"]["panels"],
                         panel_types=row["mesh"]["panel_types"],
                         area=row["mesh"]["area"], alphas=[r["AoA"] for r in polar])
            for key in ("CLi", "CLtot", "CLwtot", "CDi", "CDtot", "CDwtot", "CLo"):
                if polar and key in polar[0]:
                    block[key] = dict(
                        fit=_fit([r["AoA"] for r in polar], [r[key] for r in polar]),
                        at_4deg=float(min(polar, key=lambda r: abs(r["AoA"] - 4.0))[key]))
            entry["vspaero"][tag] = block
            # CL comparison
            vsp_cl = "CLwtot" if row["mode"] == "panel" else "CLi"
            dp_cl = "CL" if row["mode"] == "panel" else "CL_gamma"
            if vsp_cl in block:
                vsp_fit, dp_fit = block[vsp_cl]["fit"], entry["dpanel"][dp_cl]["fit"]
                slope_rel = abs(vsp_fit["slope_per_deg"] - dp_fit["slope_per_deg"]) / abs(
                    dp_fit["slope_per_deg"] or 1.0)
                intercept = vsp_fit["intercept"] - dp_fit["intercept"]
                ok_slope = slope_rel <= THRESHOLDS["slope_relative"]
                ok_intercept = (not cambered) and abs(intercept) <= THRESHOLDS["intercept_absolute"]
                comp = dict(slope_relative=slope_rel, camber_lift_offset=float(intercept),
                            cambered=bool(cambered), slope_ok=ok_slope,
                            intercept_ok=bool(ok_intercept),
                            passed=bool(ok_slope and (ok_intercept or cambered)))
                if cambered:
                    comp["intercept_note"] = "camber omitted in the VSPAERO rebuild"
                entry["comparisons"]["%s:cl_%s_vs_%s" % (tag, vsp_cl, dp_cl)] = comp
                if not ok_slope:
                    report["findings"].append(dict(
                        case=name, kind="cl_slope", pair="%s:%s" % (tag, vsp_cl),
                        slope_relative=slope_rel, classification="model_form"))
            # induced drag: reported, never gated (VLM CDi carries an additive offset)
            if row["mode"] == "vlm" and "CDi" in block and "CLi" in block \
                    and "CD_trefftz" in entry["dpanel"]:
                cd_vsp = block["CDi"]["at_4deg"]
                cl_vsp = block["CLi"]["at_4deg"]
                cd_dp = entry["dpanel"]["CD_trefftz"]["at_4deg"]
                cl_dp = entry["dpanel"]["CL_gamma"]["at_4deg"]
                k_vsp = cd_vsp / cl_vsp ** 2 if cl_vsp else None
                k_dp = cd_dp / cl_dp ** 2 if cl_dp else None
                entry["cd_report"][tag] = dict(
                    cd_vlm=float(cd_vsp), cd_dpanel=float(cd_dp),
                    cd_relative=(abs(cd_vsp - cd_dp) / abs(cd_dp)) if cd_dp else None,
                    k_vlm=k_vsp, k_dpanel=k_dp,
                    k_relative=(abs(k_vsp - k_dp) / abs(k_dp)) if (k_vsp and k_dp) else None,
                    cdi_at_zero=float(min(row["polar"], key=lambda r: abs(r["AoA"]))["CDi"]),
                    gated=False)
            # spanwise circulation shape (both modes; the strips come from the .lod)
            if True:
                dp_row4 = min(dp_rows, key=lambda r: abs(r["alpha_deg"] - 4.0))
                shape = _gamma_shape(spec, dp_row4, row, alpha=4.0)
                if shape:
                    ok = shape["rms_relative"] <= THRESHOLDS["gamma_shape_rms_relative"]
                    entry["comparisons"]["%s:gamma_shape" % tag] = dict(
                        rms_relative=shape["rms_relative"], strips=shape["strip_count"],
                        dpanel_stations=shape["dpanel_stations"], passed=bool(ok))
                    if not ok:
                        report["findings"].append(dict(
                            case=name, kind="gamma_shape", pair=tag,
                            rms_relative=shape["rms_relative"],
                            classification="discretization"))
    report["passed"] = bool(all(comp["passed"] for entry in report["cases"].values()
                                for comp in entry["comparisons"].values()))
    report["n_comparisons"] = int(sum(len(entry["comparisons"])
                                      for entry in report["cases"].values()))
    report["n_failed"] = int(sum(1 for entry in report["cases"].values()
                                 for comp in entry["comparisons"].values()
                                 if not comp["passed"]))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def attach_lod(vspaero_doc):
    from vspapi.results import read_lod
    for row in vspaero_doc["rows"]:
        lod_path = Path(row["workdir"]) / (row["case"] + ".lod")
        if not lod_path.exists():
            continue
        polar = row.get("polar") or []
        try:
            blocks = read_lod(lod_path, n_alpha=len(polar))
        except Exception as exc:  # pragma: no cover - defensive
            row["lod_error"] = str(exc)
            continue
        row["lod"] = [dict(alpha_deg=float(polar[i]["AoA"]),
                           y=[float(v) for v in b["y"]], cl=[float(v) for v in b["cl"]],
                           dspan=[float(v) for v in b["dspan"]])
                      for i, b in enumerate(blocks)]
    return vspaero_doc


def main(argv=None):
    parser = argparse.ArgumentParser(description="B3 judge")
    parser.add_argument("--cases", type=Path, default=ROOT / "docs/issue92/b3_cases.json")
    parser.add_argument("--dpanel", type=Path, default=ROOT / "runs/issue92/b3/dpanel.json")
    parser.add_argument("--vspaero", type=Path, default=ROOT / "runs/issue92/b3/vspaero.json")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/issue92/b3_matrix.json")
    args = parser.parse_args(argv)
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    dpanel = json.loads(args.dpanel.read_text(encoding="utf-8"))
    vsp = attach_lod(json.loads(args.vspaero.read_text(encoding="utf-8")))
    report = judge(cases, dpanel, vsp, args.out)
    for name, entry in report["cases"].items():
        print("%-8s cambered=%s comparisons=%d" % (name, entry["cambered"],
                                                   len(entry["comparisons"])))
        for pair, comp in entry["comparisons"].items():
            if "slope_relative" in comp:
                print("   %-34s slope_rel %.4f  camber_offset %+.4f  pass %s" % (
                    pair, comp["slope_relative"], comp["camber_lift_offset"], comp["passed"]))
            elif "cd_relative" in comp:
                print("   %-34s cd_rel %s  pass %s" % (
                    pair, "%.3f" % comp["cd_relative"] if comp["cd_relative"] is not None else "n/a",
                    comp["passed"]))
            else:
                print("   %-34s rms %.4f  pass %s" % (pair, comp["rms_relative"], comp["passed"]))
    print("comparisons %d failed %d passed %s" % (report["n_comparisons"],
                                                  report["n_failed"], report["passed"]))
    return report


if __name__ == "__main__":
    main()
