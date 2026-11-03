"""Issue #92 B3: rebuild each case in VSPAERO from the shared station table.

The wing is built section by section so that every declared station gets its own
chord, leading-edge offset and thickness ratio.  Levels change only the
tessellation (SectTess_U / Tess_W), never the geometry.

Usage (OpenVSP 3.55 python, vsppytools interpreter):
  python -m tools.issue92.b3_vspaero --cases docs/issue92/b3_cases.json \
      --out runs/benchmarks/b3_vspaero/b3/vspaero.json --levels coarse fine --modes panel vlm
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("PANEL_ROOT", Path.cwd()))

LEVELS = {"coarse": dict(tess_u=8, tess_w=24), "fine": dict(tess_u=16, tess_w=40)}


def _openvsp():
    import openvsp
    return openvsp


def _build_wing(vsp, spec, level):
    stations = spec["stations"]
    n = len(stations)
    plan = LEVELS[level]
    wid = vsp.AddGeom("WING", "")
    vsp.SetGeomName(wid, spec["name"])
    for _ in range(n):
        vsp.InsertXSec(wid, 1, vsp.XS_FOUR_SERIES)
    vsp.CutXSec(wid, 1)
    vsp.SetParmVal(wid, "Root_Chord", "XSec_1", float(stations[0]["chord"]))
    vsp.Update()
    for i in range(1, n + 1):
        sec = "XSec_%d" % i
        lo, hi = stations[i - 1], stations[min(i, n - 1)]
        dy = float(hi["y"] - lo["y"])
        dx = float(hi["le_x"] - lo["le_x"])
        sweep = math.degrees(math.atan2(dx, dy)) if dy > 0 else 0.0
        vsp.SetParmVal(wid, "SectTess_U", sec, int(plan["tess_u"]))
        vsp.SetParmVal(wid, "InCluster", sec,
                       float(stations[i - 1]["chord"] / max(stations[min(i, n - 1)]["chord"], 1e-6)))
        vsp.SetDriverGroup(wid, i, vsp.SPAN_WSECT_DRIVER, vsp.ROOTC_WSECT_DRIVER,
                           vsp.TIPC_WSECT_DRIVER)
        vsp.SetParmVal(wid, "Span", sec, dy)
        vsp.SetParmVal(wid, "Tip_Chord", sec, float(stations[min(i, n - 1)]["chord"]))
        vsp.SetParmVal(wid, "Twist", sec, float(stations[i - 1]["twist_deg"]))
        vsp.SetParmVal(wid, "Sweep", sec, sweep)
        vsp.SetParmVal(wid, "Sweep_Location", sec, 0.0)
        curve = "XSecCurve_%d" % i
        vsp.SetParmVal(wid, "ThickChord", curve, float(stations[min(i, n - 1)]["thickness"]))
        vsp.SetParmVal(wid, "Camber", curve, 0.0)
        vsp.Update()
    vsp.SetParmVal(wid, "Tess_W", "Shape", int(plan["tess_w"]))
    vsp.SetSetFlag(wid, int(getattr(vsp, "SET_FIRST_USER", 3)), True)
    return wid


def _geometry(vsp, mode):
    thick_name, thin_name = (("SET_FIRST_USER", "SET_NONE") if mode == "panel"
                             else ("SET_NONE", "SET_FIRST_USER"))
    analysis = "VSPAEROComputeGeometry"
    vsp.SetAnalysisInputDefaults(analysis)
    vsp.SetIntAnalysisInput(analysis, "Symmetry", [int(_geometry.symmetry)], 0)
    vsp.SetIntAnalysisInput(analysis, "GeomSet", [int(getattr(vsp, thick_name))], 0)
    vsp.SetIntAnalysisInput(analysis, "ThinGeomSet", [int(getattr(vsp, thin_name))], 0)
    result = vsp.ExecAnalysis(analysis)
    return Path(vsp.GetStringResults(result, "VSPGeomFileName")[0])


# Span fractions where the Cp comparison wants a cut plane.  VSPAERO only
# writes surface Cp onto CpSlice planes, so the stations have to be part of the
# model; the slice data lands in <case>.case.<n>.quad.<m>.dat with m counting
# the slices in the order they were added.
CP_STATIONS = (0.40, 0.75)


def _add_cp_slices(vsp, spec):
    half = float(spec["half_span"])
    return [dict(index=i, cut="y", location=float(frac * half))
            for i, frac in enumerate(CP_STATIONS, start=1)]


def _apply_cp_slices(vsp, spec):
    out = []
    for item in _add_cp_slices(vsp, spec):
        sid = vsp.AddCpSlice(1, item["location"])   # 1 == cut normal to y
        out.append(dict(item, id=str(sid)))
    return out


def _sweep_config(spec, wake_iters=1, wake_nodes=16):
    from vspapi.config import SweepConfig
    # The moment reference is a vector; passing only x_cg would put VSPAERO's
    # moment about a different axis line than the one DPanel integrates.
    ref = [float(v) for v in spec["moment_reference"]]
    return SweepConfig(sref=float(spec["sref"]), cref=float(spec["cref"]),
                       bref=float(2.0 * spec["half_span"]),
                       x_cg=ref[0], y_cg=ref[1], z_cg=ref[2],
                       mach=0.0, alpha=list(spec["alphas_deg"]), beta=0.0,
                       symmetry=int(spec["symmetry"]), wake_iters=int(wake_iters),
                       wake_nodes=int(wake_nodes), wake_relax=1.0, tecplot=0)


def run_case(name, spec, *, alphas, levels, modes, out_dir, exe, ncpu=8,
             cp_slices=False, tecplot=False, wake_iters=1, wake_nodes=16):
    vsp = _openvsp()
    rows = []
    for mode in modes:
        for level in levels:
            work = Path(out_dir) / ("%s_%s_%s" % (name, mode, level))
            work.mkdir(parents=True, exist_ok=True)
            vsp.ClearVSPModel()
            vsp.SetSetName(int(getattr(vsp, "SET_FIRST_USER", 3)), "wing")
            base = dict(spec, name=name, alphas_deg=list(alphas))
            _geometry.symmetry = int(base["symmetry"])
            _build_wing(vsp, base, level)
            slices = _apply_cp_slices(vsp, base) if cp_slices else []
            vsp.Update()
            vsp.WriteVSPFile(str(work / (name + ".vsp3")), int(getattr(vsp, "SET_ALL", 0)))
            geom_path = _geometry(vsp, mode)
            geom_local = work / (name + ".vspgeom")
            geom_local.write_bytes(Path(geom_path).read_bytes())
            cfg = _sweep_config(base, wake_iters=wake_iters, wake_nodes=wake_nodes)
            cfg.write(work / (name + ".vspaero"))
            cmd = [exe, "-omp", str(ncpu)]
            if tecplot:
                # Only this flag makes VSPAERO write surface Cp; the .dat
                # files are cut planes through the field, not the wall.
                cmd.append("-writetecplot")
            cmd.append(name)
            proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
            row = dict(case=name, mode=mode, level=level, exit=int(proc.returncode),
                       workdir=str(work), cp_slices=slices,
                       wake_iters=int(wake_iters), wake_nodes=int(wake_nodes))
            polar = work / (name + ".polar")
            if polar.exists():
                from vspapi.results import read_polar
                row["polar"] = [{k: (float(v) if isinstance(v, (int, float)) else v)
                                 for k, v in rec.items()} for rec in read_polar(polar)]
            else:
                row["stdout_tail"] = proc.stdout[-1200:]
            from vspapi.mesh import measure, read_vspgeom
            row["mesh"] = measure(read_vspgeom(geom_local))
            rows.append(row)
            print("%-8s %-5s %-6s exit=%d panels=%s" % (
                name, mode, level, row["exit"], row["mesh"]["panels"]), flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="B3 VSPAERO side")
    parser.add_argument("--cases", type=Path, default=ROOT / "docs/issue92/b3_cases.json")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/benchmarks/b3_vspaero/b3/vspaero.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/benchmarks/b3_vspaero/b3/vsp")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--levels", nargs="*", default=["coarse", "fine"])
    parser.add_argument("--modes", nargs="*", default=["panel", "vlm"])
    parser.add_argument("--ncpu", type=int, default=8)
    parser.add_argument("--cp-slices", action="store_true",
                        help="model Cp cut planes so surface Cp is written")
    parser.add_argument("--tecplot", action="store_true",
                        help="pass -writetecplot so surface Cp lands in <case>.tec.dat")
    parser.add_argument("--alphas", nargs="*", type=float, default=None,
                        help="override the angle list (Tecplot carries one solution)")
    parser.add_argument("--wake-iters", type=int, default=1,
                        help="wake iterations; >1 relaxes a free wake")
    parser.add_argument("--wake-nodes", type=int, default=16,
                        help="streamwise wake nodes")
    args = parser.parse_args(argv)
    doc = json.loads(args.cases.read_text(encoding="utf-8"))
    exe = os.environ.get("VSPAERO_EXE") or str(Path(
        r"D:\3D\Projects\OpenVSP-3.55.0-win64\vspaero.exe"))
    rows = []
    for name, spec in doc["cases"].items():
        if args.only and name not in args.only:
            continue
        alphas = args.alphas if args.alphas else doc.get("alphas_deg", [0., 2., 4., 6.])
        rows.extend(run_case(name, spec, alphas=alphas,
                             levels=args.levels, modes=args.modes,
                             out_dir=args.out_dir, exe=exe, ncpu=args.ncpu,
                             cp_slices=args.cp_slices, tecplot=args.tecplot,
                             wake_iters=args.wake_iters, wake_nodes=args.wake_nodes))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(rows=rows), indent=2), encoding="utf-8")
    print("wrote", args.out, "rows", len(rows))
    return rows


if __name__ == "__main__":
    main()
