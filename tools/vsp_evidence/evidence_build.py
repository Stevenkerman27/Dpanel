"""Build the issue-49 VSPAERO evidence package (tables + file hashes).

Reads:
  runs/vsp_conv/vlm_lX/{trap.polar,trap.vspgeom}   (DEP-OPT-path ladder runs)
  runs/vsp_conv/evidence/lX/{trap.vspaero,trap.tec.dat,stdout.log,run.txt}
  runs/vsp_conv/dp_s0XX/{case.json,mesh/surface.json}
Writes docs/issue49_vspaero/{levels.json,sha256.json,sections.json,summary.md}
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VSP = ROOT / "runs" / "vsp_conv"
OUT = ROOT / "docs" / "issue49_vspaero"

LEVELS = [("l1", 240), ("l2", 854), ("l3", 3120), ("l4", 12480)]
DPANEL = [("dp_s020", 0.20), ("dp_s014", 0.14), ("dp_s010", 0.10),
          ("dp_s007", 0.07), ("dp_s005", 0.05)]
STATIONS = [0.25, 0.50, 0.75, 0.95]
XFRAC = [0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]


def read_polar(path):
    lines = Path(path).read_text(errors="ignore").splitlines()
    h = next(i for i, line in enumerate(lines) if line.strip().startswith("Beta "))
    cols = lines[h].split()
    rows = []
    for line in lines[h + 1:]:
        f = line.split()
        if len(f) != len(cols):
            continue
        rows.append({c: float(v) for c, v in zip(cols, f)})
    return cols, rows


def read_tecplot(path):
    lines = Path(path).read_text(errors="ignore").splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("ZONE"))
    nnode = int(lines[header].split("N=")[1].split(",")[0])
    nelem = int(lines[header].split("E=")[1].split(",")[0])
    start = header + 1
    while start < len(lines) and not (lines[start].strip()[:1].isdigit()
                                      or lines[start].strip()[:1] in "+-."):
        start += 1
    values = np.asarray(" ".join(lines[start:]).split(), dtype=float)
    xyz = np.column_stack((values[:nnode], values[nnode:2 * nnode],
                           values[2 * nnode:3 * nnode]))
    cp = values[3 * nnode:3 * nnode + nelem]
    conn = values[3 * nnode + nelem:3 * nnode + nelem + 3 * nelem]
    return xyz, cp, conn.astype(np.int64).reshape(-1, 3) - 1


def chord_of_y(y, c_root=0.2, c_tip=0.1, half=1.0):
    return c_root + (c_tip - c_root) * (y / half)


def sections(xyz, cp, faces, speed=20.0):
    tri = xyz[faces]
    centers = tri.mean(axis=1)
    area_xy = 0.5 * np.abs(np.cross(tri[:, 1] - tri[:, 0],
                                    tri[:, 2] - tri[:, 0])[:, 2])
    ys = np.unique(np.round(centers[:, 1], 6))
    out = []
    for target in STATIONS:
        y_strip = float(ys[np.argmin(np.abs(ys - target))])
        sel = np.abs(np.round(centers[:, 1], 6) - y_strip) < 5e-7
        chord = chord_of_y(y_strip)
        xc = centers[sel, 0] / chord
        order = np.argsort(xc)
        mass = area_xy[sel][order]
        curve = cp[sel][order]
        points = []
        for frac in XFRAC:
            idx = int(np.argmin(np.abs(xc[order] - frac)))
            points.append(dict(x_over_c=float(xc[order][idx]),
                               Cp=float(curve[idx])))
        # trapezoid integral of Cp d(x/c) as an independent section cl
        grid = np.linspace(0.0, 1.0, 41)
        dense = np.interp(grid, np.clip(xc[order], 0, 1), curve)
        points_valid = points
        out.append(dict(semispan_y=y_strip, chord=chord, panels=int(sel.sum()),
                        cp_mean=float(curve.mean()), cp_min=float(curve.min()),
                        cp_max=float(curve.max()),
                        minus_integral_cp_dxc=float(-np.trapezoid(dense, grid)),
                        speed=speed, points=points_valid))
    return out


def lod_strips(path):
    """Spanwise strip loads from a VSPAERO .lod table (explicit header)."""
    lines = Path(path).read_text(errors="ignore").splitlines()
    h = next(i for i, line in enumerate(lines) if line.split()[:3] == ["Iter", "VortexSheet", "TrailVort"])
    cols = lines[h].split()
    rows = []
    for line in lines[h + 1:]:
        f = line.split()
        if len(f) != len(cols):
            continue
        try:
            rows.append({c: float(v) for c, v in zip(cols, f)})
        except ValueError:
            continue
    return cols, rows


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    levels = []
    hashes = {}
    section_payload = {}
    for tag, panels_nominal in LEVELS:
        d = VSP / ("vlm_" + tag)
        cols, rows = read_polar(d / "trap.polar")
        geom = d / "trap.vspgeom"
        pts = Path(geom).read_text(errors="ignore").splitlines()
        npnt, npnl, flag = (int(v) for v in pts[2].split()[:3])
        entry = dict(level=tag, panels_polar_run=len(rows) and None, columns=cols,
                     mesh=dict(points=npnt, panels=npnl, symmetry_flag=flag,
                               sha256=sha256(geom)),
                     polar=rows)
        ev = VSP / "evidence" / tag
        ev_polar = ev / "trap.polar"
        if ev_polar.exists():
            _, ev_rows = read_polar(ev_polar)
            entry["tecplot_run"] = dict(
                polar=ev_rows,
                vspaero_sha256=sha256(ev / "trap.vspaero"),
                exit=Path(ev / "run.txt").read_text(errors="ignore").strip(),
                log_bytes=(ev / "stdout.log").stat().st_size)
            tec = ev / "trap.tec.dat"
            if tec.exists():
                xyz, cp, faces = read_tecplot(tec)
                secs = sections(xyz, cp, faces)
                lod_path = ev / "trap.lod"
                strips = []
                if lod_path.exists():
                    cols, rows = lod_strips(lod_path)
                    strips = [dict(y=r["Yavg"], Cl=r["Cl"], Cdi=r["Cdi"],
                                   dSpan=r["dSpan"], chord=r["Chord"],
                                   V_over_Vref=r.get("V/Vref"))
                              for r in rows]
                    for sec in secs:
                        near = min(strips, key=lambda s: abs(s["y"] - sec["semispan_y"]))
                        sec["lod_Cl_nearest"] = near["Cl"]
                        sec["lod_y_nearest"] = near["y"]
                        sec["lod_Cdi_nearest"] = near["Cdi"]
                section_payload[tag] = dict(
                    nodes=int(len(xyz)), panels=int(len(faces)),
                    cp_min=float(cp.min()), cp_max=float(cp.max()),
                    cp_mean=float(cp.mean()), sections=secs,
                    lod_strips=strips)
                entry["tecplot_run"]["tecplot"] = dict(
                    file=tec.name, nodes=int(len(xyz)), panels=int(len(faces)),
                    sha256=sha256(tec))
        levels.append(entry)

    dp = []
    for tag, size in DPANEL:
        case = json.loads((VSP / tag / "case.json").read_text(encoding="utf-8"))
        surf = json.loads((VSP / tag / "mesh" / "surface.json").read_text(encoding="utf-8"))
        dp.append(dict(tag=tag, size=size,
                       wall_panels=int(case.get("wall_panels", 0)),
                       wake_panels=int(case.get("wake_panels", 0)),
                       unknowns=int(case.get("unknowns", 0)),
                       gamma_dof=(int(case["gamma_dof"]) if isinstance(case.get("gamma_dof"), int)
                                  else len(case.get("gamma_dof") or [])),
                       gamma_stations=([float(v) for v in case["gamma_stations"]]
                                       if isinstance(case.get("gamma_stations"), list) else None),
                       reference_area=float(case.get("reference_area", 0)),
                       reference_chord=float(case.get("reference_chord", 0)),
                       moment_reference=surf.get("moment_reference"),
                       stencil_rings=case.get("stencil_rings"),
                       mesh_seconds=case.get("mesh_seconds"),
                       solve_seconds_total=case.get("solve_seconds_total"),
                       rows=case.get("rows", []),
                       surface_sha256=sha256(VSP / tag / "mesh" / "surface.json"),
                       mesh_sha256=sha256(VSP / tag / "mesh" / "surface.json")))

    files = []
    for rel in ["cases/vsp/trap.vsp3", "cases/vsp/trap.vspgeom", "cases/vsp/trap.vkey",
                "cases/vsp/trap.vspaero",
                "runs/vsp_conv/cad_trap/wing.step",
                "runs/vsp_conv/cad_trap/trailing_edge.json",
                "runs/vsp_conv/cad_trap/geometry.json",
                "runs/vsp_conv/dep_vsp.py", "runs/vsp_conv/dpanel_case.py",
                "runs/vsp_conv/analyze.py", "runs/vsp_conv/evidence_build.py"]:
        p = ROOT / rel
        if p.exists():
            files.append(dict(path=rel, bytes=p.stat().st_size, sha256=sha256(p)))
    for tag, _ in LEVELS:
        for rel in ["vlm_%s/trap.vspaero" % tag, "vlm_%s/trap.vspgeom" % tag,
                    "vlm_%s/trap.polar" % tag, "evidence/%s/trap.vspaero" % tag,
                    "evidence/%s/trap.tec.dat" % tag,
                    "evidence/%s/stdout.log" % tag]:
            p = VSP / rel
            if p.exists():
                files.append(dict(path="runs/vsp_conv/" + rel, bytes=p.stat().st_size,
                                  sha256=sha256(p)))

    (OUT / "levels.json").write_text(json.dumps(levels, indent=1), encoding="utf-8")
    (OUT / "sections.json").write_text(json.dumps(section_payload, indent=1), encoding="utf-8")
    (OUT / "dpanel.json").write_text(json.dumps(dp, indent=1), encoding="utf-8")
    (OUT / "sha256.json").write_text(json.dumps(files, indent=1), encoding="utf-8")

    # console summary
    print("=== VSPAERO polar (DEP-OPT path) ===")
    for e in levels:
        print(e["level"], "panels", e["mesh"]["panels"])
        for r in e["polar"]:
            print("   a=%5.2f CL=%.6f CDi=%.6f CDtot=%.6f CMy=%.6f"
                  % (r["AoA"], r["CLtot"], r["CDi"], r["CDtot"], r["CMytot"]))
        t = e.get("tecplot_run")
        if t:
            print("   tecplot run:", t["exit"])
            for r in t["polar"]:
                print("      a=%5.2f CL=%.6f CDi=%.6f CMy=%.6f"
                      % (r["AoA"], r["CLtot"], r["CDi"], r["CMytot"]))
    print("=== DPanel ===")
    for d in dp:
        row5 = [r for r in d["rows"] if abs(r["alpha_deg"] - 5.0) < 1e-9]
        r = row5[0] if row5 else d["rows"][0]
        print("%s size %.2f wall %d wake %d gdof %d CLp %.6f CLg %.6f CDt %.6f CDp %.6f"
              % (d["tag"], d["size"], d["wall_panels"], d["wake_panels"],
                 d["gamma_dof"], r["CL_pressure"], r["CL_circulation"],
                 r["CD_trefftz"], r["CD_pressure"]))
        print("      unknowns=%d stencil_rings=%d mesh_s=%.1f solve_s=%.1f"
              % (d["unknowns"], d["stencil_rings"] or 0, d["mesh_seconds"] or 0,
                 d["solve_seconds_total"] or 0))
    print("=== sections ===")
    for tag, s in section_payload.items():
        print(tag, "panels", s["panels"], "cp range %.4f %.4f" % (s["cp_min"], s["cp_max"]))
        for sec in s["sections"]:
            print("   y=%.3f c=%.4f cp_mean %+.3f  -int(Cp)dx/c %.3f  lod_Cl %s"
                  % (sec["semispan_y"], sec["chord"], sec["cp_mean"],
                     sec["minus_integral_cp_dxc"],
                     ("%.4f" % sec["lod_Cl_nearest"]) if "lod_Cl_nearest" in sec else "n/a"))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
