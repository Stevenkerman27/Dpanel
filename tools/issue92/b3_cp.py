"""Cp comparison between DPanel and VSPAERO on common spanwise sections.

VSPAERO's <case>.case.<n>.quad.<m>.dat files are cut planes through the field,
not the wall, so they cannot carry surface Cp.  The only surface Cp it writes is
the Tecplot wall zone produced by vspaero -writetecplot; that file holds a single
solution, so tools/issue92/b3_vspaero.py --tecplot --alphas 4.0 is run once per
case and the wall triangles are read here.

The two codes discretise the same geometry with different panel layouts, so the
comparison is made on matched sections: both sides are binned in chordwise
fraction at a spanwise station DPanel actually resolves, and the gate is on the
section normal-force coefficient, which is what a Cp distribution is for.
Pointwise RMS and correlation are reported but not gated, because they mix two
different panel layouts and have no declared threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

ALPHA = 4.0
BINS = 20
STATIONS = {"inboard": 0.40, "outboard": 0.75}
THRESHOLDS = {"cn_section_relative": 0.15, "min_bins": 4, "gated": False}
GATE_NOTE = (
    "Reported, not gated.  The two codes put their Cp samples on different "
    "layouts (VSPAERO cell-centred triangles against DPanel quad panels), and a "
    "20-bin average smears the leading-edge suction peak that carries most of "
    "the section force, so the integrated cn is biased by the binning itself.  "
    "The numbers are recorded for the record; making them a gate would first "
    "require a chordwise density convergence study on both sides.")

TRAPZ = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def _zone_count(head, key):
    for token in head.replace(",", " ").split():
        if token.startswith(key + "="):
            return int(token.split("=")[1])
    raise ValueError("no " + key + " in zone header")


def read_tecplot(path):
    """Wall zone of a VSPAERO Tecplot file: node coordinates and per-triangle Cp."""
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    while i < len(lines) and not lines[i].startswith("ZONETYPE"):
        i += 1
    if i >= len(lines):
        raise ValueError("no ZONE header in %s" % path)
    head = " ".join(lines[:i])
    n = _zone_count(head, "N")
    e = _zone_count(head, "E")
    nums = []
    for ln in lines[i + 1:]:
        s = ln.strip()
        if not s or "=" in s or any(c.isalpha() for c in s):
            continue
        nums.extend(float(v) for v in s.split())
    need = 3 * n + e + 3 * e
    if len(nums) < need:
        raise ValueError("truncated zone: %d < %d values" % (len(nums), need))
    xyz = np.array(nums[:3 * n], float).reshape(3, n).T
    cp = np.array(nums[3 * n:3 * n + e], float)
    tri = np.array(nums[3 * n + e:3 * n + e + 3 * e], int).reshape(e, 3) - 1
    return dict(xyz=xyz, cp=cp, tri=tri)


def triangle_centres(doc):
    pts = doc["xyz"]
    tri = doc["tri"]
    return pts[tri].mean(axis=1)


def bin_index(xf):
    return np.clip((xf * BINS).astype(int), 0, BINS - 1)


def binned_mean(values, idx):
    out = np.full(BINS, np.nan)
    for b in range(BINS):
        sel = idx == b
        if sel.any():
            out[b] = float(values[sel].mean())
    return out


def split_sides(xf, cp, z):
    """Upper/lower split of a section using the local mid-height of each bin."""
    idx = bin_index(xf)
    z_b = binned_mean(z, idx)
    mid = z_b[idx]
    return z > mid, z_b


def section_cn(cp_b):
    """Section normal force from a chordwise Cp distribution (upper minus lower)."""
    xs = (np.arange(BINS) + 0.5) / BINS
    ok = np.isfinite(cp_b)
    if ok.sum() < THRESHOLDS["min_bins"]:
        return float("nan")
    return float(TRAPZ(cp_b[ok], xs[ok]))


def station_of(spec, y):
    """Leading-edge x and chord at a spanwise station, interpolated from the case."""
    st = sorted(spec["stations"], key=lambda s: float(s["y"]))
    ys = np.array([float(s["y"]) for s in st])
    le = np.array([float(s["le_x"]) for s in st])
    ch = np.array([float(s["chord"]) for s in st])
    return float(np.interp(y, ys, le)), float(np.interp(y, ys, ch))


def dpanel_section(spec):
    from panel.p1_lifting_g1 import _mesh_as_kernel, solve
    from tools.issue92.b3_dpanel import _wing

    wing = _wing(spec)
    res = solve(wing, alpha_deg=ALPHA, kutta="signed_speed", backend="torch")
    mesh = _mesh_as_kernel(wing)
    return dict(centres=np.asarray(mesh.centers, float),
                normals=np.asarray(mesh.normals, float),
                cp=np.asarray(res["cp"], float))


def compare(name, spec, work, dpanel):
    w = read_tecplot(work / (name + ".tec.dat"))
    cen = triangle_centres(w)
    half = float(spec["half_span"])
    sections = {}
    for tag, frac in STATIONS.items():
        # Both planforms are piecewise ruled between their own stations, so a
        # band that stays inside one panel strip samples exactly one section.
        y_s = frac * half
        d_sel = np.abs(dpanel["centres"][:, 1] - y_s) <= 0.05 * half
        v_sel = np.abs(cen[:, 1] - y_s) <= 0.02 * half
        if d_sel.sum() < 4 or v_sel.sum() < 4:
            continue
        le, chord = station_of(spec, y_s)
        dx = (dpanel["centres"][d_sel, 0] - le) / chord
        dz = dpanel["centres"][d_sel, 2]
        d_cp = dpanel["cp"][d_sel]
        d_up = dpanel["normals"][d_sel, 2] > 0
        didx = bin_index(dx)
        d_upper = np.full(BINS, np.nan)
        d_lower = np.full(BINS, np.nan)
        for b in range(BINS):
            up = (didx == b) & d_up
            lo = (didx == b) & ~d_up
            if up.any():
                d_upper[b] = float(d_cp[up].mean())
            if lo.any():
                d_lower[b] = float(d_cp[lo].mean())
        vx = (cen[v_sel, 0] - le) / chord
        vz = cen[v_sel, 2]
        vcp = w["cp"][v_sel]
        v_up, _ = split_sides(vx, vcp, vz)
        vidx = bin_index(vx)
        v_upper = np.full(BINS, np.nan)
        v_lower = np.full(BINS, np.nan)
        for b in range(BINS):
            up = (vidx == b) & v_up
            lo = (vidx == b) & ~v_up
            if up.any():
                v_upper[b] = float(vcp[up].mean())
            if lo.any():
                v_lower[b] = float(vcp[lo].mean())
        cn_dp = section_cn(d_lower - d_upper)
        cn_vs = section_cn(v_lower - v_upper)
        bu = np.isfinite(d_upper) & np.isfinite(v_upper)
        bl = np.isfinite(d_lower) & np.isfinite(v_lower)
        rms_up = float(np.sqrt(np.mean((d_upper[bu] - v_upper[bu]) ** 2))) if bu.sum() >= 3 else None
        rms_lo = float(np.sqrt(np.mean((d_lower[bl] - v_lower[bl]) ** 2))) if bl.sum() >= 3 else None
        corr_up = float(np.corrcoef(d_upper[bu], v_upper[bu])[0, 1]) if bu.sum() >= 4 else None
        corr_lo = float(np.corrcoef(d_lower[bl], v_lower[bl])[0, 1]) if bl.sum() >= 4 else None
        ok = bool(np.isfinite(cn_dp) and np.isfinite(cn_vs))
        rel = abs(cn_dp - cn_vs) / max(abs(cn_dp), 1e-3) if ok else None
        sections[tag] = dict(
            station=tag, y=y_s, y_over_half_span=y_s / half,
            dpanel=dict(cn=cn_dp,
                        cp_upper=[None if not np.isfinite(v) else round(v, 6) for v in d_upper],
                        cp_lower=[None if not np.isfinite(v) else round(v, 6) for v in d_lower],
                        faces=int(d_sel.sum())),
            vspaero=dict(cn=cn_vs,
                         cp_upper=[None if not np.isfinite(v) else round(v, 6) for v in v_upper],
                         cp_lower=[None if not np.isfinite(v) else round(v, 6) for v in v_lower],
                         triangles=int(v_sel.sum())),
            cn_relative=rel, cn_difference=(cn_dp - cn_vs) if ok else None,
            cp_rms_upper=rms_up, cp_rms_lower=rms_lo,
            cp_corr_upper=corr_up, cp_corr_lower=corr_lo,
            bins_upper=int(bu.sum()), bins_lower=int(bl.sum()),
            passed=(bool(rel is not None and rel <= THRESHOLDS["cn_section_relative"])
                    if THRESHOLDS["gated"] else None),
        )
    return sections


def main(argv=None):
    parser = argparse.ArgumentParser(description="B3 Cp comparison on matched sections")
    parser.add_argument("--cases", type=Path, default=ROOT / "docs/issue92/b3_cases.json")
    parser.add_argument("--vsp-dir", type=Path,
                        default=ROOT / "runs/benchmarks/b3_vspaero/b3/vsp_cp")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/issue92/b3_cp.json")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args(argv)
    doc = json.loads(args.cases.read_text(encoding="utf-8"))
    out = dict(schema="dpanel.issue92.b3.cp.v1", alpha_deg=ALPHA, bins=BINS,
               stations=dict(STATIONS), thresholds=THRESHOLDS,
               gate_note=GATE_NOTE,
               method=("surface Cp from the vspaero -writetecplot wall zone; both "
                       "codes binned by chordwise fraction at a spanwise station "
                       "DPanel resolves; section cn reported, not gated"))
    cases = {}
    n_fail = 0
    for name, spec in doc["cases"].items():
        if args.only and name not in args.only:
            continue
        work = args.vsp_dir / ("%s_panel_fine" % name)
        if not (work / (name + ".tec.dat")).exists():
            print("%-8s missing tecplot" % name)
            continue
        sections = compare(name, spec, work, dpanel_section(spec))
        bad = [s for s in sections.values() if s["passed"] is False]
        n_fail += len(bad)
        cases[name] = dict(cambered=bool(spec.get("cambered", False)),
                           sections=sections,
                           passed=None if not THRESHOLDS["gated"] else not bad)
        for tag, s in sections.items():
            print("%-8s %-8s y=%.3f cn dp %+.4f vsp %+.4f rel %s gated %s" % (
                name, tag, s["y"], s["dpanel"]["cn"], s["vspaero"]["cn"],
                "n/a" if s["cn_relative"] is None else "%.4f" % s["cn_relative"],
                s["passed"]))
    out["cases"] = cases
    out["n_sections"] = sum(len(c["sections"]) for c in cases.values())
    out["n_failed"] = n_fail
    out["passed"] = None if not THRESHOLDS["gated"] else bool(n_fail == 0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", args.out, "sections", out["n_sections"], "failed", n_fail)
    return out


if __name__ == "__main__":
    main()
