"""Three-direction mesh refinement study driver (issue #42).

Direction A: wall size at a fixed sizing policy (no explicit features).
Direction B: which geometric feature lines are clustered at a fixed wall size.
Direction C: how strong the clustering is (target_ratio) at a fixed wall size.

Every case is one mesh + one solve, run in its own process through
runs/peak_wrap.py so the peak working set is measured per case. Nothing here
touches panel/ defaults, the v1 CLI or any production file.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"D:\3D\Projects\panel")
PR43 = ROOT / "runs" / "pr43_git"
GMSH = ROOT / "audit" / "runtime" / "python"
STUDY = ROOT / "runs" / "mesh_study"
OUT = STUDY / "cases"
PEAK = STUDY / "peaks"
FEATURES = STUDY / "features"

ELLIPTIC = [
    # tag, size, bands, ratio, direction
    ("a_s013", 0.130, None, None, "A"),
    ("a_s010", 0.100, None, None, "A"),
    ("a_s0075", 0.075, None, None, "A"),
    ("a_s0055", 0.055, None, None, "A"),
    ("b_le", 0.100, "le", 0.25, "B"),
    ("b_te", 0.100, "te", 0.25, "B"),
    ("b_lete", 0.100, "lete", 0.25, "B"),
    ("c_r050", 0.100, "lete", 0.50, "C"),
    ("c_r125", 0.100, "lete", 0.125, "C"),
]

SWEPT = [
    ("s30_s013", 0.130, None, None, "A"),
    ("s30_s010", 0.100, None, None, "A"),
    ("s30_s0075", 0.075, None, None, "A"),
]


def feature_file(bands, ratio):
    name = "feat_%s_r%03d.json" % (bands, int(round(ratio * 1000)))
    path = FEATURES / name
    if not path.is_file():
        cmd = [sys.executable, str(STUDY / "features.py"), "--bands", bands,
               "--ratio", str(ratio), "--out", str(path)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-2000:])
    return path


def run_case(tag, case, size, features, direction):
    peak_dir = PEAK / tag
    cmd = [sys.executable, str(ROOT / "runs" / "peak_wrap.py"), "mesh_study.one_case",
           str(peak_dir), "--tag", tag, "--case", case, "--size", str(size),
           "--direction", str(direction), "--out-root", str(OUT)]
    if features is not None:
        cmd += ["--features", str(features)]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "runs"), str(PR43), str(GMSH)])
    env["TMP"] = env["TEMP"] = env["TMPDIR"] = str(ROOT / "runs" / "pipwork")
    proc = subprocess.run(cmd, cwd=str(STUDY), env=env, capture_output=True, text=True)
    print("[%s] %s" % (tag, (proc.stdout or "").strip().splitlines()[-1] if proc.stdout.strip() else "(no output)"))
    if proc.returncode != 0:
        print("[%s] FAILED rc=%d\n%s" % (tag, proc.returncode, (proc.stderr or "")[-3000:]))
    return proc.returncode


def collect(entries):
    rows = []
    for tag, *rest in entries:
        case_json = OUT / tag / "case.json"
        if not case_json.is_file():
            continue
        row = json.loads(case_json.read_text(encoding="utf-8"))
        peak = PEAK / tag / "process_peak_rss_mb.json"
        if peak.is_file():
            row["peak_rss_mb"] = json.loads(peak.read_text())["peak_rss_mb"]
        rows.append(row)
    return rows


def main():
    FEATURES.mkdir(parents=True, exist_ok=True)
    failures = []
    for tag, size, bands, ratio, direction in ELLIPTIC:
        features = feature_file(bands, ratio) if bands else None
        if run_case(tag, "elliptic", size, features, direction) != 0:
            failures.append(tag)
    for tag, size, bands, ratio, direction in SWEPT:
        if run_case(tag, "swept30", size, None, direction) != 0:
            failures.append(tag)
    elliptic = collect(ELLIPTIC)
    swept = collect(SWEPT)
    payload = dict(
        schema="dpanel.mesh_refinement_study.v1",
        physical_accuracy_accepted=False,
        environment=dict(python=sys.version.split()[0], executable=sys.executable),
        elliptic=elliptic,
        swept30=swept,
        failures=failures,
        peak_note="peak_rss_mb covers the python process that solved the case; the gmsh mesher runs in a child process and is not counted",
    )
    (STUDY / "results.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print("\n=== ELLIPTIC ===")
    for row in elliptic:
        print("%-9s s=%.3f walls=%5d LEmean=%s CL=%.6f CDp=%.6f CDt=%.6f CM=%.6f peak=%.0f" % (
            row["tag"], row["size"], row["mesh"]["wall_faces"],
            ("%.5f" % row["mesh"]["le_band_edge_mean_c"]) if row["mesh"]["le_band_edge_mean_c"] else "n/a",
            row["solve"]["CL_pressure"], row["solve"]["CD_pressure"], row["solve"]["CD_trefftz"],
            row["solve"]["CM_y"], row.get("peak_rss_mb", -1)))
    print("\n=== SWEPT30 ===")
    for row in swept:
        print("%-9s s=%.3f walls=%5d CL=%.6f CDp=%.6f CDt=%.6f CM=%.6f" % (
            row["tag"], row["size"], row["mesh"]["wall_faces"], row["solve"]["CL_pressure"],
            row["solve"]["CD_pressure"], row["solve"]["CD_trefftz"], row["solve"]["CM_y"]))
    if failures:
        print("\nFAILURES", failures)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
