"""Drive the geometric-robustness matrix: mesh, then half-native LU solve."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
WORKTREE = ROOT / "runs" / "pr34_git"
PEAK = ROOT / "runs" / "peak_wrap.py"
CAD = ROOT / "runs" / "geom_robust" / "cad"
MESH = ROOT / "runs" / "geom_robust" / "mesh"
SOLVE = ROOT / "runs" / "geom_robust" / "solve"
PEAK_DIR = ROOT / "runs" / "geom_robust" / "peaks"


def environment():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([
        str(WORKTREE), str(ROOT / "audit" / "runtime" / "python")])
    env["TMP"] = env["TEMP"] = env["TMPDIR"] = str(ROOT / "runs" / "pipwork")
    return env


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(cmd, cwd, log):
    started = time.perf_counter()
    done = subprocess.run(cmd, cwd=str(cwd), env=environment(),
                          capture_output=True, text=True, errors="replace")
    elapsed = time.perf_counter() - started
    Path(log).write_text(
        "CMD " + " ".join(str(item) for item in cmd) + "\n"
        + "EXIT %d\n" % done.returncode + "--- stdout ---\n" + done.stdout
        + "\n--- stderr ---\n" + done.stderr, encoding="utf-8")
    return done.returncode, elapsed, done.stdout, done.stderr


def tail(text, count=6):
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return "\n".join(lines[-count:])


def mesh_one(name, size, tag):
    out = MESH / tag
    cad = CAD / name
    if out.exists() and (out / "surface.json").is_file():
        record = dict(tag=tag, name=name, size=size, mesh_exit=0,
                      mesh_reused=True)
        record.update(json.loads((out / "manifest.json").read_text()))
        record["surface_sha256"] = sha256(out / "surface.json")
        data = json.loads((out / "surface.json").read_text())
        record["surface_wall"] = len(data["faces"])
        record["surface_wake"] = len(data["wake_faces"])
        record["surface_components"] = sorted(
            {int(value) for value in data["component_ids"]})
        return record
    cmd = [PY, str(PEAK), "panel.cad_half", str(PEAK_DIR / ("mesh_" + tag)),
           "--wing-cad", str(cad / "wing.step"),
           "--trailing-edge", str(cad / "trailing_edge.json"),
           "--out", str(out), "--size", str(size), "--far-size", "1.5",
           "--radius", "5.0", "--alpha-deg", "5.0", "--ncpu", "8"]
    if (cad / "body.step").is_file():
        cmd[6:6] = ["--body-cad", str(cad / "body.step")]
    code, elapsed, stdout, stderr = run(cmd, WORKTREE, MESH / (tag + ".log"))
    record = dict(tag=tag, name=name, size=size, mesh_exit=code,
                  mesh_seconds=round(elapsed, 2))
    peak_file = PEAK_DIR / ("mesh_" + tag) / "process_peak_rss_mb.json"
    if peak_file.is_file():
        record["mesh_peak_rss_mb"] = json.loads(
            peak_file.read_text())["peak_rss_mb"]
    manifest = out / "manifest.json"
    if manifest.is_file():
        record.update(json.loads(manifest.read_text()))
    failure = out / "failure.json"
    if code == 0 and not (out / "manifest.json").is_file():
        record["failure_message"] = "mesh exit 0 without manifest"
    if code != 0 and failure.is_file():
        info = json.loads(failure.read_text())
        record["failure_stage"] = info.get("stage")
        record["failure_exception"] = info.get("exception")
        record["failure_message"] = info.get("message")
        record["failure_tail"] = tail(info.get("traceback", ""), 3)
    elif code != 0:
        record["failure_message"] = "mesh exit %d without failure.json" % code
        record["failure_tail"] = tail(stderr or stdout, 6)
    surface = out / "surface.json"
    if surface.is_file():
        record["surface_sha256"] = sha256(surface)
        data = json.loads(surface.read_text())
        record["surface_wall"] = len(data["faces"])
        record["surface_wake"] = len(data["wake_faces"])
        record["surface_components"] = sorted(
            {int(value) for value in data["component_ids"]})
    return record


def solve_one(name, size, tag, record):
    out = SOLVE / tag
    surface = MESH / tag / "surface.json"
    if not surface.is_file():
        record["solve_status"] = "no_surface"
        return record
    if out.exists() and (out / "report.json").is_file():
        report = json.loads((out / "report.json").read_text())
        record["solve_reused"] = True
        record["solve_status"] = report.get("status")
        for key in ("CL", "CD_pressure", "CM", "first_seconds", "hot_seconds",
                    "right_wall_panels", "right_wake_panels",
                    "right_circulation_unknowns", "independent_state_unknowns",
                    "linear_residual_relative", "pressure_stencil_audit"):
            if key in report:
                record[key] = report[key]
        return record
    cmd = [PY, str(PEAK), "panel.run_half_native", str(PEAK_DIR / ("solve_" + tag)),
           "--surface", str(surface), "--out", str(out), "--mode", "lu",
           "--max-panels", "6000"]
    code, elapsed, stdout, stderr = run(cmd, WORKTREE, SOLVE / (tag + ".log"))
    record["solve_exit"] = code
    record["solve_wall_seconds"] = round(elapsed, 2)
    peak_file = PEAK_DIR / ("solve_" + tag) / "process_peak_rss_mb.json"
    if peak_file.is_file():
        record["solve_peak_rss_mb"] = json.loads(
            peak_file.read_text())["peak_rss_mb"]
    report_file = out / "report.json"
    if report_file.is_file():
        report = json.loads(report_file.read_text())
        record["solve_status"] = report.get("status")
        for key in ("CL", "CD_pressure", "CM", "first_seconds", "hot_seconds",
                    "right_wall_panels", "right_wake_panels",
                    "right_circulation_unknowns", "independent_state_unknowns",
                    "linear_residual_relative", "pressure_stencil_audit"):
            if key in report:
                record[key] = report[key]
    else:
        record["solve_status"] = "failed"
        record["solve_message"] = tail(stderr or stdout, 8)
        record["solve_exception"] = next(
            (line for line in (stderr or "").splitlines()
             if line.startswith(("ValueError", "MemoryError", "RuntimeError",
                                 "TypeError", "IndexError", "KeyError"))), None)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True,
                        help="comma separated name:size entries")
    parser.add_argument("--label", required=True)
    parser.add_argument("--skip-mesh", action="store_true")
    args = parser.parse_args()
    for directory in (MESH, SOLVE, PEAK_DIR, ROOT / "runs" / "pipwork"):
        directory.mkdir(parents=True, exist_ok=True)
    records = []
    for job in args.jobs.split(","):
        name, size = job.split(":")
        tag = "%s_s%03d" % (name, int(round(float(size) * 1000)))
        print("=== %s" % tag, flush=True)
        record = dict(tag=tag, name=name, size=float(size))
        if not args.skip_mesh:
            record = mesh_one(name, float(size), tag)
        if record.get("mesh_exit") == 0:
            record = solve_one(name, float(size), tag, record)
        records.append(record)
        print(json.dumps({key: record.get(key) for key in (
            "tag", "surface_wall", "CL", "CD_pressure", "CM",
            "solve_peak_rss_mb", "failure_message", "solve_message")},
            indent=None), flush=True)
        (ROOT / "runs" / "geom_robust" /
         ("matrix_%s.json" % args.label)).write_text(
            json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
