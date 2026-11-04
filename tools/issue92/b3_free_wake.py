"""Free-wake comparison for the B3 gate.

The fixed-wake matrix in docs/issue92/b3_matrix.json is the first half of the
VSPAERO comparison.  Protocol rule 4 (docs/ISSUE92_B3_VSPAERO_PROTOCOL.md) says
the free wake is compared only after the fixed wake, and only for the cases the
fixed-wake gate did not fail.  This tool runs that second half at a single
angle of attack:

  * DPanel: panel/p1_steady_free_wake.py marches the wake until the geometry
    stops moving, and reports CL, CL_gamma and CM_y from the last iteration;
  * VSPAERO: the same cases with WakeIters > 1, run by
    tools/issue92/b3_vspaero.py --wake-iters N.

The gate is on the free-wake CL and CM_y at that angle.  The fixed-to-free deltas
of both codes are recorded next to them so the two answers stay comparable with
the numbers already published for the fixed wake.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ALPHA = 4.0
THRESHOLDS = {"cl_relative": 0.15, "cm_absolute": 0.02}
GATED_CM_NOTE = (
    "CM_y stays report-only for the cambered planforms: VSPAERO rebuilds those "
    "sections with Camber=0, so the two moment references do not describe the "
    "same geometry there.")


def _polar_at(rows, case, mode, level, alpha):
    for row in rows:
        if row.get("case") != case or row.get("mode") != mode or row.get("level") != level:
            continue
        for rec in row.get("polar") or []:
            # VSPAERO's polar header spells the column AoA, not alpha.
            value = rec.get("AoA", rec.get("alpha"))
            if value is not None and abs(float(value) - alpha) < 0.05:
                return rec
    return None


def _dpanel_fixed(path, case, alpha):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    for entry in doc["cases"]:
        if entry.get("case") != case:
            continue
        for row in entry.get("rows") or []:
            if abs(float(row.get("alpha_deg", -99.0)) - alpha) < 0.05:
                return row
    return None


def _eligible(case, matrix, pair="panel/fine"):
    """Protocol rule 4: skip cases whose fixed-wake gate already failed.

    Only findings on the same mode/level pair count: a VLM coarse failure does
    not block the panel fine free-wake run, but a panel fine one does.
    """
    for item in matrix.get("findings") or []:
        if item.get("case") != case:
            continue
        found = str(item.get("pair") or "")
        if pair in found:
            return False, "%s:%s %s" % (item.get("kind"), found,
                                        item.get("classification"))
    return True, None


def run_case(name, spec, *, iterations, relaxation):
    from panel.p1_lifting_g2 import combine_wings
    from panel.p1_steady_free_wake import solve_steady_free_wake
    from tools.issue92.b3_dpanel import _wing

    wing = _wing(spec)
    model = combine_wings([wing], reference_chord=float(spec["cref"]),
                          reference_area=float(spec["sref"]),
                          reference_point=wing.reference_point)
    value = solve_steady_free_wake(model, ALPHA, iterations=iterations,
                                   relaxation=relaxation)
    hist = value["history"]
    last = hist[-1]
    return dict(
        iterations=int(value["iterations"]),
        seconds=float(value["seconds"]),
        geometry_stable=bool(value["geometry_stable"]),
        converged=bool(value["converged"]),
        wake_geometry_change=last.get("wake_geometry_change"),
        normal_slip_rms={k: v["normal_slip_rms"] for k, v in (last.get("wakes") or {}).items()},
        history=[dict(iteration=h["iteration"], CL=h["CL"], CL_gamma=h["CL_gamma"],
                      CM_y=h["CM_y"], CD_pressure=h["CD_pressure"],
                      wake_geometry_change=h.get("wake_geometry_change")) for h in hist],
        CL=float(last["CL"]), CL_gamma=float(last["CL_gamma"]), CM_y=float(last["CM_y"]),
        n_wall_faces=len(model.faces),
        wake_faces=[len(g.wake_faces) for g in model.groups],
    )


def _run_worker(name, args):
    """Run one case in a child process.

    The DPanel free-wake path dies with a native access violation on some CAD
    planforms (see the blocked entries in the report), and a crash like that
    kills the whole interpreter.  Each case therefore runs on its own, and the
    exit code is recorded instead of being lost.
    """
    tmp = Path(args.out).parent / ("_fw_%s.json" % name)
    cmd = [sys.executable, "-m", "tools.issue92.b3_free_wake", "--worker", name,
           "--cases", str(args.cases), "--out", str(tmp),
           "--iterations", str(args.iterations), "--relaxation", str(args.relaxation)]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode == 0 and tmp.exists():
        data = json.loads(tmp.read_text(encoding="utf-8"))
        tmp.unlink()
        return data, None
    tail = [ln for ln in (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]]
    return None, dict(exit=int(proc.returncode), stderr_tail=tail,
                      repro=" ".join(cmd[2:]))


def main(argv=None):
    parser = argparse.ArgumentParser(description="B3 free-wake comparison")
    parser.add_argument("--cases", type=Path, default=ROOT / "docs/issue92/b3_cases.json")
    parser.add_argument("--dpanel-fixed", type=Path,
                        default=ROOT / "runs/benchmarks/b3_vspaero/b3/dpanel_cpu.json")
    parser.add_argument("--vspaero-fixed", type=Path,
                        default=ROOT / "runs/benchmarks/b3_vspaero/b3/vspaero.json")
    parser.add_argument("--vspaero-free", type=Path,
                        default=ROOT / "runs/benchmarks/b3_vspaero/b3/vspaero_freewake.json")
    parser.add_argument("--matrix", type=Path, default=ROOT / "docs/issue92/b3_matrix.json")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/issue92/b3_free_wake.json")
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--relaxation", type=float, default=0.3)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--worker", default=None,
                        help="internal: run a single case and write its raw result")
    args = parser.parse_args(argv)
    doc = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.worker:
        spec = doc["cases"][args.worker]
        result = run_case(args.worker, spec, iterations=args.iterations,
                          relaxation=args.relaxation)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    vsp_fixed = json.loads(args.vspaero_fixed.read_text(encoding="utf-8"))["rows"]
    vsp_free = json.loads(args.vspaero_free.read_text(encoding="utf-8"))["rows"]
    cases = {}
    n_fail = 0
    for name, spec in doc["cases"].items():
        if args.only and name not in args.only:
            continue
        eligible, blocked_by = _eligible(name, matrix)
        free, crash = _run_worker(name, args)
        if free is None:
            cases[name] = dict(cambered=bool(spec.get("cambered", False)),
                               fixed_wake_eligible=eligible, blocked_by=blocked_by,
                               dpanel=None, dpanel_blocked=crash, passed=None)
            print("%-8s DPanel free wake blocked exit=%s" % (name, crash["exit"]),
                  flush=True)
            continue
        fixed_dp = _dpanel_fixed(args.dpanel_fixed, name, ALPHA) or {}
        fixed_vs = _polar_at(vsp_fixed, name, "panel", "fine", ALPHA) or {}
        free_vs = _polar_at(vsp_free, name, "panel", "fine", ALPHA) or {}
        cambered = bool(spec.get("cambered", False))
        out = dict(cambered=cambered, fixed_wake_eligible=eligible,
                   blocked_by=blocked_by, dpanel=free)
        if free_vs:
            cl_dp = free["CL"]
            cl_vs = float(free_vs.get("CLwtot", free_vs.get("CLtot")))
            rel = abs(cl_dp - cl_vs) / max(abs(cl_dp), 1e-3)
            cm_dp = free["CM_y"]
            cm_vs = float(free_vs.get("CMytot"))
            cm = abs(cm_dp - cm_vs)
            out["vspaero"] = dict(CL=cl_vs, CM_y=cm_vs,
                                  CL_fixed=float(fixed_vs.get("CLwtot", fixed_vs.get("CLtot")))
                                  if fixed_vs else None,
                                  CM_y_fixed=float(fixed_vs.get("CMytot")) if fixed_vs else None)
            out["comparison"] = dict(
                cl_relative=rel, cl_difference=cl_dp - cl_vs,
                cm_absolute=cm, cm_difference=cm_dp - cm_vs,
                # Rule 4 eligibility and the camber caveat both decide whether a
                # number is a gate or a record.
                cl_gated=eligible, cm_gated=bool(eligible and not cambered),
                cl_passed=bool(rel <= THRESHOLDS["cl_relative"]),
                cm_passed=bool(cm <= THRESHOLDS["cm_absolute"]),
            )
            out["fixed_to_free"] = dict(
                dpanel_CL_relative=(cl_dp - float(fixed_dp.get("CL"))) / abs(float(fixed_dp.get("CL")))
                if fixed_dp.get("CL") else None,
                vspaero_CL_relative=(cl_vs - out["vspaero"]["CL_fixed"]) / abs(out["vspaero"]["CL_fixed"])
                if out["vspaero"]["CL_fixed"] else None,
            )
            failed = (eligible and not out["comparison"]["cl_passed"]) or                      (out["comparison"]["cm_gated"] and not out["comparison"]["cm_passed"])
            out["passed"] = None if not eligible else (not failed)
            n_fail += 1 if failed else 0
        else:
            out["passed"] = None
        cases[name] = out
        print("%-8s CL dp %+.4f vs %+.4f eligible=%s stable=%s conv=%s" % (
            name, free["CL"], out.get("vspaero", {}).get("CL", float("nan")),
            eligible, free["geometry_stable"], free["converged"]), flush=True)
    payload = dict(schema="dpanel.issue92.b3.free_wake.v1", alpha_deg=ALPHA,
                   thresholds=THRESHOLDS, cm_note=GATED_CM_NOTE,
                   method=("DPanel steady free wake (panel/p1_steady_free_wake.py) "
                           "against VSPAERO WakeIters>1, panel mode, fine mesh, one "
                           "angle; free wake is gated only for cases whose fixed-wake "
                           "gate in b3_matrix.json did not fail"),
                   iterations=args.iterations, relaxation=args.relaxation,
                   cases=cases, n_failed=n_fail)
    passed = [c for c in cases.values() if c.get("passed") is not None]
    payload["n_gated"] = len(passed)
    payload["passed"] = bool(passed and n_fail == 0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", args.out, "gated", len(passed), "failed", n_fail)
    return payload


if __name__ == "__main__":
    main()
