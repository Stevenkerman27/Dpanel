"""Issue #92 S3: torch port of the nonlinear pressure-Kutta solve, measured on fixtures.

The production pressure-Kutta path solves

    min_t || [ B ; w * Jp(N t) ] - [ bw ; 0 ] ||,   state = N t,

with N = null_space(J, rcond=1e-11) over the Kutta rows J. Two backends exist in
production: the reference scipy.optimize.least_squares (trust-region-reflective,
ftol=xtol=gtol=1e-11, analytic Jacobian) and the numpy structured Gauss-Newton of
panel/p1_pressure_kutta_reduced.py. The torch port (slice S3) implements the
structured algorithm, because a torch process cannot call NumPy/SciPy LAPACK at all
and because the structured step is the one with a documented convergence path.

Three stages, run by different interpreters (0xC0000005 hazard, section 6f):

  python -m tools.issue92.s3_pressure_parity capture --work runs/scratch/issue92_loose/s3
  <torch python> -m tools.issue92.s3_pressure_parity torch --work runs/scratch/issue92_loose/s3
  python -m tools.issue92.s3_pressure_parity judge --work runs/scratch/issue92_loose/s3 --out ...

What is compared, and why:

* torch structured against numpy structured: same algorithm, so the decision
  quantities (status, nfev, nit, backtracks, message) are expected to agree and
  disagreeing would be a port defect;
* torch structured and numpy structured against the reference trust-region solve:
  not the same algorithm, so statuses differ by construction. What must agree is the
  *acceptance classification* produced by pressure_kutta_acceptance (accepted,
  stationary, wall at floor, Kutta satisfied), the wall/floor ratio, the pressure
  residual, and the loads;
* loads from one shared map (tools/issue92/s2_loads.py) applied to all three states,
  against the loads production reported for its own run.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(os.environ.get("PANEL_ROOT", Path.cwd()))

import numpy as np

CASES = ("g1_wing_rect", "g1_wing_camber", "g1_swept30", "g1_elliptic",
         "g2_conventional", "g2_tail_low", "g1_wingbody_cad", "g1_wingbody_cad_3x")

RCOND = 1e-11
WEIGHT = 5.0
KUTTA = "pressure"


def _manifest():
    from tools.issue92 import fixtures
    return json.loads((ROOT / "docs/issue92/fixtures_manifest.json").read_text(encoding="utf-8"))


def _build(name, entry):
    """Rebuild the production objects and the un-reduced pressure system."""
    from tools.issue92.profile_production import _multi_object, _wing_object
    import panel.p1_lifting_g1 as g1
    import panel.p1_lifting_g2 as g2

    meta = entry["meta"]
    if name.startswith("g2_"):
        mesh = _multi_object(entry)
        ops = g2.assemble_multi(mesh)
        wall = [ops["A"], ops["W"]]
        kutta_rows = ops["J"]
        pairs = [pair for group in ops["te_face_pairs"] for pair in group]
        solver = dict(mesh=mesh, wing=None, assemble=lambda: g2.assemble_multi(mesh))
        max_nfev = 80
    else:
        wing = _wing_object(name, entry)
        ops = g1.assemble(wing)
        ng = ops["W"].shape[1]
        nk = len(ops["K"])
        wall = [ops["A"], ops["W"]]
        kutta_rows = np.block([ops["C"], -ops["GH"]])
        pairs = list(ops["te_pairs"])
        solver = dict(mesh=None, wing=wing, assemble=lambda: g1.assemble(wing))
        max_nfev = 100
    areas, centers = ops["mesh"].areas, ops["mesh"].centers
    alpha = np.deg2rad(meta["alpha_deg"])
    freestream = np.array([np.cos(alpha), 0.0, np.sin(alpha)])
    weights = np.sqrt(areas / areas.mean())
    system = dict(ops=ops, B=weights[:, None] * np.column_stack(wall),
                  bw=weights * (-centers @ freestream), kutta=kutta_rows,
                  grad=ops["grad"], pairs=pairs, alpha_deg=float(meta["alpha_deg"]),
                  max_nfev=max_nfev, kind=("g2" if name.startswith("g2_") else "g1"))
    system.update(solver)
    return system


def _solve_reference(system, pressure_solver, kutta=KUTTA):
    import panel.p1_lifting_g1 as g1
    import panel.p1_lifting_g2 as g2

    started = time.perf_counter()
    if system["kind"] == "g2":
        out = g2.solve_multi(system["mesh"], system["alpha_deg"], kutta=kutta,
                             operators=system["ops"], pressure_solver=pressure_solver)
        state = np.r_[out["mu"], out["gamma"]]
    else:
        out = g1.solve(system["wing"], system["alpha_deg"], kutta=kutta,
                       pressure_solver=pressure_solver)
        state = np.r_[out["mu"], out["gamma"]]
    elapsed = time.perf_counter() - started
    return dict(state=state, seconds=elapsed, status=out.get("pressure_status"),
                success=out.get("pressure_success"),
                nfev=out.get("pressure_nfev"), message=str(out.get("pressure_message")),
                loads={key: float(out[key]) for key in ("CL", "CL_gamma", "CD_pressure", "CM_y")},
                nonlinear=out.get("nonlinear"))


def _pressure_closures(system):
    grad, pairs, nm = system["grad"], system["pairs"], system["ops"]["A"].shape[1]
    ng = int(system["B"].shape[1] - nm)

    def pressure(state):
        velocity = np.einsum("fij,j->fi", grad, state[:nm])
        return np.array([np.mean(np.sum(velocity[top] ** 2, axis=1))
                         - np.mean(np.sum(velocity[bot] ** 2, axis=1)) for top, bot in pairs])

    def jacobian(state):
        velocity = np.einsum("fij,j->fi", grad, state[:nm])
        rows = np.array([2 * np.mean(np.einsum("fi,fij->fj", velocity[top], grad[top]), axis=0)
                         - 2 * np.mean(np.einsum("fi,fij->fj", velocity[bot], grad[bot]), axis=0)
                         for top, bot in pairs])
        return np.column_stack((rows, np.zeros((rows.shape[0], ng))))

    return pressure, jacobian


def _acceptance(system, state, decision):
    """Run the production acceptance classifier on a state that came from any backend."""
    from types import SimpleNamespace
    from panel.p1_lifting_g2 import pressure_kutta_acceptance
    from scipy.linalg import null_space

    reduction = null_space(system["kutta"], rcond=RCOND)
    pressure, jacobian = _pressure_closures(system)
    z = reduction.T @ state

    def fun(t):
        value = reduction @ t
        return np.r_[system["B"] @ value - system["bw"], WEIGHT * pressure(value)]

    def jac(t):
        value = reduction @ t
        return np.vstack((system["B"], WEIGHT * jacobian(value))) @ reduction

    # Fail closed: a missing success flag silently turns every stop into the
    # "status == 0" branch and inverts the verdict (this tool shipped that bug once).
    stub = SimpleNamespace(success=bool(decision["success"]),
                           status=int(decision["status"]),
                           message=str(decision["message"]),
                           nfev=int(decision["nfev"]))
    return pressure_kutta_acceptance(result=stub, fun=fun, jac=jac, z=z,
                                     wall_operator=system["B"] @ reduction,
                                     wall_rhs=system["bw"],
                                     pressure_residual=pressure(state))


def capture(work, only=()):
    manifest = _manifest()
    work.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "dpanel.issue92.s3.capture.v1", "cases": {}}
    for name in CASES:
        if only and name not in only:
            continue
        entry = manifest["fixtures"][name]
        if entry["status"] != "ok":
            summary["cases"][name] = dict(status="unavailable", reason=entry.get("reason"))
            continue
        started = time.perf_counter()
        system = _build(name, entry)
        row = {"status": "ok", "max_nfev": system["max_nfev"], "kind": system["kind"]}
        row["assembly_seconds"] = time.perf_counter() - started
        # The linear constrained solve is the initial state of the nonlinear one; it
        # runs here because a torch process cannot call NumPy/SciPy LAPACK at all.
        signed = _solve_reference(system, None, kutta="signed_speed")
        row["signed_seconds"] = signed["seconds"]
        runs = {}
        for label in ("reference", "structured"):
            out = _solve_reference(system, label)
            if out["success"] is None:
                raise RuntimeError("%s: %s did not report pressure_success" % (name, label))
            runs[label] = dict(status=int(out["status"]), success=bool(out["success"]),
                               nfev=int(out["nfev"]),
                               message=out["message"], seconds=out["seconds"],
                               loads=out["loads"])
            np.save(work / ("%s_%s_state.npy" % (name, label)), out["state"])
            if label == "structured":
                row["structured_history"] = out["nonlinear"]
        row["runs"] = runs
        top_flat, top_ptr, bot_flat, bot_ptr = _pack_pairs(system["pairs"])
        np.savez_compressed(work / (name + "_system.npz"), B=system["B"], bw=system["bw"],
                            kutta=system["kutta"], grad=system["grad"], x0=signed["state"],
                            top_flat=top_flat, top_ptr=top_ptr, bot_flat=bot_flat,
                            bot_ptr=bot_ptr, max_nfev=np.asarray(system["max_nfev"]),
                            weight=np.asarray(WEIGHT))
        summary["cases"][name] = row
        print("%-20s reference %.1f s (nfev %d status %d) structured %.1f s (nfev %d status %d)"
              % (name, runs["reference"]["seconds"], runs["reference"]["nfev"],
                 runs["reference"]["status"], runs["structured"]["seconds"],
                 runs["structured"]["nfev"], runs["structured"]["status"]), flush=True)
    (work / "capture.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def torch(work, only=(), device="cpu"):
    """Run the torch driver on the captured systems; torch interpreter only."""
    import torch
    from panel.p1_pressure_kutta_torch_driver import solve_pressure_kutta_torch

    summary = {"schema": "dpanel.issue92.s3.torch.v1", "device": device, "cases": {}}
    for name in CASES:
        if only and name not in only:
            continue
        system_path = work / (name + "_system.npz")
        if not system_path.exists():
            continue
        # Everything comes from the capture stage: after torch runs an op in this
        # process, any NumPy/SciPy LAPACK call dies (0xC0000005), so the torch stage
        # must not assemble meshes, rebuild fixtures or solve anything else.
        with np.load(system_path) as data:
            B, bw, kutta, grad = (data[k] for k in ("B", "bw", "kutta", "grad"))
            x0 = data["x0"]
            max_nfev = int(data["max_nfev"])
            weight = float(data["weight"])
            pairs = _unpack_pairs(data["top_flat"], data["top_ptr"],
                                  data["bot_flat"], data["bot_ptr"])
        started = time.perf_counter()
        out = solve_pressure_kutta_torch(wall_operator=B, wall_rhs=bw, kutta_operator=kutta,
                                         grad=grad, te_face_pairs=pairs, x0=x0,
                                         weight=weight, max_nfev=max_nfev, device=device)
        elapsed = time.perf_counter() - started
        state = out.state.detach().to("cpu").numpy()
        np.save(work / ("%s_torch_state.npy" % name), state)
        result = out.result
        summary["cases"][name] = dict(status="ok", seconds=elapsed,
                                      status_flag=int(result.status),
                                      success=bool(result.success),
                                      nfev=int(result.nfev), nit=int(result.nit),
                                      backtracks=int(result.backtracks),
                                      message=str(result.message),
                                      optimality=float(result.optimality),
                                      cost=float(result.cost),
                                      relative_optimality=float(result.relative_optimality),
                                      n_free=int(out.reduction.shape[1]))
        print("%-20s torch %.1f s nfev %d status %d %s"
              % (name, elapsed, result.nfev, result.status, result.message), flush=True)
    (work / "torch.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def judge(work, out_path=None, only=()):
    manifest = _manifest()
    captured = json.loads((work / "capture.json").read_text(encoding="utf-8"))
    torched = json.loads((work / "torch.json").read_text(encoding="utf-8"))
    verdict = {"schema": "dpanel.issue92.s3.judge.v1", "cases": {}}
    for name in CASES:
        if only and name not in only:
            continue
        cap = captured["cases"].get(name, {})
        if cap.get("status") != "ok":
            verdict["cases"][name] = dict(status="skipped", reason=cap.get("reason", "not captured"))
            continue
        system = _build(name, manifest["fixtures"][name])
        states = {}
        for label in ("reference", "structured"):
            path = work / ("%s_%s_state.npy" % (name, label))
            if path.exists():
                states[label] = np.load(path)
        torch_state = work / ("%s_torch_state.npy" % name)
        if torch_state.exists():
            states["torch"] = np.load(torch_state)
        decisions = {label: cap["runs"][label] for label in ("reference", "structured")}
        if name in torched["cases"]:
            row = torched["cases"][name]
            decisions["torch"] = dict(status=row["status_flag"], nfev=row["nfev"],
                                      message=row["message"], seconds=row["seconds"],
                                      success=bool(row["success"]))
        from tools.issue92.s2_loads import loads_multi, loads_single

        rows = {}
        for label, state in states.items():
            if system["kind"] == "g2":
                loads = loads_multi(system["ops"], system["mesh"], system["alpha_deg"], state)
            else:
                loads = loads_single(system["ops"], system["wing"], system["alpha_deg"], state)
            acceptance = _acceptance(system, state, decisions[label])
            rows[label] = dict(status=int(decisions[label]["status"]),
                               nfev=int(decisions[label]["nfev"]),
                               message=str(decisions[label]["message"]),
                               seconds=float(decisions[label]["seconds"]),
                               loads=loads,
                               accepted=bool(acceptance["success"]),
                               accepted_at_floor=bool(acceptance["accepted_at_residual_floor"]),
                               criteria=acceptance["criteria"],
                               gradient_relative=float(acceptance["gradient_relative"]),
                               wall_over_floor=(None if acceptance["wall_over_floor"] is None
                                                else float(acceptance["wall_over_floor"])),
                               pressure_rms=float(acceptance["pressure_rms"]),
                               wall_rms=float(acceptance["wall_rms"]))
        # cross comparisons
        ref = rows.get("reference")
        table = {}
        for label, row in rows.items():
            if ref is None:
                break
            loads_delta = {key: abs(row["loads"][key] - ref["loads"][key])
                           for key in ref["loads"]}
            load_scale = max(abs(v) for v in ref["loads"].values()) or 1.0
            table[label] = dict(
                accepted_matches_reference=bool(row["accepted"] == ref["accepted"]),
                loads_absolute_delta=loads_delta,
                loads_relative=float(max(loads_delta.values()) / load_scale),
                wall_over_floor_delta=(None if row["wall_over_floor"] is None
                                       or ref["wall_over_floor"] is None
                                       else abs(row["wall_over_floor"] - ref["wall_over_floor"])),
                gradient_relative_delta=abs(row["gradient_relative"] - ref["gradient_relative"]),
                pressure_rms_delta=abs(row["pressure_rms"] - ref["pressure_rms"]))
        verdict["cases"][name] = dict(status="ok", kind=system["kind"], rows=rows,
                                      comparisons=table)
        print("%-20s ref accepted=%s floor=%.3f | structured accepted=%s floor=%.3f | torch accepted=%s floor=%.3f"
              % (name, ref["accepted"], ref["wall_over_floor"] or -1,
                 rows["structured"]["accepted"], rows["structured"]["wall_over_floor"] or -1,
                 rows["torch"]["accepted"], rows["torch"]["wall_over_floor"] or -1), flush=True)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    return verdict


def _pack_pairs(pairs):
    top_flat, top_ptr, bot_flat, bot_ptr = [], [0], [], [0]
    for top, bot in pairs:
        top_flat.extend(int(i) for i in top)
        bot_flat.extend(int(i) for i in bot)
        top_ptr.append(len(top_flat))
        bot_ptr.append(len(bot_flat))
    return (np.asarray(top_flat, int), np.asarray(top_ptr, int),
            np.asarray(bot_flat, int), np.asarray(bot_ptr, int))


def _unpack_pairs(top_flat, top_ptr, bot_flat, bot_ptr):
    tops = np.split(np.asarray(top_flat, int), np.asarray(top_ptr, int)[1:-1])
    bots = np.split(np.asarray(bot_flat, int), np.asarray(bot_ptr, int)[1:-1])
    return [(list(map(int, t)), list(map(int, b))) for t, b in zip(tops, bots)]


def bundle(name, out_path):
    """Freeze one complete torch-gated case for the test suite."""
    manifest = _manifest()
    system = _build(name, manifest["fixtures"][name])
    if system["kind"] == "g2":
        import panel.p1_lifting_g2 as g2
        signed = g2.solve_multi(system["mesh"], system["alpha_deg"], kutta="signed_speed",
                                operators=system["ops"])
    else:
        import panel.p1_lifting_g1 as g1
        signed = g1.solve(system["wing"], system["alpha_deg"], kutta="signed_speed")
    x0 = np.r_[signed["mu"], signed["gamma"]]
    structured = _solve_reference(system, "structured")
    top_flat, top_ptr, bot_flat, bot_ptr = _pack_pairs(system["pairs"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, B=system["B"], bw=system["bw"], kutta=system["kutta"],
                        grad=system["grad"], x0=x0, expected=structured["state"],
                        top_flat=top_flat, top_ptr=top_ptr, bot_flat=bot_flat,
                        bot_ptr=bot_ptr,
                        max_nfev=np.asarray(system["max_nfev"]),
                        weight=np.asarray(WEIGHT),
                        status=np.asarray(structured["status"]),
                        nfev=np.asarray(structured["nfev"]),
                        message=np.asarray(structured["message"]))
    return dict(case=name, status=int(structured["status"]), nfev=int(structured["nfev"]),
                message=structured["message"], path=str(out_path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("capture", "torch", "judge", "bundle"))
    parser.add_argument("--work", type=Path, default=ROOT / "runs/scratch/issue92_loose/s3")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--only", nargs="*", default=())
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.stage == "capture":
        capture(args.work, args.only)
    elif args.stage == "torch":
        torch(args.work, args.only, args.device)
    elif args.stage == "bundle":
        target = args.out or (ROOT / "docs/issue92/s3_reference_g1_wing_rect.npz")
        print(json.dumps(bundle("g1_wing_rect", target)))
    else:
        judge(args.work, args.out, args.only)


if __name__ == "__main__":
    main()
