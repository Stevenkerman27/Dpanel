"""Issue #92 S2: parity of the rank-aware decision points on real fixture systems.

Three stages, run by different interpreters (a NumPy LAPACK call after torch has
executed in the same process dies with 0xC0000005, section 6f of the phase-1 doc):

  python -m tools.issue92.s2_rank_parity capture --out runs/issue92/s2   (.conda_env)
  python -m tools.issue92.s2_rank_parity torch   --work runs/issue92/s2  (torch env)
  python -m tools.issue92.s2_rank_parity compare --work runs/issue92/s2  (.conda_env)

The captured object is the *initial constrained solve* of each fixture, rebuilt from
the operators the production solve actually returned:

    N = null_space(constraints, rcond=1e-11)
    t = lstsq(B @ N, bw, lapack_driver="gelsd")
    z = N @ t

That pair of calls is the only rank-sensitive decision in G1 and G2; everything
after it is nonlinear iteration (S3).

What is compared, and why it is comparable:

* z = N @ t in the original unknown space - the quantity the solver actually uses,
  and invariant under a change of null-space basis;
* rank and null-space dimension;
* the projector N N^T (basis independent) - the basis itself is NOT comparable, and
  the first measurement showed why: both backends return different orthonormal bases
  of the same null space (projector agrees to 4e-16) while individual columns rotate;
* singular values of B @ N (basis invariant);
* the margin of the decisive singular value over the truncation threshold and the gap
  ratio (smallest kept / largest discarded), which bound how far a backend difference
  can move the decision.
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


def _load_manifest():
    from tools.issue92 import fixtures
    return fixtures, json.loads((ROOT / "docs/issue92/fixtures_manifest.json").read_text(encoding="utf-8"))


def _initial_system(name, entry):
    """Solve a fixture and rebuild the un-reduced system its constrained solve used."""
    from tools.issue92.profile_production import _multi_object, _wing_object
    import panel.p1_lifting_g1 as g1
    import panel.p1_lifting_g2 as g2

    meta = entry["meta"]
    wing = None
    body = None
    mesh = None
    if name.startswith("g2_"):
        mesh = _multi_object(entry)
        # solve_multi does not return its operators, so assemble once and hand them in.
        ops = g2.assemble_multi(mesh)
        started = time.perf_counter()
        result = g2.solve_multi(mesh, meta["alpha_deg"], kutta="signed_speed", operators=ops)
        elapsed = time.perf_counter() - started
        wall = [ops["A"], ops["W"]]
        constraints = [ops["J"], ops["K"]]
        areas, centers = ops["mesh"].areas, ops["mesh"].centers
    else:
        wing = _wing_object(name, entry)
        started = time.perf_counter()
        result = g1.solve(wing, meta["alpha_deg"], kutta="signed_speed")
        elapsed = time.perf_counter() - started
        ops = result["ops"]
        wall = [ops["A"], ops["W"]]
        ng = ops["W"].shape[1]
        nk = len(ops["K"])
        constraints = [np.block([ops["C"], -ops["GH"]]),
                       np.block([ops["K"], np.zeros((nk, ng))])]
        areas, centers = ops["mesh"].areas, ops["mesh"].centers
    alpha = np.deg2rad(meta["alpha_deg"])
    freestream = np.array([np.cos(alpha), 0.0, np.sin(alpha)])
    rhs = -centers @ freestream
    weights = np.sqrt(areas / areas.mean())
    B = weights[:, None] * np.column_stack(wall)
    bw = weights * rhs
    constraints = np.vstack(constraints)
    if name.startswith("g2_"):
        body = mesh
    production = {key: float(result[key]) for key in ("CL", "CL_gamma", "CD_pressure", "CM_y")}
    return dict(B=B, bw=bw, constraints=constraints, elapsed=elapsed,
                production=production, ops=ops, wing=wing, body=body,
                alpha_deg=float(meta["alpha_deg"]))


def _loads(system, state):
    from tools.issue92.s2_loads import loads_multi, loads_single

    if system["body"] is not None:
        return loads_multi(system["ops"], system["body"], system["alpha_deg"], state)
    return loads_single(system["ops"], system["wing"], system["alpha_deg"], state)


def _numpy_decision(B, bw, constraints, z_expected=None):
    from scipy.linalg import lstsq as scipy_lstsq
    from scipy.linalg import null_space

    N = null_space(constraints, rcond=RCOND)
    z, _, rank, singular = scipy_lstsq(B @ N, bw, lapack_driver="gelsd")
    csingular = np.linalg.svd(constraints, compute_uv=False)
    threshold = csingular[0] * RCOND
    kept = csingular[csingular > threshold]
    discarded = csingular[csingular <= threshold]
    state = N @ z
    return dict(z=state, t=z, basis=N, rank=int(rank), singular=singular,
                null_dim=int(N.shape[1]), null_rank=int(kept.size),
                null_largest=float(csingular[0]),
                null_smallest_kept=float(kept.min()) if kept.size else 0.0,
                null_largest_discarded=float(discarded.max()) if discarded.size else 0.0)


def capture(work, only=()):
    fixtures, manifest = _load_manifest()
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    report = {"schema": "dpanel.issue92.s2.numpy.v1", "rcond": RCOND, "cases": {}}
    for name in (only or CASES):
        entry = manifest["fixtures"][name]
        if entry["status"] != "ok":
            report["cases"][name] = {"status": "unavailable", "reason": entry.get("reason")}
            continue
        system = _initial_system(name, entry)
        decision = _numpy_decision(system["B"], system["bw"], system["constraints"])
        np.savez(work / (name + ".npz"), B=system["B"], bw=system["bw"],
                 constraints=system["constraints"], z=decision["z"], t=decision["t"],
                 basis=decision["basis"], singular=decision["singular"])
        report["cases"][name] = {"status": "ok", "seconds": round(system["elapsed"], 3),
                                 "production_loads": system["production"],
                                 "shapes": {"wall": list(system["B"].shape),
                                            "constraints": list(system["constraints"].shape)},
                                 "rank": decision["rank"], "null_dim": decision["null_dim"],
                                 "null_rank": decision["null_rank"],
                                 "null_largest": decision["null_largest"],
                                 "null_smallest_kept": decision["null_smallest_kept"],
                                 "null_largest_discarded": decision["null_largest_discarded"],
                                 "singular": [float(v) for v in decision["singular"][:6]],
                                 "state_norm": float(np.linalg.norm(decision["z"]))}
        print("%-20s wall=%s constraints=%s rank=%-5d null=%-5d seconds=%6.2f" % (
            name, tuple(system["B"].shape), tuple(system["constraints"].shape),
            decision["rank"], decision["null_dim"], system["elapsed"]))
    (work / "s2_rank_numpy.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def torch_stage(work, device="cpu"):
    import torch

    from panel.p1_rank_aware_torch import constrained_solve

    work = Path(work)
    numpy_report = json.loads((work / "s2_rank_numpy.json").read_text(encoding="utf-8"))
    report = {"schema": "dpanel.issue92.s2.torch.v1", "device": device, "cases": {}}
    for name, entry in numpy_report["cases"].items():
        if entry.get("status") != "ok":
            report["cases"][name] = entry
            continue
        data = np.load(work / (name + ".npz"))
        started = time.perf_counter()
        state, decisions = constrained_solve(data["B"], data["bw"], data["constraints"],
                                             rcond_null=RCOND, device=device)
        elapsed = time.perf_counter() - started
        np.savez(work / "torch" / (name + ".npz"), z=state.detach().cpu().numpy())
        report["cases"][name] = {"status": "ok", "seconds": round(elapsed, 3),
                                 "null_space": decisions["null_space"],
                                 "least_squares": {key: value for key, value in decisions["least_squares"].items()
                                                   if key != "singular"},
                                 "singular": [float(v) for v in decisions["least_squares"]["singular"][:6]],
                                 "state_norm": float(torch.linalg.vector_norm(state))}
    (work / "torch" / "s2_rank_torch.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def loads_stage(work, out):
    """Apply one load map to both states, and check that map against production."""
    from tools.issue92.s2_loads import loads_multi, loads_single

    fixtures, manifest = _load_manifest()
    work = Path(work)
    numpy_report = json.loads((work / "s2_rank_numpy.json").read_text(encoding="utf-8"))
    result = {"schema": "dpanel.issue92.s2.loads.v1", "cases": {}}
    for name, entry in numpy_report["cases"].items():
        if entry.get("status") != "ok":
            result["cases"][name] = {"status": "skipped"}
            continue
        system = _initial_system(name, manifest["fixtures"][name])
        z_numpy = np.load(work / (name + ".npz"))["z"]
        z_torch = np.load(work / "torch" / (name + ".npz"))["z"]
        numpy_loads = _loads(system, z_numpy)
        torch_loads = _loads(system, z_torch)
        helper_gap = {key: abs(numpy_loads[key] - system["production"][key]) for key in numpy_loads}
        result["cases"][name] = {
            "status": "ok",
            "production": system["production"],
            "numpy": numpy_loads,
            "torch": torch_loads,
            "helper_minus_production": helper_gap,
            "relative": {key: abs(numpy_loads[key] - torch_loads[key])
                         / max(abs(numpy_loads[key]), 1e-12) for key in numpy_loads},
        }
        print("%-20s helper|d|=%s rel=%s" % (
            name, {"%s" % key: "%.1e" % value for key, value in helper_gap.items()},
            {"%s" % key: "%.1e" % value for key, value in result["cases"][name]["relative"].items()}))
    Path(out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def compare(work, out):
    numpy_report = json.loads((Path(work) / "s2_rank_numpy.json").read_text(encoding="utf-8"))
    torch_report = json.loads((Path(work) / "torch" / "s2_rank_torch.json").read_text(encoding="utf-8"))
    result = {"schema": "dpanel.issue92.s2.parity.v1", "cases": {}}
    for name, entry in numpy_report["cases"].items():
        other = torch_report["cases"].get(name, {})
        if entry.get("status") != "ok" or other.get("status") != "ok":
            result["cases"][name] = {"status": "skipped"}
            continue
        data = np.load(Path(work) / (name + ".npz"))
        torch_data = np.load(Path(work) / "torch" / (name + ".npz"))
        z_numpy, z_torch = data["z"], torch_data["z"]
        scale = max(float(np.linalg.norm(z_numpy)), 1e-300)
        basis = data["basis"]
        own = other["least_squares"]["report"]
        singular_numpy = np.array(entry["singular"], float)
        singular_torch = np.array(other["singular"], float)
        gap_numpy = (entry["null_smallest_kept"] / entry["null_largest_discarded"]
                     if entry["null_largest_discarded"] else None)
        gap_torch = other["null_space"]["gap_ratio"]
        result["cases"][name] = {
            "status": "ok",
            "rank_numpy": entry["rank"], "rank_torch": own["rank"],
            "null_dim_numpy": entry["null_dim"], "null_dim_torch": other["null_space"]["null_dim"],
            "state_relative": float(np.linalg.norm(z_numpy - z_torch) / scale),
            "state_max_relative": float(np.max(np.abs(z_numpy - z_torch)) / max(float(np.max(np.abs(z_numpy))), 1e-300)),
            "singular_relative": float(np.max(np.abs(singular_numpy - singular_torch))
                                       / max(float(singular_numpy[0]), 1e-300)),
            # margin = how many times the decisive singular value clears the truncation
            # threshold; both sides are normalised the same way.
            "margin_over_threshold_numpy": (entry["null_smallest_kept"]
                                            / (entry["null_largest"] * RCOND)
                                            if entry["null_largest"] else None),
            "margin_over_threshold_torch": other["null_space"]["margin"],
            "gap_ratio_numpy": gap_numpy,
            "gap_ratio_torch": (None if gap_torch == float("inf") else gap_torch),
            "state_norm": entry["state_norm"],
        }
    Path(out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--work", type=Path, default=ROOT / "runs/issue92/s2")
    cap.add_argument("--only", nargs="*", default=None)
    tor = sub.add_parser("torch")
    tor.add_argument("--work", type=Path, default=ROOT / "runs/issue92/s2")
    tor.add_argument("--device", default="cpu")
    loa = sub.add_parser("loads")
    loa.add_argument("--work", type=Path, default=ROOT / "runs/issue92/s2")
    loa.add_argument("--out", type=Path, default=ROOT / "runs/issue92/s2/s2_loads.json")
    com = sub.add_parser("compare")
    com.add_argument("--work", type=Path, default=ROOT / "runs/issue92/s2")
    com.add_argument("--out", type=Path, default=ROOT / "docs/issue92/s2_rank_parity.json")
    args = parser.parse_args(argv)
    if args.stage == "capture":
        report = capture(args.work, only=args.only or ())
    elif args.stage == "torch":
        report = torch_stage(args.work, device=args.device)
    elif args.stage == "loads":
        report = loads_stage(args.work, args.out)
    else:
        report = compare(args.work, args.out)
    print(json.dumps({name: value.get("status") for name, value in report["cases"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
