"""Issue #92 S7: freeze the production reference solution before the old path goes.

Route B ends by deleting the NumPy implementations from `panel/` once the torch chain
owns the default. That deletion is only safe if the answers of the before state are
recoverable, so this tool writes one small bundle per frozen fixture holding the
production default solve (SciPy trust-region pressure Kutta, `pressure_solver='reference'`)
together with the decisions that identify it and the rank spectrum that decided the
constraint truncation, plus a `verify` stage that re-runs the live code against those
bundles.

Two stages, both need SciPy:

  python -m tools.issue92.s7_reference_export export
  python -m tools.issue92.s7_reference_export verify

The bundles are `docs/issue92/references/<fixture>.npz` and the index, with one sha256
per bundle and the gap table, is `docs/issue92/s7_reference_index.json`.

What a bundle keeps, and why:

* `mu`, `gamma`, `cp` - the full state the loads and every downstream consumer see;
* `CL`, `CL_gamma`, `CD_pressure`, `CM_y` plus the four residuals - the delivered
  quantities;
* `pressure_status/success/nfev/message` - the decisions a port must reproduce exactly;
* the constraint spectrum with `rcond`, the truncation threshold and the margin of the
  decisive singular value over it - the only place where a 1e-14 backend difference can
  legitimately change a decision (section 3 of the port map).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("PANEL_ROOT", Path.cwd()))

import numpy as np

from tools.issue92.s3_pressure_parity import CASES, RCOND, _build, _manifest

SCHEMA = "dpanel.issue92.s7.reference.v1"
OUT = ROOT / "docs/issue92/references"
INDEX = ROOT / "docs/issue92/s7_reference_index.json"
LOADS = ("CL", "CL_gamma", "CD_pressure", "CM_y")
RESIDUALS = ("wall_rms", "jump_rms", "signed_kutta_rms", "pressure_kutta_rms")


def _spectrum(system, backend="numpy", device="cpu"):
    """Singular values of the constraint matrix and the margin over the rcond cut.

    The torch backend takes the spectrum in torch as well: a NumPy LAPACK call after
    a torch kernel kills the process (0xC0000005), and this function runs after the
    solve on every case of the verify loop.
    """
    constraint = system["kutta"]
    if backend == "torch":
        import torch

        singular = torch.linalg.svdvals(
            torch.as_tensor(np.asarray(constraint, float), dtype=torch.float64,
                            device=device)).detach().cpu().numpy()
    else:
        singular = np.linalg.svd(constraint, compute_uv=False)
    threshold = singular[0] * RCOND
    kept = singular >= threshold
    kept_values = singular[kept]
    discarded = singular[~kept]
    margin = float(kept_values[-1] / threshold) if kept.any() else 0.0
    gap = (float(kept_values[-1] / discarded[0]) if discarded.size else None)
    return dict(singular=singular, threshold=float(threshold), rank=int(kept.sum()),
                margin_over_threshold=margin, gap_ratio=gap,
                smallest_kept=float(kept_values[-1]) if kept.any() else None,
                largest_discarded=float(discarded[0]) if discarded.size else None)


def _solve(system, pressure_solver="reference", backend="numpy", device="cpu"):
    import panel.p1_lifting_g1 as g1
    import panel.p1_lifting_g2 as g2

    if backend == "torch":
        # the torch path implements the structured solver only
        pressure_solver = None
    if system["kind"] == "g2":
        out = g2.solve_multi(system["mesh"], system["alpha_deg"], kutta="pressure",
                             operators=system["ops"], pressure_solver=pressure_solver,
                             backend=backend, device=device)
        rank = out.get("linear_constrained_rank")
    else:
        out = g1.solve(system["wing"], system["alpha_deg"], kutta="pressure",
                       pressure_solver=pressure_solver, backend=backend, device=device,
                       wall_formulation="collocation")
        rank = out.get("constrained_rank")
    return out, rank


def _sha256(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def stage_export(cases, force):
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = _manifest()["fixtures"]
    index = {"schema": SCHEMA, "rcond": RCOND, "kutta": "pressure",
             "pressure_solver": "reference", "cases": {}}
    for name in cases:
        entry = manifest[name]
        system = _build(name, entry)
        spectrum = _spectrum(system)
        out, rank = _solve(system)
        arrays = dict(mu=np.asarray(out["mu"], dtype=np.float64),
                      gamma=np.asarray(out["gamma"], dtype=np.float64),
                      cp=np.asarray(out["cp"], dtype=np.float64),
                      singular=spectrum["singular"])
        scalars = {key: float(out[key]) for key in LOADS + RESIDUALS}
        decisions = dict(pressure_status=out.get("pressure_status"),
                         pressure_success=None if out.get("pressure_success") is None
                         else bool(out["pressure_success"]),
                         pressure_nfev=out.get("pressure_nfev"),
                         pressure_message=str(out.get("pressure_message")),
                         constrained_rank=rank, spectrum_rank=spectrum["rank"])
        path = OUT / ("%s.npz" % name)
        if path.exists() and not force:
            raise SystemExit("%s exists; pass --force to overwrite" % path)
        np.savez_compressed(path, alpha_deg=np.float64(system["alpha_deg"]),
                            kind=np.array(system["kind"]), **arrays,
                            **{key: np.float64(value) for key, value in scalars.items()},
                            **{key: np.array(value) for key, value in decisions.items()})
        index["cases"][name] = dict(
            sha256=_sha256(path), bytes=path.stat().st_size, kind=system["kind"],
            alpha_deg=system["alpha_deg"], loads=scalars, decisions=decisions,
            spectrum={key: spectrum[key] for key in
                      ("threshold", "rank", "margin_over_threshold", "gap_ratio",
                       "smallest_kept", "largest_discarded")})
        print("%-22s rank=%-5d margin=%.3e gap=%s loads CL=%.12f"
              % (name, spectrum["rank"], spectrum["margin_over_threshold"],
                 "inf" if spectrum["gap_ratio"] is None else "%.3e" % spectrum["gap_ratio"],
                 scalars["CL"]))
    INDEX.write_text(json.dumps(index, indent=1), encoding="utf-8")
    print("wrote %s" % INDEX)


def stage_verify(cases, tolerances, backend="numpy", device="cpu"):
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    manifest = _manifest()["fixtures"]
    failures = []
    for name in cases:
        record = index["cases"][name]
        path = OUT / ("%s.npz" % name)
        if _sha256(path) != record["sha256"]:
            failures.append((name, "bundle hash"))
            print("%-22s FAIL bundle hash" % name)
            continue
        bundle = np.load(path, allow_pickle=False)
        system = _build(name, manifest[name])
        spectrum = _spectrum(system, backend, device)
        out, rank = _solve(system, backend=backend, device=device)
        problems = []
        # status/nfev/message belong to the algorithm that produced the bundle (SciPy
        # trust-region-reflective). The torch backend implements the structured solver,
        # so only the decision that both algorithms agree on - acceptance - is compared.
        # S3 section 10.5 shows the stopping rule that fires is roundoff, not a property.
        keys = ("constrained_rank",) if backend != "numpy" else (
            "pressure_status", "pressure_nfev", "pressure_message", "constrained_rank")
        for key in keys:
            recorded = bundle[key].item() if bundle[key].ndim == 0 else bundle[key].tolist()
            live = {"pressure_status": out.get("pressure_status"),
                    "pressure_nfev": out.get("pressure_nfev"),
                    "pressure_message": str(out.get("pressure_message")),
                    "constrained_rank": rank}[key]
            if recorded != live:
                problems.append("%s %r != %r" % (key, recorded, live))
        recorded_success = bundle["pressure_success"].item()
        live_success = out.get("pressure_success")
        if recorded_success != (None if live_success is None else bool(live_success)):
            problems.append("pressure_success")
        load_error = 0.0
        load_limit = (tolerances["loads_relative"] if backend == "numpy"
                      else tolerances["torch_loads_relative"])
        for key in LOADS:
            recorded = float(bundle[key])
            live = float(np.asarray(out[key]))
            error = abs(live - recorded) / max(abs(recorded), 1e-30)
            load_error = max(load_error, error)
            if error > load_limit:
                problems.append("%s %.3e" % (key, error))
        state_error = 0.0
        for key in ("mu", "gamma", "cp"):
            recorded = np.asarray(bundle[key])
            live = np.asarray(out[key])
            error = float(np.linalg.norm(live - recorded)
                          / max(np.linalg.norm(recorded), 1e-30))
            state_error = max(state_error, error)
            limit = (tolerances["cp_relative"] if key == "cp"
                     else tolerances["state_relative"])
            if backend != "numpy":
                limit = (tolerances["torch_cp_relative"] if key == "cp"
                         else tolerances["torch_state_relative"])
            if error > limit:
                problems.append("%s %.3e" % (key, error))
        # relative: the margin is ~1e10, so an absolute 1e-12 tolerance would fail on
        # the last-bit difference between two SVD implementations.
        recorded_margin = record["spectrum"]["margin_over_threshold"]
        if abs(spectrum["margin_over_threshold"] - recorded_margin) > (
                1e-12*max(abs(recorded_margin), 1e-30)):
            problems.append("spectrum margin %.6e != %.6e"
                            % (spectrum["margin_over_threshold"], recorded_margin))
        if problems:
            failures.append((name, "; ".join(problems)))
            print("%-22s FAIL %s" % (name, "; ".join(problems)))
        else:
            print("%-22s pass  loads=%.1e state=%.1e rank=%d margin=%.3e"
                  % (name, load_error, state_error, spectrum["rank"],
                     spectrum["margin_over_threshold"]))
    print("backend=%s device=%s failures: %r"
          % (backend, device, [name for name, _ in failures]))
    return 0 if not failures else 1


def main(argv=None):
    global INDEX
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("export", "verify"))
    parser.add_argument("--case", action="append", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--loads-relative", type=float, default=1e-9)
    parser.add_argument("--state-relative", type=float, default=1e-9)
    parser.add_argument("--cp-relative", type=float, default=1e-9)
    parser.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-loads-relative", type=float, default=1e-6)
    parser.add_argument("--torch-state-relative", type=float, default=1e-6)
    parser.add_argument("--torch-cp-relative", type=float, default=1e-4)
    parser.add_argument("--index", default=str(INDEX))
    args = parser.parse_args(argv)
    INDEX = Path(args.index)
    cases = tuple(args.case) if args.case else CASES
    if args.stage == "export":
        stage_export(cases, args.force)
        return 0
    return stage_verify(cases, dict(loads_relative=args.loads_relative,
                                    state_relative=args.state_relative,
                                    cp_relative=args.cp_relative,
                                    torch_loads_relative=args.torch_loads_relative,
                                    torch_state_relative=args.torch_state_relative,
                                    torch_cp_relative=args.torch_cp_relative),
                        backend=args.backend, device=args.device)


if __name__ == "__main__":
    raise SystemExit(main())
