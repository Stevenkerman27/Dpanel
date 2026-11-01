"""Parity and timing for the batched P1 influence kernel (issue 92, Phase 1 step 1).

For every frozen fixture this compares the reference NumPy assembly with the batched
float64 kernel (A and W separately), then re-solves with the batched matrices to compare
loads, so a parity claim is anchored on the quantity the solver actually consumes.
Timing is reported per stage, with the reference and the batched path measured on the
same process and the same geometry.

Usage:
  python -m tools.issue92.influence_parity --out docs/issue92/influence_parity.json
  python -m tools.issue92.influence_parity --only g1_wing_rect --repeat 5
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

from panel import p1_lifting_g1
from panel.p1_batched_influence import (DEFAULT_SOURCE_BLOCK, DEFAULT_TARGET_BLOCK,
                                    influence_matrix, wake_matrix)
from panel.p1_lifting_g1 import _mesh_as_kernel, assemble, solve
from panel.p1_lifting_g2 import assemble_multi, solve_multi
from tools.issue92 import fixtures, profile_production

FINGERPRINT_KEYS = ("CL", "CL_gamma", "CD_pressure", "CM_y", "wall_rms",
                    "pressure_kutta_rms", "signed_kutta_rms", "jump_rms",
                    "constrained_rank")


def _versions():
    out = {"python": platform.python_version(), "numpy": np.__version__,
           "cpu": platform.processor() or platform.machine()}
    try:
        import torch
        out["torch"] = torch.__version__
        out["torch_cuda"] = bool(torch.cuda.is_available())
    except ModuleNotFoundError:
        out["torch"] = None
    return out


def _time_call(func, repeat):
    func()
    best = float("inf")
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        out = func()
        best = min(best, time.perf_counter() - t0)
    return out, best


def _compare(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("shape mismatch " + str(a.shape) + " vs " + str(b.shape))
    diff = float(np.max(np.abs(a - b))) if a.size else 0.0
    scale = float(np.max(np.abs(a))) if a.size else 0.0
    denom = np.maximum(np.abs(a), 1e-300)
    return {"shape": list(a.shape), "max_abs": diff,
            "max_rel_to_reference_scale": diff / scale if scale else 0.0,
            "max_elementwise_relative": float(np.max(np.abs(a - b) / denom)) if a.size else 0.0,
            "reference_max_abs": scale}


def _wing_pair(name, entry, repeat, target_block, source_block, save_dir=None):
    wing = profile_production._wing_object(name, entry)
    mesh = _mesh_as_kernel(wing)
    nface = len(wing.faces)
    self_faces = np.arange(nface)
    dofs = np.asarray(wing.dof_face, dtype=np.int64)
    ndof = int(wing.ndof_mu)

    A_ref, t_ref_a = _time_call(
        lambda: p1_lifting_g1._doublet_matrix(mesh.triangles, mesh.centers,
                                            self_faces=self_faces, face_dofs=dofs, ndof=ndof),
        repeat)

    def reference_wake():
        W = np.zeros((nface, wing.gamma_hat.shape[1]))
        for f in wing.wake_faces:
            D = p1_lifting_g1._doublet_matrix(wing.wake_xyz[f][None], mesh.centers,
                                              face_dofs=np.arange(3)[None], ndof=3)
            W = W + D @ wing.gamma_hat[f]
        return W

    W_ref, t_ref_w = _time_call(reference_wake, max(1, repeat - 1))
    A_bat, t_bat_a = _time_call(
        lambda: influence_matrix(mesh.triangles, mesh.centers, self_faces=self_faces,
                                 face_dofs=dofs, ndof=ndof, target_block=target_block,
                                 source_block=source_block),
        repeat)
    W_bat, t_bat_w = _time_call(
        lambda: wake_matrix(wing.wake_xyz, wing.wake_faces, mesh.centers,
                            target_block=target_block, source_block=source_block)
                @ np.asarray(wing.gamma_hat, dtype=np.float64),
        repeat)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_dir / (name + "_reference.npz"), A=A_ref, W=W_ref)
    ops_ref = assemble(wing)
    record = {"kind": "wing", "A": _compare(A_ref, A_bat), "W": _compare(W_ref, W_bat),
              "seconds": {"reference_A": t_ref_a, "batched_A": t_bat_a,
                          "reference_W": t_ref_w, "batched_W": t_bat_w,
                          "reference_W_note": "per-wake-triangle reference loop",
                          "speedup_A": t_ref_a / t_bat_a if t_bat_a else None,
                          "speedup_W": t_ref_w / t_bat_w if t_bat_w else None}}
    record["A_matches_assembled"] = _compare(ops_ref["A"], A_bat)
    record["W_matches_assembled"] = _compare(ops_ref["W"], W_bat)
    ops_torch, t_prod_torch = _time_call(lambda: assemble(wing, backend="torch"), repeat)
    _, t_prod_numpy = _time_call(lambda: assemble(wing), repeat)
    record["assembly_backend"] = {
        "numpy": t_prod_numpy, "torch": t_prod_torch,
        "speedup": t_prod_numpy / t_prod_torch if t_prod_torch else None,
        "A": _compare(ops_ref["A"], ops_torch["A"]),
        "W": _compare(ops_ref["W"], ops_torch["W"])}
    if nface <= 2000:
        sv = np.linalg.svd(A_ref, compute_uv=False)
        sv_b = np.linalg.svd(A_bat, compute_uv=False)
        record["singular_values"] = {"max_abs": float(np.max(np.abs(sv - sv_b))),
                                     "sigma_max": float(sv[0]),
                                     "sigma_min": float(sv[-1]),
                                     "condition": float(sv[0] / sv[-1])}
    ops_bat = dict(ops_ref)
    ops_bat["A"] = A_bat
    ops_bat["W"] = W_bat
    original = p1_lifting_g1.assemble
    try:
        p1_lifting_g1.assemble = lambda w, **kw: ops_bat
        bat_result = solve(wing, entry["meta"]["alpha_deg"], kutta=entry["meta"]["kutta"])
    finally:
        p1_lifting_g1.assemble = original
    ref_result = solve(wing, entry["meta"]["alpha_deg"], kutta=entry["meta"]["kutta"])
    record["loads"] = {}
    for key in FINGERPRINT_KEYS:
        a = ref_result.get(key)
        b = bat_result.get(key)
        if a is None or b is None:
            record["loads"][key] = {"reference": a, "batched": b, "delta": None}
            continue
        record["loads"][key] = {"reference": float(a), "batched": float(b),
                                "delta": float(b) - float(a)}
    record["state"] = {}
    for key in ("mu", "gamma", "cp"):
        a, b = ref_result.get(key), bat_result.get(key)
        if a is None or b is None:
            record["state"][key] = None
            continue
        record["state"][key] = _compare(np.asarray(a, float), np.asarray(b, float))
    return record


def _multi_pair(name, entry, repeat, target_block, source_block, save_dir=None):
    mesh = profile_production._multi_object(entry)
    ops_ref = assemble_multi(mesh)
    ms = ops_ref["mesh"]
    nface = len(mesh.faces)
    dofs = np.asarray(mesh.dof_face, dtype=np.int64)
    ndof = int(mesh.ndof_mu)
    A_ref, t_ref_a = _time_call(
        lambda: p1_lifting_g1._doublet_matrix(ms.triangles, ms.centers, self_faces=np.arange(nface),
                                            face_dofs=dofs, ndof=ndof), repeat)

    def reference_wake():
        parts = []
        for group in mesh.groups:
            wake_tris = np.asarray(group.wake_xyz, float)[np.asarray(group.wake_faces, int)]
            coefficients = p1_lifting_g1._doublet_matrix(
                wake_tris, ms.centers, face_dofs=np.asarray(group.wake_faces, int),
                ndof=len(group.wake_xyz))
            parts.append(coefficients @ np.asarray(group.gamma_hat, float))
        return np.column_stack(parts)

    W_ref, t_ref_w = _time_call(reference_wake, max(1, repeat - 1))
    A_bat, t_bat_a = _time_call(
        lambda: influence_matrix(ms.triangles, ms.centers, self_faces=np.arange(nface),
                                 face_dofs=dofs, ndof=ndof, target_block=target_block,
                                 source_block=source_block), repeat)
    W_bat, t_bat_w = _time_call(
        lambda: np.column_stack([wake_matrix(group.wake_xyz, group.wake_faces, ms.centers,
                                             target_block=target_block,
                                             source_block=source_block)
                                 @ np.asarray(group.gamma_hat, float)
                                 for group in mesh.groups]), repeat)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_dir / (name + "_reference.npz"), A=A_ref, W=W_ref)
    ops_torch, t_prod_torch = _time_call(lambda: assemble_multi(mesh, backend="torch"), repeat)
    _, t_prod_numpy = _time_call(lambda: assemble_multi(mesh), repeat)
    alpha = entry["meta"]["alpha_deg"]
    kutta = entry["meta"]["kutta"]
    ref_solve = solve_multi(mesh, alpha, kutta=kutta)
    bat_solve = solve_multi(mesh, alpha, kutta=kutta, operators=ops_torch)
    loads = {}
    for key in FINGERPRINT_KEYS:
        a, b = ref_solve.get(key), bat_solve.get(key)
        loads[key] = ({"reference": None, "torch": None, "delta": None} if a is None
                      else {"reference": float(a), "torch": float(b),
                            "delta": float(b) - float(a)})
    return {"kind": "multi", "A": _compare(A_ref, A_bat), "W": _compare(W_ref, W_bat),
            "A_matches_assembled": _compare(ops_ref["A"], A_bat),
            "W_matches_assembled": _compare(ops_ref["W"], W_bat),
            "assembly_backend": {"numpy": t_prod_numpy, "torch": t_prod_torch,
                                 "speedup": t_prod_numpy / t_prod_torch if t_prod_torch else None,
                                 "A": _compare(ops_ref["A"], ops_torch["A"]),
                                 "W": _compare(ops_ref["W"], ops_torch["W"])},
            "loads": loads,
            "seconds": {"reference_A": t_ref_a, "batched_A": t_bat_a,
                        "reference_W": t_ref_w, "batched_W": t_bat_w,
                        "speedup_A": t_ref_a / t_bat_a if t_bat_a else None,
                        "speedup_W": t_ref_w / t_bat_w if t_bat_w else None}}


def main(argv=None):
    parser = argparse.ArgumentParser(description="batched influence kernel parity")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/issue92/influence_parity.json")
    parser.add_argument("--only", default=None)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--target-block", type=int, default=DEFAULT_TARGET_BLOCK)
    parser.add_argument("--source-block", type=int, default=DEFAULT_SOURCE_BLOCK)
    parser.add_argument("--save-reference", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=None,
                        help="torch CPU threads; the block sweep found 8 fastest on this host")
    args = parser.parse_args(argv)
    import torch
    if args.threads is not None:
        torch.set_num_threads(int(args.threads))
    manifest = fixtures.load_manifest()
    names = ([item.strip() for item in args.only.split(",") if item.strip()]
             if args.only else sorted(manifest["fixtures"]))
    payload = {"status": "ok", "versions": _versions(),
               "torch_threads": int(torch.get_num_threads()),
               "block": {"target_block": args.target_block,
                         "source_block": args.source_block},
               "repeat": args.repeat, "fixtures": {}}
    for name in names:
        entry = manifest["fixtures"][name]
        if not fixtures.available(name):
            payload["fixtures"][name] = {"status": "fixture-missing"}
            continue
        kind = entry["meta"]["kind"]
        t0 = time.perf_counter()
        if kind == "wing":
            record = _wing_pair(name, entry, args.repeat, args.target_block, args.source_block,
                                save_dir=args.save_reference)
        else:
            record = _multi_pair(name, entry, args.repeat, args.target_block, args.source_block,
                                 save_dir=args.save_reference)
        record["seconds"]["whole_fixture"] = time.perf_counter() - t0
        record["status"] = "ok"
        payload["fixtures"][name] = record
        print(name, kind,
              "A max_abs=%.3e" % record["A"]["max_abs"],
              "W max_abs=%.3e" % record["W"]["max_abs"],
              "speedup A=%.2fx W=%.2fx" % (record["seconds"]["speedup_A"] or float("nan"),
                                            record["seconds"]["speedup_W"] or float("nan")),
              flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())