"""Issue #92 S4: torch port of the P1 assembly operators (A, W, grad).

The production assembly in `panel/p1_lifting_g1.py::assemble` and
`panel/p1_lifting_g2.py::assemble_multi` spends 12.2 s of the 22.9 s elliptic solve in
`_doublet_matrix` (docs/ISSUE92_PHASE0_BASELINE.md). This slice ports exactly those three
operators to torch (`panel/p1_assembly_torch.py`) and measures them on the eight frozen
fixtures:

* A  - wall doublet operator (targets = wall face centroids, self-face corrected);
* W  - wake doublet contraction, in both production shapes: G1 contracts per wake face with
       three private DOFs, G2 contracts one nodal operator per group with shared wake nodes;
* grad - face gradient map (pure scatter, expected to be exact).

Stages, run by different interpreters (0xC0000005 hazard, ISSUE92_PHASE1_WAKE_EVALUATOR 6f):

  python -m tools.issue92.s4_assembly_parity capture --work runs/issue92/s4
  <torch python> -m tools.issue92.s4_assembly_parity torch --work runs/issue92/s4
  python -m tools.issue92.s4_assembly_parity judge --work runs/issue92/s4 --out ...

Thresholds are declared before the measurement: the two operators are algebraically
identical transcriptions, so the comparison is floating-point reassociation only. A is
compared with max |dA| / max |A| and with a Frobenius ratio; grad is a scatter with no
arithmetic and is required to be bit-identical.
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

THRESHOLDS = {
    "a_max_relative": 1e-12,
    "a_frobenius_relative": 1e-13,
    "w_max_relative": 1e-12,
    "grad_max_abs": 0.0,
}


def _manifest():
    from tools.issue92 import fixtures  # noqa: F401  (registers the fixture cases)
    manifest = json.loads((ROOT / "docs/issue92/fixtures_manifest.json").read_text(encoding="utf-8"))
    return manifest["fixtures"]


def _source(name, system):
    """Wing/multi mesh that owns faces and dof_face for this case."""
    return system["mesh"] if system["kind"] == "g2" else system["wing"]


def capture(work: Path):
    from tools.issue92.s3_pressure_parity import _build

    work.mkdir(parents=True, exist_ok=True)
    manifest = _manifest()
    rows = {}
    for name in CASES:
        started = time.perf_counter()
        system = _build(name, manifest[name])
        elapsed = time.perf_counter() - started
        ops = system["ops"]
        source = _source(name, system)
        disk = ops["mesh"]
        payload = dict(
            name=name, kind=system["kind"],
            triangles=disk.triangles, centers=disk.centers,
            basis_gradients=disk.basis_gradients,
            faces=np.asarray(source.faces), dof_face=np.asarray(source.dof_face),
            A=ops["A"], grad=ops["grad"], W=ops["W"],
            numpy_assemble_seconds=elapsed)
        if system["kind"] == "g1":
            # G1 looks per-face (D_j @ gamma_hat[wake_faces[j]]), but the strengths are the
            # shared wake-node values, so its operator is nodal too (ndof = wake nodes).
            wing = system["wing"]
            payload.update(
                wake_mode="nodal", n_groups=1,
                g0_wake_triangles=np.asarray(wing.wake_xyz, float)[np.asarray(wing.wake_faces)],
                g0_wake_dofs=np.asarray(wing.wake_faces, int),
                g0_wake_gamma=np.asarray(wing.gamma_hat, float),
                g0_wake_ndof=np.int64(len(wing.wake_xyz)))
        else:
            mesh = system["mesh"]
            payload.update(wake_mode="nodal", n_groups=len(mesh.groups))
            for gi, group in enumerate(mesh.groups):
                payload["g%d_wake_triangles" % gi] = np.asarray(group.wake_xyz, float)[
                    np.asarray(group.wake_faces, int)]
                payload["g%d_wake_dofs" % gi] = np.asarray(group.wake_faces, int)
                payload["g%d_wake_gamma" % gi] = np.asarray(group.gamma_hat, float)
                payload["g%d_wake_ndof" % gi] = np.int64(len(group.wake_xyz))
        np.savez_compressed(work / ("%s.npz" % name), **payload)
        rows[name] = dict(kind=system["kind"],
                          n_wall_faces=int(payload["A"].shape[0]),
                          n_dof=int(payload["A"].shape[1]),
                          n_gamma=int(payload["W"].shape[1]),
                          wake_mode=payload["wake_mode"],
                          numpy_assemble_seconds=elapsed)
        print("%-18s kind=%s wall=%d dof=%d gamma=%d numpy=%.2fs"
              % (name, system["kind"], payload["A"].shape[0], payload["A"].shape[1],
                 payload["W"].shape[1], elapsed))
    (work / "capture.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return rows


def _relative(diff, scale):
    denom = float(np.max(np.abs(scale)))
    return (float(np.max(np.abs(diff))) / denom) if denom else float(np.max(np.abs(diff)))


def torch_stage(work: Path):
    import torch
    from panel.p1_assembly_torch import doublet_matrix, gradient_map, wake_operator

    manifest = _manifest()
    rows = {}
    for name in CASES:
        data = np.load(work / ("%s.npz" % name))
        kind = str(data["kind"])
        started = time.perf_counter()
        A = doublet_matrix(
            torch.from_numpy(data["triangles"]), torch.from_numpy(data["centers"]),
            self_faces=torch.arange(data["triangles"].shape[0]),
            face_dofs=torch.from_numpy(data["dof_face"]),
            ndof=int(data["A"].shape[1]))
        grad = gradient_map(torch.from_numpy(data["faces"]),
                            torch.from_numpy(data["dof_face"]),
                            torch.from_numpy(data["basis_gradients"]),
                            ndof=int(data["A"].shape[1]))
        if str(data["wake_mode"]) == "per_face":
            W = wake_operator(torch.from_numpy(data["wake_triangles"]),
                              torch.from_numpy(data["centers"]),
                              torch.from_numpy(data["wake_gamma"]), mode="per_face")
        else:
            parts = []
            for gi in range(int(data["n_groups"])):
                parts.append(wake_operator(
                    torch.from_numpy(data["g%d_wake_triangles" % gi]),
                    torch.from_numpy(data["centers"]),
                    torch.from_numpy(data["g%d_wake_gamma" % gi]), mode="nodal",
                    face_dofs=torch.from_numpy(data["g%d_wake_dofs" % gi]),
                    ndof=int(data["g%d_wake_ndof" % gi])))
            W = torch.cat(parts, dim=1)
        elapsed = time.perf_counter() - started
        A_np, W_np, grad_np = A.numpy(), W.numpy(), grad.numpy()
        rows[name] = dict(
            kind=kind,
            a_max_relative=_relative(A_np - data["A"], data["A"]),
            a_frobenius_relative=(float(np.linalg.norm(A_np - data["A"]))
                                  / float(np.linalg.norm(data["A"]))),
            w_max_relative=_relative(W_np - data["W"], data["W"]),
            grad_max_abs=float(np.max(np.abs(grad_np - data["grad"]))) if grad_np.size else 0.0,
            torch_assembly_seconds=elapsed,
            numpy_assemble_seconds=float(data["numpy_assemble_seconds"]))
        print("%-18s maxrel A=%.2e W=%.2e grad=%.2e torch=%.2fs numpy=%.2fs"
              % (name, rows[name]["a_max_relative"], rows[name]["w_max_relative"],
                 rows[name]["grad_max_abs"], elapsed,
                 rows[name]["numpy_assemble_seconds"]))
    (work / "torch.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return rows


def judge(work: Path, out: Path):
    capture_rows = json.loads((work / "capture.json").read_text(encoding="utf-8"))
    torch_rows = json.loads((work / "torch.json").read_text(encoding="utf-8"))
    cases = {}
    for name in CASES:
        row = dict(capture_rows[name])
        row.update(torch_rows[name])
        row["passed"] = bool(
            row["a_max_relative"] <= THRESHOLDS["a_max_relative"]
            and row["a_frobenius_relative"] <= THRESHOLDS["a_frobenius_relative"]
            and row["w_max_relative"] <= THRESHOLDS["w_max_relative"]
            and row["grad_max_abs"] <= THRESHOLDS["grad_max_abs"])
        cases[name] = row
    payload = dict(stage="s4_assembly", thresholds=THRESHOLDS, cases=cases,
                   passed=all(row["passed"] for row in cases.values()),
                   claims=[
                       "A, W and grad are reproduced entrywise from geometry tensors alone",
                       "grad is bit-identical because it is a scatter with no arithmetic",
                   ],
                   non_claims=[
                       "assembly wall time on CPU is not a speed claim; the S5 solve stage is",
                       "loads and Cp post-processing are not covered here",
                   ])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("s4 passed=%s -> %s" % (payload["passed"], out))
    for name, row in cases.items():
        print("  %-18s A=%.2e W=%.2e grad=%.1e %.2f s torch / %.2f s numpy"
              % (name, row["a_max_relative"], row["w_max_relative"], row["grad_max_abs"],
                 row["torch_assembly_seconds"], row["numpy_assemble_seconds"]))
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("capture", "torch", "judge"))
    parser.add_argument("--work", default=str(ROOT / "runs/issue92/s4"))
    parser.add_argument("--out", default=str(ROOT / "docs/issue92/s4_assembly_parity.json"))
    args = parser.parse_args(argv)
    work, out = Path(args.work), Path(args.out)
    if args.stage == "capture":
        capture(work)
    elif args.stage == "torch":
        torch_stage(work)
    else:
        judge(work, out)


if __name__ == "__main__":
    main()
