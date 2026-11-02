"""Micro-benchmark of the batched influence kernel over block sizes and thread counts."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
from panel.p1_batched_influence import influence_matrix
from tools.issue92 import fixtures


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="g1_elliptic")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/issue92/block_sweep.json")
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args(argv)
    import torch
    arrays, _entry = fixtures.load(args.fixture)
    tri = np.asarray(arrays["xyz"], float)[np.asarray(arrays["faces"], int)]
    centers = tri.mean(axis=1)
    dofs = np.asarray(arrays["dof_face"], int)
    ndof = int(arrays["ndof_mu"])
    print("faces", len(tri), "ndof", ndof, "threads", torch.get_num_threads(), flush=True)
    payload = {"fixture": args.fixture, "n_faces": int(len(tri)), "ndof": ndof,
               "n_samples": len(tri), "repeat": int(args.repeat),
               "threads_default": int(torch.get_num_threads()), "rows": []}
    configs = [(128, 64), (256, 128), (512, 128), (256, 256), (512, 256), (1024, 256)]
    for threads in (1, 8, 16, 32):
        torch.set_num_threads(threads)
        for target_block, source_block in configs:
            influence_matrix(tri, centers, self_faces=np.arange(len(tri)), face_dofs=dofs,
                             ndof=ndof, target_block=target_block, source_block=source_block)
            best = float("inf")
            for _ in range(args.repeat):
                t0 = time.perf_counter()
                influence_matrix(tri, centers, self_faces=np.arange(len(tri)), face_dofs=dofs,
                                 ndof=ndof, target_block=target_block, source_block=source_block)
                best = min(best, time.perf_counter() - t0)
            payload["rows"].append({"threads": threads, "target_block": target_block,
                                    "source_block": source_block, "seconds": best})
            print("threads", threads, "block", target_block, source_block, "%.3f s" % best, flush=True)
    best = min(payload["rows"], key=lambda r: r["seconds"])
    payload["best"] = best
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("best", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())