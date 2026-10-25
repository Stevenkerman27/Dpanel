"""Export the *actual* VortexAD dense AIC and RHS for isolated solver comparisons.

Run in the verified isolated VortexAD/CSDL-alpha/JAX environment:
  python test/export_vortexad_aic.py --out runs/backend_probes/vortexad_aic.npz

This measures compilation+assembly, NOT raw-Cp, gradients, total aircraft
performance, or a matrix-free boundary-element operator.
"""
import argparse
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-n", type=int, default=3000)
    args = ap.parse_args()

    import csdl_alpha as csdl
    from VortexAD import PanelMethod, SAMPLE_GEOMETRY_PATH

    mesh = Path(SAMPLE_GEOMETRY_PATH) / "pm" / "naca0012_LE_TE_cluster.stl"
    if not mesh.is_file():
        raise FileNotFoundError(f"Sample mesh not found in source checkout: {mesh}")

    t0 = time.perf_counter()
    rec = csdl.Recorder(inline=False)
    rec.start()
    alpha = csdl.Variable(value=np.array([5.0]))
    panel = PanelMethod({
        "Mach": 0.2, "alpha": alpha, "mesh_path": str(mesh),
        "ref_area": 10.0, "BC": "Dirichlet", "drag_type": "Trefftz",
        "Cp cutoff": -5.0,  # irrelevant for AIC and RHS; raw Cp not evaluated
    })
    panel.declare_outputs(["AIC_mu", "RHS"])
    results = panel.evaluate()
    rec.stop()
    build_s = time.perf_counter() - t0

    aic = results["AIC_mu"]
    rhs = results["RHS"]
    shape = aic.shape
    if len(shape) not in (2, 3) or shape[-1] != shape[-2]:
        raise ValueError(f"Unexpected AIC shape: {shape}")
    if shape[-1] > args.max_n:
        raise MemoryError(f"Refusing {shape[-1]} dense unknowns: max-n={args.max_n}")

    t1 = time.perf_counter()
    sim = csdl.experimental.JaxSimulator(
        recorder=rec, additional_inputs=[alpha],
        additional_outputs=[aic, rhs], gpu=False,
    )
    sim.run()
    compile_and_eval_s = time.perf_counter() - t1
    A = np.asarray(sim[aic], dtype=np.float64).squeeze()
    b = np.asarray(sim[rhs], dtype=np.float64).reshape(-1)
    if A.ndim != 2 or A.shape[0] != len(b) or A.shape[0] != A.shape[1]:
        raise ValueError(f"AIC/RHS mismatch: A={A.shape}, b={b.shape}")
    if not np.isfinite(A).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite AIC/RHS, refusing to export")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, A=A, b=b,
        mesh_source=str(mesh),
        source_kind="VortexAD original dense AIC_mu and RHS",
        alpha_deg=5.0, mach=0.2,
    )
    print(f"Saved {args.out} n={len(b)} A_bytes={A.nbytes}")
    print(f"graph_build_seconds={build_s:.6f}")
    print(f"jax_compile_and_first_evaluation_seconds={compile_and_eval_s:.6f}")
    print("This is a fully stored matrix; not a matrix-free BEM benchmark.")


if __name__ == "__main__":
    main()
