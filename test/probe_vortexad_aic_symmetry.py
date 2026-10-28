"""Decompose the VortexAD full-aircraft AIC mirror discrepancy by term.

Usage (isolated, pinned VortexAD/CSDL/JAX environment):
 python test/probe_vortexad_aic_symmetry.py --out runs/backend_probes/aic_symmetry.json

Diagnostic only. Do NOT symmetrize matrices for the production solver. It
compares:
  AIC_mu_orig       body doublet
  AIC_mu-AIC_mu_orig wake/Kutta adjustment
  AIC_sigma         body source
  AIC_mu            full body + wake
  RHS               true boundary right-hand side

The source mesh must be symmetric about y=0 and have no panel *centers*
on the plane. Matching uses the measured panel centers and an involutive
bijection, not assumptions about panel index order or an input STL's triangles.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def mirror_pairs(centers, normals, *, tol=1e-10):
    centers = np.asarray(centers, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or normals.shape != centers.shape:
        raise ValueError("Need aligned [N,3] centers and outward normals")
    if not np.isfinite(centers).all() or not np.isfinite(normals).all():
        raise ValueError("Nonfinite mesh geometry")
    scale = max(float(np.ptp(centers, axis=0).max()), 1.0)
    eps = tol * scale
    if np.any(np.abs(centers[:, 1]) < eps):
        raise ValueError("Panel center lies on y=0; cannot half-pair it")
    reflected = centers * np.array([1., -1., 1.])
    distance, match = cKDTree(centers).query(reflected, k=1)
    if np.max(distance) > eps:
        raise ValueError(f"Mirror centroid mismatch {np.max(distance):.4g} > {eps:.4g}")
    if len(np.unique(match)) != len(match) or not np.array_equal(match[match], np.arange(len(match))):
        raise ValueError("Mirror mapping is not one-to-one and involutive")
    reflected_normal = normals * np.array([1., -1., 1.])
    normal_err = np.linalg.norm(reflected_normal - normals[match], axis=1)
    if np.max(normal_err) > 1e-8:
        raise ValueError(f"Mirror normal mismatch max={np.max(normal_err):.4g}")
    if np.count_nonzero(centers[:, 1] > 0) != np.count_nonzero(centers[:, 1] < 0):
        raise ValueError("Unequal half-model panel counts")
    return match, {
        "panel_count": len(centers),
        "half_count": int(np.count_nonzero(centers[:, 1] > 0)),
        "max_centroid_mirror_distance": float(np.max(distance)),
        "max_normal_mirror_distance": float(np.max(normal_err)),
        "scale": scale,
    }


def matrix_report(A, mirror, *, detail=5):
    A = np.asarray(A, dtype=np.float64)
    n = len(mirror)
    if A.shape != (n, n) or not np.isfinite(A).all():
        raise ValueError("Expected finite square A aligned with mirror map")
    dif = A - A[np.ix_(mirror, mirror)]
    d = np.abs(dif)
    largest = np.argpartition(d.ravel(), -min(detail, d.size))[-min(detail, d.size):]
    largest = largest[np.argsort(d.ravel()[largest])[::-1]]
    offenders = []
    for flat in largest:
        i, j = np.unravel_index(int(flat), d.shape)
        offenders.append({
            "target": int(i), "source": int(j),
            "mirrored_target": int(mirror[i]), "mirrored_source": int(mirror[j]),
            "coefficient": float(A[i, j]), "mirror_coefficient": float(A[mirror[i], mirror[j]]),
            "absolute_delta": float(d[i, j]),
        })
    diag_delta = np.abs(np.diag(dif))
    peak = float(np.max(np.abs(A)))
    return {
        "max_abs_delta": float(np.max(d)),
        "max_delta_relative_to_global_max": float(np.max(d) / max(peak, 1e-300)),
        "rms_abs_delta": float(np.sqrt(np.mean(np.square(d)))),
        "median_row_max_abs_delta": float(np.median(np.max(d, axis=1))),
        "max_diagonal_abs_delta": float(np.max(diag_delta)),
        "nonzero_gt_1e_12": int(np.count_nonzero(d > 1e-12)),
        "matrix_max_abs": peak,
        "worst_pairs": offenders,
    }


def vector_report(b, mirror):
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if b.shape != (len(mirror),):
        raise ValueError("RHS not aligned to panels")
    d = np.abs(b - b[mirror])
    return {
        "max_abs_delta": float(np.max(d)),
        "relative_to_max": float(np.max(d) / max(float(np.max(np.abs(b))), 1e-300)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", default="naca0012_LE_TE_cluster.stl")
    parser.add_argument("--max-n", type=int, default=3000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import csdl_alpha as csdl
    from VortexAD import PanelMethod, SAMPLE_GEOMETRY_PATH

    mesh = Path(SAMPLE_GEOMETRY_PATH) / "pm" / args.mesh
    if not mesh.is_file():
        raise FileNotFoundError(f"VortexAD source-checkout mesh missing: {mesh}")
    rec = csdl.Recorder(inline=False)
    rec.start()
    alpha = csdl.Variable(value=np.array([5.0]))
    model = PanelMethod({
        "Mach": 0.2, "alpha": alpha, "mesh_path": str(mesh),
        "ref_area": 10.0, "BC": "Dirichlet", "drag_type": "Trefftz",
        "Cp cutoff": -5.0,  # AIC and RHS do not depend on Cp post-processing
    })
    keys = ("AIC_mu", "AIC_mu_orig", "AIC_sigma", "RHS", "panel_centers", "panel_normal")
    model.declare_outputs(list(keys))
    outputs = model.evaluate()
    n = outputs["AIC_mu"].shape[-1]
    if n > args.max_n:
        raise MemoryError(f"Refusing {n} unknowns; max-n={args.max_n}")
    rec.stop()
    begin = time.perf_counter()
    sim = csdl.experimental.JaxSimulator(
        recorder=rec, additional_inputs=[alpha],
        additional_outputs=[outputs[k] for k in keys],
        gpu=False,
    )
    sim.run()
    warmup_seconds = time.perf_counter() - begin

    values = {k: np.asarray(sim[outputs[k]], dtype=np.float64).squeeze() for k in keys}
    centers = values["panel_centers"].reshape(n, 3)
    normals = values["panel_normal"].reshape(n, 3)
    mirror, geometry = mirror_pairs(centers, normals)
    full = values["AIC_mu"].reshape(n, n)
    body = values["AIC_mu_orig"].reshape(n, n)
    source = values["AIC_sigma"].reshape(n, n)
    wake = full - body
    report = {
        "mesh": args.mesh, "n": n, "alpha_degrees": 5.0,
        "jax_compile_first_run_seconds": warmup_seconds,
        "geometry": geometry,
        "body_doublet": matrix_report(body, mirror),
        "wake_kutta_adjustment": matrix_report(wake, mirror),
        "body_source": matrix_report(source, mirror),
        "full_aic": matrix_report(full, mirror),
        "rhs": vector_report(values["RHS"], mirror),
        "warning": "Pure diagnostic. No modifications to upstream matrices. Symmetrization is NOT a validated production fix.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
