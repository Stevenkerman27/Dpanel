"""Offline, non-production comparison of pressure-gradient reconstructions.

Read the immutable fields saved by panel/solve.py. Use the same panel centers,
tangent frames, cut-aware neighbor stencils, mu and reference quantities for:
  1. unweighted QR affine fit (equivalent to the solver's normal equations);
  2. inverse-distance weighted QR affine fit (experimental candidate).

There is no pressure clipping, artificial source regularization, SVD
truncation, or silent fallback for rank-deficient stencils. This tool never
changes the main solver, and cannot certify physical accuracy.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def reconstruct_gradients(centers, mu, neighbors, tangent_l, tangent_m, *, method):
    """Return [N,2] tangential gradients and conditioning diagnostics.

    The stencil contains only face IDs from the real, cut-aware adjacency in
    panel/solve.py. A repeated self-ID is padding and contributes no equation.
    """
    if method not in ("affine_qr", "weighted_affine_qr", "quadratic_qr"):
        raise ValueError("Unknown diagnostic method")
    centers = np.asarray(centers, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    neighbors = np.asarray(neighbors, dtype=np.int64)
    tangent_l = np.asarray(tangent_l, dtype=np.float64)
    tangent_m = np.asarray(tangent_m, dtype=np.float64)
    n = len(mu)
    if (centers.shape != (n, 3) or neighbors.ndim != 2 or
            neighbors.shape[0] != n or tangent_l.shape != (n, 3) or
            tangent_m.shape != (n, 3)):
        raise ValueError("Inconsistent [N,3] or [N,K] saved panel arrays")
    if neighbors.min() < 0 or neighbors.max() >= n:
        raise ValueError("Out-of-range stencil face index")
    if not all(np.isfinite(a).all() for a in (centers, mu, tangent_l, tangent_m)):
        raise ValueError("Nonfinite potential/geometry")
    gradients = np.empty((n, 2), dtype=np.float64)
    ratios = np.empty(n, dtype=np.float64)
    relative_residual = np.empty(n, dtype=np.float64)
    for i in range(n):
        faces = neighbors[i][neighbors[i] != i]
        required = 5 if method == "quadratic_qr" else 2
        if len(faces) < required or len(np.unique(faces)) != len(faces):
            raise ValueError(f"Insufficient or duplicated tangent neighbors at face {i}")
        delta = centers[faces] - centers[i]
        X = np.column_stack((delta @ tangent_l[i], delta @ tangent_m[i]))
        y = mu[faces] - mu[i]
        if method == "quadratic_qr":
            u, v = X[:, 0], X[:, 1]
            X = np.column_stack((u, v, 0.5 * u**2, u * v, 0.5 * v**2))
        distance = np.linalg.norm(X[:, :2], axis=1)
        if np.any(distance == 0) or not np.isfinite(distance).all():
            raise ValueError(f"Coincident or nonfinite tangent stencil at face {i}")
        if method == "weighted_affine_qr":
            weights = np.mean(distance) / distance
        else:
            weights = np.ones(len(distance))
        design = X * weights[:, None]
        target = y * weights
        singular = np.linalg.svd(design, compute_uv=False)
        ratios[i] = singular[-1] / singular[0] if singular[0] > 0 else 0.0
        if ratios[i] <= np.finfo(np.float64).eps * max(design.shape):
            raise ValueError(f"Numerically rank-deficient tangent stencil at face {i}")
        Q, R = np.linalg.qr(design, mode="reduced")
        fit = np.linalg.solve(R, Q.T @ target)
        gradients[i] = fit[:2]
        relative_residual[i] = np.linalg.norm(X @ fit - y) / max(np.linalg.norm(y), 1e-300)
    return gradients, {
        "minimum_singular_ratio": float(ratios.min()),
        "median_singular_ratio": float(np.median(ratios)),
        "maximum_fit_relative_residual": float(relative_residual.max()),
        "median_fit_relative_residual": float(np.median(relative_residual)),
    }


def pressure_from_tangent_gradient(gradients, tangent_l, tangent_m, alpha_degrees):
    """Incompressible, normalized unit-speed Cp without clipping."""
    angle = np.deg2rad(float(alpha_degrees))
    flow = np.array([np.cos(angle), 0.0, np.sin(angle)])
    vl = np.asarray(tangent_l, dtype=np.float64) @ flow + gradients[:, 0]
    vm = np.asarray(tangent_m, dtype=np.float64) @ flow + gradients[:, 1]
    return 1.0 - vl**2 - vm**2


def normalized_pressure_loads(cp, centers, normal, area, alpha, reference, sref):
    """Exactly the dimensionless pressure-force and moment convention of solve.py."""
    cp = np.asarray(cp, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    area = np.asarray(area, dtype=np.float64)
    forces = -cp[:, None] * area[:, None] * normal
    moments = np.cross(centers - reference, forces)
    a = np.deg2rad(alpha)
    lift_axis = np.array([-np.sin(a), 0.0, np.cos(a)])
    return {
        "CL": float(forces.sum(axis=0) @ lift_axis / sref),
        "CM_y": float(moments.sum(axis=0)[1] / sref),
        "force_normalized": forces,
        "moment_normalized": moments,
    }


def circulation_cl(surface, gamma):
    """Trapezoidal integral of the exported wake-station circulation hat map."""
    chord = float(surface["reference_chord"])
    origin = np.asarray(surface["origin"], dtype=np.float64)
    y = (np.asarray(surface["gamma_stations"], dtype=np.float64) - origin[1]) / chord
    dof = np.asarray(surface["gamma_station_dof"], dtype=np.int64)
    gamma = np.asarray(gamma, dtype=np.float64)
    if y.ndim != 1 or dof.shape != y.shape or len(y) < 2 or not np.all(np.diff(y) > 0):
        raise ValueError("Invalid ordered circulation station map")
    if (np.any(dof < -1) or np.any(dof >= len(gamma)) or
            set(dof[dof >= 0]) != set(range(len(gamma)))):
        raise ValueError("Incomplete circulation DOF mapping")
    value = np.zeros(len(y))
    value[dof >= 0] = gamma[dof[dof >= 0]]
    sref = float(surface["reference_area"]) / chord**2
    return float(2.0 * np.sum(0.5 * (value[1:] + value[:-1]) * np.diff(y)) / sref)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True, help="Existing panel/solve.py output directory")
    p.add_argument("--quadratic", action="store_true", help="Also test unregularized 5-term quadratic QR; fails if any stencil is rank-deficient")
    p.add_argument("--out", type=Path, required=True, help="New JSON report path; never overwrite")
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite an existing report: {args.out}")
    result = json.loads((args.run / "result.json").read_text(encoding="utf-8"))
    surface = json.loads((args.run / "surface_input.json").read_text(encoding="utf-8"))
    with np.load(args.run / "fields.npz", allow_pickle=False) as f:
        data = {key: f[key] for key in (
            "panel_center", "mu", "neighbor_indices", "panel_x_dir",
            "panel_y_dir", "panel_normal", "panel_area", "Cp_raw", "gamma",
            "component_ids", "panel_forces"
        )}
    chord = float(surface["reference_chord"])
    sref = float(surface["reference_area"]) / chord**2
    reference = (np.asarray(surface["moment_reference"]) - np.asarray(surface["origin"])) / chord
    alpha = float(result["parameters"]["alpha_deg"])
    center, normal, area = data["panel_center"], data["panel_normal"], data["panel_area"]
    raw = np.asarray(data["Cp_raw"], dtype=np.float64)
    baseline = normalized_pressure_loads(raw, center, normal, area, alpha, reference, sref)
    # The saved pressure/force ledger must close before a reconstruction study.
    baseline_panel_forces = baseline["force_normalized"]
    np.testing.assert_allclose(
        baseline_panel_forces, data["panel_forces"], rtol=1e-9, atol=1e-10,
        err_msg="Saved Cp and panel_forces disagree",
    )
    np.testing.assert_allclose(
        baseline_panel_forces.sum(axis=0) * chord**2,
        np.asarray(result["force"]), rtol=1e-9, atol=1e-10,
        err_msg="Saved total pressure force is inconsistent with panels",
    )
    np.testing.assert_allclose(
        baseline["moment_normalized"].sum(axis=0) * chord**3,
        np.asarray(result["moment"]), rtol=1e-9, atol=1e-10,
        err_msg="Saved total pressure moment is inconsistent with panel arms",
    )
    for name, mask in (("wing", data["component_ids"] != 20),
                       ("body", data["component_ids"] == 20)):
        np.testing.assert_allclose(
            baseline_panel_forces[mask].sum(axis=0) * chord**2,
            np.asarray(result["components"][name]["force"]),
            rtol=1e-9, atol=1e-10,
            err_msg=f"Saved {name} force does not close",
        )
        np.testing.assert_allclose(
            baseline["moment_normalized"][mask].sum(axis=0) * chord**3,
            np.asarray(result["components"][name]["moment"]),
            rtol=1e-9, atol=1e-10,
            err_msg=f"Saved {name} moment does not close",
        )
    if not np.isclose(baseline["CL"], result["CL"], rtol=1e-9, atol=1e-10):
        raise ValueError("Saved Cp, geometry and result.json disagree on CL; do not compare reconstructions")
    if not np.isclose(baseline["CM_y"], result["CM"][1], rtol=1e-9, atol=1e-10):
        raise ValueError("Saved Cp, geometry and result.json disagree on CM_y")

    cl_gamma = circulation_cl(surface, data["gamma"])
    report = {
        "source_run": str(args.run), "source_sha256": result.get("source_sha256"),
        "physical_accuracy_accepted": result.get("physical_accuracy_accepted", False),
        "wall_panels": len(raw),
        "CL_circulation_not_external_truth": cl_gamma,
        "reference_area_normalized": sref,
        "baseline": {
            "CL": baseline["CL"], "CM_y": baseline["CM_y"],
            "CL_minus_circulation": baseline["CL"] - cl_gamma,
            "Cp_min": float(raw.min()), "Cp_max": float(raw.max()),
        },
        "experimental": {},
    }
    original_force = baseline["force_normalized"]
    methods = ("affine_qr", "weighted_affine_qr") + (("quadratic_qr",) if args.quadratic else ())
    for method in methods:
        gradients, conditioning = reconstruct_gradients(
            center, data["mu"], data["neighbor_indices"],
            data["panel_x_dir"], data["panel_y_dir"], method=method,
        )
        cp = pressure_from_tangent_gradient(
            gradients, data["panel_x_dir"], data["panel_y_dir"], alpha
        )
        if not np.isfinite(cp).all():
            raise ValueError(f"Nonfinite Cp from {method}; no clipping")
        loads = normalized_pressure_loads(cp, center, normal, area, alpha, reference, sref)
        face_force_error = loads["force_normalized"] - original_force
        changed = np.argsort(np.abs(cp - raw))[-5:][::-1]
        report["experimental"][method] = {
            **conditioning,
            "CL": loads["CL"], "CM_y": loads["CM_y"],
            "CL_minus_circulation": loads["CL"] - cl_gamma,
            "CL_delta_from_baseline": loads["CL"] - baseline["CL"],
            "Cp_min": float(cp.min()), "Cp_max": float(cp.max()),
            "Cp_area_weighted_RMSE_delta": float(np.sqrt(np.average((cp - raw)**2, weights=area))),
            "panel_force_L2_relative_delta": float(
                np.linalg.norm(face_force_error) / max(np.linalg.norm(original_force), 1e-300)
            ),
            "largest_changed_faces": [
                {"face": int(i), "component": int(data["component_ids"][i]),
                 "Cp_baseline": float(raw[i]), "Cp_candidate": float(cp[i])}
                for i in changed
            ],
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Diagnostic only. No physical-accuracy acceptance or production solver change.")


if __name__ == "__main__":
    main()
