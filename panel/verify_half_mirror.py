"""Independent mirrored full-aircraft oracle for a native half-domain CAD mesh.

Builds a full mirrored mesh ONLY within this opt-in validation driver. The
production half-native solver never needs full left-side unknowns. The oracle
uses the previously validated dense full JAX solver on an *identical symmetric
discretization*, not the old independent-left/right Gmsh mesh.

Run:
 python -m panel.verify_half_mirror --surface runs/HALF_CAD/surface.json \
   --out runs/NEW_MIRROR_AUDIT.json
Add --derivatives for a same-topology geometry/station VJP + FD comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .differentiable import PanelTopology, solve_fields
from .half_native import solve_half_fields, validate_half_configuration
from .run_half_native import load_half_surface


_REFLECTION = jnp.asarray([1., -1., 1.])


def make_full_mirror_topology(half, station_dof):
    """Mirror topology once; no two-sided Gmsh or coordinate-nearest pairing.

    Positive-side DOFs appear first and preserve their column IDs. One mirror
    copy of each non-root circulation DOF is appended. A root-contact DOF,
    including several stations belonging to a single root contact interval,
    is shared across the y=0 symmetry plane.
    """
    t = half["topology"]
    x = np.asarray(half["vertices"])
    y = np.asarray(half["station_y"])
    dof = np.asarray(station_dof, dtype=np.int64)
    n = len(t.faces)
    ng = t.hat_map.shape[1]
    if (dof.shape != y.shape or dof[0] < 0 or dof[-1] != -1
            or len(set(dof[dof >= 0])) != ng):
        raise ValueError("Invalid half circulation station mapping")
    tol = 1e-10*max(float(np.ptp(x, axis=0).max()), 1.)
    if abs(y[0]) > tol or np.any(x[:, 1] < -tol):
        raise ValueError("Requires y>=0 with the central station at y=0")
    positive_nodes = np.flatnonzero(x[:, 1] > tol)
    node_reflection = np.arange(len(x), dtype=np.int32)
    node_reflection[positive_nodes] = (
        len(x)+np.arange(len(positive_nodes), dtype=np.int32))
    faces = np.asarray(t.faces)
    wake = np.asarray(t.wake_faces)
    te = np.asarray(t.te_edges)
    full_faces = np.vstack((faces, node_reflection[faces[:, ::-1]]))
    full_wake = np.vstack((wake, node_reflection[wake[:, ::-1]]))
    full_te = np.vstack((te, node_reflection[te[:, ::-1]]))
    full_upper = np.r_[np.asarray(t.upper_te), np.asarray(t.upper_te)+n]
    full_lower = np.r_[np.asarray(t.lower_te), np.asarray(t.lower_te)+n]

    # The right-half pressure stencil already contains virtual reflected
    # neighbor IDs 0..2n-1, so these IDs become real test-only left faces.
    neighbor_right = np.asarray(t.neighbor_ghost_ids, dtype=np.int32)
    if neighbor_right.size and (neighbor_right.min() < 0 or
                                neighbor_right.max() >= 2*n):
        raise ValueError("Half pressure ghost ID out of range")
    neighbor_left = np.where(
        neighbor_right < n, neighbor_right+n, neighbor_right-n)
    full_neighbors = np.vstack((neighbor_right, neighbor_left))

    # Mirror y locations while retaining the single central station.
    full_y = np.r_[-y[:0:-1], y]
    root = int(dof[0])
    other = [i for i in range(ng) if i != root]
    mirrored_label = {root: root}
    mirrored_label.update({value: ng+j for j, value in enumerate(other)})
    full_dof = np.r_[
        [(-1 if int(k) < 0 else mirrored_label[int(k)])
         for k in dof[:0:-1]],
        dof,
    ].astype(np.int32)
    full_ng = ng+len(other)
    if set(full_dof[full_dof >= 0]) != set(range(full_ng)):
        raise ValueError("Mirrored circulation DOF labels not contiguous")
    hat = np.zeros((len(full_dof), full_ng), dtype=np.float64)
    mask = full_dof >= 0
    hat[np.flatnonzero(mask), full_dof[mask]] = 1.

    x_full = np.vstack((x, x[positive_nodes]*[1., -1., 1.]))
    centers_wake = x_full[full_wake].mean(axis=1)[:, 1]
    wi = np.searchsorted(full_y, centers_wake, side="right")-1
    quad = np.asarray([.5-np.sqrt(3.)/6, .5+np.sqrt(3.)/6])
    ends = x_full[full_te]
    y_quad = ends[:, 0, 1, None]*(1-quad)+ends[:, 1, 1, None]*quad
    ti = np.searchsorted(full_y, y_quad, side="right")-1
    if (np.any(wi < 0) or np.any(wi >= len(full_y)-1)
            or np.any(ti < 0) or np.any(ti >= len(full_y)-1)
            or np.any(ti[:, 0] != ti[:, 1])):
        raise ValueError("Full mirror wake/TE station interval invalid")

    topology = PanelTopology(*(jnp.asarray(a) for a in (
        full_faces, full_wake, full_te, full_upper, full_lower, hat,
        wi, ti, full_neighbors, np.tile(np.asarray(t.component_ids), 2),
    )))
    # E replicates right-side wall state and gamma to exact left images.
    E = np.zeros((2*n+full_ng, n+ng), dtype=np.float64)
    E[:n, :n] = np.eye(n)
    E[n:2*n, :n] = np.eye(n)
    E[2*n:2*n+ng, n:n+ng] = np.eye(ng)
    for label, copy in mirrored_label.items():
        if label != root:
            E[2*n+copy, n+label] = 1.
    # R picks the independent positive-y collocation + Kutta equations.
    R = np.r_[np.arange(n), 2*n+np.arange(ng)].astype(np.int32)
    return dict(
        full_topology=topology,
        positive_nodes=jnp.asarray(positive_nodes),
        E=E, R=R, half_wall_count=n, half_gamma_count=ng,
        full_gamma_count=full_ng, full_nodes=len(x_full),
    )


def mirrored_full_input(half, static, x=None, y=None):
    """Differentiable half-coordinate -> full mirrored-coordinate map."""
    x = half["vertices"] if x is None else x
    y = half["station_y"] if y is None else y
    full_x = jnp.concatenate(
        (x, x[static["positive_nodes"]]*_REFLECTION), axis=0)
    full_y = jnp.concatenate((-y[:0:-1], y))
    return dict(
        **{key: half[key] for key in (
            "freestream", "density", "reference_area",
            "reference_chord", "reference_point")},
        vertices=full_x, station_y=full_y,
        topology=static["full_topology"],
    )


def _error(reference, candidate):
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape:
        raise ValueError(f"Oracle shape mismatch: {a.shape} vs {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite field in real mirrored oracle")
    delta = a-b
    return dict(
        max_abs=float(np.max(np.abs(delta))),
        relative_l2=float(
            np.linalg.norm(delta)/max(np.linalg.norm(a), 1e-300)),
    )


def verify(surface_path, *, derivatives=False, max_half_panels=1000):
    half = load_half_surface(surface_path)
    raw = json.loads(Path(surface_path).read_text(encoding="utf-8"))
    validate_half_configuration(
        half["vertices"], half["station_y"], half["topology"],
        half["freestream"], half["reference_point"])
    n = len(half["topology"].faces)
    if n > max_half_panels:
        raise ValueError("Half mesh exceeds explicit full-oracle memory guard")
    plan = make_full_mirror_topology(half, raw["gamma_station_dof"])
    full_input = mirrored_full_input(half, plan)

    half_out = jax.jit(solve_half_fields)(**half)
    full_out = jax.jit(solve_fields)(**full_input)
    E, R = plan["E"], plan["R"]
    nf = plan["half_wall_count"]
    ng = plan["half_gamma_count"]
    full_A = np.asarray(full_out["matrix"])
    half_A = np.asarray(half_out["matrix_half"])
    full_b = np.asarray(full_out["rhs"])
    half_b = np.asarray(half_out["rhs_half"])
    full_state = np.r_[
        np.asarray(full_out["mu"]), np.asarray(full_out["gamma"])]
    half_state = np.r_[
        np.asarray(half_out["mu_half"]), np.asarray(half_out["gamma_half"])]
    stats = dict(
        status="mirror_oracle_numerical_check",
        half_wall_panels=nf, full_oracle_wall_panels=2*nf,
        half_circulation_unknowns=ng,
        full_oracle_circulation_unknowns=plan["full_gamma_count"],
        half_independent_state=len(half_state),
        full_oracle_state=len(full_state),
        half_vs_RAE_matrix=_error(half_A, full_A[R]@E),
        half_vs_full_RHS=_error(half_b, full_b[R]),
        half_mu_vs_full_positive=_error(
            half_out["mu_half"], full_out["mu"][:nf]),
        half_gamma_vs_full_positive=_error(
            half_out["gamma_half"], full_out["gamma"][:ng]),
        expanded_half_state_vs_full=_error(full_state, E@half_state),
        Cp_raw_right_vs_full=_error(
            full_out["Cp_raw"][:nf], half_out["Cp_raw_half"]),
        pressure_forces_right=_error(
            full_out["panel_forces"][:nf], half_out["panel_forces_half"]),
        pressure_moments_right=_error(
            full_out["panel_moments"][:nf], half_out["panel_moments_half"]),
        full_force=_error(full_out["force"], half_out["force"]),
        full_moment=_error(full_out["moment"], half_out["moment"]),
        CL=_error(full_out["CL"], half_out["CL"]),
        CM=_error(full_out["CM"], half_out["CM"]),
        full_lift_coefficient=float(full_out["CL"]),
        half_lift_coefficient=float(half_out["CL"]),
        physical_accuracy_accepted=False,
    )
    # Assembly comparison is the strictest: do not accept a reduced system
    # that only agrees in total lift due to an accidental cancellation.
    if stats["half_vs_RAE_matrix"]["relative_l2"] > 1e-10:
        raise AssertionError(f"Half image A does not equal full RAE: {stats['half_vs_RAE_matrix']}")
    if stats["half_vs_full_RHS"]["relative_l2"] > 1e-10:
        raise AssertionError("Half image RHS does not equal full mirrored RHS")
    for key in ("Cp_raw_right_vs_full", "full_force", "full_moment",
                "expanded_half_state_vs_full"):
        if stats[key]["relative_l2"] > 1e-8:
            raise AssertionError(f"Mirrored full field differs in {key}: {stats[key]}")

    if derivatives:
        # Fixed-connectivity, mirror-compatible perturbations; nodes on
        # y=0 do not move in y and the central station remains at y=0.
        x, y = half["vertices"], half["station_y"]
        dx = jnp.stack(
            (.02*x[:, 0], .01*x[:, 1], .02*x[:, 2]), axis=1)
        dy = .01*y
        weights = jnp.asarray(
            np.random.default_rng(20261026).standard_normal((nf, 3)))
        def objectives(scale):
            xx, yy = x+scale*dx, y+scale*dy
            half_o = solve_half_fields(**{**half, "vertices":xx, "station_y":yy})
            full_o = solve_fields(**mirrored_full_input(half, plan, xx, yy))
            a = (jnp.sum(weights*half_o["panel_forces_half"])
                 +.2*half_o["CM"][1])
            b = (jnp.sum(weights*full_o["panel_forces"][:nf])
                 +.2*full_o["CM"][1])
            return a, b
        fhalf = jax.jit(lambda v: objectives(v)[0])
        ffull = jax.jit(lambda v: objectives(v)[1])
        gh = float(jax.jit(jax.grad(fhalf))(jnp.array(0.)))
        gf = float(jax.jit(jax.grad(ffull))(jnp.array(0.)))
        checks = []
        for h in (1e-3, 1e-4, 1e-5):
            ph = float((fhalf(h)-fhalf(-h))/(2*h))
            pf = float((ffull(h)-ffull(-h))/(2*h))
            checks.append(dict(h=h, half_ad=gh, full_ad=gf,
                               half_fd=ph, full_fd=pf,
                               half_abs_error=abs(gh-ph),
                               full_abs_error=abs(gf-pf)))
            np.testing.assert_allclose(gh, ph, rtol=5e-4, atol=5e-7)
            np.testing.assert_allclose(gf, pf, rtol=5e-4, atol=5e-7)
        np.testing.assert_allclose(gh, gf, rtol=1e-7, atol=1e-8)
        stats["symmetric_geometry_station_derivatives"] = checks
    stats["status"] = "exact_mirror_discretization_verified"
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--surface", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-half-panels", type=int, default=1000)
    p.add_argument("--derivatives", action="store_true")
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite existing mirror-oracle audit")
    jax.config.update("jax_enable_x64", True)
    report = dict(status="started", source=str(args.surface),
                  source_sha256=hashlib.sha256(args.surface.read_bytes()).hexdigest(),
                  JAX=jax.__version__, physical_accuracy_accepted=False)
    try:
        report.update(verify(args.surface, derivatives=args.derivatives,
                             max_half_panels=args.max_half_panels))
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n",
                            encoding="utf-8")
        raise
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n",
                        encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
