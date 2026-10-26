"""Read-only pressure / circulation / ideal-Trefftz drag consistency audit.

This deliberately compares DIFFERENT estimators: finite fixed-wake panel
pressure integration and a semi-infinite flat-wake lifting-line approximation.
A mismatch is reported, never concealed by editing Cp or forcing equality.

The pure mathematical functions require only NumPy. The opt-in CLI imports
the existing half-native JAX solver lazily; it never invokes CAD/remeshing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def ideal_trefftz_piecewise_linear(stations, gamma, *, U, density, reference_area):
    """Semi-infinite flat-wake induced CD from a full-span Gamma(y).

    Gamma is linearly interpolated between increasing full-span stations
    with zero values at both physical tips. The induced normal velocity
    uses an analytic per-segment Cauchy principal-value logarithm, evaluated
    at segment midpoints. Spanwise force integration uses midpoint rule.

    For this convention: w=-PV integral(Gamma'(s)/(y-s) ds)/(4*pi)
    and Di=-rho integral(Gamma(y)*w(y) dy). The 1/(4*pi) corresponds
    to near-wing semi-infinite trailing-filament downwash, not a direct
    finite panel-wake momentum flux. An elliptic Gamma reproduces
    CDi=CL**2/(pi*AR) under refinement.
    """
    y = np.asarray(stations, dtype=np.float64)
    g = np.asarray(gamma, dtype=np.float64)
    if (y.ndim != 1 or len(y) < 3 or g.shape != y.shape
            or not np.isfinite(y).all() or not np.isfinite(g).all()
            or np.any(np.diff(y) <= 0)
            or not np.isfinite([U, density, reference_area]).all()
            or min(U, density, reference_area) <= 0):
        raise ValueError("Ascending full stations, Gamma, positive SI references required")
    limit = 1e-10 * max(np.max(np.abs(g)), 1.)
    if max(abs(g[0]), abs(g[-1])) > limit:
        raise ValueError("Gamma must vanish at both tips")
    a, b = y[:-1], y[1:]
    width = b-a
    centers = (a+b)/2
    slope = np.diff(g)/width
    # Self-interval analytic PV gives log(1) = 0. No denominator core or
    # arbitrary diagonal epsilon is introduced into the ideal estimator.
    with np.errstate(divide="raise", invalid="raise"):
        logs = np.log(np.abs((centers[:, None]-a[None, :])/
                             (centers[:, None]-b[None, :])))
    w = -(logs@slope)/(4*np.pi)
    gmid = (g[:-1]+g[1:])/2
    seg_drag = -density*gmid*w*width
    qS = .5*density*U*U*reference_area
    return dict(
        CDi_ideal=float(seg_drag.sum()/qS),
        CL_circulation=float(density*U*np.sum(gmid*width)/qS),
        induced_drag_N=float(seg_drag.sum()),
        circulation_lift_N=float(density*U*np.sum(gmid*width)),
        station_count=int(len(y)),
        downwash_min=float(w.min()),
        downwash_max=float(w.max()),
        signed_segment_drag_N=seg_drag.tolist(),
    )


def symmetric_half_gamma(positive_stations, positive_dofs, gamma_dofs):
    """Build even full-span gamma from frozen half root-contact DOF labels."""
    y = np.asarray(positive_stations, dtype=np.float64)
    dof = np.asarray(positive_dofs, dtype=np.int64)
    gamma = np.asarray(gamma_dofs, dtype=np.float64)
    if (y.ndim != 1 or len(y) < 2 or dof.shape != y.shape
            or gamma.ndim != 1 or not np.isfinite(y).all()
            or not np.isfinite(gamma).all()
            or y[0] != 0 or dof[0] < 0
            or np.any(np.diff(y) <= 0) or np.any(dof < -1)
            or dof[-1] != -1
            or np.any(dof[dof >= 0] >= len(gamma))):
        raise ValueError("Invalid half circulation station/DOF mapping")
    values = np.zeros_like(y)
    occupied = dof >= 0
    values[occupied] = gamma[dof[occupied]]
    if len(set(dof[occupied].tolist())) != len(gamma):
        raise ValueError("Missing half circulation unknown")
    return np.r_[-y[:0:-1], y], np.r_[values[:0:-1], values]


def pressure_integral_audit(vertices, faces, Cp, freestream,
                            *, density, reference_area, component_ids=None):
    """Independently integrate raw Cp; decompose cancellation without clipping."""
    x = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    cp = np.asarray(Cp, dtype=np.float64)
    flow = np.asarray(freestream, dtype=np.float64)
    if (x.ndim != 2 or x.shape[1] != 3
            or f.ndim != 2 or f.shape[1] != 3 or len(f) == 0
            or cp.shape != (len(f),) or flow.shape != (3,)
            or not np.isfinite(x).all() or not np.isfinite(cp).all()
            or not np.isfinite(flow).all()
            or np.any(f < 0) or np.any(f >= len(x))
            or not np.isfinite([density, reference_area]).all()
            or min(density, reference_area) <= 0):
        raise ValueError("Invalid half pressure geometry or references")
    speed = float(np.linalg.norm(flow))
    if speed <= 0 or abs(flow[1])/speed > 1e-12:
        raise ValueError("Only zero-sideslip symmetry is supported")
    tri = x[f]
    av = .5*np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    if np.any(np.linalg.norm(av, axis=1) <= 0):
        raise ValueError("Degenerate physical wall triangle")
    qS = .5*density*speed**2*reference_area
    force = -(.5*density*speed**2)*cp[:, None]*av
    # Mirror doubles streamwise/vertical forces for zero sideslip; no
    # second set of independent wall states is created.
    drag_axis = flow/speed
    lift_axis = np.cross(drag_axis, [0., 1., 0.])
    lift_axis /= np.linalg.norm(lift_axis)
    face_drag = 2*(force@drag_axis)/qS
    face_lift = 2*(force@lift_axis)/qS
    cd = float(face_drag.sum())
    abs_sum = float(np.abs(face_drag).sum())
    denom = abs(cd)
    x_part = float(2*force[:, 0].sum()*drag_axis[0]/qS)
    z_part = float(2*force[:, 2].sum()*drag_axis[2]/qS)
    tags = np.asarray(component_ids if component_ids is not None
                      else np.zeros(len(f), dtype=np.int64))
    if tags.shape != (len(f),):
        raise ValueError("Invalid component IDs")
    by_component = {}
    for tag in np.unique(tags):
        mask = tags == tag
        by_component[str(int(tag))] = dict(
            wall_panels=int(mask.sum()),
            CD_pressure=float(face_drag[mask].sum()),
            CL_pressure=float(face_lift[mask].sum()),
            sum_abs_drag=float(np.abs(face_drag[mask]).sum()),
        )
    return dict(
        CD_pressure=cd, CL_pressure=float(face_lift.sum()),
        drag_x_projection=x_part, drag_z_projection=z_part,
        face_positive_drag=float(face_drag[face_drag > 0].sum()),
        face_negative_drag=float(face_drag[face_drag < 0].sum()),
        face_sum_abs_drag=abs_sum,
        face_cancellation_factor=(abs_sum/denom if denom else None),
        projection_cancellation_factor=(
            (abs(x_part)+abs(z_part))/denom if denom else None),
        max_abs_delta_CD_if_1pct_relative_face_drag_error=.01*abs_sum,
        Cp_raw_min=float(cp.min()), Cp_raw_max=float(cp.max()),
        component_summary=by_component,
    )


def wake_alignment_audit(vertices, wake_faces, freestream):
    """Check freestream normal flux on wake; NOT total wake kinematic residual."""
    x = np.asarray(vertices, dtype=float)
    f = np.asarray(wake_faces, dtype=int)
    u = np.asarray(freestream, dtype=float)
    if (f.ndim != 2 or f.shape[1] != 3 or len(f) == 0
            or u.shape != (3,) or np.linalg.norm(u) == 0
            or np.any(f < 0) or np.any(f >= len(x))):
        raise ValueError("Invalid half wake geometry")
    tri = x[f]
    area = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    length = np.linalg.norm(area, axis=1)
    if np.any(length <= 0):
        raise ValueError("Zero-area wake triangle")
    normal = area/length[:, None]
    ratio = np.abs(normal@(u/np.linalg.norm(u)))
    return dict(
        max_abs_Udotn_over_U=float(ratio.max()),
        rms_Udotn_over_U=float(np.sqrt(np.average(ratio**2, weights=length))),
        area_weighted_mean_abs_Udotn_over_U=float(
            np.average(ratio, weights=length)),
        test_type="freestream_flux_only_not_full_local_wake_tangency",
    )


def audit_saved_half_state(fixture, fields):
    """Report competing estimators without assuming they must be identical."""
    rho = float(fixture["density"])
    area = float(fixture["reference_area"])
    flow = np.asarray(fixture["freestream"])
    pressure = pressure_integral_audit(
        fixture["coordinates"], fixture["faces"], fields["Cp_raw_half"],
        flow, density=rho, reference_area=area,
        component_ids=fixture["component_ids"])
    y, g = symmetric_half_gamma(
        fixture["gamma_stations"], fixture["gamma_station_dof"],
        fields["gamma_half"])
    far = ideal_trefftz_piecewise_linear(
        y, g, U=float(np.linalg.norm(flow)), density=rho,
        reference_area=area)
    wake = wake_alignment_audit(
        fixture["coordinates"], fixture["wake_faces"], flow)
    near_cd, far_cd = pressure["CD_pressure"], far["CDi_ideal"]
    return dict(
        half_wall_count=len(fixture["faces"]),
        half_wake_count=len(fixture["wake_faces"]),
        half_gamma_unknowns=len(np.asarray(fields["gamma_half"])),
        near_pressure=pressure, ideal_trefftz=far, wake_alignment=wake,
        CD_near_minus_ideal_far=float(near_cd-far_cd),
        CD_near_over_ideal_far=float(near_cd/far_cd)
            if abs(far_cd) > 1e-15 else None,
        CL_near_minus_circulation=float(
            pressure["CL_pressure"]-far["CL_circulation"]),
        ideal_trefftz_is_not_exact_finite_wake_momentum=True,
        physical_accuracy_accepted=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-half-panels", type=int, default=1600)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Do not overwrite a previous drag or mesh audit")
    if args.max_half_panels < 1:
        raise ValueError("Positive dense LU guard required")
    fixture = json.loads(args.surface.read_text(encoding="utf-8"))
    if fixture.get("schema") != "dpanel.half_planar_cut.v1":
        raise ValueError("Require genuine dpanel.half_planar_cut.v1")
    if len(fixture["faces"]) > args.max_half_panels:
        raise MemoryError("Explicitly set max-half-panels for a larger dense solve")

    from .run_half_native import load_half_surface
    from .half_native import solve_half_fields
    import jax
    jax.config.update("jax_enable_x64", True)
    inputs = load_half_surface(args.surface)
    result_fields = jax.jit(solve_half_fields)(**inputs)
    result_fields["CL"].block_until_ready()
    state = {k: np.asarray(v) for k, v in result_fields.items()
             if k in ("Cp_raw_half", "gamma_half", "mu_half", "force")}
    result = audit_saved_half_state(fixture, state)
    result.update(
        source_sha256=hashlib.sha256(args.surface.read_bytes()).hexdigest(),
        JAX=jax.__version__, solver="half-native dense LU, immutable CAD mesh")
    flow = np.asarray(fixture["freestream"])
    speed = float(np.linalg.norm(flow))
    qS = .5*float(fixture["density"])*speed**2*float(fixture["reference_area"])
    cd_solver = float(np.dot(state["force"], flow/speed)/qS)
    result["CD_pressure_solver"] = cd_solver
    result["independent_pressure_integration_max_abs_error"] = abs(
        cd_solver-result["near_pressure"]["CD_pressure"])
    if result["independent_pressure_integration_max_abs_error"] > 1e-10:
        raise AssertionError("Independent Cp force integration differs from solver")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
