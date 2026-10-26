"""Issue #49 Phase 0: independent reference metrics on one frozen half solve.

Every number here is evaluated on the *total* perturbation field

    phi(p) = -(S+S_mirror) sigma + (D+D_mirror) mu
             + [ (W+W_mirror) basis + (M-M_mirror) slope ] gamma

with the wall source sheet, the wall doublet sheet, the trailing wake doublet
sheet and their y-mirrors included, and the gradient taken by forward-mode AD
through the same analytic triangle kernel.  Nothing is a freestream
projection, nothing is a Trefftz functional, and no second discretisation is
solved: these are properties of the one solution that was already produced.

The source sign is the assembly convention: the system solves
(D+D_i) mu + W gamma = (S+S_i) sigma, i.e. the interior condition
-S sigma + D mu + W gamma = 0.  An interior probe confirms it (|phi| rms
7e-5 just inside the wall with the minus sign, 5.6e-2 with the plus sign).

Metric groups
  linear_residual   ||A x - b|| / ||b|| rebuilt from the returned matrix
  interior_trace    |phi| just inside every wall panel (assembly consistency)
  wall_normal_trace |V.n|/U one panel-normal offset range off every panel
  te_limits         Cp on both sides of every true TE edge, offset sweep
  wake_trace        V.n_wake/U and the Cp jump across the wake sheet
  wake_transport    dGamma/dy against the mean tangential wake velocity
  circulation_lift  Kutta-Joukowski lift of the converged gamma distribution
  nearfield_drag    signed CD_p, sum of |per-face drag| and per component

Off-surface traces are reported as a sweep, never as one number: the constant
doublet panel has an edge singularity, so the smallest offsets are dominated
by |V| ~ 1/distance to the panel edge and no limit may be claimed from them.
The production solver default is untouched; no accuracy gate is changed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .kernel import triangle_potential
from .half_native import (
    _half_wake_te_geometry, _mirror_triangles, solve_half_fields,
)
from .run_half_native import load_half_surface

WALL_OFFSETS = (1e-4, 1e-3, 1e-2)
INTERIOR_OFFSET = 1e-3
TE_OFFSETS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
WAKE_OFFSETS = (1e-3, 3e-3, 1e-2)


class HalfField:
    """Total perturbation potential and velocity of one frozen half state.

    The class holds no solver logic.  It rebuilds exactly the influence
    operators that the assembly used (``half_native.solve_half_fields``) and
    evaluates them at arbitrary points, so traces can be taken off-surface.
    """

    def __init__(self, fields, result):
        self.x = jnp.asarray(fields["vertices"])
        self.t = fields["topology"]
        self.y = jnp.asarray(fields["station_y"])
        self.flow = jnp.asarray(fields["freestream"])
        self.speed = float(np.linalg.norm(np.asarray(self.flow)))
        tri = self.x[self.t.faces]
        area_vector = jnp.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) / 2
        self.tri = tri
        self.areas = jnp.linalg.norm(area_vector, axis=1)
        self.normals = area_vector / self.areas[:, None]
        self.centers = tri.mean(axis=1)
        self.mu = jnp.asarray(result["mu_half"])
        self.gamma = jnp.asarray(result["gamma_half"])
        self.sigma = jnp.asarray(result["sigma_half"])
        (self.wake, self.basis, self.slope, self.T, self.M,
         self.te_basis) = _half_wake_te_geometry(self.x, self.y, self.t)
        self.mirror_tri = _mirror_triangles(self.tri)
        self.mirror_wake = _mirror_triangles(self.wake)

    def potential(self, points):
        """Total perturbation potential at ``points`` (n,3), shape (n,)."""
        p = jnp.asarray(points)
        single, doublet = triangle_potential(p, self.tri)
        single_i, doublet_i = triangle_potential(p, self.mirror_tri)
        phi = (-(single + single_i) @ self.sigma
               + (doublet + doublet_i) @ self.mu)
        if self.wake.shape[0]:
            wake_d, wake_m = triangle_potential(
                p, self.wake, source=False, linear_y=True)
            wake_di, wake_mi = triangle_potential(
                p, self.mirror_wake, source=False, linear_y=True)
            phi = phi + (((wake_d + wake_di) @ self.basis
                         + (wake_m - wake_mi) @ self.slope) @ self.gamma)
        return phi

    def _point_potential(self, point):
        return self.potential(point[None])[0]

    def velocity(self, points, *, chunk=256):
        """Total velocity U_inf + grad(phi) at ``points`` (n,3).

        The gradient is analytic (forward mode through the kernel), never a
        finite difference of the potential.  Targets are evaluated in chunks
        so the kernel working set stays O(chunk * panels).
        """
        pts = jnp.asarray(points, dtype=jnp.float64)
        evaluate = jax.jit(jax.vmap(jax.jacfwd(self._point_potential)))
        pieces = [evaluate(pts[start:start + chunk])
                  for start in range(0, pts.shape[0], chunk)]
        grad = jnp.concatenate(pieces, axis=0) if len(pieces) > 1 else pieces[0]
        return self.flow[None, :] + grad

    def speed_ratio(self, points):
        v = self.velocity(points)
        return jnp.sqrt(jnp.sum(v * v, axis=1)) / self.speed

    def cp(self, points):
        ratio = self.speed_ratio(points)
        return 1 - ratio * ratio


def _f(values):
    return np.asarray(values, dtype=float)


def _describe(values):
    values = _f(values)
    absolute = np.abs(values)
    return {
        "rms": float(np.sqrt(np.mean(values * values))),
        "mean": float(np.mean(values)),
        "median_abs": float(np.median(absolute)),
        "p90_abs": float(np.quantile(absolute, 0.9)),
        "max_abs": float(np.max(absolute)),
    }


def linear_residual(result):
    """||A x - b|| / ||b|| rebuilt from the returned matrix."""
    matrix = _f(result["matrix_half"])
    rhs = _f(result["rhs_half"])
    state = np.concatenate((_f(result["mu_half"]).ravel(),
                            _f(result["gamma_half"]).ravel()))
    residual = matrix @ state - rhs
    return {"absolute": float(np.linalg.norm(residual)),
            "relative": float(np.linalg.norm(residual) / np.linalg.norm(rhs)),
            "unknowns": int(state.size)}


def interior_trace(field, *, chord, offset=INTERIOR_OFFSET):
    """|phi| just inside every wall panel: the assembled interior condition.

    A correct reconstruction makes this small; it is small because the system
    enforces it, so it checks the reconstruction, not the physics.
    """
    points = _f(field.centers) - (offset * chord) * _f(field.normals)
    phi = _f(field.potential(points)).ravel()
    scale = field.speed * chord
    return {"offset_chords": offset,
            "rms": float(np.sqrt(np.mean(phi * phi))),
            "max_abs": float(np.max(np.abs(phi))),
            "normalized_rms_over_Uc": float(np.sqrt(np.mean(phi * phi)) / scale)}


def wall_normal_trace(field, *, chord, offsets=WALL_OFFSETS):
    """True exterior trace V.n/U off every wall panel centroid."""
    normals = _f(field.normals)
    centers = _f(field.centers)
    out = {"offsets_chords": list(offsets), "panels": int(centers.shape[0]),
           "sweep": []}
    for distance in offsets:
        velocity = _f(field.velocity(centers + (distance * chord) * normals))
        flux = np.sum(velocity * normals, axis=1) / field.speed
        entry = _describe(flux)
        entry["offset_chords"] = distance
        out["sweep"].append(entry)
    out["rms"] = [entry["rms"] for entry in out["sweep"]]
    out["max_abs"] = [entry["max_abs"] for entry in out["sweep"]]
    out["median_abs"] = [entry["median_abs"] for entry in out["sweep"]]
    return out


def te_limits(field, *, chord, offsets=TE_OFFSETS):
    """Two-sided Cp limits at the true trailing edge, by offset sweep.

    Points leave the TE edge midpoint along the outward normal of the upper
    and of the lower adjacent panel, so neither ray crosses the body.  All
    offsets are reported; a linear intercept is also reported but is marked
    invalid when the smallest offset is more than ten times the largest.
    """
    ends = _f(field.x[field.t.te_edges])
    if ends.size == 0:
        return {"edges": 0, "note": "no true TE edges in this fixture"}
    mid = ends.mean(axis=1)
    normals = _f(field.normals)
    upper_normal = normals[np.asarray(field.t.upper_te)]
    lower_normal = normals[np.asarray(field.t.lower_te)]
    cp_upper, cp_lower = [], []
    for distance in offsets:
        cp_upper.append(_f(field.cp(mid + (distance * chord) * upper_normal)))
        cp_lower.append(_f(field.cp(mid + (distance * chord) * lower_normal)))
    cp_upper, cp_lower = np.stack(cp_upper), np.stack(cp_lower)
    jump = cp_upper[0] - cp_lower[0]
    design = np.stack([np.ones(len(offsets)), np.asarray(offsets)]).T
    upper_fit, *_ = np.linalg.lstsq(design, cp_upper, rcond=None)
    lower_fit, *_ = np.linalg.lstsq(design, cp_lower, rcond=None)
    singular = bool(np.max(np.abs(cp_upper[0])) > 10 * np.max(np.abs(cp_upper[-1])))
    worst = int(np.argmax(np.abs(jump)))
    return {
        "edges": int(mid.shape[0]),
        "offsets_chords": list(offsets),
        "cp_upper": cp_upper.tolist(),
        "cp_lower": cp_lower.tolist(),
        "cp_upper_mean": cp_upper.mean(axis=1).tolist(),
        "cp_lower_mean": cp_lower.mean(axis=1).tolist(),
        "cp_jump_mean": (cp_upper.mean(axis=1) - cp_lower.mean(axis=1)).tolist(),
        "cp_jump_rms": float(np.sqrt(np.mean(jump * jump))),
        "cp_jump_max_abs": float(np.max(np.abs(jump))),
        "worst_edge": worst,
        "worst_edge_cp_upper": cp_upper[:, worst].tolist(),
        "worst_edge_cp_lower": cp_lower[:, worst].tolist(),
        "edge_singular": singular,
        "linear_extrapolation_valid": (not singular),
        "extrapolated_cp_upper_mean": float(np.mean(upper_fit[0])),
        "extrapolated_cp_lower_mean": float(np.mean(lower_fit[0])),
        "extrapolated_cp_jump_mean": float(np.mean(upper_fit[0] - lower_fit[0])),
    }


def wake_trace(field, *, chord, offsets=WAKE_OFFSETS):
    """V.n/U and Cp jump across the wake doublet sheet, both sides measured.

    A doublet sheet is a vortex sheet: its normal velocity is continuous and
    its tangential velocity jumps.  The mean of the two sides is therefore
    the sheet-normal flow, and the difference is the sheet strength.
    """
    wake = _f(field.wake)
    if wake.size == 0:
        return {"panels": 0, "note": "no wake panels in this fixture"}
    area_vector = np.cross(wake[:, 1] - wake[:, 0], wake[:, 2] - wake[:, 0]) / 2
    areas = np.linalg.norm(area_vector, axis=1)
    normals = area_vector / areas[:, None]
    centers = wake.mean(axis=1)
    out = {"panels": int(centers.shape[0]), "offsets_chords": list(offsets),
           "sweep": []}
    for distance in offsets:
        offset = (distance * chord) * normals
        plus = _f(field.velocity(centers + offset))
        minus = _f(field.velocity(centers - offset))
        flux_plus = np.sum(plus * normals, axis=1) / field.speed
        flux_minus = np.sum(minus * normals, axis=1) / field.speed
        cp_plus = 1 - np.sum(plus * plus, axis=1) / field.speed ** 2
        cp_minus = 1 - np.sum(minus * minus, axis=1) / field.speed ** 2
        out["sweep"].append({
            "offset_chords": distance,
            "normal_flux": _describe(0.5 * (flux_plus + flux_minus)),
            "normal_flux_jump": _describe(flux_plus - flux_minus),
            "cp_jump_lower_minus_upper": _describe(cp_minus - cp_plus),
        })
    return out


def wake_transport(field, *, chord, offset_chords=1e-3):
    """Measured tangential flow and spanwise strength gradient on the sheet.

    The wake doublet density is piecewise constant along the span, so the
    tangential velocity jump across the sheet vanishes *inside* every strip by
    construction: for such a representation the jump is not a test of the
    steady transport identity.  What is measured instead is the mean
    tangential velocity across the sheet, the spanwise gradient of the
    circulation, and the ratio between them, which states how far the
    discrete sheet is from the continuous vortex-sheet picture.  This is a
    representation diagnostic, never an acceptance criterion.
    """
    wake = _f(field.wake)
    if wake.size == 0:
        return {"panels": 0, "note": "no wake panels in this fixture"}
    edges = np.stack([wake[:, 1] - wake[:, 0], wake[:, 2] - wake[:, 1],
                      wake[:, 0] - wake[:, 2]], axis=1)
    span = np.abs(edges[:, :, 1]).argmin(axis=1)
    tangent = edges[np.arange(len(wake)), span]
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    centers = wake.mean(axis=1)
    area_vector = np.cross(wake[:, 1] - wake[:, 0], wake[:, 2] - wake[:, 0]) / 2
    normals = area_vector / np.linalg.norm(area_vector, axis=1)[:, None]
    offset = (offset_chords * chord) * normals
    plus = _f(field.velocity(centers + offset))
    minus = _f(field.velocity(centers - offset))
    mean_tangential = np.sum(0.5 * (plus + minus) * tangent, axis=1)
    jump_tangential = np.sum((plus - minus) * tangent, axis=1)
    gradient = _f(field.slope) @ _f(field.gamma)
    speed = field.speed
    gradient_rms = float(np.sqrt(np.mean(gradient ** 2)))
    jump_rms = float(np.sqrt(np.mean(jump_tangential ** 2)))
    return {
        "panels": int(centers.shape[0]),
        "offset_chords": offset_chords,
        "tangential_velocity_over_U": _describe(mean_tangential / speed),
        "tangential_jump_over_U": _describe(jump_tangential / speed),
        "spanwise_gradient_over_U": _describe(gradient / speed),
        "jump_over_gradient_rms": (jump_rms / gradient_rms
                                   if gradient_rms > 0 else None),
        "interpretation": (
            "piecewise-constant doublet wake: the tangential jump vanishes "
            "inside a strip by construction, so the jump/gradient ratio "
            "measures representation error, not a transport residual"),
    }


def circulation_lift(field, *, density, area, chord, speed):
    """Kutta-Joukowski lift implied by the converged gamma distribution.

    The TE fan gives int(hat_j) dy per circulation dof; the half model is
    doubled about y=0.  L = rho U int_(full span) Gamma dy.
    """
    ends = _f(field.x[field.t.te_edges])
    if ends.size == 0:
        return {"CL_gamma": None, "note": "no true TE edges in this fixture"}
    length = np.linalg.norm(ends[:, 1] - ends[:, 0], axis=1)
    basis = _f(field.te_basis)
    measure = (0.5 * length[:, None, None] * basis).sum(axis=(0, 1))
    gamma = _f(field.gamma)
    lift = 2.0 * density * speed * float(measure @ gamma)
    dynamic = 0.5 * density * speed ** 2
    return {"CL_gamma": lift / (dynamic * area),
            "lift": lift,
            "span_measure": measure.tolist(),
            "gamma": gamma.tolist()}


def nearfield_drag(field, result, *, density, area, chord, speed):
    """Signed near-field drag and the gross |per-face| drag sum.

    The mirrored panel force is f*(1,-1,1) and the drag axis has no y
    component, so a mirrored panel contributes the same signed drag: totals
    are twice the half-model sums, signed and absolute alike.
    """
    axis = _f(field.flow) / field.speed
    per_face = _f(result["panel_forces_half"]) @ axis
    dynamic = 0.5 * density * speed ** 2
    tags = np.asarray(field.t.component_ids)
    by_component = {}
    for tag in sorted(set(int(v) for v in tags)):
        mask = tags == tag
        by_component[str(tag)] = {
            "CD_pressure": 2.0 * float(per_face[mask].sum()) / (dynamic * area),
            "CD_pressure_sum_abs": (2.0 * float(np.abs(per_face[mask]).sum())
                                    / (dynamic * area)),
            "panels": int(mask.sum()),
        }
    return {
        "CD_pressure": 2.0 * float(per_face.sum()) / (dynamic * area),
        "CD_pressure_sum_abs": 2.0 * float(np.abs(per_face).sum()) / (dynamic * area),
        "panels": int(per_face.size),
        "by_component": by_component,
    }


def audit(fields, result, *, chord, area, density, speed,
          wall_offsets=WALL_OFFSETS, te_offsets=TE_OFFSETS,
          wake_offsets=WAKE_OFFSETS):
    field = HalfField(fields, result)
    return {
        "linear_residual": linear_residual(result),
        "interior_trace": interior_trace(field, chord=chord),
        "wall_normal_trace": wall_normal_trace(
            field, chord=chord, offsets=wall_offsets),
        "te_limits": te_limits(field, chord=chord, offsets=te_offsets),
        "wake_trace": wake_trace(field, chord=chord, offsets=wake_offsets),
        "wake_transport": wake_transport(field, chord=chord),
        "circulation_lift": circulation_lift(
            field, density=density, area=area, chord=chord, speed=speed),
        "nearfield_drag": nearfield_drag(
            field, result, density=density, area=area, chord=chord, speed=speed),
    }


def solve_case(fields, *, max_panels=6000):
    """One deterministic dense LU half solve with a warm timing repeat."""
    nf = len(fields["topology"].faces)
    if nf > max_panels:
        raise MemoryError("Refusing oversized experimental half panel solve")
    solve = jax.jit(solve_half_fields)
    started = time.perf_counter()
    result = solve(**fields)
    result["CL"].block_until_ready()
    first = time.perf_counter() - started
    started = time.perf_counter()
    result = solve(**fields)
    result["CL"].block_until_ready()
    return result, {"first_seconds": first,
                    "hot_seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(
        description="Issue #49 Phase 0 reference audit of one half fixture")
    parser.add_argument("--case", type=Path, required=True,
                        help="directory holding surface.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rings", type=int, default=2)
    parser.add_argument("--max-stencil-rings", type=int, default=6)
    parser.add_argument("--max-panels", type=int, default=6000)
    args = parser.parse_args()
    surface = args.case / "surface.json"
    if not surface.exists():
        raise FileNotFoundError(str(surface))
    if args.out.exists():
        raise FileExistsError("Use a new report path; preserve existing evidence")
    args.out.mkdir(parents=True)
    jax.config.update("jax_enable_x64", True)
    fields, stencil = load_half_surface(
        surface, stencil_rings=args.rings,
        max_stencil_rings=args.max_stencil_rings, return_stencil_audit=True)
    result, timing = solve_case(fields, max_panels=args.max_panels)
    fixture = json.loads(surface.read_text(encoding="utf-8"))
    density = float(fixture["density"])
    area = float(fixture["reference_area"])
    chord = float(fixture["reference_chord"])
    speed = float(np.linalg.norm(np.asarray(fixture["freestream"])))
    metrics = audit(fields, result, chord=chord, area=area,
                    density=density, speed=speed)
    from .baseline_manifest import build_manifest
    residual = metrics["linear_residual"]
    manifest = build_manifest(
        surface_path=surface, fixture=fixture, result=result,
        residual_absolute=residual["absolute"],
        residual_relative=residual["relative"], metrics=metrics,
        case_id=args.case.name)
    np.savez(
        args.out / "state.npz",
        mu_half=_f(result["mu_half"]), gamma_half=_f(result["gamma_half"]),
        sigma_half=_f(result["sigma_half"]), Cp_raw_half=_f(result["Cp_raw_half"]))
    report = {
        "status": "ok",
        "case": str(args.case),
        "surface": str(surface),
        "schema": fixture.get("schema"),
        "timing": timing,
        "wall_panels": int(len(fields["topology"].faces)),
        "wake_panels": int(len(fields["topology"].wake_faces)),
        "circulation_unknowns": int(fields["topology"].hat_map.shape[1]),
        "CL": float(result["CL"]),
        "CD_pressure": float(result["CD_pressure"]),
        "CM": [float(v) for v in np.asarray(result["CM"]).ravel()],
        "pressure_stencil_audit": stencil,
        "metrics": metrics,
        "code_digest": manifest["code"]["digest"],
        "physical_accuracy_accepted": False,
    }
    for name, payload in (("metrics.json", metrics), ("manifest.json", manifest),
                          ("report.json", report)):
        (args.out / name).write_text(
            json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "wall_panels": report["wall_panels"],
        "CL": report["CL"],
        "CD_pressure": report["CD_pressure"],
        "CL_gamma": metrics["circulation_lift"]["CL_gamma"],
        "residual_relative": residual["relative"],
        "wall_flux_rms": metrics["wall_normal_trace"]["rms"],
        "te_cp_jump_rms": metrics["te_limits"].get("cp_jump_rms"),
        "out": str(args.out),
    }, indent=1))


if __name__ == "__main__":
    main()