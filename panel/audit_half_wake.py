"""Read-only two-sided total-flow residuals on a native half-domain wake sheet.

This driver closes the structural gap recorded in docs/pr26_audit_a/README.md.
The half-native solver never writes a full-surface run directory, so the legacy
panel/audit.py wake diagnostics (total-flow tangency, material circulation
transport and the two-sided pressure jump) cannot be evaluated on a half case.
Nothing here is production code: no solver, pressure reconstruction, wake shape
or default mode is modified, and no production module is imported for mutation.

Method
------
1. Re-solve the same frozen half fixture with panel.half_native.solve_half_fields.
2. Expand the half state to the exact mirrored full discretization with the
   already-accepted panel.verify_half_mirror maps (make_full_mirror_topology).
   No nearest-neighbour pairing and no independent left-side solve.
3. Evaluate the analytic single/double-layer field with the numba kernel in
   core/triangle_kernel.py. JAX is used only to re-derive the same potentials
   as an independent cross-kernel check, never as the reported field, and the
   solver's tangential affine Cp reconstruction is never used.

Reported wake diagnostics, per sample point and area weighted:
  flux      = Vmean . n / U            (total-flow tangency, NOT Uinf . n)
  transport = Vmean . grad(Gamma) / U^2 (material circulation transport)
  dCp       = Cp(+) - Cp(-)             (two-sided static pressure jump)
with Vmean = (V(+) + V(-))/2 at +/- eps along the +z wake normal, and the
algebraic identity dCp = -2 Vmean . (V(+) - V(-)) / U^2 verified separately.
The velocity magnitude used for Cp is the full analytic field, so dCp here is
not a reconstructed Cp and is not required to match the solver's Cp_raw.

Run:
  python -m panel.audit_half_wake --surface runs/HALF/surface.json \
    --out runs/HALF_WAKE/audit.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

_CORE = Path(__file__).resolve().parents[1] / "core"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

from triangle_kernel import influence  # noqa: E402

TRACE_OFF_SURFACE = -1
WAKE_NORMAL = np.asarray([0.0, 0.0, 1.0])
REFLECT = np.asarray([1.0, -1.0, 1.0])
# Three interior barycentric points per triangle; the centroid alone would hide
# within-panel variation of the two-sided jumps.
BARYCENTRIC = (
    np.asarray([0.6, 0.2, 0.2]),
    np.asarray([0.2, 0.6, 0.2]),
    np.asarray([0.2, 0.2, 0.6]),
)
DEFAULT_OFFSETS = (1e-2, 1e-3, 1e-4)
SCHEMA = "dpanel.half_wake_audit.v1"


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha256_arrays(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        block = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
        digest.update(str(block.shape).encode("ascii"))
        digest.update(block.tobytes())
    return digest.hexdigest()


def _max_abs_error(reference, candidate):
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    peak = max(float(np.max(np.abs(a))), 1e-300)
    return dict(max_abs=float(np.max(np.abs(a - b))),
                relative_to_peak=float(np.max(np.abs(a - b)) / peak))


def build_mirror_state(surface_path, *, max_half_panels=2000):
    """Solve the frozen half fixture and expand it to the mirrored full state."""
    import jax

    from .differentiable import solve_fields
    from .half_native import solve_half_fields
    from .run_half_native import load_half_surface
    from .verify_half_mirror import make_full_mirror_topology, mirrored_full_input

    if not jax.config.x64_enabled:
        raise ValueError("Explicitly enable JAX float64 before auditing")
    raw = json.loads(Path(surface_path).read_text(encoding="utf-8"))
    half = load_half_surface(surface_path)
    nf = len(half["topology"].faces)
    if nf > max_half_panels:
        raise MemoryError(
            "Half mesh exceeds the explicit full-oracle guard; raise "
            "--max-half-panels deliberately")
    plan = make_full_mirror_topology(half, raw["gamma_station_dof"])
    full_input = mirrored_full_input(half, plan)
    half_out = jax.jit(solve_half_fields)(**half)
    full_out = jax.jit(solve_fields)(**full_input)
    half_state = np.r_[np.asarray(half_out["mu_half"]),
                       np.asarray(half_out["gamma_half"])]
    full_state = np.r_[np.asarray(full_out["mu"]), np.asarray(full_out["gamma"])]
    expanded = np.asarray(plan["E"]) @ half_state
    topology = plan["full_topology"]
    coordinates = np.asarray(full_input["vertices"], dtype=np.float64)
    faces = np.asarray(topology.faces, dtype=np.int64)
    wake_faces = np.asarray(topology.wake_faces, dtype=np.int64)
    triangles = np.ascontiguousarray(coordinates[faces])
    wake = np.ascontiguousarray(coordinates[wake_faces])
    area_vector = np.cross(triangles[:, 1] - triangles[:, 0],
                           triangles[:, 2] - triangles[:, 0]) / 2
    wake_area_vector = np.cross(wake[:, 1] - wake[:, 0],
                                wake[:, 2] - wake[:, 0]) / 2
    flow = np.asarray(full_input["freestream"], dtype=np.float64)
    half_gamma = int(plan["half_gamma_count"])
    half_wake_count = len(half["topology"].wake_faces)
    gamma_full = np.asarray(full_out["gamma"], dtype=np.float64)
    gamma_half = np.asarray(half_out["gamma_half"], dtype=np.float64)
    # The half basis maps to the half circulation dofs, so only the assembled
    # fields (basis @ gamma) are comparable across the two discretizations.
    wake_doublet_full = (np.asarray(full_out["wake_basis"], dtype=np.float64)
                         @ gamma_full)[:half_wake_count]
    wake_doublet_half = (np.asarray(half_out["wake_basis_half"], dtype=np.float64)
                         @ gamma_half)
    wake_slope_full = (np.asarray(full_out["wake_slope"], dtype=np.float64)
                       @ gamma_full)[:half_wake_count]
    wake_slope_half = (np.asarray(half_out["wake_slope_half"], dtype=np.float64)
                       @ gamma_half)
    return dict(
        surface_path=str(Path(surface_path).resolve()),
        surface_sha256=_sha256_file(surface_path),
        raw=raw,
        plan=plan,
        topology=topology,
        coordinates=coordinates,
        faces=faces,
        wake_faces=wake_faces,
        triangles=triangles,
        normals=np.ascontiguousarray(area_vector / np.linalg.norm(
            area_vector, axis=1)[:, None]),
        area=np.linalg.norm(area_vector, axis=1),
        wake=wake,
        wake_centers=wake.mean(axis=1),
        wake_area=np.linalg.norm(wake_area_vector, axis=1),
        wake_normals=np.tile(WAKE_NORMAL, (len(wake), 1)),
        mu=np.asarray(full_out["mu"], dtype=np.float64),
        sigma=np.asarray(full_out["sigma"], dtype=np.float64),
        gamma=np.asarray(full_out["gamma"], dtype=np.float64),
        wake_basis=np.asarray(full_out["wake_basis"], dtype=np.float64),
        wake_slope=np.asarray(full_out["wake_slope"], dtype=np.float64),
        flow=flow,
        speed=float(np.linalg.norm(flow)),
        chord=float(np.asarray(full_input["reference_chord"])),
        half_panels=nf,
        half_wake=len(half["topology"].wake_faces),
        half_gamma=half_gamma,
        full_state=full_state,
        half_state=half_state,
        mirror_check=dict(
            expanded_state_vs_full=_max_abs_error(full_state, expanded),
            half_mu_vs_full_positive=_max_abs_error(
                np.asarray(full_out["mu"])[:nf],
                np.asarray(half_out["mu_half"])),
            half_gamma_vs_full_positive=_max_abs_error(
                np.asarray(full_out["gamma"])[:half_gamma],
                np.asarray(half_out["gamma_half"])),
            half_wake_doublet_vs_full_positive=_max_abs_error(
                wake_doublet_full, wake_doublet_half),
            half_wake_slope_vs_full_positive=_max_abs_error(
                wake_slope_full, wake_slope_half),
        ),
    )


def _subset(context, include_image):
    """Full mirrored discretization, or its positive half only (diagnostic)."""
    if include_image:
        return dict(panels=context["triangles"], normals=context["normals"],
                    mu=context["mu"], sigma=context["sigma"],
                    wake=context["wake"], wake_normals=context["wake_normals"],
                    wake_centers=context["wake_centers"],
                    wake_mu=context["wake_basis"] @ context["gamma"],
                    wake_slope=context["wake_slope"] @ context["gamma"])
    nf, nw = context["half_panels"], context["half_wake"]
    return dict(panels=context["triangles"][:nf], normals=context["normals"][:nf],
                mu=context["mu"][:nf], sigma=context["sigma"][:nf],
                wake=context["wake"][:nw], wake_normals=context["wake_normals"][:nw],
                wake_centers=context["wake_centers"][:nw],
                wake_mu=(context["wake_basis"] @ context["gamma"])[:nw],
                wake_slope=(context["wake_slope"] @ context["gamma"])[:nw])


def _moment_terms(targets, subset):
    """Constant and linearly varying wake doublet contributions.

    The linear term differentiates the analytic line-integral moment identity of
    panel/kernel.py; it is evaluated with the numba kernel, never with JAX.
    """
    wake, wake_normals = subset["wake"], subset["wake_normals"]
    centers = subset["wake_centers"]
    trace = np.full(len(targets), TRACE_OFF_SURFACE, dtype=np.int64)
    single, grad_single = influence(targets, wake, wake_normals, trace, 0)
    doublet, grad_doublet = influence(targets, wake, wake_normals, trace, 1)
    relative_y = targets[:, None, 1] - centers[None, :, 1]
    height = targets[:, None, 2] - centers[None, :, 2]
    moment = relative_y * doublet + height * grad_single[:, :, 1]
    hessian_y = np.zeros(grad_single.shape)
    for k, q in ((0, 1), (1, 2), (2, 0)):
        edge = wake[:, q] - wake[:, k]
        length = np.linalg.norm(edge, axis=1)
        outward = np.cross(edge / length[:, None], wake_normals)
        r0 = wake[None, :, k] - targets[:, None]
        r1 = wake[None, :, q] - targets[:, None]
        rad0 = np.linalg.norm(r0, axis=2)
        rad1 = np.linalg.norm(r1, axis=2)
        dot = np.sum(r0 * r1, axis=2)
        product = rad0 * rad1
        plus = product + dot
        opposite = dot < 0
        plus[opposite] = (np.sum(np.cross(r0, r1) ** 2, axis=2)[opposite]
                          / (product - dot)[opposite])
        gradient_log = (length[None, :, None] / plus[:, :, None]
                        * (r0 / rad0[:, :, None] + r1 / rad1[:, :, None]))
        hessian_y -= outward[None, :, 1, None] * gradient_log / (4 * np.pi)
    moment_gradient = (relative_y[:, :, None] * grad_doublet
                       + height[:, :, None] * hessian_y)
    moment_gradient[:, :, 1] += doublet
    moment_gradient[:, :, 2] += grad_single[:, :, 1]
    return dict(single=single, grad_single=grad_single, doublet=doublet,
                grad_doublet=grad_doublet, moment=moment,
                moment_gradient=moment_gradient)


def evaluate_field(targets, context, *, include_image=True, chunk=512,
                   trace_ids=None):
    """Total velocity and perturbation potential at off-surface targets.

    include_image=False drops the mirrored half as a diagnostic of how much of
    the field the image carries; it is never used for a reported residual.
    trace_ids selects the coincident-panel trace (wall centroids only).
    """
    targets = np.ascontiguousarray(np.asarray(targets, dtype=np.float64))
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("Targets must be an (N,3) array")
    subset = _subset(context, include_image)
    panels, panel_normals = subset["panels"], subset["normals"]
    mu, sigma = subset["mu"], subset["sigma"]
    wake_mu, wake_slope = subset["wake_mu"], subset["wake_slope"]
    velocity = np.empty((len(targets), 3))
    potential = np.empty(len(targets))
    for start in range(0, len(targets), chunk):
        stop = min(start + chunk, len(targets))
        block = targets[start:stop]
        if trace_ids is None:
            trace = np.full(len(block), TRACE_OFF_SURFACE, dtype=np.int64)
        else:
            trace = np.ascontiguousarray(trace_ids[start:stop], dtype=np.int64)
        single, grad_single = influence(block, panels, panel_normals, trace, 0)
        doublet, grad_doublet = influence(block, panels, panel_normals, trace, 1)
        moments = _moment_terms(block, subset)
        velocity[start:stop] = (
            context["flow"]
            - np.einsum("ijk,j->ik", grad_single, sigma)
            + np.einsum("ijk,j->ik", grad_doublet, mu)
            + np.einsum("ijk,j->ik", moments["grad_doublet"], wake_mu)
            + np.einsum("ijk,j->ik", moments["moment_gradient"], wake_slope))
        potential[start:stop] = (
            -single @ sigma + doublet @ mu + moments["doublet"] @ wake_mu
            + moments["moment"] @ wake_slope)
    return velocity, potential


def reference_potential(targets, context, *, chunk=256):
    """Same perturbation potential through the JAX kernel (cross-check only)."""
    import jax.numpy as jnp

    from .kernel import triangle_potential

    targets = np.asarray(targets, dtype=np.float64)
    panels = jnp.asarray(context["triangles"])
    wake = jnp.asarray(context["wake"])
    wake_mu = jnp.asarray(context["wake_basis"] @ context["gamma"])
    wake_slope = jnp.asarray(context["wake_slope"] @ context["gamma"])
    mu = jnp.asarray(context["mu"])
    sigma = jnp.asarray(context["sigma"])
    out = np.empty(len(targets))
    for start in range(0, len(targets), chunk):
        stop = min(start + chunk, len(targets))
        block = jnp.asarray(targets[start:stop])
        single, doublet = triangle_potential(block, panels, block_size=64)
        constant, moment = triangle_potential(block, wake, source=False,
                                              linear_y=True, block_size=64)
        value = (-single @ sigma + doublet @ mu
                 + constant @ wake_mu + moment @ wake_slope)
        out[start:stop] = np.asarray(value)
    return out


def wake_samples(context, *, include_image=True):
    """Centroid plus three interior barycentric points of every wake triangle."""
    wake = context["wake"] if include_image else context["wake"][:context["half_wake"]]
    centroid = np.asarray([1 / 3, 1 / 3, 1 / 3])
    samples = [np.einsum("ijk,j->ik", wake, centroid)]
    for bary in BARYCENTRIC:
        samples.append(np.einsum("ijk,j->ik", wake, bary))
    return np.stack(samples, axis=1)


def wake_labels(context):
    """Streamwise distance from the trailing edge and the near/far split."""
    ends = context["coordinates"][np.asarray(context["topology"].te_edges)]
    te_mid = ends.mean(axis=1)
    centers = context["wake_centers"]
    index = np.array([int(np.argmin(np.abs(te_mid[:, 1] - center[1])))
                      for center in centers], dtype=np.int64)
    gap = centers[:, 0] - te_mid[index, 0]
    # The streamwise extent of a truncated wake is a user choice, so the near /
    # far split is taken relative to that extent; an absolute 0.5 chord cut is
    # reported alongside for meshes whose wake is longer than the chord.
    extent = max(float(np.max(np.abs(gap))), 1e-300)
    fraction = gap / extent
    return dict(x_gap=gap, gap_fraction=fraction, near=fraction <= 0.5,
                near_absolute=gap <= 0.5 * context["chord"],
                te_index=index, te_mid=te_mid)


def two_sided_metrics(velocity_plus, velocity_minus, speed):
    mean = 0.5 * (velocity_plus + velocity_minus)
    difference = velocity_plus - velocity_minus
    return dict(
        mean=mean,
        difference=difference,
        flux=np.sum(mean * WAKE_NORMAL, axis=1) / speed,
        flux_plus=np.sum(velocity_plus * WAKE_NORMAL, axis=1) / speed,
        flux_minus=np.sum(velocity_minus * WAKE_NORMAL, axis=1) / speed,
        cp_plus=1 - np.sum(velocity_plus ** 2, axis=1) / speed ** 2,
        cp_minus=1 - np.sum(velocity_minus ** 2, axis=1) / speed ** 2,
        dcp=(np.sum(velocity_minus ** 2, axis=1)
             - np.sum(velocity_plus ** 2, axis=1)) / speed ** 2,
        identity=-np.sum(mean * difference, axis=1) * 2 / speed ** 2,
    )


def _weighted_rms(values, weights):
    return float(np.sqrt(np.average(np.asarray(values) ** 2,
                                    weights=np.asarray(weights))))


def _wake_transport(mean, context, include_image, per_panel):
    """Vmean . grad(Gamma) / U^2, repeated from panel to sample order."""
    slope = context["wake_slope"] @ context["gamma"]
    if not include_image:
        slope = slope[:context["half_wake"]]
    slope = np.repeat(slope, per_panel)
    return mean[:, 1] * slope / context["speed"] ** 2


def summarize(metrics, context, weights, mask, include_image, label, per_panel):
    if not np.any(mask):
        return None
    transport = _wake_transport(metrics["mean"], context, include_image,
                                per_panel)
    return dict(
        label=label,
        samples=int(np.count_nonzero(mask)),
        area_fraction=float(weights[mask].sum() / weights.sum()),
        flux_rms=_weighted_rms(metrics["flux"][mask], weights[mask]),
        flux_mean=float(np.average(metrics["flux"][mask], weights=weights[mask])),
        flux_plus_rms=_weighted_rms(metrics["flux_plus"][mask], weights[mask]),
        flux_minus_rms=_weighted_rms(metrics["flux_minus"][mask], weights[mask]),
        dcp_rms=_weighted_rms(metrics["dcp"][mask], weights[mask]),
        dcp_mean=float(np.average(metrics["dcp"][mask], weights=weights[mask])),
        dcp_abs_max=float(np.max(np.abs(metrics["dcp"][mask]))),
        transport_rms=_weighted_rms(transport[mask], weights[mask]),
        transport_mean=float(np.average(transport[mask], weights=weights[mask])),
        identity_abs_max=float(np.max(np.abs(
            metrics["dcp"][mask] - metrics["identity"][mask]))),
    )


def audit_wake(context, *, offsets=DEFAULT_OFFSETS, chunk=512,
               include_image=True):
    samples = wake_samples(context, include_image=include_image)
    panels, per_panel = samples.shape[0], samples.shape[1]
    labels = wake_labels(context)
    if not include_image:
        labels = dict(x_gap=labels["x_gap"][:panels],
                      near=labels["near"][:panels],
                      te_index=labels["te_index"][:panels])
    area = (context["wake_area"] if include_image
            else context["wake_area"][:context["half_wake"]])
    weights = np.repeat(area / per_panel, per_panel)
    near_mask = np.repeat(labels["near"], per_panel)
    by_offset, stored = [], None
    for factor in offsets:
        offset = (factor * np.sqrt(area))[:, None] * WAKE_NORMAL[None, :]
        plus = np.empty((panels, per_panel, 3))
        minus = np.empty((panels, per_panel, 3))
        for s in range(per_panel):
            plus[:, s, :], _ = evaluate_field(
                samples[:, s, :] + offset, context, include_image=include_image,
                chunk=chunk)
            minus[:, s, :], _ = evaluate_field(
                samples[:, s, :] - offset, context, include_image=include_image,
                chunk=chunk)
        metrics = two_sided_metrics(plus.reshape(-1, 3), minus.reshape(-1, 3),
                                    context["speed"])
        block = dict(
            offset_over_sqrt_area=factor,
            all=summarize(metrics, context, weights,
                          np.ones_like(near_mask), include_image, "all",
                          per_panel),
            near=summarize(metrics, context, weights, near_mask,
                           include_image, "near: first half of the wake "
                           "streamwise extent", per_panel),
            far=summarize(metrics, context, weights, ~near_mask,
                          include_image, "far: second half of the wake "
                          "streamwise extent", per_panel),
        )
        by_offset.append(block)
        if stored is None:
            stored = dict(metrics=metrics, plus=plus, minus=minus,
                          weights=weights, near_mask=near_mask,
                          labels=labels, area=area)
    sensitivity = dict(
        offsets=[float(f) for f in offsets],
        flux_rms=[block["all"]["flux_rms"] for block in by_offset],
        dcp_rms=[block["all"]["dcp_rms"] for block in by_offset],
        transport_rms=[block["all"]["transport_rms"] for block in by_offset],
        flux_rms_coarsest_minus_finest=(
            by_offset[0]["all"]["flux_rms"] - by_offset[-1]["all"]["flux_rms"]),
        dcp_rms_coarsest_minus_finest=(
            by_offset[0]["all"]["dcp_rms"] - by_offset[-1]["all"]["dcp_rms"]),
        # A converged two-sided average is O(eps) in the offset; the ratio of the
        # successive differences is reported instead of an absolute threshold.
        flux_rms_successive=[by_offset[i]["all"]["flux_rms"]
                             - by_offset[i + 1]["all"]["flux_rms"]
                             for i in range(len(by_offset) - 1)],
        dcp_rms_successive=[by_offset[i]["all"]["dcp_rms"]
                            - by_offset[i + 1]["all"]["dcp_rms"]
                            for i in range(len(by_offset) - 1)],
    )
    return dict(by_offset=by_offset, sensitivity=sensitivity, stored=stored)


def probe_points(context):
    """Off-surface probes: wake stations lifted clear of the sheet and walls."""
    count = len(context["wake_centers"])
    picks = np.unique(np.linspace(0, count - 1, 12).round().astype(int))
    probes = []
    for height in (0.15, 0.4):
        for pick in picks:
            center = context["wake_centers"][pick].copy()
            probes.append(center + np.asarray([0.0, 0.0, height * context["chord"]]))
    probes.append(context["flow"] * 0.0 + np.asarray(
        [6.0 * context["chord"], 1.5 * context["chord"], 3.0 * context["chord"]]))
    return np.asarray(probes)


def check_potential_gradient(context, *, steps=(1e-3, 1e-4), chunk=256):
    """Velocity must equal the central-difference gradient of the potential."""
    probes = probe_points(context)
    velocity, _ = evaluate_field(probes, context, chunk=chunk)
    results = []
    for step in steps:
        h = step * context["chord"]
        gradient = np.empty((len(probes), 3))
        for axis in range(3):
            basis = np.zeros(3)
            basis[axis] = h
            _, ahead = evaluate_field(probes + basis, context, chunk=chunk)
            _, behind = evaluate_field(probes - basis, context, chunk=chunk)
            # Total potential includes the freestream term flow . x.
            total_ahead = ahead + (probes + basis) @ context["flow"]
            total_behind = behind + (probes - basis) @ context["flow"]
            gradient[:, axis] = (total_ahead - total_behind) / (2 * h)
        scale = max(float(np.max(np.linalg.norm(velocity, axis=1))), 1e-300)
        results.append(dict(
            step_over_chord=step,
            max_abs=float(np.max(np.abs(gradient - velocity))),
            relative_to_peak_speed=float(np.max(np.abs(gradient - velocity))
                                         / scale),
            rms=float(np.sqrt(np.mean((gradient - velocity) ** 2)))))
    return dict(probes=len(probes), steps=results)


def check_potential_kernel(context, *, chunk=256):
    """Numba potentials against the JAX analytic kernel on the same targets."""
    probes = probe_points(context)
    _, numba = evaluate_field(probes, context, chunk=chunk)
    jax_potential = reference_potential(probes, context, chunk=chunk)
    error = _max_abs_error(jax_potential, numba)
    return dict(targets=len(probes), max_abs=error["max_abs"],
                relative_to_peak=error["relative_to_peak"],
                potential_peak=float(np.max(np.abs(numba))))


def check_wall_tangency(context, *, chunk=256):
    """Analytic total-flow normal velocity at wall centroids (exterior trace)."""
    nf = context["half_panels"]
    centers = (context["triangles"][:nf, 0] + context["triangles"][:nf, 1]
               + context["triangles"][:nf, 2]) / 3
    normal = context["normals"][:nf]
    velocity, potential = evaluate_field(
        centers, context, chunk=chunk,
        trace_ids=np.arange(nf, dtype=np.int64))
    residual = np.sum(velocity * normal, axis=1) / context["speed"]
    area = context["area"][:nf]
    return dict(
        cells=nf,
        normal_rms=float(np.sqrt(np.average(residual ** 2, weights=area))),
        normal_max=float(np.max(np.abs(residual))),
        interior_potential_rms=float(np.sqrt(np.average(
            (potential - context["mu"][:nf]) ** 2, weights=area))),
        note="exterior trace; interior potential = potential - mu",
    )


def check_mirror_equivariance(context, stored):
    """Face i and its mirror partner i+W/2 must carry the reflected field.

    Under y -> -y the velocity components transform as (Vx, -Vy, Vz) and the
    spanwise circulation gradient flips sign, so both are *antisymmetric* while
    the flux V.n, |V|^2 and the scalar invariant V.grad(Gamma) are *symmetric*.
    The check asserts each of those separately instead of assuming one rule.
    """
    half = context["half_wake"]
    metrics = stored["metrics"]
    per_panel = stored["plus"].shape[1]
    flux = metrics["flux"].reshape(-1, per_panel).mean(axis=1)
    dcp = metrics["dcp"].reshape(-1, per_panel).mean(axis=1)
    transport = _wake_transport(metrics["mean"], context, True,
                                per_panel).reshape(-1, per_panel).mean(axis=1)
    vy = metrics["mean"][:, 1].reshape(-1, per_panel).mean(axis=1)
    slope = context["wake_slope"] @ context["gamma"]
    speed = context["speed"]
    return dict(
        pairs=half,
        flux_max_abs=float(np.max(np.abs(flux[:half] - flux[half:2 * half]))),
        dcp_max_abs=float(np.max(np.abs(dcp[:half] - dcp[half:2 * half]))),
        transport_symmetry_max_abs=float(np.max(np.abs(
            transport[:half] - transport[half:2 * half]))),
        vy_antisymmetry_max_abs=float(np.max(np.abs(
            vy[:half] + vy[half:2 * half])) / speed),
        slope_sum_max_abs=float(np.max(np.abs(
            (slope[:half] + slope[half:2 * half])
            / max(float(np.max(np.abs(slope[:half]))), 1e-300)))),
        mu_symmetry_max_abs=float(np.max(np.abs(
            context["mu"][:context["half_panels"]]
            - context["mu"][context["half_panels"]:2 * context["half_panels"]]))),
    )


def check_image_contribution(context, stored, *, chunk=512):
    """How much of the reported field is carried by the mirrored half."""
    samples = wake_samples(context)[:, 0, :]
    offset = (1e-3 * np.sqrt(stored["area"]))[:, None] * WAKE_NORMAL[None, :]
    targets = np.vstack((samples + offset, samples - offset))
    full, _ = evaluate_field(targets, context, chunk=chunk)
    direct, _ = evaluate_field(targets, context, include_image=False, chunk=chunk)
    difference = full - direct
    return dict(
        targets=len(targets),
        image_rms_over_speed=float(np.sqrt(np.mean(np.sum(difference ** 2,
                                                          axis=1)))
                                   / context["speed"]),
        image_plus_rms_over_speed=float(np.sqrt(np.mean(np.sum(
            difference[:len(samples)] ** 2, axis=1))) / context["speed"]),
        flux_shift_rms=float(np.sqrt(np.mean(
            (np.sum(full[:len(samples)] * WAKE_NORMAL, axis=1)
             - np.sum(direct[:len(samples)] * WAKE_NORMAL, axis=1)) ** 2))
            / context["speed"]),
        note="full mirrored field minus positive-half-only field",
    )


def audit(surface_path, *, max_half_panels=2000, offsets=DEFAULT_OFFSETS,
          chunk=512, steps=(1e-3, 1e-4)):
    started = time.perf_counter()
    context = build_mirror_state(surface_path, max_half_panels=max_half_panels)
    wake = audit_wake(context, offsets=offsets, chunk=chunk)
    half_wake = audit_wake(context, offsets=offsets[:1], chunk=chunk,
                           include_image=False)
    stored = wake["stored"]
    report = dict(
        schema=SCHEMA,
        status="independently_audited",
        physical_accuracy_accepted=False,
        source_surface=context["surface_path"],
        surface_sha256=context["surface_sha256"],
        state_sha256=_sha256_arrays(context["mu"], context["gamma"],
                                    context["sigma"]),
        code_sha256=_sha256_file(__file__),
        environment=dict(python=platform.python_version(),
                         numpy=np.__version__, jax=_jax_version()),
        geometry=dict(half_panels=context["half_panels"],
                      half_wake=context["half_wake"],
                      half_gamma=context["half_gamma"],
                      full_panels=len(context["triangles"]),
                      full_wake=len(context["wake"]),
                      full_gamma=len(context["gamma"])),
        reference=dict(freestream=context["flow"].tolist(),
                       speed=context["speed"], chord=context["chord"]),
        mirror_oracle=context["mirror_check"],
        wake_report=dict(by_offset=wake["by_offset"],
                         sensitivity=wake["sensitivity"],
                         near_far_rule="near = the first half of the "
                                       "truncated wake streamwise extent "
                                       "(x_gap <= 0.5*max x_gap), x_TE at the "
                                       "nearest trailing-edge midpoint in y; "
                                       "an absolute 0.5*chord cut is recorded "
                                       "separately and can be empty",
                         sampling="4 points per wake triangle (centroid plus "
                                  "three interior barycentric points), area "
                                  "weighted, both mirrored halves"),
        positive_half_only=dict(
            warning="diagnostic only: fails exactly because the image is "
                    "missing",
            near_far=half_wake["by_offset"][0]),
        checks=dict(
            velocity_equals_potential_gradient=check_potential_gradient(
                context, steps=steps, chunk=chunk),
            potential_numba_vs_jax=check_potential_kernel(context, chunk=chunk),
            wall_tangency=check_wall_tangency(context, chunk=chunk),
            mirror_equivariance=check_mirror_equivariance(context, stored),
            image_contribution=check_image_contribution(context, stored,
                                                        chunk=chunk),
            units=dict(
                flux="dimensionless (V . n / U)",
                transport="dimensionless (V . grad(Gamma) / U^2, Gamma[s]=m^2/s)",
                dcp="dimensionless (2 dP / (rho U^2))",
                constraint="n is the discretized wake normal +z, never "
                           "replaced by the freestream direction",
            ),
        ),
        elapsed_seconds=time.perf_counter() - started,
    )
    return report


def _jax_version():
    try:
        import jax
        return jax.__version__
    except Exception:  # pragma: no cover - the report must stay writable
        return None


def _print_summary(report):
    wake = report["wake_report"]["by_offset"]
    print(json.dumps(dict(
        schema=report["schema"],
        surface_sha256=report["surface_sha256"],
        geometry=report["geometry"],
        near=wake[-1]["near"], far=wake[-1]["far"], all=wake[-1]["all"],
        sensitivity=report["wake_report"]["sensitivity"],
        checks=report["checks"],
        elapsed_seconds=report["elapsed_seconds"],
    ), indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Read-only two-sided wake audit for a native half solve")
    parser.add_argument("--surface", required=True,
                        help="half surface.json (schema dpanel.half_planar_cut.v1)")
    parser.add_argument("--out", required=True, help="new report path")
    parser.add_argument("--max-half-panels", type=int, default=2000,
                        help="explicit guard for the dense full-oracle solve")
    parser.add_argument("--offsets", default="1e-2,1e-3,1e-4",
                        help="two-sided offsets as multiples of sqrt(area)")
    parser.add_argument("--chunk", type=int, default=512,
                        help="targets per kernel block")
    parser.add_argument("--steps", default="1e-3,1e-4",
                        help="central-difference steps as multiples of chord")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"Do not overwrite a previous wake audit: {out}")
    offsets = tuple(float(value) for value in args.offsets.split(",") if value)
    steps = tuple(float(value) for value in args.steps.split(",") if value)
    import jax
    jax.config.update("jax_enable_x64", True)
    report = audit(args.surface, max_half_panels=args.max_half_panels,
                   offsets=offsets, chunk=args.chunk, steps=steps)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
