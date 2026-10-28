"""Read-only two-sided total-flow residuals on a native half-domain wake sheet.

This driver closes the structural gap recorded in docs/pr26_audit_a/README.md.
The half-native solver never writes a full-surface run directory, so the legacy
panel/audit.py wake diagnostics (total-flow tangency, material circulation
transport and the two-sided pressure jump) cannot be evaluated on a half case.
Nothing here is production code: no solver, pressure reconstruction, wake shape
or default mode is modified, and no production module is imported for mutation.

Method
------
1. Solve the same frozen half fixture once with
   panel.half_native.solve_half_fields.
2. Mirror that single state by index maps: the left half carries the same
   mu/sigma under y -> -y, the wake doublet is unchanged and the wake
   circulation slope flips sign, exactly as panel.half_native does for its own
   image term. No full-model solve, no dense expansion matrix, no
   nearest-neighbour pairing. panel.verify_half_mirror.make_full_mirror_topology
   is used only by the explicit small-mesh --full-oracle cross-check.
3. Evaluate the analytic single/double-layer field with the numba kernel in
   core/triangle_kernel.py. JAX is used only to re-derive the same potentials
   as an independent cross-kernel check, never as the reported field, and the
   solver's tangential affine Cp reconstruction is never used.

Reported wake diagnostics, per sample point and area weighted:
  flux      = Vmean . n / U            (total-flow tangency, NOT Uinf . n)
  transport = Vmean . grad(Gamma) / U^2 (material circulation transport)
  dCp       = Cp(+) - Cp(-)             (two-sided static pressure jump)
with Vmean = (V(+) + V(-))/2 at +/- eps along the +z wake normal. Three
physical closures are reported next to the algebraic identity
dCp = -2 Vmean . (V(+) - V(-)) / U^2:
  potential jump   phi(+) - phi(-)  against the wake doublet strength
  normal jump      (V(+) - V(-)) . n
  tangential jump  (V(+) - V(-))_t  against +/- grad_s(Gamma)
The sign of the tangential relation is measured, never preset. The velocity
magnitude used for Cp is the full analytic field, so dCp here is not a
reconstructed Cp and is not required to match the solver's Cp_raw.

Streamwise bands are fixed fractions of x_gap / Cref measured from the
trailing edge, so an empty band is reported as null instead of being redefined
until it is non-empty. The older relative half-extent split is kept only as a
historical control.

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


def _mirror_arrays(half, station_dof):
    """O(N) index map from the half surface to its exact mirrored model.

    Same construction as panel.verify_half_mirror.make_full_mirror_topology, but
    the dense expansion matrix E is never formed: full face n+i is the mirror
    partner of face i, so a state is mirrored by duplicating it.
    """
    topology = half["topology"]
    x = np.asarray(half["vertices"], dtype=np.float64)
    y = np.asarray(half["station_y"], dtype=np.float64)
    dof = np.asarray(station_dof, dtype=np.int64)
    faces = np.asarray(topology.faces, dtype=np.int64)
    wake = np.asarray(topology.wake_faces, dtype=np.int64)
    te = np.asarray(topology.te_edges, dtype=np.int64)
    tol = 1e-10 * max(float(np.ptp(x, axis=0).max()), 1.0)
    if abs(y[0]) > tol or np.any(x[:, 1] < -tol):
        raise ValueError("Requires y>=0 with the central station at y=0")
    if dof.shape != y.shape or dof[-1] != -1:
        raise ValueError("Invalid half circulation station mapping")
    positive = np.flatnonzero(x[:, 1] > tol)
    reflect = np.arange(len(x), dtype=np.int64)
    reflect[positive] = len(x) + np.arange(len(positive), dtype=np.int64)
    return dict(
        vertices=np.vstack((x, x[positive] * REFLECT)),
        faces=np.vstack((faces, reflect[faces[:, ::-1]])),
        wake=np.vstack((wake, reflect[wake[:, ::-1]])),
        te=np.vstack((te, reflect[te[:, ::-1]])),
        positive=positive,
        half_panels=len(faces),
        half_wake=len(wake),
        half_gamma=int(topology.hat_map.shape[1]),
    )


def _triangles(vertices, faces):
    triangles = np.ascontiguousarray(vertices[faces])
    area_vector = np.cross(triangles[:, 1] - triangles[:, 0],
                           triangles[:, 2] - triangles[:, 0]) / 2
    area = np.linalg.norm(area_vector, axis=1)
    normals = area_vector / np.maximum(area[:, None], 1e-300)
    return triangles, normals, area, triangles.mean(axis=1)


def _frozen_halves(path, surface_sha256, half):
    """Read a previously saved half state; never re-solve the flow."""
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        if "surface_sha256" in keys:
            stored = str(np.asarray(data["surface_sha256"]).reshape(-1)[0])
            if stored != surface_sha256:
                raise ValueError("Frozen state surface SHA does not match "
                                 "--surface")
        missing = {"mu_half", "gamma_half", "sigma_half"} - keys
        if missing:
            raise ValueError(f"Frozen state is missing arrays: {sorted(missing)}")
        mu = np.asarray(data["mu_half"], dtype=np.float64)
        gamma = np.asarray(data["gamma_half"], dtype=np.float64)
        sigma = np.asarray(data["sigma_half"], dtype=np.float64)
    nf = len(half["topology"].faces)
    ng = int(half["topology"].hat_map.shape[1])
    if mu.shape != (nf,) or sigma.shape != (nf,) or gamma.shape != (ng,):
        raise ValueError("Frozen state shapes do not match this half surface")
    return mu, gamma, sigma


def _full_oracle_check(half, raw, mu, sigma, wake_mu, wake_slope):
    """Explicit small-mesh full solve; only a cross-check of the index map."""
    import jax

    from .differentiable import solve_fields
    from .verify_half_mirror import make_full_mirror_topology, mirrored_full_input

    plan = make_full_mirror_topology(half, raw["gamma_station_dof"])
    full_input = mirrored_full_input(half, plan)
    out = jax.jit(solve_fields)(**full_input)
    nf = len(half["topology"].faces)
    nw = len(half["topology"].wake_faces)
    gamma = np.asarray(out["gamma"], dtype=np.float64)
    doublet = np.asarray(out["wake_basis"], dtype=np.float64) @ gamma
    slope = np.asarray(out["wake_slope"], dtype=np.float64) @ gamma

    def gap(reference, candidate):
        reference = np.asarray(reference)
        candidate = np.asarray(candidate)
        delta = reference - candidate
        peak = max(float(np.max(np.abs(candidate))), 1e-300)
        return dict(max_abs=float(np.max(np.abs(delta))),
                    relative_to_peak=float(np.max(np.abs(delta)) / peak))

    return dict(performed=True, half_panels=nf,
                mu=gap(np.asarray(out["mu"], dtype=np.float64), mu),
                sigma=gap(np.asarray(out["sigma"], dtype=np.float64), sigma),
                wake_doublet=gap(doublet[:2 * nw], wake_mu),
                wake_slope=gap(slope[:2 * nw], wake_slope))


def build_mirror_state(surface_path, *, max_half_panels=2000,
                       full_oracle=False, full_oracle_panels=700,
                       frozen_state=None):
    """One frozen half state, mirrored by index; no full-model solve by default."""
    import jax
    import jax.numpy as jnp

    from . import half_native
    from .run_half_native import load_half_surface

    if not jax.config.x64_enabled:
        raise ValueError("Explicitly enable JAX float64 before auditing")
    raw = json.loads(Path(surface_path).read_text(encoding="utf-8"))
    half = load_half_surface(surface_path)
    sha = _sha256_file(surface_path)
    nf = len(half["topology"].faces)
    if nf > max_half_panels:
        raise MemoryError(
            "Half mesh exceeds the explicit half-solve guard; raise "
            "--max-half-panels deliberately")
    mirror = _mirror_arrays(half, raw["gamma_station_dof"])
    if frozen_state is None:
        solved = jax.jit(half_native.solve_half_fields)(**half)
        mu_half = np.asarray(solved["mu_half"], dtype=np.float64)
        sigma_half = np.asarray(solved["sigma_half"], dtype=np.float64)
        gamma_half = np.asarray(solved["gamma_half"], dtype=np.float64)
        residual = float(np.max(np.abs(np.asarray(
            solved["linear_residual_half"], dtype=np.float64))))
        origin = "fresh half solve"
    else:
        mu_half, gamma_half, sigma_half = _frozen_halves(frozen_state, sha,
                                                         half)
        residual = None
        origin = "reused frozen state (no flow solve)"
    # The wake basis and slope are pure geometry and are assembled here the same
    # way for a fresh and for a replayed state, so a replay reproduces the same
    # wake operators (and the same state hash) bit for bit.
    _, basis, slope = half_native._half_wake_te_geometry(
        jnp.asarray(half["vertices"]), jnp.asarray(half["station_y"]),
        half["topology"])[:3]
    doublet_pos = np.asarray(basis, dtype=np.float64) @ gamma_half
    slope_pos = np.asarray(slope, dtype=np.float64) @ gamma_half
    if frozen_state is None:
        operator_gap = max(
            float(np.max(np.abs(np.asarray(solved["wake_basis_half"],
                                           dtype=np.float64) @ gamma_half
                                - doublet_pos))),
            float(np.max(np.abs(np.asarray(solved["wake_slope_half"],
                                           dtype=np.float64) @ gamma_half
                                - slope_pos))))
    else:
        operator_gap = None
    triangles, normals, area, centers = _triangles(mirror["vertices"],
                                                   mirror["faces"])
    wake, wake_normals, wake_area, wake_centers = _triangles(
        mirror["vertices"], mirror["wake"])
    # The mirrored half is the same state under y -> -y: mu and sigma are
    # scalars and are copied, the doublet sheet value is copied, only the
    # spanwise circulation slope changes sign.
    mu = np.r_[mu_half, mu_half]
    sigma = np.r_[sigma_half, sigma_half]
    wake_mu = np.r_[doublet_pos, doublet_pos]
    wake_slope = np.r_[slope_pos, -slope_pos]
    half_wake = mirror["half_wake"]
    mirror_check = dict(
        map="index: full face n+i is the mirror partner of half face i",
        panel_normal_partner_max_abs=float(np.max(np.abs(
            normals[:nf] * REFLECT - normals[nf:]))),
        wake_normal_partner_max_abs=float(np.max(np.abs(
            wake_normals[:half_wake] * REFLECT - wake_normals[half_wake:]))),
        wake_centroid_partner_max_abs=float(np.max(np.abs(
            wake_centers[:half_wake] * REFLECT - wake_centers[half_wake:]))),
        dense_expansion_matrix="not formed",
        wake_operator_vs_solver_max_abs=operator_gap,
    )
    if full_oracle:
        if nf > full_oracle_panels:
            raise MemoryError(
                "Full oracle refused: half mesh above --full-oracle-panels; "
                "the default path never needs it")
        mirror_check["full_oracle"] = _full_oracle_check(
            half, raw, mu, sigma, wake_mu, wake_slope)
    flow = np.asarray(half["freestream"], dtype=np.float64)
    return dict(
        surface_path=str(Path(surface_path).resolve()),
        surface_sha256=sha,
        raw=raw,
        vertices=mirror["vertices"],
        faces=mirror["faces"],
        te_edges=mirror["te"],
        wake_node_ids=mirror["wake"],
        triangles=triangles,
        centers=centers,
        normals=normals,
        area=area,
        wake=wake,
        wake_centers=wake_centers,
        wake_area=wake_area,
        wake_normals=wake_normals,
        mu=mu,
        sigma=sigma,
        wake_mu=wake_mu,
        wake_slope=wake_slope,
        flow=flow,
        speed=float(np.linalg.norm(flow)),
        chord=float(np.asarray(half["reference_chord"])),
        half_panels=nf,
        half_wake=half_wake,
        half_gamma=mirror["half_gamma"],
        half_state=np.r_[mu_half, gamma_half],
        linear_residual=residual,
        state_origin=origin,
        frozen_state=None if frozen_state is None else str(
            Path(frozen_state).resolve()),
        full_oracle=bool(full_oracle),
        mirror_check=mirror_check,
    )


def _subset(context, include_image):
    """Full mirrored discretization, or its positive half only (diagnostic)."""
    if include_image:
        return dict(panels=context["triangles"], normals=context["normals"],
                    mu=context["mu"], sigma=context["sigma"],
                    wake=context["wake"], wake_normals=context["wake_normals"],
                    wake_centers=context["wake_centers"],
                    wake_mu=context["wake_mu"],
                    wake_slope=context["wake_slope"])
    nf, nw = context["half_panels"], context["half_wake"]
    return dict(panels=context["triangles"][:nf], normals=context["normals"][:nf],
                mu=context["mu"][:nf], sigma=context["sigma"][:nf],
                wake=context["wake"][:nw],
                wake_normals=context["wake_normals"][:nw],
                wake_centers=context["wake_centers"][:nw],
                wake_mu=context["wake_mu"][:nw],
                wake_slope=context["wake_slope"][:nw])


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
    wake_mu = jnp.asarray(context["wake_mu"])
    wake_slope = jnp.asarray(context["wake_slope"])
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


WAKE_BANDS = ((0.0, 0.05), (0.05, 0.1), (0.1, 0.25),
              (0.25, 0.5), (0.5, 1.0), (1.0, None))
WAKE_BAND_NAMES = tuple(
    f"[{low:g},{high:g})" if high is not None else f"[{low:g},inf)"
    for low, high in WAKE_BANDS)


def wake_labels(context):
    """Streamwise bands tied to the trailing edge, plus the historical split.

    x_gap is the wake-triangle centroid x minus the x of the trailing-edge edge
    midpoint at the same spanwise station (the nearest edge midpoint in y).

    This per-triangle view is kept for the historical control split and for the
    first-row flag. The reported streamwise bands are per sample (see
    sample_bands): the wake strips are long triangles, so a centroid band would
    place a sample that sits right behind the trailing edge in a far band.
    """
    te = np.asarray(context["te_edges"])
    te_mid = context["vertices"][te].mean(axis=1)
    centers = context["wake_centers"]
    index = np.array([int(np.argmin(np.abs(te_mid[:, 1] - center[1])))
                      for center in centers], dtype=np.int64)
    gap = centers[:, 0] - te_mid[index, 0]
    over_chord = gap / context["chord"]
    extent = max(float(np.max(np.abs(gap))), 1e-300)
    fraction = gap / extent
    te_nodes = np.unique(te)
    first_row = np.any(np.isin(context["wake_node_ids"], te_nodes), axis=1)
    return dict(
        x_gap=gap, gap_over_chord=over_chord, gap_fraction=fraction,
        relative_near=fraction <= 0.5,
        near_absolute=gap <= 0.5 * context["chord"],
        first_row=first_row, te_index=index, te_mid=te_mid,
        definitions=dict(
            gap="wake triangle centroid x minus the trailing-edge edge "
                "midpoint x at the nearest spanwise station",
            first_row="wake triangles sharing at least one node with a "
                      "declared trailing-edge edge",
            historical="relative_near = centroid x_gap <= 0.5*max|x_gap| "
                       "(control only, centroid based)",
        ),
    )


def sample_bands(samples, context):
    """Per-sample streamwise bands, each sample measured from the trailing edge.

    The sample's own x distance to the nearest trailing-edge midpoint decides its
    band, and its parent triangle's area is shared by the four samples, so the
    bands partition the sheet without any band being widened to avoid emptiness.
    """
    flat = np.asarray(samples, dtype=np.float64).reshape(-1, 3)
    te_mid = context["vertices"][context["te_edges"]].mean(axis=1)
    nearest = np.abs(te_mid[None, :, 1] - flat[:, 1:2]).argmin(axis=1)
    gap = flat[:, 0] - te_mid[nearest, 0]
    over_chord = gap / context["chord"]
    masks = []
    for index, (low, high) in enumerate(WAKE_BANDS):
        if index == 0:
            mask = over_chord < high
        elif high is None:
            mask = over_chord >= low
        else:
            mask = (over_chord >= low) & (over_chord < high)
        masks.append(mask)
    return dict(
        gap=gap, gap_over_chord=over_chord, te_index=nearest,
        below_range=int(np.count_nonzero(over_chord < 0.0)),
        masks=masks, names=WAKE_BAND_NAMES,
        definitions=dict(
            gap="sample x minus the trailing-edge midpoint x at the nearest "
                "spanwise station",
            bands="fixed x_gap / Cref intervals evaluated per sample; the "
                  "first band also absorbs any sample with x_gap < 0",
            empty="a band with no samples is reported as null; the band edges "
                  "are never moved"),
    )


def two_sided_metrics(velocity_plus, velocity_minus, potential_plus,
                      potential_minus, normal, speed):
    mean = 0.5 * (velocity_plus + velocity_minus)
    difference = velocity_plus - velocity_minus
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-300)
    normal_part = np.sum(difference * normal, axis=1)
    tangential = difference - normal_part[:, None] * normal
    return dict(
        mean=mean,
        difference=difference,
        normal=normal,
        flux=np.sum(mean * normal, axis=1) / speed,
        flux_plus=np.sum(velocity_plus * normal, axis=1) / speed,
        flux_minus=np.sum(velocity_minus * normal, axis=1) / speed,
        cp_plus=1 - np.sum(velocity_plus ** 2, axis=1) / speed ** 2,
        cp_minus=1 - np.sum(velocity_minus ** 2, axis=1) / speed ** 2,
        dcp=(np.sum(velocity_minus ** 2, axis=1)
             - np.sum(velocity_plus ** 2, axis=1)) / speed ** 2,
        identity=-np.sum(mean * difference, axis=1) * 2 / speed ** 2,
        potential_jump=potential_plus - potential_minus,
        normal_jump=normal_part / speed,
        tangential_jump=tangential / speed,
    )


def _weighted_rms(values, weights):
    return float(np.sqrt(np.average(np.asarray(values) ** 2,
                                    weights=np.asarray(weights))))


def _sample_slope(context, include_image, per_panel):
    slope = context["wake_slope"]
    if not include_image:
        slope = slope[:context["half_wake"]]
    return np.repeat(slope, per_panel)


def _sample_doublet(context, include_image, per_panel):
    doublet = context["wake_mu"]
    if not include_image:
        doublet = doublet[:context["half_wake"]]
    return np.repeat(doublet, per_panel)


def _wake_transport(mean, context, include_image, per_panel):
    """Vmean . grad(Gamma) / U^2, repeated from panel to sample order."""
    return (mean[:, 1] * _sample_slope(context, include_image, per_panel)
            / context["speed"] ** 2)


def summarize(metrics, context, weights, mask, include_image, label, per_panel):
    if not np.any(mask):
        return None
    speed = context["speed"]
    transport = _wake_transport(metrics["mean"], context, include_image,
                                per_panel)
    slope = _sample_slope(context, include_image, per_panel)
    doublet = _sample_doublet(context, include_image, per_panel)
    tangential = metrics["tangential_jump"] * speed
    gradient = np.zeros_like(tangential)
    gradient[:, 1] = slope
    plus = np.linalg.norm(tangential + gradient, axis=1) / speed
    minus = np.linalg.norm(tangential - gradient, axis=1) / speed
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
        potential_jump_rms=_weighted_rms(metrics["potential_jump"][mask],
                                         weights[mask]),
        doublet_rms=_weighted_rms(doublet[mask], weights[mask]),
        normal_jump_rms=_weighted_rms(metrics["normal_jump"][mask],
                                      weights[mask]),
        tangential_jump_rms=_weighted_rms(
            np.linalg.norm(tangential, axis=1)[mask] / speed, weights[mask]),
        tangential_plus_slope_rms=_weighted_rms(plus[mask], weights[mask]),
        tangential_minus_slope_rms=_weighted_rms(minus[mask], weights[mask]),
        sign_reference="tangential_minus_slope" if (
            _weighted_rms(minus[mask], weights[mask])
            <= _weighted_rms(plus[mask], weights[mask]))
            else "tangential_plus_slope",
        identity_abs_max=float(np.max(np.abs(
            metrics["dcp"][mask] - metrics["identity"][mask]))),
    )


def _jump_closure(metrics, weights, slope, doublet, speed):
    """Physical jump closures; the tangential sign is measured, not preset."""
    tangential = metrics["tangential_jump"] * speed
    gradient = np.zeros_like(tangential)
    gradient[:, 1] = slope
    plus = np.linalg.norm(tangential + gradient, axis=1) / speed
    minus = np.linalg.norm(tangential - gradient, axis=1) / speed
    jump = metrics["potential_jump"]
    plus_rms = _weighted_rms(plus, weights)
    minus_rms = _weighted_rms(minus, weights)
    return dict(
        potential_jump_rms=_weighted_rms(jump, weights),
        doublet_rms=_weighted_rms(doublet, weights),
        potential_jump_minus_doublet_rms=_weighted_rms(jump - doublet, weights),
        potential_jump_plus_doublet_rms=_weighted_rms(jump + doublet, weights),
        normal_jump_rms=_weighted_rms(metrics["normal_jump"], weights),
        tangential_jump_rms=_weighted_rms(
            np.linalg.norm(tangential, axis=1) / speed, weights),
        tangential_plus_slope_rms=plus_rms,
        tangential_minus_slope_rms=minus_rms,
        smallest_residual=("tangential_minus_slope" if minus_rms <= plus_rms
                           else "tangential_plus_slope"),
        note="the smaller residual names the measured jump convention; no "
             "sign is preset and transport is not dismissed by its size",
    )


def _sensitivity(by_offset, offsets):
    ratios = [offsets[i] / offsets[i + 1] for i in range(len(offsets) - 1)]
    flux = [block["all"]["flux_rms"] for block in by_offset]
    dcp = [block["all"]["dcp_rms"] for block in by_offset]

    def successive(values):
        return [values[i] - values[i + 1] for i in range(len(values) - 1)]

    def order(values):
        delta = successive(values)
        out = []
        for i in range(len(delta) - 1):
            small, large = abs(delta[i + 1]), abs(delta[i])
            if small <= 0 or large <= 0:
                out.append(None)
            else:
                out.append(float(np.log(large / small) / np.log(ratios[i])))
        return out

    return dict(
        offsets=[float(value) for value in offsets],
        offset_ratios=[float(value) for value in ratios],
        flux_rms=flux,
        dcp_rms=dcp,
        transport_rms=[block["all"]["transport_rms"] for block in by_offset],
        flux_rms_successive=successive(flux),
        dcp_rms_successive=successive(dcp),
        flux_rms_estimated_order=order(flux),
        dcp_rms_estimated_order=order(dcp),
        note="orders are empirical for this mixed edge/corner sampling; no "
             "universal order is guaranteed and none is assumed",
    )


def audit_wake(context, *, offsets=DEFAULT_OFFSETS, chunk=512,
               include_image=True):
    samples = wake_samples(context, include_image=include_image)
    panels, per_panel = samples.shape[0], samples.shape[1]
    labels = wake_labels(context)
    # Bands are decided per sample, not per wake-triangle centroid: the strips
    # are long triangles, so a centroid band would file a sample that sits right
    # behind the trailing edge under a far band.
    sample = sample_bands(samples, context)
    if not include_image:
        labels = dict(
            gap_over_chord=labels["gap_over_chord"][:panels],
            relative_near=labels["relative_near"][:panels],
            near_absolute=labels["near_absolute"][:panels],
            first_row=labels["first_row"][:panels],
            te_index=labels["te_index"][:panels],
            definitions=labels["definitions"])
    area = (context["wake_area"] if include_image
            else context["wake_area"][:context["half_wake"]])
    weights = np.repeat(area / per_panel, per_panel)
    normal = (context["wake_normals"] if include_image
              else context["wake_normals"][:context["half_wake"]])
    normal = np.repeat(normal, per_panel, axis=0)
    slope = _sample_slope(context, include_image, per_panel)
    doublet = _sample_doublet(context, include_image, per_panel)

    def repeated(mask):
        return np.repeat(np.asarray(mask, dtype=bool), per_panel)

    by_offset, stored = [], None
    for factor in offsets:
        offset = (factor * np.sqrt(area))[:, None] * WAKE_NORMAL[None, :]
        plus = np.empty((panels, per_panel, 3))
        minus = np.empty((panels, per_panel, 3))
        plus_potential = np.empty((panels, per_panel))
        minus_potential = np.empty((panels, per_panel))
        for s in range(per_panel):
            plus[:, s, :], plus_potential[:, s] = evaluate_field(
                samples[:, s, :] + offset, context, include_image=include_image,
                chunk=chunk)
            minus[:, s, :], minus_potential[:, s] = evaluate_field(
                samples[:, s, :] - offset, context, include_image=include_image,
                chunk=chunk)
        metrics = two_sided_metrics(
            plus.reshape(-1, 3), minus.reshape(-1, 3),
            plus_potential.reshape(-1), minus_potential.reshape(-1),
            normal, context["speed"])
        summary = dict(
            all=summarize(metrics, context, weights,
                          np.ones(panels * per_panel, dtype=bool),
                          include_image, "all", per_panel),
            bands=[summarize(metrics, context, weights, mask,
                             include_image, name, per_panel)
                   for name, mask in zip(sample["names"], sample["masks"])],
            first_row=summarize(metrics, context, weights,
                                repeated(labels["first_row"]), include_image,
                                "first wake layer adjacent to the trailing "
                                "edge", per_panel),
            near_absolute=summarize(
                metrics, context, weights,
                sample["gap"] <= 0.5 * context["chord"], include_image,
                "sample x_gap <= 0.5 * Cref", per_panel),
            historical_near=summarize(metrics, context, weights,
                                      repeated(labels["relative_near"]),
                                      include_image,
                                      "historical control: first half of the "
                                      "truncated wake extent", per_panel),
            historical_far=summarize(metrics, context, weights,
                                     repeated(~labels["relative_near"]),
                                     include_image,
                                     "historical control: second half of the "
                                     "truncated wake extent", per_panel),
            jump_closure=_jump_closure(metrics, weights, slope, doublet,
                                       context["speed"]),
        )
        block = dict(offset_over_sqrt_area=factor, **summary)
        by_offset.append(block)
        if stored is None:
            stored = dict(metrics=metrics, plus=plus, minus=minus,
                          weights=weights, normal=normal,
                          labels=labels, area=area)
    return dict(by_offset=by_offset,
                sensitivity=_sensitivity(by_offset, offsets),
                band_rule=dict(
                    bands=sample["definitions"],
                    below_range=sample["below_range"],
                    names=list(sample["names"]),
                    gap_summary=dict(
                        minimum_over_chord=float(np.min(sample["gap_over_chord"])),
                        p10_over_chord=float(np.percentile(sample["gap_over_chord"], 10)),
                        median_over_chord=float(np.median(sample["gap_over_chord"])),
                        maximum_over_chord=float(np.max(sample["gap_over_chord"])),
                        note="the smallest reachable distance is set by the "
                             "wake triangle size, because samples are interior "
                             "points of their triangle; a band below the "
                             "smallest reachable gap is empty by resolution, "
                             "not because the flux there is zero")),
                stored=stored,
                definitions=dict(labels["definitions"],
                                 **sample["definitions"]))


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
    slope = context["wake_slope"]
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
          chunk=512, steps=(1e-3, 1e-4), full_oracle=False,
          full_oracle_panels=700, frozen_state=None):
    started = time.perf_counter()
    context = build_mirror_state(
        surface_path, max_half_panels=max_half_panels,
        full_oracle=full_oracle, full_oracle_panels=full_oracle_panels,
        frozen_state=frozen_state)
    wake = audit_wake(context, offsets=offsets, chunk=chunk)
    half_wake = audit_wake(context, offsets=offsets[:1], chunk=chunk,
                           include_image=False)
    stored = wake["stored"]
    mirror_oracle = dict(context["mirror_check"])
    mirror_oracle.setdefault("full_oracle", dict(
        performed=False,
        note="the default path never solves or imports the full model; pass "
             "--full-oracle on a small mesh for the explicit cross-check"))
    report = dict(
        schema=SCHEMA,
        status="independently_audited",
        physical_accuracy_accepted=False,
        source_surface=context["surface_path"],
        surface_sha256=context["surface_sha256"],
        state_sha256=_sha256_arrays(context["mu"], context["wake_mu"],
                                    context["wake_slope"], context["sigma"]),
        code_sha256=_sha256_file(__file__),
        environment=dict(python=platform.python_version(),
                         numpy=np.__version__, jax=_jax_version()),
        geometry=dict(half_panels=context["half_panels"],
                      half_wake=context["half_wake"],
                      half_gamma=context["half_gamma"],
                      full_panels=len(context["triangles"]),
                      full_wake=len(context["wake"])),
        state_origin=context["state_origin"],
        frozen_state=context["frozen_state"],
        linear_residual=context["linear_residual"],
        reference=dict(freestream=context["flow"].tolist(),
                       speed=context["speed"], chord=context["chord"]),
        mirror_oracle=mirror_oracle,
        wake_report=dict(
            by_offset=wake["by_offset"],
            sensitivity=wake["sensitivity"],
            band_rule=wake["band_rule"],
            definitions=wake["definitions"],
            sampling="4 points per wake triangle (centroid plus three "
                     "interior barycentric points), area weighted, both "
                     "mirrored halves; every sample is banded by its own "
                     "x_gap / Cref and the smallest reachable gap is set by "
                     "the size of its wake triangle",
        ),
        positive_half_only=dict(
            warning="diagnostic only: evaluated exactly to show what the "
                    "missing image does, never used for a reported residual",
            offset_over_sqrt_area=half_wake["by_offset"][0][
                "offset_over_sqrt_area"],
            all=half_wake["by_offset"][0]["all"],
            bands=half_wake["by_offset"][0]["bands"],
            first_row=half_wake["by_offset"][0]["first_row"],
        ),
        implementation_checks=dict(
            velocity_equals_potential_gradient=check_potential_gradient(
                context, steps=steps, chunk=chunk),
            potential_numba_vs_jax=check_potential_kernel(context, chunk=chunk),
            mirror_map=context["mirror_check"],
            image_contribution=check_image_contribution(context, stored,
                                                        chunk=chunk),
        ),
        physical_residuals=dict(
            wall_tangency=check_wall_tangency(context, chunk=chunk),
            mirror_equivariance=check_mirror_equivariance(context, stored),
            units=dict(
                flux="dimensionless (V . n / U)",
                transport="dimensionless (V . grad(Gamma) / U^2, "
                          "Gamma[s] = m^2/s)",
                dcp="dimensionless (2 dP / (rho U^2))",
                potential_jump="m^2/s; doublet strength has the same unit",
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
    wake = report["wake_report"]["by_offset"][-1]
    print(json.dumps(dict(
        schema=report["schema"],
        surface_sha256=report["surface_sha256"],
        state_sha256=report["state_sha256"],
        geometry=report["geometry"],
        all=wake["all"],
        bands=wake["bands"],
        first_row=wake["first_row"],
        jump_closure=wake["jump_closure"],
        sensitivity=report["wake_report"]["sensitivity"],
        implementation_checks=report["implementation_checks"],
        physical_residuals=report["physical_residuals"],
        elapsed_seconds=report["elapsed_seconds"],
    ), indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Read-only two-sided wake audit for a native half solve")
    parser.add_argument("--surface", required=True,
                        help="half surface.json (schema dpanel.half_planar_cut.v1)")
    parser.add_argument("--out", required=True, help="new report path")
    parser.add_argument("--max-half-panels", type=int, default=2000,
                        help="explicit guard for the single half solve")
    parser.add_argument("--full-oracle", action="store_true",
                        help="also solve the full mirrored model once; small "
                             "meshes only, a cross-check of the index map")
    parser.add_argument("--full-oracle-panels", type=int, default=700,
                        help="refuse --full-oracle above this half-panel count")
    parser.add_argument("--frozen-state", default=None,
                        help="frozen_state.npz from a previous half solve; the "
                             "flow is not solved again")
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
                   offsets=offsets, chunk=args.chunk, steps=steps,
                   full_oracle=args.full_oracle,
                   full_oracle_panels=args.full_oracle_panels,
                   frozen_state=args.frozen_state)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
