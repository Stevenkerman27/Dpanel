"""Experimental fixed-topology, vertex-input JAX panel evaluator.

Equations and unweighted pressure reconstruction are extracted from solve.py.
This module has no CLI/import-time execution or global dtype/device changes.
Topology is frozen; geometry, wake/TE basis, references and loads are dynamic.

The initial topology is bootstrapped from an already validated legacy run.
CAD Booleans, remeshing, changing contact graph and nonplanar wakes are NOT
made differentiable. A correct VJP does not establish aerodynamic accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np

from .kernel import triangle_potential


class PanelTopology(NamedTuple):
    """Only index maps / basis connectivity; no cached metric quantities."""
    faces: object
    wake_faces: object
    te_edges: object
    upper_te: object
    lower_te: object
    hat_map: object
    wake_interval: object
    te_interval: object
    neighbors: object
    component_ids: object


def linear_solve(matrix, rhs):
    """Dense LU reused for implicit forward and transposed solves.

    Differentiation acts on matvec's matrix, NOT on pivots/LU operations.
    This is a dense solver, not a matrix-free acceleration claim.
    """
    factor = jla.lu_factor(jax.lax.stop_gradient(matrix))
    return jax.lax.custom_linear_solve(
        lambda x: matrix @ x, rhs,
        solve=lambda _, b: jla.lu_solve(factor, b),
        transpose_solve=lambda _, b: jla.lu_solve(factor, b, trans=1),
        symmetric=False,
    )


def solve_fields(vertices, station_y, topology, freestream, density,
                 reference_area, reference_chord, reference_point):
    """Pure JAX: continuous arrays -> raw Cp, SI loads and state diagnostics.

    vertices: all wall AND wake nodes in physical length units, [V,3].
    station_y: current spanwise circulation stations, same units, [S].
    freestream: geometry-axis velocity [3]; density: mass/length^3.
    No pressure clipping, normal-velocity projection used as validation,
    regularization or finite-difference backward is introduced.

    Call validate_configuration outside tracing at actual optimization iterates.
    station_y must move consistently with the vertices: arbitrary independent
    perturbations that move a wake triangle outside its strip are not legal.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Enable jax_enable_x64 in the caller for this research evaluator")
    vertices, station_y = jnp.asarray(vertices), jnp.asarray(station_y)
    flow = jnp.asarray(freestream)
    t = topology
    tri = vertices[t.faces]
    av = jnp.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) / 2
    area = jnp.linalg.norm(av, axis=1)
    normal = av / area[:, None]
    center = tri.mean(axis=1)
    sigma = -(normal @ flow)
    single, doublet = triangle_potential(center, tri, interior_diagonal=True)
    wake = vertices[t.wake_faces]
    dw, my = triangle_potential(center, wake, source=False, linear_y=True)

    # Recompute from current geometry; never freeze baseline metric basis values.
    interval = t.wake_interval
    dy = station_y[interval + 1] - station_y[interval]
    fraction = (wake.mean(axis=1)[:, 1] - station_y[interval]) / dy
    basis = (1 - fraction[:, None]) * t.hat_map[interval] + fraction[:, None] * t.hat_map[interval + 1]
    slope = (t.hat_map[interval + 1] - t.hat_map[interval]) / dy[:, None]
    wake_operator = dw @ basis + my @ slope

    gauss = jnp.asarray([.5 - jnp.sqrt(3.) / 6, .5 + jnp.sqrt(3.) / 6])
    ends = vertices[t.te_edges]
    te_y = ends[:, 0, 1, None] * (1 - gauss) + ends[:, 1, 1, None] * gauss
    ti = t.te_interval
    tf = (te_y - station_y[ti]) / (station_y[ti + 1] - station_y[ti])
    te_basis = (1 - tf[..., None]) * t.hat_map[ti] + tf[..., None] * t.hat_map[ti + 1]
    length = jnp.linalg.norm(ends[:, 1] - ends[:, 0], axis=1)
    weights = jnp.broadcast_to(length[:, None] / 2, te_y.shape)
    mass = jnp.einsum("eq,eqi,eqj->ij", weights, te_basis, te_basis)
    integrals = jnp.einsum("eq,eqi->ei", weights, te_basis)
    measure = integrals.sum(axis=0)
    ng, nf = t.hat_map.shape[1], tri.shape[0]
    incidence = jnp.zeros((len(t.te_edges), nf), dtype=vertices.dtype)
    rows = jnp.arange(len(t.te_edges))
    incidence = incidence.at[rows, t.upper_te].add(1.)
    incidence = incidence.at[rows, t.lower_te].add(-1.)
    tested = (integrals.T @ incidence) / measure[:, None]
    mass = mass / measure[:, None]
    matrix = jnp.block([[doublet, wake_operator], [-tested, mass]])
    rhs = jnp.concatenate((single @ sigma, jnp.zeros(ng, dtype=vertices.dtype)))
    state = linear_solve(matrix, rhs)
    mu, gamma = state[:nf], state[nf:]

    # Same 2-column unweighted fit as legacy CLI; candidates from PR #10 rejected.
    l = tri[:, 1] - tri[:, 0]
    l = l / jnp.linalg.norm(l, axis=1)[:, None]
    m = jnp.cross(normal, l)
    delta = center[t.neighbors] - center[:, None]
    dl, dm = jnp.sum(delta * l[:, None], axis=2), jnp.sum(delta * m[:, None], axis=2)
    dmu = mu[t.neighbors] - mu[:, None]
    ll, mm, lm = (jnp.sum(a, axis=1) for a in (dl*dl, dm*dm, dl*dm))
    bl, bm = jnp.sum(dl*dmu, axis=1), jnp.sum(dm*dmu, axis=1)
    determinant = ll*mm - lm*lm
    gl, gm = (bl*mm - bm*lm)/determinant, (bm*ll - bl*lm)/determinant
    velocity = (l @ flow + gl)[:, None]*l + (m @ flow + gm)[:, None]*m
    speed2 = jnp.dot(flow, flow)
    cp = 1 - jnp.sum(velocity*velocity, axis=1)/speed2
    q = .5*density*speed2
    forces = -q*cp[:, None]*av
    moments = jnp.cross(center - reference_point, forces)
    force, moment = forces.sum(axis=0), moments.sum(axis=0)
    drag_axis = flow / jnp.sqrt(speed2)
    lift_axis = jnp.cross(drag_axis, jnp.asarray([0., 1., 0.]))
    lift_axis = lift_axis / jnp.linalg.norm(lift_axis)
    component_force = jnp.stack([jnp.sum(jnp.where((t.component_ids == k)[:, None], forces, 0.), axis=0)
                                 for k in (10, 20, 30)])
    component_moment = jnp.stack([jnp.sum(jnp.where((t.component_ids == k)[:, None], moments, 0.), axis=0)
                                  for k in (10, 20, 30)])
    return dict(
        Cp_raw=cp, panel_forces=forces, panel_moments=moments,
        force=force, moment=moment, component_forces=component_force,
        component_moments=component_moment, panel_center=center,
        panel_area=area, panel_normal=normal, surface_velocity=velocity,
        CL=force @ lift_axis/(q*reference_area),
        CD_pressure=force @ drag_axis/(q*reference_area),
        CM=moment/(q*reference_area*reference_chord),
        mu=mu, gamma=gamma, sigma=sigma, matrix=matrix, rhs=rhs,
        wake_basis=basis, wake_slope=slope, te_basis=te_basis,
        linear_residual=matrix @ state - rhs,
        tangent_determinant=determinant,
    )


def validate_configuration(vertices, station_y, topology):
    """Host-side check of the fixed planar-strip geometry before tracing.

    This is not a substitute for the legacy CAD/wall-intersection validator.
    After large deformations, rerun the independent geometric audit. There is
    deliberately no automatic remeshing, topology change or silent projection.
    """
    x, y = np.asarray(vertices), np.asarray(station_y)
    t = topology
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(y) <= 0):
        raise ValueError("Nonfinite geometry or unordered circulation stations")
    tri, wake = x[np.asarray(t.faces)], x[np.asarray(t.wake_faces)]
    if np.any(np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1) <= 0):
        raise ValueError("Degenerate wall panel")
    scale = max(float(np.ptp(x, axis=0).max()), np.finfo(float).tiny)
    tol = 1e-10*scale
    if np.ptp(wake[..., 2]) > tol or np.any(np.cross(wake[:, 1]-wake[:, 0], wake[:, 2]-wake[:, 0])[:, 2] <= 0):
        raise ValueError("Only a consistently +z-oriented planar wake is supported")
    wi, ti = np.asarray(t.wake_interval), np.asarray(t.te_interval)[:, 0]
    for coord, ix in ((wake[..., 1], wi), (x[np.asarray(t.te_edges), 1], ti)):
        if np.any(coord.min(axis=1) < y[ix]-tol) or np.any(coord.max(axis=1) > y[ix+1]+tol):
            raise ValueError("Geometry left frozen wake/TE strips; update geometry map, not just vertices")
    centers = tri.mean(axis=1)
    l = tri[:, 1]-tri[:, 0]; l /= np.linalg.norm(l, axis=1)[:, None]
    normal = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    m = np.cross(normal, l)
    delta = centers[np.asarray(t.neighbors)]-centers[:, None]
    design = np.stack((np.sum(delta*l[:, None], axis=2), np.sum(delta*m[:, None], axis=2)), axis=-1)
    singular = np.linalg.svd(design, compute_uv=False)
    if np.any(singular[:, 0] <= 0) or np.any(singular[:, -1]/singular[:, 0] < 1e-6):
        raise ValueError("Deformation made pressure stencils rank deficient")


def load_prepared_run(run_dir):
    """Bootstrap topology from a previously accepted CLI run; no new solve.

    Return (input dictionary, immutable source metadata). Only the selected
    fields are loaded. Inputs are physical coordinates and reference quantities;
    their numerical values are not closed-over constants in solve_fields.
    """
    root = Path(run_dir)
    result = json.loads((root/'result.json').read_text(encoding='utf-8'))
    surface = json.loads((root/'surface_input.json').read_text(encoding='utf-8'))
    if surface.get('schema') != 'dpanel.planar_cut.v1' or result.get('input_validation', {}).get('status') != 'input_accepted':
        raise ValueError('Require a geometry-validated panel/solve.py run')
    with np.load(root/'fields.npz', allow_pickle=False) as f:
        current = np.array(f['coordinates'])
        neighbors = np.array(f['neighbor_indices'])
    chord = float(surface['reference_chord'])
    origin = np.asarray(surface['origin'])
    vertices = current*chord + origin
    span = float(result['parameters']['span_factor'])
    station_y = (np.asarray(surface['gamma_stations'])-origin[1])*span+origin[1]
    dof = np.asarray(surface['gamma_station_dof'], dtype=np.int64)
    hat = np.zeros((len(dof), int(dof.max())+1))
    hat[np.flatnonzero(dof >= 0), dof[dof >= 0]] = 1
    faces, wf, edges = (np.asarray(surface[k], dtype=np.int64) for k in ('faces','wake_faces','trailing_edges'))
    wi = np.minimum(np.searchsorted(station_y, vertices[wf].mean(axis=1)[:, 1], side='right')-1, len(station_y)-2)
    gauss = np.array([.5-np.sqrt(3)/6, .5+np.sqrt(3)/6])
    te_y = vertices[edges[:, 0], 1, None]*(1-gauss)+vertices[edges[:, 1], 1, None]*gauss
    ti = np.minimum(np.searchsorted(station_y, te_y, side='right')-1, len(station_y)-2)
    if np.any(wi < 0) or np.any(ti < 0) or np.any(ti[:, 0] != ti[:, 1]):
        raise ValueError('Saved geometry incompatible with circulation strip map')
    topology = PanelTopology(*(jnp.asarray(a) for a in (
        faces,wf,edges,np.asarray(surface['upper_te_faces']),np.asarray(surface['lower_te_faces']),
        hat,wi,ti,neighbors,np.asarray(surface['component_ids']),
    )))
    validate_configuration(vertices, station_y, topology)
    alpha = np.deg2rad(result['parameters']['alpha_deg'])
    inputs = dict(vertices=jnp.asarray(vertices), station_y=jnp.asarray(station_y), topology=topology,
                  freestream=jnp.asarray([np.cos(alpha),0.,np.sin(alpha)]), density=jnp.asarray(2.),
                  reference_area=jnp.asarray(surface['reference_area']), reference_chord=jnp.asarray(chord),
                  reference_point=jnp.asarray(surface['moment_reference']))
    return inputs, dict(result=result, surface=surface)
