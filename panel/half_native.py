"""Right-half-only lifting panels with exact image source, doublet and wake.

The input consists solely of y>=0 wetted wall and fluid-side wake triangles,
their true TE topology and nonnegative circulation stations. Image geometry
is created transiently in the JAX integration kernel, never as independent
physical unknowns. This module initially provides a small dense reference
and a streamed A/A.T for later GMRES; it does not differentiate CAD Booleans
or claim that the low-order pressure model meets physical accuracy gates.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .kernel import triangle_potential
from .differentiable import linear_solve


class HalfTopology(NamedTuple):
    faces: object
    wake_faces: object
    te_edges: object
    upper_te: object
    lower_te: object
    hat_map: object
    wake_interval: object
    te_interval: object
    neighbor_ghost_ids: object
    component_ids: object


_REFLECT = jnp.asarray([1., -1., 1.])
_AXIAL_REFLECT = jnp.asarray([-1., 1., -1.])


def _mirror_triangles(triangles):
    """Reflect positions and reverse vertex winding to retain outward normal."""
    return triangles[:, (0, 2, 1), :] * _REFLECT


def _pressure_neighbors_from_half(vertices, faces, te_edges, rings):
    """Virtual O(N) connectivity for pressure stencils, not mirrored DOFs.

    Virtual left triangles share the exact y=0 seam IDs. Only right-side face
    neighbor IDs [0,2*N) are returned, including virtual left *ghost* IDs.
    """
    if rings < 1:
        raise ValueError("Need at least one cut-aware adjacency ring")
    x = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int32)
    ys = x[:, 1]
    tol = 1e-10 * max(float(np.ptp(x, axis=0).max()), 1.)
    if np.any(ys < -tol):
        raise ValueError("Half vertices must lie at y >= 0")
    other = np.flatnonzero(ys > tol)
    mirror = np.arange(len(x))
    mirror[other] = len(x) + np.arange(len(other))
    virtual = np.concatenate((f, mirror[f[:, ::-1]]))
    edges, winding = {}, {}
    for i, face in enumerate(virtual):
        for a, b in zip(face, np.roll(face, -1)):
            a, b = int(a), int(b)
            key = tuple(sorted((a, b)))
            edges.setdefault(key, []).append(i)
            winding.setdefault(key, []).append((a, b))
    if any(len(items) != 2 for items in edges.values()):
        raise ValueError("Mirrored physical wall is not a closed 2-manifold")
    if any(a[0] != a[1][::-1] for a in winding.values()):
        raise ValueError("Mirrored wall faces have inconsistent winding")
    full_xyz = np.concatenate((x, x[other] * [1., -1., 1.]))
    tri = full_xyz[virtual]
    av = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])/2
    if np.sum(np.einsum("ij,ij->i", tri.mean(axis=1), av))/3 <= 0:
        raise ValueError("Public wall must have positive outward volume")
    cuts = {tuple(sorted(map(int, e))) for e in te_edges}
    cuts.update(tuple(sorted(map(int, mirror[e]))) for e in te_edges)
    adjacent = [set() for _ in virtual]
    for edge, (i, j) in edges.items():
        if edge not in cuts:
            adjacent[i].add(j)
            adjacent[j].add(i)
    stencils = []
    for i in range(len(f)):
        seen = {i}
        frontier = {i}
        for _ in range(rings):
            frontier = set().union(*(adjacent[j] for j in frontier)) - seen
            seen |= frontier
        stencils.append(sorted(seen-{i}))
    if any(len(s) < 2 for s in stencils):
        raise ValueError("Insufficient pressure stencil after TE cut")
    width = max(map(len, stencils))
    # Repeated self index pads the row with zero differences; it is not a
    # source of independent pressure neighbors.
    return np.asarray([s+[i]*(width-len(s)) for i, s in enumerate(stencils)],
                      dtype=np.int32)


def prepare_half_topology(vertices, faces, wake_faces, te_edges, upper_te,
                          lower_te, station_y, station_dof, component_ids,
                          *, stencil_rings=2):
    """Construct frozen index maps from an already conforming right-half mesh.

    This is a host-only, discrete operation. It does not create or persist
    left-side vertices, faces or physical state unknowns.
    """
    x = np.asarray(vertices, dtype=np.float64)
    f, w = (np.asarray(v, dtype=np.int32) for v in (faces, wake_faces))
    e = np.asarray(te_edges, dtype=np.int32)
    up, lo, tags = (np.asarray(v, dtype=np.int32)
                    for v in (upper_te, lower_te, component_ids))
    ys = np.asarray(station_y, dtype=np.float64)
    dof = np.asarray(station_dof, dtype=np.int32)
    if (x.ndim != 2 or x.shape[1] != 3 or len(x) == 0
            or f.ndim != 2 or f.shape[1] != 3 or len(f) == 0
            or w.ndim != 2 or w.shape[1] != 3 or len(w) == 0
            or e.ndim != 2 or e.shape[1] != 2 or len(e) == 0
            or ys.ndim != 1 or len(ys) < 2 or dof.shape != ys.shape
            or len(up) != len(e) or len(lo) != len(e)
            or len(tags) != len(f)):
        raise ValueError("Half mesh/topology shape mismatch")
    if (not np.isfinite(x).all() or not np.isfinite(ys).all()
            or np.any(np.diff(ys) <= 0)):
        raise ValueError("Half geometry/stations must be finite and increasing")
    tol = 1e-10 * max(float(np.ptp(x, axis=0).max()), 1.)
    if abs(ys[0]) > tol or np.any(x[:, 1] < -tol):
        raise ValueError("Station origin must be y=0 and all nodes y>=0")
    if (min(f.min(), w.min(), e.min()) < 0
            or max(f.max(), w.max(), e.max()) >= len(x)
            or min(up.min(), lo.min()) < 0
            or max(up.max(), lo.max()) >= len(f)):
        raise ValueError("Half-mesh index out of range")
    if (np.any(x[f].mean(axis=1)[:, 1] <= tol)
            or np.any(x[w].mean(axis=1)[:, 1] <= tol)):
        raise ValueError("Wall/wake triangles must lie on the positive side")
    wake_tri = x[w]
    if np.ptp(wake_tri[:, :, 2]) > tol:
        raise ValueError("Only a constant-z planar wake is supported")
    wake_an = np.cross(wake_tri[:, 1]-wake_tri[:, 0],
                       wake_tri[:, 2]-wake_tri[:, 0])
    if np.any(wake_an[:, 2] <= 0):
        raise ValueError("Wake triangles must have +z normals")
    if np.any(dof < -1) or dof[-1] != -1:
        raise ValueError("Outboard circulation must vanish at the tip")
    ng = int(dof.max())+1
    if ng < 1 or set(dof[dof >= 0].tolist()) != set(range(ng)):
        raise ValueError("Half circulation DOFs must be contiguous and present")
    hat = np.zeros((len(ys), ng), dtype=np.float64)
    rows = np.flatnonzero(dof >= 0)
    hat[rows, dof[rows]] = 1.
    wc = wake_tri.mean(axis=1)[:, 1]
    wi = np.searchsorted(ys, wc, side="right") - 1
    ends = x[e]
    gauss = np.array([.5-np.sqrt(3.)/6, .5+np.sqrt(3.)/6])
    ty = ends[:, 0, 1, None]*(1-gauss) + ends[:, 1, 1, None]*gauss
    ti = np.searchsorted(ys, ty, side="right")-1
    if (np.any(wi < 0) or np.any(wi >= len(ys)-1)
            or np.any(ti < 0) or np.any(ti >= len(ys)-1)
            or np.any(ti[:, 0] != ti[:, 1])
            or np.any(wake_tri[:, :, 1].min(axis=1) < ys[wi]-tol)
            or np.any(wake_tri[:, :, 1].max(axis=1) > ys[wi+1]+tol)):
        raise ValueError("Wake/TE left the fixed circulation strips")
    neighbors = _pressure_neighbors_from_half(x, f, e, stencil_rings)
    return HalfTopology(*(jnp.asarray(a) for a in
        (f, w, e, up, lo, hat, wi, ti, neighbors, tags)))


def validate_half_configuration(vertices, station_y, topology,
                                freestream, reference_point):
    """Call outside JAX tracing at each actual design iterate."""
    x, y = np.asarray(vertices), np.asarray(station_y)
    flow, origin = np.asarray(freestream), np.asarray(reference_point)
    if (x.ndim != 2 or x.shape[1] != 3 or y.ndim != 1
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or np.any(np.diff(y) <= 0) or not np.isfinite(flow).all()
            or not np.isfinite(origin).all()):
        raise ValueError("Invalid continuous half geometry or flight state")
    tol = 1e-10*max(float(np.ptp(x, axis=0).max()), 1.)
    if (np.any(x[:, 1] < -tol) or abs(float(y[0])) > tol
            or abs(float(flow[1])) > 1e-12*np.linalg.norm(flow)
            or abs(float(origin[1])) > tol):
        raise ValueError("Half model requires y>=0, zero sideslip and reference on symmetry plane")
    tri, wake = x[np.asarray(topology.faces)], x[np.asarray(topology.wake_faces)]
    if (np.any(tri[:, :, 1].mean(axis=1) <= tol)
            or np.any(wake[:, :, 1].mean(axis=1) <= tol)):
        raise ValueError("Half-model triangles cross or lie on symmetry plane")
    if np.ptp(wake[:, :, 2]) > tol:
        raise ValueError("Half-model wake is not planar")
    cross = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    if np.any(np.linalg.norm(cross, axis=1) <= 0):
        raise ValueError("Degenerate wall triangles")
    normal = cross/np.linalg.norm(cross, axis=1)[:, None]
    tangent = tri[:, 1]-tri[:, 0]
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    other = np.cross(normal, tangent)
    centers = tri.mean(axis=1)
    augmented = np.concatenate((centers, centers * [1., -1., 1.]))
    delta = augmented[np.asarray(topology.neighbor_ghost_ids)]-centers[:, None]
    design = np.stack((np.sum(delta*tangent[:, None], axis=2),
                       np.sum(delta*other[:, None], axis=2)), axis=-1)
    singular = np.linalg.svd(design, compute_uv=False)
    if np.any(singular[:, 0] <= 0) or np.any(singular[:, -1]/singular[:, 0] < 1e-6):
        raise ValueError("Half pressure reconstruction stencil rank deficient")
    wi, ti = np.asarray(topology.wake_interval), np.asarray(topology.te_interval)
    te = x[np.asarray(topology.te_edges)][:, :, 1]
    if (np.any(wake[:, :, 1].min(axis=1) < y[wi]-tol)
            or np.any(wake[:, :, 1].max(axis=1) > y[wi+1]+tol)
            or np.any(te.min(axis=1) < y[ti[:, 0]]-tol)
            or np.any(te.max(axis=1) > y[ti[:, 0]+1]+tol)):
        raise ValueError("Half geometry left fixed wake/TE station strips")


def _half_loads(x, t, flow, density, area_ref, chord_ref, point, mu):
    """One-sided raw pressure, exact mirrored physical force and axial moment."""
    tri = x[t.faces]
    av = jnp.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])/2
    normals = av/jnp.linalg.norm(av, axis=1)[:, None]
    centers = tri.mean(axis=1)
    virtual_centers = jnp.concatenate((centers, centers*_REFLECT))
    virtual_mu = jnp.concatenate((mu, mu))
    tangent = tri[:, 1]-tri[:, 0]
    tangent = tangent/jnp.linalg.norm(tangent, axis=1)[:, None]
    second = jnp.cross(normals, tangent)
    delta = virtual_centers[t.neighbor_ghost_ids]-centers[:, None]
    dl = jnp.sum(delta*tangent[:, None], axis=2)
    dm = jnp.sum(delta*second[:, None], axis=2)
    dmu = virtual_mu[t.neighbor_ghost_ids]-mu[:, None]
    ll, mm, lm = (jnp.sum(a, axis=1) for a in (dl*dl, dm*dm, dl*dm))
    bl, bm = jnp.sum(dl*dmu, axis=1), jnp.sum(dm*dmu, axis=1)
    determinant = ll*mm-lm*lm
    grad_l = (bl*mm-bm*lm)/determinant
    grad_m = (bm*ll-bl*lm)/determinant
    velocity = ((tangent@flow+grad_l)[:, None]*tangent
                + (second@flow+grad_m)[:, None]*second)
    speed2 = jnp.dot(flow, flow)
    cp = 1-jnp.sum(velocity*velocity, axis=1)/speed2
    q = .5*density*speed2
    forces = -q*cp[:, None]*av
    moments = jnp.cross(centers-point, forces)
    one_f, one_m = forces.sum(axis=0), moments.sum(axis=0)
    all_f = one_f + one_f*_REFLECT
    all_m = one_m + one_m*_AXIAL_REFLECT
    drag_axis = flow/jnp.sqrt(speed2)
    lift_axis = jnp.cross(drag_axis, jnp.asarray([0., 1., 0.]))
    lift_axis = lift_axis/jnp.linalg.norm(lift_axis)
    tags = t.component_ids
    cf = jnp.stack([jnp.sum(jnp.where((tags==k)[:, None], forces, 0.), axis=0)
                    for k in (10, 20, 30)])
    cm = jnp.stack([jnp.sum(jnp.where((tags==k)[:, None], moments, 0.), axis=0)
                    for k in (10, 20, 30)])
    return dict(
        Cp_raw_half=cp, panel_forces_half=forces, panel_moments_half=moments,
        panel_centers_half=centers, panel_areas_half=jnp.linalg.norm(av, axis=1),
        panel_normals_half=normals, surface_velocity_half=velocity,
        tangent_determinant=determinant, force=all_f, moment=all_m,
        component_forces=cf+cf*_REFLECT,
        component_moments=cm+cm*_AXIAL_REFLECT,
        CL=all_f@lift_axis/(q*area_ref),
        CD_pressure=all_f@drag_axis/(q*area_ref),
        CM=all_m/(q*area_ref*chord_ref),
    )


def _half_wake_te_geometry(x, y, t):
    """Re-evaluate wake basis, weak TE jumps and circulation mass in JAX."""
    wake = x[t.wake_faces]
    wi = t.wake_interval
    dy = y[wi+1]-y[wi]
    frac = (wake.mean(axis=1)[:, 1]-y[wi])/dy
    basis = (1-frac[:, None])*t.hat_map[wi] + frac[:, None]*t.hat_map[wi+1]
    slope = (t.hat_map[wi+1]-t.hat_map[wi])/dy[:, None]
    gauss = jnp.asarray([.5-jnp.sqrt(3.)/6, .5+jnp.sqrt(3.)/6])
    ends = x[t.te_edges]
    te_y = ends[:, 0, 1, None]*(1-gauss)+ends[:, 1, 1, None]*gauss
    ti = t.te_interval
    tf = (te_y-y[ti])/(y[ti+1]-y[ti])
    te_basis = (1-tf[..., None])*t.hat_map[ti] + tf[..., None]*t.hat_map[ti+1]
    length = jnp.linalg.norm(ends[:, 1]-ends[:, 0], axis=1)
    weights = jnp.broadcast_to(length[:, None]/2, te_y.shape)
    mass = jnp.einsum("eq,eqi,eqj->ij", weights, te_basis, te_basis)
    integrals = jnp.einsum("eq,eqi->ei", weights, te_basis)
    measure = integrals.sum(axis=0)
    nf = len(t.faces)
    incidence = jnp.zeros((len(t.te_edges), nf), dtype=x.dtype)
    rows = jnp.arange(len(t.te_edges))
    incidence = incidence.at[rows, t.upper_te].add(1.)
    incidence = incidence.at[rows, t.lower_te].add(-1.)
    return wake, basis, slope, (integrals.T@incidence)/measure[:, None], \
           mass/measure[:, None], te_basis


def solve_half_fields(vertices, station_y, topology, freestream, density,
                      reference_area, reference_chord, reference_point):
    """Differentiable half *dense reference* (N_half+G_half unknowns only).

    All image influences are included. No full physical mesh or full AIC is
    constructed, and no source/wake influence is clipped or approximated.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Explicitly enable JAX float64")
    x, y, flow = jnp.asarray(vertices), jnp.asarray(station_y), jnp.asarray(freestream)
    t = topology
    tri = x[t.faces]
    av = jnp.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])/2
    sigma = -(av/jnp.linalg.norm(av, axis=1)[:, None])@flow
    centers = tri.mean(axis=1)
    Sd, Dd = triangle_potential(centers, tri, interior_diagonal=True)
    Si, Di = triangle_potential(centers, _mirror_triangles(tri))
    wake, basis, slope, T, M, tebasis = _half_wake_te_geometry(x, y, t)
    dw, my = triangle_potential(centers, wake, source=False, linear_y=True)
    wi, mi = triangle_potential(centers, _mirror_triangles(wake),
                                source=False, linear_y=True)
    # Reflection reverses y derivative of an even circulation distribution.
    W = (dw+wi)@basis + (my-mi)@slope
    A = jnp.block([[Dd+Di, W], [-T, M]])
    rhs = jnp.concatenate(((Sd+Si)@sigma,
                           jnp.zeros(t.hat_map.shape[1], dtype=x.dtype)))
    state = linear_solve(A, rhs)
    nf = len(tri)
    mu, gamma = state[:nf], state[nf:]
    out = _half_loads(x, t, flow, density, reference_area, reference_chord,
                      reference_point, mu)
    out.update(mu_half=mu, gamma_half=gamma, sigma_half=sigma,
               matrix_half=A, rhs_half=rhs,
               linear_residual_half=A@state-rhs, wake_basis_half=basis,
               wake_slope_half=slope, te_basis_half=tebasis)
    return out


class HalfPanelOperator:
    """Streamed half-only Av and exact A.Tv; no global AIC in these methods.

    This class consumes right-half geometry and differentiable half station
    positions. Each target chunk computes direct + image influences afresh.
    O(N^2) arithmetic; O(block*N) kernel working storage, no FMM.
    """

    def __init__(self, vertices, station_y, topology, freestream, *, block_size=32):
        if not jax.config.x64_enabled:
            raise ValueError("Explicitly enable JAX float64")
        if block_size < 1:
            raise ValueError("Positive target block required")
        self.x, self.y = jnp.asarray(vertices), jnp.asarray(station_y)
        self.t, self.flow = topology, jnp.asarray(freestream)
        self.tri = self.x[topology.faces]
        self.image = _mirror_triangles(self.tri)
        self.center = self.tri.mean(axis=1)
        av = jnp.cross(self.tri[:, 1]-self.tri[:, 0],
                       self.tri[:, 2]-self.tri[:, 0])/2
        self.sigma = -(av/jnp.linalg.norm(av, axis=1)[:, None])@self.flow
        self.nf, self.ng = len(self.tri), topology.hat_map.shape[1]
        self.block = min(int(block_size), self.nf)
        self.padding = (-self.nf) % self.block
        self.ids = jnp.pad(jnp.arange(self.nf), (0, self.padding), mode="edge")
        self.starts = jnp.arange(0, len(self.ids), self.block)
        self.sources = jnp.arange(self.nf)
        wake, self.basis, self.slope, self.T, self.M, _ = \
            _half_wake_te_geometry(self.x, self.y, topology)
        self.wake, self.wake_image = wake, _mirror_triangles(wake)

    @property
    def state_size(self):
        return self.nf+self.ng

    def _ids(self, start):
        return jax.lax.dynamic_slice_in_dim(self.ids, start, self.block)

    def coefficients(self, rows):
        targets = self.center[rows]
        Sd, Dd = triangle_potential(
            targets, self.tri, interior_diagonal=True, block_size=0,
            target_panel_ids=rows, source_panel_ids=self.sources)
        Si, Di = triangle_potential(targets, self.image, block_size=0)
        dw, my = triangle_potential(
            targets, self.wake, source=False, linear_y=True, block_size=0)
        wi, mi = triangle_potential(
            targets, self.wake_image, source=False, linear_y=True, block_size=0)
        return Sd+Si, Dd+Di, (dw+wi)@self.basis+(my-mi)@self.slope

    def rhs(self):
        def one(start):
            S, _, _ = jax.checkpoint(self.coefficients)(self._ids(start))
            return S@self.sigma
        body = jax.lax.map(one, self.starts).reshape(-1)[:self.nf]
        return jnp.concatenate((body, jnp.zeros(self.ng, dtype=self.x.dtype)))

    def matvec(self, state):
        mu, gamma = state[:self.nf], state[self.nf:]
        def one(start):
            _, D, W = jax.checkpoint(self.coefficients)(self._ids(start))
            return D@mu+W@gamma
        body = jax.lax.map(one, self.starts).reshape(-1)[:self.nf]
        return jnp.concatenate((body, -self.T@mu+self.M@gamma))

    def rmatvec(self, seed):
        padded = jnp.pad(seed[:self.nf], (0, self.padding))
        def one(carry, start):
            _, D, W = jax.checkpoint(self.coefficients)(self._ids(start))
            v = jax.lax.dynamic_slice_in_dim(padded, start, self.block)
            dm, dg = carry
            return (dm+D.T@v, dg+W.T@v), None
        init = (jnp.zeros(self.nf, dtype=self.x.dtype),
                jnp.zeros(self.ng, dtype=self.x.dtype))
        (mu, gamma), _ = jax.lax.scan(one, init, self.starts)
        return jnp.concatenate((mu-self.T.T@seed[self.nf:],
                                gamma+self.M.T@seed[self.nf:]))
