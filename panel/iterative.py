"""Opt-in exact blocked panel operator, GMRES and XZ-even state reduction.

No dense wall AIC, source AIC, wall/wake operator or LU is constructed here.
Pairwise work is still quadratic: this is NOT FMM or an asymptotic speed claim.
Only frozen-topology, planar-wake, float64 models supported by differentiable.py
are supported. Half mode additionally requires an exactly mirrored discretization.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from jax.scipy.sparse.linalg import gmres
from jax.custom_transpose import custom_transpose
import numpy as np
from scipy.spatial import cKDTree

from .kernel import triangle_potential
from .differentiable import recover_loads, validate_configuration


@dataclass(frozen=True)
class OperatorPlan:
    """Host-only immutable index planning. No current geometry or metric cache."""
    body_rows: np.ndarray
    body_inverse: np.ndarray
    gamma_rows: np.ndarray
    gamma_inverse: np.ndarray
    near_blocks: tuple
    block_size: int
    symmetry: bool
    vertex_mirror: np.ndarray | None
    plane_y: float

    @property
    def shape(self):
        n = len(self.body_rows) + len(self.gamma_rows)
        return n, n


def _orbits(permutation):
    ids = np.arange(len(permutation))
    if not np.array_equal(permutation[permutation], ids):
        raise ValueError("Reflection must be an involution")
    representative = np.minimum(ids, permutation)
    rows, inverse = np.unique(representative, return_inverse=True)
    return rows.astype(np.int32), inverse.astype(np.int32)


def _geometric_blocks(points, ids, size):
    if len(ids) <= size:
        return (ids,)
    axis = np.ptp(points[ids], axis=0).argmax()
    order = ids[np.argsort(points[ids, axis], kind='stable')]
    mid = len(order)//2
    return (_geometric_blocks(points, order[:mid], size)
            + _geometric_blocks(points, order[mid:], size))


def _face_reflection(faces, vertex_map):
    lookup = {tuple(sorted(face)): i for i, face in enumerate(faces)}
    if len(lookup) != len(faces):
        raise ValueError("Duplicate panels cannot define a mirror map")
    try:
        return np.array([lookup[tuple(sorted(vertex_map[face]))] for face in faces])
    except KeyError as exc:
        raise ValueError("Mesh triangles are not exact mirror pairs; do not average AIC") from exc


def make_plan(vertices, station_y, topology, freestream, *, block_size,
              preconditioner_block_size, symmetry, plane_y=0.0):
    """Freeze pairing and spatial blocks once, outside JAX tracing.

    Plane vertices may be self-paired; wall panels on the plane are rejected.
    A station at the symmetry plane remains an independent *self* orbit.
    Full coordinates are retained O(N); only state unknowns/rows are reduced.
    """
    if block_size <= 0 or preconditioner_block_size <= 0:
        raise ValueError("Block sizes must be positive")
    x, y = np.asarray(vertices), np.asarray(station_y)
    validate_configuration(x, y, topology)
    f = np.asarray(topology.faces)
    n, ng = len(f), topology.hat_map.shape[1]
    center = x[f].mean(axis=1)
    bperm, gperm, vperm = np.arange(n), np.arange(ng), None
    if symmetry:
        scale = max(float(np.ptp(x, axis=0).max()), np.finfo(float).tiny)
        tol = 1e-10*scale
        reflected = x.copy(); reflected[:, 1] = 2*plane_y-x[:, 1]
        distance, vperm = cKDTree(x).query(reflected)
        if (distance.max() > tol or len(np.unique(vperm)) != len(x)
                or not np.array_equal(vperm[vperm], np.arange(len(x)))):
            raise ValueError("Need bijective mirrored vertices, including wake nodes")
        if np.any(np.abs(center[:, 1]-plane_y) <= tol):
            raise ValueError("Wall panel center on symmetry plane; no fictitious cap")
        bperm = _face_reflection(f, vperm)
        av = np.cross(x[f[:,1]]-x[f[:,0]], x[f[:,2]]-x[f[:,0]])
        normal = av/np.linalg.norm(av,axis=1)[:,None]
        if not np.allclose(normal[bperm], normal*[1,-1,1], rtol=0, atol=1e-9):
            raise ValueError("Reflected wall winding must preserve outward normals")
        if not np.array_equal(np.asarray(topology.component_ids)[bperm], topology.component_ids):
            raise ValueError("Component tags must respect reflection")
        _face_reflection(np.asarray(topology.wake_faces), vperm)
        if not np.allclose(y, 2*plane_y-y[::-1], rtol=0, atol=tol):
            raise ValueError("Circulation stations are not mirrored")
        hat = np.asarray(topology.hat_map)
        pairs = []
        for column in hat.T:
            hits = np.flatnonzero(np.all(hat == column[::-1,None], axis=0))
            if len(hits) != 1:
                raise ValueError("Circulation DOF basis has no unique mirror partner")
            pairs.append(hits[0])
        gperm = np.array(pairs)
        edges = np.asarray(topology.te_edges)
        te_lookup = {tuple(sorted(edge)): i for i, edge in enumerate(edges)}
        upper, lower = np.asarray(topology.upper_te), np.asarray(topology.lower_te)
        for i, edge in enumerate(edges):
            key = tuple(sorted(vperm[edge]))
            if key not in te_lookup:
                raise ValueError("TE edge has no mirror counterpart")
            j = te_lookup[key]
            if upper[j] != bperm[upper[i]] or lower[j] != bperm[lower[i]]:
                raise ValueError("TE upper/lower banks do not respect reflection")
    br, bi = _orbits(bperm)
    gr, gi = _orbits(gperm)
    blocks = _geometric_blocks(center[br], np.arange(len(br), dtype=np.int32), preconditioner_block_size)
    plan = OperatorPlan(br, bi, gr, gi, blocks, int(block_size), bool(symmetry), vperm, float(plane_y))
    validate_plan(vertices, station_y, topology, freestream, plan)
    return plan


def validate_plan(vertices, station_y, topology, freestream, plan):
    """Call at actual iterates, outside tracing; no topology changes are repaired."""
    validate_configuration(vertices, station_y, topology)
    u = np.asarray(freestream)
    if not np.isfinite(u).all() or np.linalg.norm(u) == 0:
        raise ValueError("Require finite nonzero freestream")
    if plan.symmetry:
        x, y = np.asarray(vertices), np.asarray(station_y)
        scale = max(float(np.ptp(x, axis=0).max()), np.finfo(float).tiny)
        reflected = x.copy(); reflected[:, 1] = 2*plan.plane_y-x[:, 1]
        if (not np.allclose(x[plan.vertex_mirror], reflected, rtol=0, atol=1e-10*scale)
                or not np.allclose(y, 2*plan.plane_y-y[::-1], rtol=0, atol=1e-10*scale)
                or abs(u[1]) > 1e-12*np.linalg.norm(u)):
            raise ValueError("Even half model requires symmetric geometry/stations and zero sideslip")


class PanelOperator:
    """Exact full or R A E operator with analytic block-recomputed transpose."""
    def __init__(self, vertices, station_y, topology, freestream, plan):
        if not jax.config.x64_enabled:
            raise ValueError("Explicitly enable JAX x64")
        self.t, self.plan = topology, plan
        self.x, self.y, self.flow = jnp.asarray(vertices), jnp.asarray(station_y), jnp.asarray(freestream)
        self.tri, self.wake = self.x[topology.faces], self.x[topology.wake_faces]
        self.center = self.tri.mean(axis=1)
        av = jnp.cross(self.tri[:,1]-self.tri[:,0], self.tri[:,2]-self.tri[:,0])
        self.sigma = -(av/jnp.linalg.norm(av,axis=1)[:,None]) @ self.flow
        self.nf, self.ng = len(self.tri), topology.hat_map.shape[1]
        self.nb, self.nr = len(plan.body_rows), len(plan.gamma_rows)
        self.block = min(plan.block_size, self.nb)
        self.rows = jnp.asarray(plan.body_rows)
        self.bi, self.gi = jnp.asarray(plan.body_inverse), jnp.asarray(plan.gamma_inverse)
        self.gr = jnp.asarray(plan.gamma_rows)
        padding = (-self.nb) % self.block
        self.padded_rows = jnp.pad(self.rows, (0,padding), mode='edge')
        self.starts = jnp.arange(0, len(self.padded_rows), self.block)
        wi = topology.wake_interval
        self.wdy = self.y[wi+1]-self.y[wi]
        self.wf = (self.wake.mean(axis=1)[:,1]-self.y[wi])/self.wdy
        gauss = jnp.array([.5-jnp.sqrt(3.)/6, .5+jnp.sqrt(3.)/6])
        ends = self.x[topology.te_edges]
        ty = ends[:,0,1,None]*(1-gauss)+ends[:,1,1,None]*gauss
        ti = topology.te_interval
        self.tf = (ty-self.y[ti])/(self.y[ti+1]-self.y[ti])
        self.weights = jnp.broadcast_to(jnp.linalg.norm(ends[:,1]-ends[:,0],axis=1)[:,None]/2, ty.shape)
        self.measure = self.te_scatter(self.weights)

    def expand(self, vector):
        return vector[:self.nb][self.bi], vector[self.nb:][self.gi]

    def te_scatter(self, values):
        ti = self.t.te_interval
        stations = jnp.zeros(len(self.y), dtype=self.x.dtype)
        stations = stations.at[ti.ravel()].add((values*(1-self.tf)).ravel())
        stations = stations.at[(ti+1).ravel()].add((values*self.tf).ravel())
        return self.t.hat_map.T @ stations

    def te_sample(self, gamma):
        g = self.t.hat_map @ gamma; ti = self.t.te_interval
        return (1-self.tf)*g[ti]+self.tf*g[ti+1]

    def kutta(self, mu, gamma):
        jump = mu[self.t.upper_te]-mu[self.t.lower_te]
        return self.te_scatter(self.weights*(self.te_sample(gamma)-jump[:,None]))/self.measure

    def kutta_transpose(self, seed):
        sample = self.te_sample(seed/self.measure)
        edge_seed = jnp.sum(self.weights*sample, axis=1)
        mu = jnp.zeros(self.nf, dtype=self.x.dtype)
        mu = mu.at[self.t.upper_te].add(-edge_seed).at[self.t.lower_te].add(edge_seed)
        return mu, self.te_scatter(self.weights*sample)

    def wake_sample(self, gamma):
        g = self.t.hat_map @ gamma; wi = self.t.wake_interval
        return ((1-self.wf)*g[wi]+self.wf*g[wi+1], (g[wi+1]-g[wi])/self.wdy)

    def wake_transpose(self, value_seed, slope_seed):
        wi = self.t.wake_interval
        stations = jnp.zeros(len(self.y), dtype=self.x.dtype)
        stations = stations.at[wi].add((1-self.wf)*value_seed-slope_seed/self.wdy)
        stations = stations.at[wi+1].add(self.wf*value_seed+slope_seed/self.wdy)
        return self.t.hat_map.T @ stations

    def _ids(self, start):
        return jax.lax.dynamic_slice_in_dim(self.padded_rows,start,self.block)

    def body_block(self, rows, source_ids, *, source=False):
        return triangle_potential(self.center[rows], self.tri[source_ids],
            interior_diagonal=True, source=source, block_size=0,
            target_panel_ids=rows, source_panel_ids=source_ids)

    def coefficients(self, rows):
        D = self.body_block(rows, jnp.arange(self.nf))
        W, Y = triangle_potential(self.center[rows], self.wake,
                                  source=False, linear_y=True, block_size=0)
        return D, W, Y

    def rhs(self):
        def block(start):
            S, _ = self.body_block(self._ids(start), jnp.arange(self.nf), source=True)
            return S @ self.sigma
        body = jax.lax.map(jax.checkpoint(block), self.starts).reshape(-1)[:self.nb]
        return jnp.concatenate((body,jnp.zeros(self.nr,dtype=self.x.dtype)))

    def matvec(self, vector):
        mu, gamma = self.expand(vector)
        strength, slope = self.wake_sample(gamma)
        def block(carry, start):
            cm, cs, cg = carry
            D,W,Y = jax.checkpoint(self.coefficients)(self._ids(start))
            return carry, D@cm + W@cs + Y@cg
        _, values = jax.lax.scan(block, (mu,strength,slope), self.starts)
        body = values.reshape(-1)[:self.nb]
        return jnp.concatenate((body,self.kutta(mu,gamma)[self.gr]))

    def rmatvec(self, vector):
        padded_seed = jnp.pad(vector[:self.nb],(0,len(self.padded_rows)-self.nb))
        def add_block(carry,start):
            D,W,Y = jax.checkpoint(self.coefficients)(self._ids(start))
            mu, gamma, all_seed = carry
            seed = jax.lax.dynamic_slice_in_dim(all_seed,start,self.block)
            return (mu+D.T@seed, gamma+self.wake_transpose(W.T@seed,Y.T@seed), all_seed), None
        (mu,gamma,_),_ = jax.lax.scan(add_block,
            (jnp.zeros(self.nf,dtype=self.x.dtype),jnp.zeros(self.ng,dtype=self.x.dtype),padded_seed),self.starts)
        # R^T scatters rows; E^T sums mirror cotangents (not just half selection).
        seed_g = jnp.zeros(self.ng,dtype=self.x.dtype).at[self.gr].set(vector[self.nb:], unique_indices=True)
        km,kg = self.kutta_transpose(seed_g)
        m = jnp.zeros(self.nb,dtype=self.x.dtype).at[self.bi].add(mu+km)
        g = jnp.zeros(self.nr,dtype=self.x.dtype).at[self.gi].add(gamma+kg)
        return jnp.concatenate((m,g))

    def linear_action(self, *, transpose=False):
        """Register the analytic transpose of the streamed map explicitly.

        The custom JVP differentiates geometry through the ordinary analytic
        kernels; only the vector-linear part uses the custom transpose. This
        avoids asking JAX to transpose a rematerialized loop with linear
        closed-over carry (unsupported in some JAX releases).
        """
        plan = self.plan
        def raw(res, vector, reverse):
            x, y, t, flow = res
            op = PanelOperator(x,y,t,flow,plan)
            return op.rmatvec(vector) if reverse else op.matvec(vector)
        @custom_transpose
        def linear(res, vector):
            return raw(res,vector,transpose)
        @linear.def_transpose
        def linear_reverse(res, cotangent):
            return raw(res,cotangent,not transpose)
        @jax.custom_jvp
        def action(x,y,t,flow,vector):
            aval = jax.core.ShapedArray(vector.shape,vector.dtype)
            return linear(aval,(x,y,t,flow),vector)
        @partial(action.defjvp, symbolic_zeros=True)
        def action_jvp(primals,tangents):
            x,y,t,flow,vector = primals
            dx,dy,_,dflow,dvector = tangents
            zero = jax.custom_derivatives.SymbolicZero
            res = (x,y,t,flow)
            primal = raw(res,vector,transpose)
            if all(isinstance(v,zero) for v in (dx,dy,dflow)):
                geometry_dot = jnp.zeros_like(primal)
            else:
                tangent = tuple(jnp.zeros_like(p) if isinstance(d,zero) else d
                                for p,d in zip((x,y,flow),(dx,dy,dflow)))
                _, geometry_dot = jax.jvp(
                    lambda xx,yy,uu: raw((xx,yy,t,uu),vector,transpose),
                    (x,y,flow),tangent)
            if isinstance(dvector,zero):
                vector_dot = jnp.zeros_like(primal)
            else:
                aval = jax.core.ShapedArray(vector.shape,vector.dtype)
                vector_dot = linear(aval,res,dvector)
            return primal, geometry_dot + vector_dot
        return lambda v: action(self.x,self.y,self.t,self.flow,v)

    def preconditioner(self):
        """P=[[blockdiag(D),0],[-T,M]] and P^-T. No global wall factor."""
        factors = []
        for ids in self.plan.near_blocks:
            sources = np.flatnonzero(np.isin(self.plan.body_inverse,ids)).astype(np.int32)
            lookup = np.full(self.nb,-1); lookup[ids] = np.arange(len(ids))
            columns = jnp.asarray(lookup[self.plan.body_inverse[sources]])
            coefficient = self.body_block(self.rows[ids],jnp.asarray(sources))
            B = jnp.zeros((len(ids),len(ids)),dtype=self.x.dtype).at[:,columns].add(coefficient)
            factors.append((jnp.asarray(ids),jla.lu_factor(jax.lax.stop_gradient(B))))
        zero_mu = jnp.zeros(self.nf,dtype=self.x.dtype)
        def mass_action(g):
            return self.kutta(zero_mu,g[self.gi])[self.gr]
        M = jax.vmap(mass_action, in_axes=1, out_axes=1)(jnp.eye(self.nr,dtype=self.x.dtype))
        mf = jla.lu_factor(jax.lax.stop_gradient(M))
        def body_solve(rhs,transpose):
            out = jnp.zeros_like(rhs)
            for ids,factor in factors:
                out = out.at[ids].set(jla.lu_solve(factor,rhs[ids],trans=int(transpose)), unique_indices=True)
            return out
        def apply(rhs):
            m = body_solve(rhs[:self.nb],False)
            jump = self.kutta(m[self.bi],jnp.zeros(self.ng,dtype=self.x.dtype))[self.gr]
            g = jla.lu_solve(mf,rhs[self.nb:]-jump)
            return jnp.concatenate((m,g))
        def transpose(rhs):
            g = jla.lu_solve(mf,rhs[self.nb:],trans=1)
            seed_g = jnp.zeros(self.ng,dtype=self.x.dtype).at[self.gr].set(g, unique_indices=True)
            km,_ = self.kutta_transpose(seed_g)
            lower_transpose = jnp.zeros(self.nb,dtype=self.x.dtype).at[self.bi].add(km)
            m = body_solve(rhs[:self.nb]-lower_transpose,True)
            return jnp.concatenate((m,g))
        return apply,transpose


def solve_state(operator, rhs, *, tolerance, absolute_tolerance, restart,
                max_cycles, preconditioned):
    """Implicit AD through exact matvec. Nonconverged solves return NaN (fail closed).

    JAX GMRES info is not a portable iteration counter; independently inspect
    original residuals, including in the transpose callback. Each callback is
    given its own correctly oriented preconditioner.
    """
    if not 0 < tolerance < 1 or absolute_tolerance < 0 or min(restart,max_cycles) < 1:
        raise ValueError("Invalid GMRES tolerances or iteration limits")
    forward_pc,transpose_pc = operator.preconditioner() if preconditioned else (None,None)
    def checked(mv,b,pc):
        x,_ = gmres(mv,b,tol=tolerance,atol=absolute_tolerance,
                    restart=restart,maxiter=max_cycles,M=pc,solve_method='incremental')
        error = jnp.linalg.norm(mv(x)-b)
        threshold = jnp.maximum(tolerance*jnp.linalg.norm(b),absolute_tolerance)
        good = jnp.isfinite(error) & (error <= threshold)
        return jnp.where(good,x,jnp.full_like(x,jnp.nan))
    forward_action = operator.linear_action()
    transpose_action = operator.linear_action(transpose=True)
    state = jax.lax.custom_linear_solve(forward_action,rhs,
        solve=lambda mv,b: checked(mv,b,forward_pc),
        transpose_solve=lambda _,b: checked(transpose_action,b,transpose_pc),
        symmetric=False)
    return state


def solve_fields_iterative(vertices, station_y, topology, freestream, density,
                           reference_area, reference_chord, reference_point,
                           *, plan, tolerance, absolute_tolerance, restart,
                           max_cycles, preconditioned):
    """Opt-in alternative. Same pressure law; no matrix returned or constructed.

    Validate geometry/half-plan at every external iterate. In half mode only
    mirror-preserving geometry directions are supported, not arbitrary full-X
    gradients of a nonsymmetric aircraft. No physical precision certificate.
    """
    op = PanelOperator(vertices,station_y,topology,freestream,plan)
    rhs = op.rhs()
    state = solve_state(op,rhs,tolerance=tolerance,absolute_tolerance=absolute_tolerance,
                        restart=restart,max_cycles=max_cycles,preconditioned=preconditioned)
    mu,gamma = op.expand(state)
    out = recover_loads(vertices,topology,freestream,density,reference_area,
                        reference_chord,reference_point,mu)
    out.update(mu=mu,gamma=gamma,sigma=op.sigma,state=state,rhs=rhs,
               linear_residual=op.matvec(state)-rhs)
    return out
