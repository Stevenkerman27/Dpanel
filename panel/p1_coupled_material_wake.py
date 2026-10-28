"""Issue #65 B2: opt-in, coupled P1 material-coordinate wake experiment.

The wall and every wake use the SAME P1 potential representation.  Phase A's
velocity has the opposite sign to influence_block's double-layer potential.
The vectorized evaluator below preserves Phase A's equation and quadrature,
not a P0 ring approximation.  Epsilon and edge-quadrature sensitivity are
measured separately; neither finite test proves a Cauchy principal value.

Geometry uses a rooted, fixed-material-arclength Picard map, not time shedding.
A uniform aligned wake is a fixed point; all xyz coordinates can change.  A
rejected node/step is never counted as evidence of convergence.  This module
does not change the production solver, certify physical accuracy, or supply a
geometry derivative through the free-wake iteration.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import hashlib
import time
from types import SimpleNamespace

import numpy as np


def triangle_data(xyz, faces):
    """Geometry and P1 shape gradients; reject degenerate/nonfinite triangles."""
    xyz, faces = np.asarray(xyz, float), np.asarray(faces, int)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or faces.ndim != 2
            or faces.shape[1] != 3 or not len(faces)
            or faces.min() < 0 or faces.max() >= len(xyz)
            or not np.isfinite(xyz).all()):
        raise ValueError("invalid triangular geometry")
    tri = xyz[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    a2 = np.linalg.norm(cross, axis=1)
    if np.any(a2 <= 0) or not np.isfinite(a2).all():
        raise ValueError("degenerate triangular geometry")
    normal = cross / a2[:, None]
    g1 = np.cross(tri[:, 2] - tri[:, 0], normal) / a2[:, None]
    g2 = np.cross(normal, tri[:, 1] - tri[:, 0]) / a2[:, None]
    edges = np.stack([np.linalg.norm(tri[:, (i+1) % 3] - tri[:, i], axis=1)
                      for i in range(3)], axis=1)
    return tri, normal, a2 / 2, np.stack((-g1-g2, g1, g2), axis=1), edges


class TotalVelocity:
    """U + wall + ALL wakes, with actual per-face P1 nodal values.

    Arrays are copied so a solved state cannot be mutated underneath an audit.
    `order=48` is the existing wake_velocity.sheet_velocity quadrature order.
    No self faces, source terms, or cross-group interactions are omitted.
    """
    def __init__(self, mesh, state):
        parts = [(mesh.xyz, mesh.faces, np.asarray(state['mu'])[mesh.dof_face])]
        offset = 0
        for group in mesh.groups:
            ng = group.gamma_hat.shape[1]
            nodal = group.gamma_hat @ np.asarray(state['gamma'])[offset:offset+ng]
            parts.append((group.wake_xyz, group.wake_faces, nodal[group.wake_faces]))
            offset += ng
        if offset != len(state['gamma']):
            raise ValueError("Gamma length does not match the independent groups")
        triangles, normals, means, gradients = [], [], [], []
        for xyz, faces, values in parts:
            tri, n, _a, shape, _e = triangle_data(xyz, faces)
            triangles.append(tri)
            normals.append(n)
            means.append(np.mean(values, axis=1))
            gradients.append(np.einsum('fa,fai->fi', values, shape))
        self.triangles = np.concatenate(triangles)
        self.normals = np.concatenate(normals)
        self.mu_face = np.concatenate(means)
        self.grad_face = np.concatenate(gradients)
        theta = np.deg2rad(float(state['alpha_deg']))
        self.free = np.array([np.cos(theta), 0., np.sin(theta)])
        self._quadrature = {}
        if not np.isfinite(self.mu_face).all() or not np.isfinite(self.grad_face).all():
            raise FloatingPointError("nonfinite solved P1 strength")

    def __call__(self, point, *, order=48):
        p = np.asarray(point, float)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError("field point must be a finite (3,) array")
        if order not in self._quadrature:
            z, w = np.polynomial.legendre.leggauss(order)
            self._quadrature[order] = ((z+1)/2, w/2)
        z, w = self._quadrature[order]
        tri, n = self.triangles, self.normals
        center = tri.mean(axis=1)
        # Match Phase A's tangential projection, including roundoff handling.
        grad = self.grad_face - n*np.einsum('fi,fi->f', self.grad_face, n)[:, None]
        boundary = np.zeros_like(n)
        surface = np.zeros_like(n)
        for k in range(3):
            a, b = tri[:, k], tri[:, (k+1) % 3]
            dl = b-a
            length = np.linalg.norm(dl, axis=1)
            tangent = dl/length[:, None]
            points = a[:, None, :] + z[None, :, None]*dl[:, None, :]
            rel = p - points
            radius = np.linalg.norm(rel, axis=2)
            if np.any(radius == 0):
                raise FloatingPointError("field point hits an edge quadrature node")
            mu = self.mu_face[:, None] + np.einsum('fqi,fi->fq',
                                                       points-center[:, None], grad)
            weights = length[:, None]*w
            boundary += np.sum((weights*mu/radius**3)[:, :, None]
                               * np.cross(tangent[:, None, :], rel), axis=1)
            surface += np.cross(tangent, n)*np.sum(weights/radius, axis=1)[:, None]
        r = tri-p
        lengths = np.linalg.norm(r, axis=2)
        num = np.einsum('fi,fi->f', r[:, 0], np.cross(r[:, 1], r[:, 2]))
        den = np.prod(lengths, axis=1)
        for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            den += np.einsum('fi,fi->f', r[:, i], r[:, j])*lengths[:, k]
        omega = 2*np.arctan2(num, den)
        induced = (boundary + np.cross(np.cross(grad, n),
                                        surface-omega[:, None]*n)).sum(axis=0)/(4*np.pi)
        # Production D = -Omega/(4*pi); apply this sign ONCE, to wall and wakes.
        return self.free-induced


def limits(field, points, normals, scales, *, eps_fraction, ladder, order):
    """Finite one-sided samples, with independent epsilon/quadrature checks."""
    factors = np.asarray(ladder, float)
    if (factors.ndim != 1 or len(factors) < 2 or not np.isfinite(factors).all()
            or np.any(factors <= 0) or np.any(np.diff(factors) >= 0)
            or not np.isfinite(eps_fraction) or not 0 < eps_fraction < .5
            or not isinstance(order, int) or order < 8):
        raise ValueError("invalid positive decreasing epsilon ladder/quadrature")
    points, normals, scales = map(np.asarray, (points, normals, scales))
    if (points.shape != normals.shape or points.ndim != 2 or points.shape[1] != 3
            or scales.shape != (len(points),) or np.any(scales <= 0)
            or not np.isfinite(scales).all()
            or not np.allclose(np.linalg.norm(normals, axis=1), 1., atol=1e-10)):
        raise ValueError("invalid sampling points/normals/scales")
    plus, minus = np.empty_like(points, dtype=float), np.empty_like(points, dtype=float)
    errors, qerrors = np.zeros(len(points)), np.zeros(len(points))
    uref = float(np.linalg.norm(field.free))
    for i, (p, n, scale) in enumerate(zip(points, normals, scales)):
        previous = None
        for factor in factors:
            h = eps_fraction*scale*factor
            up, um = field(p+h*n, order=order), field(p-h*n, order=order)
            mean = (up+um)/2
            if previous is not None:
                errors[i] = np.linalg.norm(mean-previous)/uref
            previous = mean
        plus[i], minus[i] = up, um
        refined = (field(p+h*n, order=2*order)+field(p-h*n, order=2*order))/2
        qerrors[i] = np.linalg.norm(refined-mean)/uref
    if not all(np.isfinite(a).all() for a in (plus, minus, errors, qerrors)):
        raise FloatingPointError("nonfinite velocity/epsilon/quadrature diagnostic")
    return dict(plus=plus, minus=minus, mean=(plus+minus)/2,
                eps=eps_fraction*scales*factors[-1], epsilon_error=errors,
                quadrature_error=qerrors)


def material_paths(material):
    """Frozen cumulative arclength and root for each original material path."""
    s = np.zeros(len(material.reference_xyz))
    roots = np.full(len(s), -1, int)
    for node in material.order:
        p = int(material.predecessor[node])
        if p < 0:
            roots[node] = node
        else:
            if roots[p] < 0:
                raise ValueError("material order is not topological")
            s[node] = s[p] + np.linalg.norm(material.reference_xyz[node]
                                            - material.reference_xyz[p])
            roots[node] = roots[p]
    return s, roots


def streamline_target(xyz, material, velocity, eligible):
    """Rooted fixed-arclength Picard map; no division by Vx or uniform drift.

    Use the downstream tangent at a TE root (the root is not a free sheet
    node). Else use the average endpoint velocity. Rejected predecessors stay
    in place; every rejected node remains explicitly in the convergence gate.
    """
    target = np.asarray(xyz, float).copy()
    target[material.roots] = material.reference_xyz[material.roots]
    for node in material.order:
        p = int(material.predecessor[node])
        if p < 0 or not eligible[node]:
            continue
        v = velocity[node].copy()
        if material.predecessor[p] >= 0 and eligible[p]:
            v = (v+velocity[p])/2
        speed = np.linalg.norm(v)
        if not np.isfinite(speed) or speed <= 1e-12:
            raise ValueError("undefined streamline tangent")
        ds = np.linalg.norm(material.reference_xyz[node]-material.reference_xyz[p])
        target[node] = target[p]+ds*v/speed
    return target


def area_diagnostics(sample, normal, area, material_s, *, bands=3):
    """Unmasked face diagnostics, in frozen relative material-distance bands."""
    plus, minus, mean = sample['plus'], sample['minus'], sample['mean']
    slip = np.einsum('fi,fi->f', mean, normal)
    jump = plus-minus
    dcp = np.sum(minus**2, axis=1)-np.sum(plus**2, axis=1)
    identity = dcp + 2*np.einsum('fi,fi->f', mean, jump)
    njump = np.einsum('fi,fi->f', jump, normal)
    if bands < 1 or not np.isfinite(material_s).all():
        raise ValueError("invalid material bands")
    band_id = np.minimum((np.clip(material_s, 0, 1)*bands).astype(int), bands-1)
    def aggregate(mask):
        weight = area[mask]
        result = dict(faces=int(mask.sum()), area=float(weight.sum()),
                      area_fraction=float(weight.sum()/area.sum()))
        for name, value in (('slip', slip), ('dcp', dcp), ('normal_jump', njump),
                            ('identity_residual', identity)):
            result[name+'_rms'] = (float(np.sqrt(np.average(value[mask]**2, weights=weight)))
                                  if len(weight) else None)
            result[name+'_max'] = float(np.max(np.abs(value[mask]))) if len(weight) else None
        return result
    return dict(overall=aggregate(np.ones(len(area), bool)),
                bands=[dict(s_min=k/bands, s_max=(k+1)/bands,
                            **aggregate(band_id == k)) for k in range(bands)],
                epsilon_error_max=float(np.max(sample['epsilon_error'])),
                quadrature_error_max=float(np.max(sample['quadrature_error'])),
                epsilon_min=float(np.min(sample['eps'])),
                epsilon_max=float(np.max(sample['eps'])))


@dataclass(frozen=True)
class GeometryGates:
    max_area_ratio: float = 4.
    max_fold_deg: float = 75.
    max_step_fraction: float = .25
    max_collision_pairs: int = 200000
    tolerance_fraction: float = 1e-10


def _bounds_overlap(p, q, tol):
    return not (np.any(p.max(axis=0) < q.min(axis=0)-tol)
                or np.any(q.max(axis=0) < p.min(axis=0)-tol))


def convex_overlap(p, q, tol):
    """Conservative SAT overlap of small convex point hulls, including coplanar.

    All possible hull face normals and edge-cross-edge axes are included.
    Additional in-plane axes handle flat hulls. A swept triangle is contained
    in the hull of its old/new vertices: false-positive rejection is possible,
    but vertex-only collision detection is not used as a clearance proof.
    """
    p, q = np.asarray(p, float), np.asarray(q, float)
    if not _bounds_overlap(p, q, tol):
        return False
    def vectors(x):
        edges = np.asarray([x[j]-x[i] for i, j in combinations(range(len(x)), 2)])
        normals = np.asarray([np.cross(x[j]-x[i], x[k]-x[i])
                              for i, j, k in combinations(range(len(x)), 3)]).reshape(-1, 3)
        return edges.reshape(-1, 3), normals
    ep, np_ = vectors(p)
    eq, nq = vectors(q)
    normals = np.concatenate((np_, nq))
    axes = [normals, np.cross(ep[:, None], eq[None, :]).reshape(-1, 3)]
    if len(normals):
        axes.append(np.cross(normals[:, None], np.concatenate((ep, eq))[None]).reshape(-1, 3))
    axes = np.concatenate(axes)
    norm = np.linalg.norm(axes, axis=1)
    axes = axes[norm > 1e-30]/norm[norm > 1e-30, None]
    if not len(axes):
        return True
    pp, qq = p@axes.T, q@axes.T
    return not np.any((pp.max(axis=0) < qq.min(axis=0)-tol)
                     | (qq.max(axis=0) < pp.min(axis=0)-tol))


def geometry_failure(mesh, candidates, materials, gates):
    """Validate the ENTIRE candidate atomically. Return a refusal reason or None."""
    tol = gates.tolerance_fraction*mesh.reference_chord
    swept, vertex_sets = [], []
    for gi, group in enumerate(mesh.groups):
        old, new = group.wake_xyz, candidates[group.name]
        mat = materials[group.name]
        if not np.isfinite(new).all():
            return 'nonfinite'
        if not np.array_equal(new[mat.roots], mat.reference_xyz[mat.roots]):
            return 'root'
        _s, root_ids = material_paths(mat)
        free = np.setdiff1d(mat.used, mat.roots)
        ref = mat.reference_xyz[free]-mat.reference_xyz[root_ids[free]]
        if np.any(np.einsum('ij,ij->i', new[free]-mat.reference_xyz[root_ids[free]], ref) <= 0):
            return 'attachment'
        tri0, n0, a0, _g0, e0 = triangle_data(old, group.wake_faces)
        try:
            tri1, n1, a1, _g1, _e1 = triangle_data(new, group.wake_faces)
        except ValueError:
            return 'area'
        scale = np.full(len(old), np.inf)
        for k in range(3):
            np.minimum.at(scale, group.wake_faces[:, k], e0.min(axis=1))
        if np.any(np.linalg.norm(new[mat.used]-old[mat.used], axis=1)
                  > gates.max_step_fraction*scale[mat.used]):
            return 'step'
        # Check both incremental quality and accumulated collapse/stretch.
        ratio = np.r_[a1/a0, a1/mat.panel_area]
        if np.any((ratio < 1/gates.max_area_ratio) | (ratio > gates.max_area_ratio)):
            return 'area'
        if np.any(np.einsum('fi,fi->f', n0, n1) < np.cos(np.deg2rad(gates.max_fold_deg))):
            return 'fold'
        # Remove only a tiny boundary strip to permit declared TE contact.
        # The resulting clearance tolerance is explicit, not a vortex core.
        shrink = 1e-8
        t0 = (1-shrink)*tri0 + shrink*tri0.mean(axis=1)[:, None]
        t1 = (1-shrink)*tri1 + shrink*tri1.mean(axis=1)[:, None]
        swept.extend(np.concatenate((t0, t1), axis=1))
        vertex_sets.extend([{(gi, int(n)) for n in f} for f in group.wake_faces])
    wall = mesh.xyz[mesh.faces]
    pairs = 0
    for i, hull in enumerate(swept):
        for tri in wall:
            if not _bounds_overlap(hull, tri, tol):
                continue
            pairs += 1
            if pairs > gates.max_collision_pairs:
                return 'intersection_budget'
            if convex_overlap(hull, tri, tol):
                return 'wall'
        for j in range(i):
            if vertex_sets[i] & vertex_sets[j]:  # local fold gate covers neighbours
                continue
            if not _bounds_overlap(hull, swept[j], tol):
                continue
            pairs += 1
            if pairs > gates.max_collision_pairs:
                return 'intersection_budget'
            if convex_overlap(hull, swept[j], tol):
                return 'intersection'
    return None


def snapshot_hash(arrays):
    """Content hash independent of NPZ ZIP timestamps; includes names/shapes/dtypes."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        if a.dtype.hasobject:
            raise ValueError("object arrays are forbidden in evidence snapshots")
        h.update(name.encode()+b'\0'+a.dtype.str.encode()+b'\0')
        h.update(str(a.shape).encode()+b'\0'+a.tobytes())
    return h.hexdigest()


def solve_coupled_material_wake(mesh, alpha_deg=5., *, kutta='pressure', iterations=4,
                                relaxation=.15, eps_fraction=.05,
                                ladder=(1., .5, .25, .125), order=48,
                                epsilon_tol=.02, quadrature_tol=.002,
                                geometry_tol=.002, slip_tol=.01,
                                max_nfev=80, max_backtracks=8, gates=None,
                                save_callback=None):
    """Coupled B2 experiment; callback(iteration, mesh, state, record, arrays).

    The returned `converged` is deliberately False: a kinematic candidate is
    not a pressure-continuous, mesh/epsilon-converged physical free wake.
    The complete accepted geometry is re-solved before a state is returned.
    """
    from .p1_lifting_g2 import assemble_multi, solve_multi
    from .p1_steady_free_wake import (prepare_wake_material, reassemble_wake_only,
                                     with_wake_xyz)
    if (type(iterations) is not int or iterations < 0 or type(max_backtracks) is not int
            or max_backtracks < 0 or not 0 < relaxation <= 1
            or not all(np.isfinite(v) and v > 0 for v in
                       (epsilon_tol, quadrature_tol, geometry_tol, slip_tol))):
        raise ValueError("invalid iteration or convergence controls")
    gates = GeometryGates() if gates is None else gates
    if (not np.isfinite(gates.max_area_ratio) or gates.max_area_ratio <= 1
            or not 0 < gates.max_fold_deg < 90 or not gates.max_step_fraction > 0
            or gates.max_collision_pairs < 1 or not gates.tolerance_fraction > 0):
        raise ValueError("invalid geometry gates")
    started = time.perf_counter()
    materials = {g.name: prepare_wake_material(g) for g in mesh.groups}
    static = assemble_multi(mesh)
    fixed = solve_multi(mesh, alpha_deg, kutta=kutta, max_nfev=max_nfev, operators=static)
    current, history, incoming = mesh, [], None
    status = 'iteration_limit'
    load_names = ('CL', 'CL_gamma', 'CD_pressure', 'CM_y')
    initial_equivalence = None
    for iteration in range(iterations+1):
        # Reassemble even iteration 0: explicitly test W-only assembly equivalence.
        ops = reassemble_wake_only(current, static)
        state = solve_multi(current, alpha_deg, kutta=kutta, max_nfev=max_nfev, operators=ops)
        if iteration == 0:
            initial_equivalence = {k: float(abs(state[k]-fixed[k])/max(1., abs(fixed[k])))
                                   for k in load_names}
            initial_equivalence['W_relative'] = float(np.linalg.norm(ops['W']-static['W'])
                                                      / max(1., np.linalg.norm(static['W'])))
            if max(initial_equivalence.values()) > 1e-10:
                raise RuntimeError('iteration-0 fixed-wake equivalence failed: '+str(initial_equivalence))
        field = TotalVelocity(current, state)
        arrays = dict(mu=state['mu'], gamma=state['gamma'], cp=state['cp'],
                      alpha_deg=np.asarray(alpha_deg), wall_xyz=current.xyz,
                      wall_faces=current.faces, wall_dof_face=current.dof_face,
                      sampling_eps_fraction=np.asarray(eps_fraction),
                      sampling_ladder=np.asarray(ladder, float), sampling_order=np.asarray(order))
        record = {k: state[k] for k in (*load_names, 'wall_rms', 'jump_rms',
                  'pressure_kutta_rms', 'signed_kutta_rms',
                  'linear_constrained_rank', 'linear_singular_min', 'nonlinear')}
        record.update(iteration=iteration, groups=state['group_results'],
                      CLp_minus_CLgamma=float(state['CL']-state['CL_gamma']),
                      CL_relative_gap=float((state['CL']-state['CL_gamma'])/
                                            max(abs(state['CL_gamma']), 1e-12)),
                      wakes={}, incoming_update=incoming)
        constraint = ops['J'] if kutta == 'pressure' else np.vstack((ops['J'], ops['K']))
        sv = np.linalg.svd(constraint, compute_uv=False)
        record['constraint_rank'] = int(np.sum(sv > 1e-11*sv[0]))
        record['constraint_singular_min'] = float(sv[-1])
        targets, counts = {}, {}
        all_eligible, map_residual = True, 0.
        for gi, group in enumerate(current.groups):
            mat = materials[group.name]
            tri, normal, area, _shape, edges = triangle_data(group.wake_xyz, group.wake_faces)
            face_samples = limits(field, tri.mean(axis=1), normal, edges.min(axis=1),
                                  eps_fraction=eps_fraction, ladder=ladder, order=order)
            distance, _roots = material_paths(mat)
            relative_s = distance[group.wake_faces].mean(axis=1)/max(distance.max(), 1e-30)
            diag = area_diagnostics(face_samples, normal, area, relative_s)
            normals = np.zeros_like(group.wake_xyz)
            scales = np.full(len(normals), np.inf)
            for k in range(3):
                np.add.at(normals, group.wake_faces[:, k], normal*area[:, None])
                np.minimum.at(scales, group.wake_faces[:, k], edges.min(axis=1))
            free_nodes = np.setdiff1d(mat.used, mat.roots)
            lengths = np.linalg.norm(normals[free_nodes], axis=1)
            if np.any(lengths <= 1e-14):
                raise ValueError(group.name+': undefined area-weighted node normal')
            samples = limits(field, group.wake_xyz[free_nodes], normals[free_nodes]/lengths[:, None],
                             scales[free_nodes], eps_fraction=eps_fraction, ladder=ladder, order=order)
            velocity = np.zeros_like(group.wake_xyz)
            velocity[free_nodes] = samples['mean']
            finite_limit = ((samples['epsilon_error'] <= epsilon_tol)
                            & (samples['quadrature_error'] <= quadrature_tol))
            fast = np.linalg.norm(samples['mean'], axis=1) > 1e-6
            eligible = np.zeros(len(velocity), bool)
            eligible[free_nodes] = finite_limit & fast
            # Opposed endpoint velocities have an undefined average tangent.
            tangent_rejections = 0
            for node in mat.order:
                pred = int(mat.predecessor[node])
                if (pred >= 0 and mat.predecessor[pred] >= 0 and eligible[node]
                        and eligible[pred] and np.linalg.norm(velocity[node]+velocity[pred]) <= 2e-6):
                    eligible[node] = False
                    tangent_rejections += 1
            all_eligible &= bool(np.all(eligible[free_nodes]))
            target = streamline_target(group.wake_xyz, mat, velocity, eligible)
            map_residual = max(map_residual, float(np.max(np.linalg.norm(
                              target[mat.used]-group.wake_xyz[mat.used], axis=1))/mesh.reference_chord))
            targets[group.name] = target
            counts[group.name] = dict(root=len(mat.roots), eligible=int(eligible.sum()),
                                      convergence=int((~finite_limit).sum()),
                                      speed=int((finite_limit & ~fast).sum())+tangent_rejections,
                                      rejected=int((~eligible[free_nodes]).sum()))
            diag.update(area_min=float(area.min()), aspect_p90=float(np.percentile(
                        edges.max(axis=1)/edges.min(axis=1), 90)),
                        fixed_roots=len(mat.roots), node_epsilon_error_max=float(samples['epsilon_error'].max()),
                        node_quadrature_error_max=float(samples['quadrature_error'].max()),
                        eligibility=counts[group.name])
            record['wakes'][group.name] = diag
            prefix = 'g%d_' % gi
            for name, value in dict(xyz=group.wake_xyz, faces=group.wake_faces,
                                    gamma_hat=group.gamma_hat, roots=mat.roots,
                                    reference_xyz=mat.reference_xyz, predecessor=mat.predecessor,
                                    order=mat.order, used=mat.used, panel_area=mat.panel_area,
                                    free_nodes=free_nodes, eligible=eligible,
                                    material_s=relative_s).items():
                arrays[prefix+name] = np.asarray(value)
            arrays.update({prefix+'node_'+k: v for k, v in samples.items()})
            arrays.update({prefix+'face_'+k: v for k, v in face_samples.items()})
        record['map_residual_chord'] = map_residual
        face_checks = all(d['epsilon_error_max'] <= epsilon_tol
                          and d['quadrature_error_max'] <= quadrature_tol
                          and d['overall']['slip_rms'] <= slip_tol for d in record['wakes'].values())
        nonlinear_ok = state['nonlinear'] is None or state['nonlinear']['success']
        candidate = bool(all_eligible and face_checks and map_residual <= geometry_tol and nonlinear_ok)
        record['kinematic_candidate'] = candidate
        record['state_sha256'] = snapshot_hash(arrays)
        record['outgoing_update'] = None
        stop = iteration == iterations or candidate or not nonlinear_ok
        if candidate:
            status = 'kinematic_candidate_not_physical_acceptance'
        elif not nonlinear_ok:
            status = 'blocked_nonlinear_solve'
        if not stop:
            update = dict(accepted=False, relaxation=None, trial_reasons=[],
                          eligible=counts, displacement_max=0., displacement_rms=0.)
            maximum = max(float(np.max(np.linalg.norm(targets[g.name]-g.wake_xyz, axis=1)))
                          for g in current.groups)
            if maximum <= 1e-15*mesh.reference_chord:
                status, stop = 'blocked_velocity_or_zero_update', True
            else:
                for backtrack in range(max_backtracks+1):
                    factor = relaxation*2.**(-backtrack)
                    proposed = {g.name: g.wake_xyz+factor*(targets[g.name]-g.wake_xyz)
                                for g in current.groups}
                    for g in current.groups:
                        proposed[g.name][materials[g.name].roots] = materials[g.name].reference_xyz[materials[g.name].roots]
                    reason = geometry_failure(current, proposed, materials, gates)
                    if reason is not None:
                        update['trial_reasons'].append(reason)
                        continue
                    ds = np.concatenate([np.linalg.norm(proposed[g.name][materials[g.name].used]
                                            -g.wake_xyz[materials[g.name].used], axis=1) for g in current.groups])
                    update.update(accepted=True, relaxation=factor,
                                  displacement_max=float(ds.max()), displacement_rms=float(np.sqrt(np.mean(ds**2))),
                                  moved_nodes=int(np.count_nonzero(ds > 0)))
                    break
                if not update['accepted']:
                    status, stop = 'blocked_geometry', True
            record['outgoing_update'] = update
        update = record['outgoing_update']
        if update is not None:
            update['reason_counts'] = {name: update['trial_reasons'].count(name) for name in
                ('root', 'speed', 'step', 'area', 'fold', 'wall', 'intersection',
                 'convergence', 'nonfinite', 'attachment', 'intersection_budget')}
            update['reason_counts']['root'] = sum(c['root'] for c in counts.values())
            for name in ('speed', 'convergence'):
                update['reason_counts'][name] = sum(c[name] for c in counts.values())
        history.append(record)
        if save_callback is not None:
            save_callback(iteration, current, state, record, arrays)
        if stop:
            break
        incoming = record['outgoing_update']
        current = with_wake_xyz(current, proposed)
    trends = {g.name: dict(slip_nonincreasing=bool(np.all(np.diff([
        row['wakes'][g.name]['overall']['slip_rms'] for row in history]) <= 0)),
        rejected_nodes=[row['wakes'][g.name]['eligibility']['rejected'] for row in history])
        for g in current.groups}
    return dict(mesh=current, state=state, history=history, status=status, trends=trends,
                iterations=len(history)-1, initial_equivalence=initial_equivalence,
                seconds=time.perf_counter()-started, converged=False,
                kinematic_candidate=history[-1]['kinematic_candidate'],
                physical_accuracy_accepted=False,
                method='P1-coupled-rooted-material-arclength-Picard',
                kernel='Phase-A-equation; finite Gauss edges; epsilon and order checks')


def replay_snapshot(arrays, expected_sha):
    """Re-evaluate frozen total-flow samples without re-solving/changing mu/Gamma.

    A mismatch in the recorded content SHA is an error. Bitwise equality is
    reported, never promised across NumPy/BLAS/platform changes.
    """
    if snapshot_hash(arrays) != expected_sha:
        raise ValueError("frozen-state SHA mismatch")
    groups = []
    gi = 0
    while 'g%d_xyz' % gi in arrays:
        p = 'g%d_' % gi
        groups.append(SimpleNamespace(name=p, wake_xyz=arrays[p+'xyz'],
                      wake_faces=arrays[p+'faces'], gamma_hat=arrays[p+'gamma_hat']))
        gi += 1
    if not groups:
        raise ValueError("snapshot has no wake groups")
    mesh = SimpleNamespace(xyz=arrays['wall_xyz'], faces=arrays['wall_faces'],
                           dof_face=arrays['wall_dof_face'], groups=tuple(groups))
    field = TotalVelocity(mesh, dict(mu=arrays['mu'], gamma=arrays['gamma'],
                                   alpha_deg=float(arrays['alpha_deg'])))
    kwargs = dict(eps_fraction=float(arrays['sampling_eps_fraction']),
                  ladder=arrays['sampling_ladder'], order=int(arrays['sampling_order']))
    differences = {}
    equal = True
    for gi, group in enumerate(groups):
        p = 'g%d_' % gi
        tri, normal, area, _shape, edges = triangle_data(group.wake_xyz, group.wake_faces)
        normals = np.zeros_like(group.wake_xyz)
        scale = np.full(len(normals), np.inf)
        for k in range(3):
            np.add.at(normals, group.wake_faces[:, k], normal*area[:, None])
            np.minimum.at(scale, group.wake_faces[:, k], edges.min(axis=1))
        free = arrays[p+'free_nodes']
        normals[free] /= np.linalg.norm(normals[free], axis=1)[:, None]
        samples = dict(face=limits(field, tri.mean(axis=1), normal, edges.min(axis=1), **kwargs),
                       node=limits(field, group.wake_xyz[free], normals[free], scale[free], **kwargs))
        for location, values in samples.items():
            for key, value in values.items():
                name = p+location+'_'+key
                equal &= bool(np.array_equal(value, arrays[name]))
                differences[name] = float(np.max(np.abs(value-arrays[name])))
    return dict(state_sha256=expected_sha, bitwise_equal=equal, max_differences=differences,
                physical_accuracy_accepted=False)
