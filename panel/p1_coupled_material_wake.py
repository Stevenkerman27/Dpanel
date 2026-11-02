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

from .p1_geometry_contact import (classify_new_triangle, classify_swept_wall,
                                  declared_contact_pairs,
                                 swept_hull)
from .wake_velocity_methods import (ACCEPTED_METHODS, AREA_QUADRATURES,
                                    ARCHITECTURE_DOC, DEFAULT_METHOD, describe,
                                    wiring_error)
from dataclasses import dataclass
from itertools import combinations
import hashlib
import inspect
import time
from types import SimpleNamespace

import numpy as np

from . import velocity_profile as _profile


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
    """U + wall + ALL wakes, preserving each sheet's P1 gradient.

    Two wire-ready models, both on the same P1 doublet-sheet representation
    (see docs/ARCHITECTURE_WAKE_MODELS.md):

    * analytic_edges -- method 1, unregularized, production default/reference.
    * rm_regularized_p1 -- method 2, one explicit Rosenhead–Moore core applied
      to BOTH the P1 edge traces and the in-panel distributed-vorticity term.

    area_quadrature (legacy_duffy | projection_fan | projection_fan_adaptive) is
    an integrator choice *inside* method 2, not a velocity method. legacy_gauss
    is a deprecated backend kept for frozen-snapshot replay. method 3
    (Lagrangian filaments) is documented and not wired. Both models change only
    velocity reconstruction for wake kinematics; solve_multi's potential
    operators remain unregularized and unchanged.
    """
    def __init__(self, mesh, state, *, method="analytic_edges", core_radius=None,
                 area_quadrature="legacy_duffy", fan_order=16, fan_strips=8,
                 fan_refinement_order=None, fan_refinement_strips=None,
                 near_ratio=2.0, adaptive_far_order=8, adaptive_far_levels=1,
                 adaptive_budget=2e-4, adaptive_max_escalations=2,
                 adaptive_subdivision_levels=0, adaptive_subdivision_fan=(48, 8),
                 analytic_backend="loop", analytic_device="cpu", analytic_chunk=64):
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
        self.method = method
        self.core_radius = None
        self.area_quadrature = area_quadrature
        self.fan_order = fan_order
        self.fan_strips = fan_strips
        self.fan_refinement_order = fan_refinement_order
        self.fan_refinement_strips = fan_refinement_strips
        # Fail closed: an area-quadrature choice that the selected evaluator cannot
        # honour would otherwise be accepted and silently ignored.
        if method not in ACCEPTED_METHODS:
            raise ValueError(
                "velocity_method must be one of " + ", ".join(ACCEPTED_METHODS)
                + f"; quadrature settings belong to area_quadrature; see {ARCHITECTURE_DOC}")
        self.method_spec = describe(method)
        if area_quadrature not in AREA_QUADRATURES:
            raise ValueError("area_quadrature must be " + " or ".join(AREA_QUADRATURES))
        if type(fan_order) is not int or fan_order < 2:
            raise ValueError("fan_order must be an integer >= 2")
        if type(fan_strips) is not int or fan_strips < 1:
            raise ValueError("fan_strips must be an integer >= 1")
        if area_quadrature != "legacy_duffy" and method != "rm_regularized_p1":
            raise ValueError(
                "area_quadrature=projection_fan requires method='rm_regularized_p1'")
        if method == "analytic_edges":
            from .p1_analytic_sheet_velocity import AnalyticSheetField
            self._analytic = AnalyticSheetField(parts, freestream=self.free)
        elif method == "rm_regularized_p1":
            if core_radius is None or not np.isfinite(core_radius) or core_radius <= 0:
                raise ValueError("rm_regularized_p1 requires an explicit positive core_radius")
            from .p1_regularized_sheet_velocity import RegularizedP1SheetField
            self.core_radius = float(core_radius)
            self._regularized = RegularizedP1SheetField(
                parts, freestream=self.free, core_radius=self.core_radius,
                area_quadrature=area_quadrature, fan_order=fan_order,
                fan_strips=fan_strips, fan_refinement_order=fan_refinement_order,
                fan_refinement_strips=fan_refinement_strips, near_ratio=near_ratio,
                adaptive_far_order=adaptive_far_order,
                adaptive_far_levels=adaptive_far_levels,
                adaptive_budget=adaptive_budget,
                adaptive_max_escalations=adaptive_max_escalations,
                adaptive_subdivision_levels=adaptive_subdivision_levels,
                adaptive_subdivision_fan=adaptive_subdivision_fan)
            # The gate plan is a property of the field, not of the caller: limits()
            # must compare the base fan against the refined fan, never the same
            # configuration twice (issue #84 task 1).
            self.fan_refinement_order = self._regularized.fan_refinement_order
            self.fan_refinement_strips = self._regularized.fan_refinement_strips
            self.near_ratio = self._regularized.near_ratio
            self.adaptive_far_order = self._regularized.adaptive_far_order
            self.adaptive_far_levels = self._regularized.adaptive_far_levels
            self.adaptive_budget = self._regularized.adaptive_budget
            self.adaptive_max_escalations = self._regularized.adaptive_max_escalations
            self.adaptive_subdivision_levels = \
                self._regularized.adaptive_subdivision_levels
            self.adaptive_subdivision_fan = \
                self._regularized.adaptive_subdivision_fan
            self.quadrature_is_internal = self._regularized.quadrature_is_internal
        elif method != "legacy_gauss":
            raise ValueError(wiring_error(method) or
                             "unknown P1 sheet-velocity integration method")
        # The analytic evaluator has three interchangeable execution backends. 'loop'
        # is the original per-target path and stays the default so that nothing that
        # was recorded before changes; 'numpy' batches targets through the same closed
        # form (bitwise identical, one call per chunk); 'torch' runs the batched kernel
        # on an explicit device. A backend that cannot be honoured raises here instead
        # of falling back, because a silent fallback would make a GPU timing a CPU one.
        if analytic_backend not in ("loop", "numpy", "torch"):
            raise ValueError("analytic_backend must be 'loop', 'numpy' or 'torch'")
        if analytic_device not in ("cpu", "cuda"):
            raise ValueError("analytic_device must be 'cpu' or 'cuda'")
        if type(analytic_chunk) is not int or analytic_chunk < 1:
            raise ValueError("analytic_chunk must be a positive integer")
        if analytic_backend != "torch" and analytic_device != "cpu":
            raise ValueError("analytic_device='cuda' requires analytic_backend='torch'")
        if analytic_backend != "loop" and method != "analytic_edges":
            raise ValueError("analytic_backend applies only to method='analytic_edges'")
        self.analytic_backend = analytic_backend
        self.analytic_device = analytic_device
        self.analytic_chunk = analytic_chunk
        self._torch_analytic = None
        if analytic_backend == "torch":
            from .p1_torch_analytic_sheet import GEOMETRY_KEYS, TorchAnalyticSheetField
            arrays = {key: getattr(self._analytic, key) for key in GEOMETRY_KEYS}
            self._torch_analytic = TorchAnalyticSheetField.from_arrays(
                arrays, freestream=self.free, device=analytic_device)
            self.analytic_device = self._torch_analytic.device.type
        if not np.isfinite(self.mu_face).all() or not np.isfinite(self.grad_face).all():
            raise FloatingPointError("nonfinite solved P1 strength")

    def pop_adaptive_estimates(self):
        '''Per-target a-posteriori estimates from the internal adaptive gate.'''
        if self.method == "rm_regularized_p1":
            return self._regularized.pop_adaptive_estimates()
        return []

    def quadrature_arguments(self, order):
        '''Base and refined quadrature keywords for the sampling gate.'''
        if self.method == "rm_regularized_p1":
            return self._regularized.quadrature_arguments(order)
        return dict(order=order), dict(order=2*order)

    def quadrature_plan(self, order):
        '''JSON description of the gate actually run by limits().'''
        if self.method == "rm_regularized_p1":
            return self._regularized.quadrature_plan(order)
        return dict(method=self.method, base=dict(order=int(order)),
                    refined=dict(order=2*int(order)), legacy_order=int(order))

    def __call__(self, point, *, order=48, fan_order=None, fan_strips=None):
        p = np.asarray(point, float)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError("field point must be a finite (3,) array")
        _profile.add("total.call.count")
        if fan_order is not None or fan_strips is not None:
            if (self.method != "rm_regularized_p1" or self.area_quadrature
                    not in ("projection_fan", "projection_fan_adaptive")):
                raise ValueError("fan overrides require rm_regularized_p1 with a fan "
                                 "area quadrature")
        if self.method == "analytic_edges":
            if self.analytic_backend == "torch":
                return self._torch_analytic.velocity(p[None, :], chunk=1)[0]
            if self.analytic_backend == "numpy":
                return self._analytic.evaluate_many(p[None, :], chunk=1)[0]
            return self._analytic(p)
        if self.method == "rm_regularized_p1":
            return self._regularized(p, order=order, fan_order=fan_order,
                                     fan_strips=fan_strips)
        if order not in self._quadrature:
            z, w = np.polynomial.legendre.leggauss(order)
            self._quadrature[order] = ((z+1)/2, w/2)
        z, w = self._quadrature[order]
        tri, n = self.triangles, self.normals
        center = tri.mean(axis=1)
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
        return self.free-induced


    def evaluate_many(self, points, *, order=48, fan_order=None, fan_strips=None):
        pts = np.asarray(points, float)
        if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
            raise ValueError("points must be a finite (N,3) array")
        _profile.add("total.evaluate_many.calls")
        _profile.add("total.evaluate_many.targets", len(pts))
        _profile.add("total.evaluate_many.method."+self.method)
        if fan_order is not None or fan_strips is not None:
            if (self.method != "rm_regularized_p1" or self.area_quadrature
                    not in ("projection_fan", "projection_fan_adaptive")):
                raise ValueError("fan overrides require rm_regularized_p1 with a fan "
                                 "area quadrature")
        if self.method == "rm_regularized_p1":
            return self._regularized.evaluate_many(pts, order=order,
                                                   fan_order=fan_order,
                                                   fan_strips=fan_strips)
        if self.analytic_backend == "numpy":
            return self._analytic.evaluate_many(pts, chunk=self.analytic_chunk)
        if self.analytic_backend == "torch":
            return self._torch_analytic.velocity(pts, chunk=self.analytic_chunk)
        return np.stack([self(point, order=order) for point in pts])


def _accepts_quadrature_order(field):
    '''True when the field's velocity call takes an order keyword.'''
    target = field.evaluate_many if hasattr(field, "evaluate_many") else field
    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return True
    if "order" in parameters:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())


def _plan_label(kwargs):
    '''Deterministic compact label for one quadrature configuration.'''
    return ",".join("%s=%s" % (key, kwargs[key]) for key in sorted(kwargs))


def limits(field, points, normals, scales, *, eps_fraction, ladder, order, tag="",
           audit_quadrature=True):
    """Finite one-sided samples: epsilon sensitivity plus legacy Gauss-order audit.

    In analytic-edge mode, edge quadrature does not exist: its reported
    quadrature_error is 0 (NOT APPLICABLE), not a PV-convergence certificate.

    tag only labels the profiling counters (wake_face, free_node, replay_*): it never
    changes a value, so a labelled and an unlabelled call are bitwise identical.
    """
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
    base_kwargs, refined_kwargs = dict(order=order), dict(order=2*order)
    accepts_order = _accepts_quadrature_order(field)
    analytic = (getattr(field, "method", None) == "analytic_edges"
                or not accepts_order)
    internal = bool(getattr(field, "quadrature_is_internal", False))
    if not accepts_order:
        base_kwargs, refined_kwargs = {}, {}
    if internal:
        base_kwargs, refined_kwargs = {}, {}
    if analytic:
        plan = dict(method=getattr(field, "method", None) or "analytic_edges",
                    base=dict(base_kwargs), refined=dict(base_kwargs), legacy_order=None,
                    note="edge quadrature does not exist; quadrature_error is not applicable")
    elif internal:
        if hasattr(field, "quadrature_plan"):
            plan = field.quadrature_plan(order)
        else:
            plan = dict(method="projection_fan_adaptive", base={}, refined={},
                        legacy_order=None, internal=True)
    else:
        if hasattr(field, "quadrature_plan"):
            plan = field.quadrature_plan(order)
        else:
            plan = dict(method="legacy_gauss", base=dict(base_kwargs),
                        refined=dict(refined_kwargs), legacy_order=int(order))
        if hasattr(field, "quadrature_arguments"):
            base_kwargs, refined_kwargs = field.quadrature_arguments(order)
        if dict(base_kwargs) == dict(refined_kwargs):
            raise ValueError("quadrature gate would compare a configuration with itself")
    _profile.add("limits.calls")
    _profile.add("limits.points", len(points))
    _profile.add("limits.tag."+str(tag)+".calls")
    _profile.add("limits.tag."+str(tag)+".points", len(points))
    plus, minus = np.empty_like(points, dtype=float), np.empty_like(points, dtype=float)
    errors, qerrors = np.zeros(len(points)), np.zeros(len(points))
    reference = getattr(field, "free", None)
    if reference is None:
        reference = getattr(field, "freestream", None)
    if reference is None:
        raise ValueError("field must expose a reference speed as free or freestream")
    uref = float(np.linalg.norm(reference))
    if hasattr(field, "evaluate_many"):
        previous = None
        for factor in factors:
            h = eps_fraction*scales*factor
            targets = np.concatenate((points+h[:, None]*normals,
                                      points-h[:, None]*normals), axis=0)
            _profile.add("limits.ladder_steps")
            _profile.add("limits.targets", len(targets))
            _profile.add("limits.tag."+str(tag)+".targets", len(targets))
            values = field.evaluate_many(targets, **base_kwargs)
            up, um = values[:len(points)], values[len(points):]
            mean = (up+um)/2
            if previous is not None:
                errors = np.linalg.norm(mean-previous, axis=1)/uref
            previous = mean
        plus[:], minus[:] = up, um
        if analytic:
            qerrors[:] = 0.
        elif internal:
            _profile.add("limits.internal_gate")
            estimates = (field.pop_adaptive_estimates()
                         if hasattr(field, "pop_adaptive_estimates") else [])
            if len(estimates) < 2*len(points):
                raise ValueError("internal quadrature field reported no per-target "
                                 "a-posteriori estimates")
            recent = np.asarray(estimates[-2*len(points):], float)
            qerrors = .5*(recent[:len(points)]+recent[len(points):])
        elif not audit_quadrature:
            # gate_mode='off': the refined evaluation exists only to audit the
            # quadrature; skip it and report qerrors=0 (not applicable).
            qerrors[:] = 0.
        else:
            h = eps_fraction*scales*factors[-1]
            targets = np.concatenate((points+h[:, None]*normals,
                                      points-h[:, None]*normals), axis=0)
            _profile.add("limits.refined_calls")
            _profile.add("limits.refined_targets", len(targets))
            _profile.add("limits.tag."+str(tag)+".refined_targets", len(targets))
            refined = field.evaluate_many(targets, **refined_kwargs)
            refined_mean = (refined[:len(points)]+refined[len(points):])/2
            qerrors = np.linalg.norm(refined_mean-mean, axis=1)/uref
    else:
        _profile.add("limits.scalar_points", len(points))
        for i, (p, n, scale) in enumerate(zip(points, normals, scales)):
            previous = None
            for factor in factors:
                _profile.add("limits.ladder_steps")
                _profile.add("limits.targets", 2)
                h = eps_fraction*scale*factor
                up, um = (field(p+h*n, **base_kwargs),
                          field(p-h*n, **base_kwargs))
                mean_i = (up+um)/2
                if previous is not None:
                    errors[i] = np.linalg.norm(mean_i-previous)/uref
                previous = mean_i
            plus[i], minus[i] = up, um
            if analytic:
                qerrors[i] = 0.
            elif internal:
                _profile.add("limits.internal_gate")
                estimates = (field.pop_adaptive_estimates()
                             if hasattr(field, "pop_adaptive_estimates") else [])

                if len(estimates) < 2:
                    raise ValueError("internal quadrature field reported no "
                                     "a-posteriori estimate")
                qerrors[i] = .5*(estimates[-2]+estimates[-1])
            else:
                refined = (field(p+h*n, **refined_kwargs)
                           +field(p-h*n, **refined_kwargs))/2
                qerrors[i] = np.linalg.norm(refined-mean_i)/uref
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


def streamline_target(xyz, material, velocity, eligible, *, step_scale=None,
                     truncation_policy='advance'):
    """Rooted fixed-arclength Picard map; no division by Vx or uniform drift.

    Use the downstream tangent at a TE root (the root is not a free sheet
    node). Else use the average endpoint velocity. Rejected predecessors stay
    in place; every rejected node remains explicitly in the convergence gate.

    The map marches one REFERENCE LINK per node, so a link that is far longer
    than the sheet's own panels (a truncation row) asks for a displacement of
    link*|u_hat-t_ref| chord -- measured 15-23 chord on the G2 CAD wake for a
    single 26-chord link. That request is not a velocity artifact and no
    per-node step cap can accept it, so such rows are boundary markers:
    truncation_policy='freeze' holds them (and reports them), 'advance' keeps
    the legacy behaviour.
    """
    if truncation_policy not in ('advance', 'freeze'):
        raise ValueError("truncation_policy must be 'advance' or 'freeze'")
    if truncation_policy == 'freeze' and step_scale is None:
        raise ValueError("truncation_policy='freeze' requires the per-node step_scale")
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
    if truncation_policy == 'freeze':
        hold = np.asarray(step_scale['truncation'], bool)
        target[hold] = np.asarray(xyz, float)[hold]
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


def _controlling_nodes(detail, ledgers=None):
    """Name the specific nodes and pairs that refused the last candidate."""
    detail = detail or {}
    contacts = detail.get('contacts', {})
    out = dict(groups=detail.get('groups', {}),
               contacts=dict(kinds=contacts.get('kinds', {}),
                             violations_count=contacts.get('violations_count', 0),
                             worst_depth=contacts.get('worst_depth', 0.),
                             violations=list(contacts.get('violations', []))[:10]),
               clip={})
    if ledgers:
        out['clip'] = {name: dict(clipped=ledger['clipped'], raw_max=ledger['raw_max'],
                                  accepted_max=ledger['accepted_max'],
                                  clipped_ids=ledger['clipped_ids'][:10])
                       for name, ledger in ledgers.items()}
    return out


def node_step_scale(wake_xyz, wake_faces):
    """Shortest incident wake edge per node (inf when the node has no face)."""
    _tri, _n, _a, _s, edges = triangle_data(np.asarray(wake_xyz, float),
                                            np.asarray(wake_faces, int))
    scale = np.full(len(wake_xyz), np.inf)
    for k in range(3):
        np.minimum.at(scale, np.asarray(wake_faces, int)[:, k], edges.min(axis=1))
    return scale


def wake_step_scale(wake_xyz, wake_faces, material, *, truncation_ratio=4.):
    """Per-node convection scale of the rooted arclength map.

    'scale' is the SHORTEST incident triangle edge: a step cap only bounds strain
    when it is expressed in the mesh's own resolution.  Using the reference link
    as the cap (0.25 * a 26-chord truncation link = 6.5 chord, see
    docs/ISSUE84_STREAMLINE_STEP_UNIT.md) accepts a displacement that deforms the
    sheet far beyond the area/fold gate, so the link is reported for diagnostics
    ('link', 'truncation') and never used as the cap.

    'truncation' marks links longer than truncation_ratio times the MEDIAN link
    of the sheet.  The ratio is relative, not absolute: a uniformly coarse sheet
    is self-similar under the map and must not be flagged, while one outlier row
    (G2 CAD: 102 nodes on 26.0-chord links, median link 0.666) is.  Those nodes
    are reported, never silently frozen and never exempted from the gate.
    """
    xyz = np.asarray(wake_xyz, float)
    faces = np.asarray(wake_faces, int)
    geometric = node_step_scale(xyz, faces)
    predecessor = np.asarray(material.predecessor, int)
    reference = np.asarray(material.reference_xyz, float)
    link = np.zeros(len(xyz))
    known = predecessor >= 0
    link[known] = np.linalg.norm(reference[known]-reference[predecessor[known]], axis=1)
    finite = np.isfinite(geometric)
    scale = np.where(finite, geometric, np.maximum(link, 1e-12))
    positive = link[known & (link > 0)]
    median_link = float(np.median(positive)) if len(positive) else 0.
    truncation = (known & (link > truncation_ratio*median_link)
                  if median_link > 0 else np.zeros(len(xyz), bool))
    return dict(scale=scale, link=link, geometric=geometric, truncation=truncation,
                link_median=median_link)


def clip_step_update(old, target, caps, *, roots=None, reference=None,
                     unit='geometric_edge'):
    """Cap every node displacement; roots stay pinned. Returns (xyz, ledger)."""
    old = np.asarray(old, float)
    delta = np.asarray(target, float)-old
    magnitude = np.linalg.norm(delta, axis=1)
    caps = np.asarray(caps, float)
    limited = np.isfinite(caps) & (magnitude > caps)
    factor = np.ones(len(old))
    # A cap applied exactly at the gate threshold (cap/magnitude rounding) can
    # still read back as 0.25+8e-15 and fail the step gate; keep a relative
    # margin so the accepted step is strictly inside the limit.
    factor[limited] = (1.-1e-9)*caps[limited]/np.maximum(magnitude[limited], 1e-300)
    new = old+delta*factor[:, None]
    if roots is not None:
        roots = np.asarray(roots, int)
        new[roots] = np.asarray(reference, float)[roots]
    ids = np.nonzero(limited)[0]
    accepted = np.linalg.norm(new-old, axis=1)
    ledger = dict(mode="per_node_step_cap", unit=unit, clipped=int(len(ids)),
                  clipped_ids=[int(i) for i in ids],
                  clipped_fraction=float(len(ids)/max(len(old), 1)),
                  raw_max=float(magnitude.max()) if len(magnitude) else 0.,
                  raw_rms=float(np.sqrt(np.mean(magnitude**2))) if len(magnitude) else 0.,
                  accepted_max=float(accepted.max()) if len(old) else 0.,
                  accepted_rms=float(np.sqrt(np.mean(accepted**2))) if len(old) else 0.,
                  cap_min=(float(caps[np.isfinite(caps)].min())
                           if np.isfinite(caps).any() else None))
    return new, ledger


def geometry_audit(mesh, candidates, materials, gates, *, declared=None, step_scale=None):
    """Validate the ENTIRE candidate atomically; return reason plus audit detail.

    The wall test classifies every swept wake triangle against every
    overlapping wall triangle with a SIGNED plane distance
    (panel.p1_geometry_contact) instead of a boolean SAT overlap.  Pairs whose
    contact is declared by reference topology may touch; any protrusion through
    the wall plane inside the wall footprint is refused for declared and
    undeclared pairs alike.
    """
    tol = gates.tolerance_fraction*mesh.reference_chord
    detail = dict(tolerance=float(tol), groups={}, declared_pairs=len(declared or {}),
                  contacts=dict(tested=0, declared=0, undeclared=0, pairs_tested=0,
                                kinds={}, violations=[], violations_count=0, worst_depth=0.,
                                pre_existing=0, pre_existing_pairs=[]))
    swept, vertex_sets, owners = [], [], []
    for gi, group in enumerate(mesh.groups):
        old, new = group.wake_xyz, np.asarray(candidates[group.name], float)
        mat = materials[group.name]
        if not np.isfinite(new).all():
            return dict(reason='nonfinite', detail=detail)
        if not np.array_equal(new[mat.roots], mat.reference_xyz[mat.roots]):
            return dict(reason='root', detail=detail)
        _s, root_ids = material_paths(mat)
        free = np.setdiff1d(mat.used, mat.roots)
        ref = mat.reference_xyz[free]-mat.reference_xyz[root_ids[free]]
        attachment = np.einsum('ij,ij->i', new[free]-mat.reference_xyz[root_ids[free]], ref) <= 0
        if np.any(attachment):
            detail['groups'][group.name] = dict(
                attachment_failures=int(attachment.sum()),
                attachment_nodes=[int(v) for v in free[attachment]][:20])
            return dict(reason='attachment', detail=detail)
        tri0, n0, a0, _g0, e0 = triangle_data(old, group.wake_faces)
        try:
            tri1, n1, a1, _g1, _e1 = triangle_data(new, group.wake_faces)
        except ValueError:
            return dict(reason='area', detail=detail)
        # The step gate must use the SAME length unit as the stepper that
        # produced the candidate (the map arclength for wake nodes), otherwise a
        # legitimate fraction of one map step reads as a huge geometric step.
        scale = (node_step_scale(old, group.wake_faces) if step_scale is None
                 else np.asarray(step_scale[group.name], float))
        step = np.linalg.norm(new[mat.used]-old[mat.used], axis=1)/np.maximum(
            scale[mat.used], 1e-300)
        over_step = step > gates.max_step_fraction
        if np.any(over_step):
            detail['groups'][group.name] = dict(
                step_fraction_max=float(step.max()), step_limit=float(gates.max_step_fraction),
                step_nodes=[int(v) for v in mat.used[over_step]][:20])
            return dict(reason='step', detail=detail)
        ratio = np.r_[a1/a0, a1/mat.panel_area]
        if np.any((ratio < 1/gates.max_area_ratio) | (ratio > gates.max_area_ratio)):
            detail['groups'][group.name] = dict(area_ratio_min=float(ratio.min()),
                                                area_ratio_max=float(ratio.max()))
            return dict(reason='area', detail=detail)
        fold = np.einsum('fi,fi->f', n0, n1)
        if np.any(fold < np.cos(np.deg2rad(gates.max_fold_deg))):
            detail['groups'][group.name] = dict(fold_cos_min=float(fold.min()))
            return dict(reason='fold', detail=detail)
        # Remove only a tiny boundary strip to permit declared TE contact.
        # The resulting clearance tolerance is explicit, not a vortex core.
        shrink = 1e-8
        t0 = (1-shrink)*tri0 + shrink*tri0.mean(axis=1)[:, None]
        t1 = (1-shrink)*tri1 + shrink*tri1.mean(axis=1)[:, None]
        swept.extend(np.concatenate((t0, t1), axis=1))
        owners.extend([(group.name, i) for i in range(len(group.wake_faces))])
        vertex_sets.extend([{(gi, int(n)) for n in f} for f in group.wake_faces])
    wall = mesh.xyz[mesh.faces]
    old_coords = {g.name: np.asarray(g.wake_xyz, float) for g in mesh.groups}
    group_faces = {g.name: g.wake_faces for g in mesh.groups}
    static_cache = {}
    wall_edge_min = np.minimum(np.minimum(
        np.linalg.norm(wall[:, 1]-wall[:, 0], axis=1),
        np.linalg.norm(wall[:, 2]-wall[:, 1], axis=1)),
        np.linalg.norm(wall[:, 0]-wall[:, 2], axis=1))
    pairs = 0
    for i, hull in enumerate(swept):
        label, face_id = owners[i]
        for j, tri in enumerate(wall):
            if not _bounds_overlap(hull, tri, tol):
                continue
            pairs += 1
            if pairs > gates.max_collision_pairs:
                detail['contacts']['pairs_tested'] = pairs
                return dict(reason='intersection_budget', detail=detail)
            entry = None if declared is None else declared.get((label, face_id, j))
            declared_kind = None if entry is None else entry[0]
            detail['contacts']['declared' if declared_kind is not None else 'undeclared'] += 1
            region = None
            if declared_kind is not None:
                # The WALL half of the shared simplex: the attachment is fixed,
                # so a wake node may leave it only by the graze corridor.
                region = np.asarray(mesh.xyz, float)[list(entry[1])]
            graze = max(20.*tol, 0.02*float(wall_edge_min[j]))
            cls = classify_swept_wall(hull, tri, tol=tol, declared=declared_kind,
                                      declared_region=region, graze_tol=graze if region is not None else 0.)
            detail['contacts']['tested'] += 1
            if cls.kind != 'disjoint':
                kinds = detail['contacts']['kinds']
                kinds[cls.kind] = kinds.get(cls.kind, 0)+1
                detail['contacts']['worst_depth'] = max(detail['contacts']['worst_depth'],
                                                        float(cls.depth))
            if cls.violation:
                # An inherited crossing is attributed, not silently excused:
                # it must already violate in the geometry the step starts from,
                # keep the same class, and not get deeper. It is reported every
                # iteration until the geometry is repaired, and a new class or a
                # deeper protrusion still blocks.
                key = (label, int(face_id), int(j))
                old_cls = static_cache.get(key)
                if old_cls is None:
                    old_cls = classify_new_triangle(
                        old_coords[label][group_faces[label][face_id]], tri, tol=tol,
                        declared=declared_kind, declared_region=region,
                        graze_tol=graze if region is not None else 0.)
                    static_cache[key] = old_cls
                if (old_cls.violation and old_cls.kind == cls.kind
                        and cls.depth <= old_cls.depth + 10.*tol):
                    detail['contacts']['pre_existing'] += 1
                    if len(detail['contacts']['pre_existing_pairs']) < 200:
                        detail['contacts']['pre_existing_pairs'].append(dict(
                            group=label, wake_face=int(face_id), wall_face=int(j),
                            kind=cls.kind, depth=float(cls.depth)))
                    continue
                detail['contacts']['violations_count'] += 1
                if len(detail['contacts']['violations']) < 200:
                    detail['contacts']['violations'].append(dict(
                        group=label, wake_face=int(face_id), wall_face=int(j),
                        kind=cls.kind, depth=float(cls.depth), area=float(cls.area),
                        declared=cls.declared, old_kind=old_cls.kind,
                        old_depth=float(old_cls.depth), old_violation=bool(old_cls.violation)))
    detail['contacts']['pairs_tested'] = pairs
    if detail['contacts']['violations_count']:
        return dict(reason='wall', detail=detail)
    for i, hull in enumerate(swept):
        for j in range(i):
            if vertex_sets[i] & vertex_sets[j]:  # local fold gate covers neighbours
                continue
            if not _bounds_overlap(hull, swept[j], tol):
                continue
            pairs += 1
            if pairs > gates.max_collision_pairs:
                detail['contacts']['pairs_tested'] = pairs
                return dict(reason='intersection_budget', detail=detail)
            if convex_overlap(hull, swept[j], tol):
                return dict(reason='intersection', detail=detail)
    detail['contacts']['pairs_tested'] = pairs
    return dict(reason=None, detail=detail)


def geometry_failure(mesh, candidates, materials, gates):
    """Validate the ENTIRE candidate atomically. Return a refusal reason or None."""
    return geometry_audit(mesh, candidates, materials, gates)['reason']


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
                                velocity_method="analytic_edges", core_radius=None,
                                area_quadrature="legacy_duffy", fan_order=16,
                                fan_strips=8, fan_refinement_order=None,
                                fan_refinement_strips=None, near_ratio=2.0,
                                adaptive_far_order=8, adaptive_far_levels=1,
                                adaptive_budget=2e-4, adaptive_max_escalations=2,
                                gate_mode="audit", truncation_policy="advance",
                                 analytic_backend="loop", analytic_device="cpu",
                                 analytic_chunk=64, solve_backend="numpy",
                                 solve_device="cpu", pressure_solver=None,
                                 save_callback=None):
    """Coupled B2 experiment; callback(iteration, mesh, state, record, arrays).

    The returned `converged` is deliberately False: a kinematic candidate is
    not a pressure-continuous, mesh/epsilon-converged physical free wake.
    The complete accepted geometry is re-solved before a state is returned.
    """
    from .p1_lifting_g2 import assemble_multi, solve_multi
    from .p1_steady_free_wake import (prepare_wake_material, reassemble_wake_only,
                                     with_wake_xyz)

    def _solve(target, operators_for_solve):
        """Solve the potential system on the selected backend.

        'numpy' keeps the production NumPy/SciPy path. 'torch' uses the ported
        torch.linalg chain (S5) and returns host arrays for everything downstream,
        because a NumPy LAPACK call after a torch kernel crashes the process.
        """
        if solve_backend == "numpy":
            return solve_multi(target, alpha_deg, kutta=kutta, max_nfev=max_nfev,
                               operators=operators_for_solve,
                               pressure_solver=pressure_solver or "reference")
        from .p1_solve_torch import solve_multi_torch
        solved = solve_multi_torch(target, alpha_deg, kutta=kutta, max_nfev=max_nfev,
                                   operators=operators_for_solve, device=solve_device)
        for key in ("mu", "gamma", "cp"):
            solved[key] = np.asarray(solved[key])
        solved["force"] = np.asarray(solved["force"], float)
        return solved
    if (type(iterations) is not int or iterations < 0 or type(max_backtracks) is not int
            or max_backtracks < 0 or not 0 < relaxation <= 1
            or not all(np.isfinite(v) and v > 0 for v in
                       (epsilon_tol, quadrature_tol, geometry_tol, slip_tol))):
        raise ValueError("invalid iteration or convergence controls")
    if gate_mode not in ("enforce", "audit", "off", "safe"):
        raise ValueError("gate_mode must be 'enforce', 'audit', 'off' or 'safe'")
    if truncation_policy not in ("advance", "freeze"):
        raise ValueError("truncation_policy must be 'advance' or 'freeze'")
    audit_quadrature = gate_mode != 'off'
    if velocity_method not in ACCEPTED_METHODS:
        raise ValueError(
            "velocity_method must be one of " + ", ".join(ACCEPTED_METHODS)
            + f"; quadrature settings belong to area_quadrature; see {ARCHITECTURE_DOC}")
    if velocity_method == "rm_regularized_p1":
        if core_radius is None or not np.isfinite(core_radius) or core_radius <= 0:
            raise ValueError("rm_regularized_p1 requires an explicit positive core_radius")
    elif core_radius is not None:
        raise ValueError("core_radius is only valid with velocity_method='rm_regularized_p1'")
    if analytic_backend not in ("loop", "numpy", "torch"):
        raise ValueError("analytic_backend must be 'loop', 'numpy' or 'torch'")
    if analytic_backend != "loop" and velocity_method != "analytic_edges":
        raise ValueError("analytic_backend applies only to velocity_method='analytic_edges'")
    if solve_backend not in ("numpy", "torch"):
        raise ValueError("solve_backend must be 'numpy' or 'torch'")
    if solve_device not in ("cpu", "cuda"):
        raise ValueError("solve_device must be 'cpu' or 'cuda'")
    if solve_backend != "torch" and solve_device != "cpu":
        raise ValueError("solve_device='cuda' requires solve_backend='torch'")
    if pressure_solver not in (None, "reference", "structured"):
        raise ValueError("pressure_solver must be None, 'reference' or 'structured'")
    if solve_backend == "torch" and pressure_solver == "reference":
        raise ValueError("the torch backend implements only pressure_solver='structured'")
    gates = GeometryGates() if gates is None else gates
    if (not np.isfinite(gates.max_area_ratio) or gates.max_area_ratio <= 1
            or not 0 < gates.max_fold_deg < 90 or not gates.max_step_fraction > 0
            or gates.max_collision_pairs < 1 or not gates.tolerance_fraction > 0):
        raise ValueError("invalid geometry gates")
    started = time.perf_counter()
    materials = {g.name: prepare_wake_material(g) for g in mesh.groups}
    # Reference topology declares which wake/wall pairs are allowed to touch.
    # Identity comes from shared vertex ids, never from coordinate proximity.
    declared = declared_contact_pairs([g.wake_faces for g in mesh.groups], mesh.faces,
                                      group_labels=[g.name for g in mesh.groups])
    static = assemble_multi(mesh)
    fixed = _solve(mesh, static)
    current, history, incoming = mesh, [], None
    status = 'iteration_limit'
    load_names = ('CL', 'CL_gamma', 'CD_pressure', 'CM_y')
    initial_equivalence = None
    for iteration in range(iterations+1):
        # Reassemble even iteration 0: explicitly test W-only assembly equivalence.
        ops = reassemble_wake_only(current, static)
        if iteration == 0:
            # Same unchanged geometry: verify that W-only assembly is identical
            # as an OPERATOR and reuse the canonical fixed-wake state. A second
            # nonlinear least_squares call may stop at a different nfev even
            # for a roundoff-sized W change (local receipt: 5 vs 6 nfev).
            for key in ("A", "J", "K", "grad"):
                if not np.array_equal(ops[key], static[key]):
                    raise RuntimeError("iteration-0 fixed operator changed: "+key)
            w_relative = float(np.linalg.norm(ops['W']-static['W'])
                               / max(1., np.linalg.norm(static['W'])))
            if w_relative > 1e-12:
                raise RuntimeError("iteration-0 wake influence differs: "+str(w_relative))
            state = fixed
            initial_equivalence = {k: 0. for k in load_names}
            initial_equivalence["W_relative"] = w_relative
        else:
            state = _solve(current, ops)
        field = TotalVelocity(current, state, method=velocity_method, core_radius=core_radius,
                              area_quadrature=area_quadrature,
                              fan_order=fan_order, fan_strips=fan_strips,
                              fan_refinement_order=fan_refinement_order,
                              fan_refinement_strips=fan_refinement_strips,
                              near_ratio=near_ratio,
                              adaptive_far_order=adaptive_far_order,
                              adaptive_far_levels=adaptive_far_levels,
                              adaptive_budget=adaptive_budget,
                              adaptive_max_escalations=adaptive_max_escalations,
                              analytic_backend=analytic_backend,
                              analytic_device=analytic_device,
                              analytic_chunk=analytic_chunk)
        plan = (field.quadrature_plan(order) if hasattr(field, "quadrature_plan")
                else dict(method=getattr(field, "method", "injected_field"),
                          base={}, refined={}, legacy_order=int(order)))
        arrays = dict(mu=state['mu'], gamma=state['gamma'], cp=state['cp'],
                      alpha_deg=np.asarray(alpha_deg), wall_xyz=current.xyz,
                      wall_faces=current.faces, wall_dof_face=current.dof_face,
                      sampling_eps_fraction=np.asarray(eps_fraction),
                      sampling_ladder=np.asarray(ladder, float), sampling_order=np.asarray(order),
                      sampling_kernel=np.asarray(field.method),
                      sampling_core_radius=np.asarray(np.nan if core_radius is None else core_radius),
                      sampling_area_quadrature=np.asarray(area_quadrature),
                      sampling_fan_order=np.asarray(fan_order),
                      sampling_fan_strips=np.asarray(fan_strips),
                      sampling_quadrature_base=np.asarray(_plan_label(plan['base'])),
                      sampling_quadrature_refined=np.asarray(_plan_label(plan['refined'])),
                      sampling_fan_refinement_order=np.asarray(
                          -1 if fan_refinement_order is None else int(fan_refinement_order)),
                      sampling_fan_refinement_strips=np.asarray(
                          -1 if fan_refinement_strips is None else int(fan_refinement_strips)),
                      sampling_near_ratio=np.asarray(float(near_ratio)),
                      sampling_adaptive_far_order=np.asarray(int(adaptive_far_order)),
                      sampling_adaptive_far_levels=np.asarray(int(adaptive_far_levels)),
                      sampling_adaptive_budget=np.asarray(float(adaptive_budget)),
                      sampling_adaptive_max_escalations=np.asarray(
                          int(adaptive_max_escalations)),
                      sampling_analytic_backend=np.asarray(analytic_backend),
                      sampling_analytic_device=np.asarray(
                          getattr(field, 'analytic_device', 'cpu')),
                      sampling_analytic_chunk=np.asarray(int(analytic_chunk)))
        record = {k: state[k] for k in (*load_names, 'wall_rms', 'jump_rms',
                  'pressure_kutta_rms', 'signed_kutta_rms',
                  'linear_constrained_rank', 'linear_singular_min', 'nonlinear')}
        record.update(iteration=iteration, velocity_method=velocity_method,
                      solve_backend=solve_backend, solve_device=solve_device,
                      pressure_solver=(None if kutta != "pressure" else
                                       (pressure_solver or
                                        ("structured" if solve_backend == "torch"
                                         else "reference"))),
                      analytic_backend=analytic_backend,
                      analytic_device=getattr(field, 'analytic_device', 'cpu'),
                      area_quadrature=area_quadrature, fan_order=fan_order,
                      fan_strips=fan_strips, quadrature_plan=plan,
                      core_radius=core_radius, strength_solve_kernel="unregularized_P1_BEM_operators",
                      groups=state['group_results'],
                      CLp_minus_CLgamma=float(state['CL']-state['CL_gamma']),
                      CL_relative_gap=float((state['CL']-state['CL_gamma'])/
                                            max(abs(state['CL_gamma']), 1e-12)),
                      wakes={}, incoming_update=incoming)
        constraint = ops['J'] if kutta == 'pressure' else np.vstack((ops['J'], ops['K']))
        if solve_backend == "torch":
            import torch as _torch
            _c = _torch.as_tensor(np.asarray(constraint, float), dtype=_torch.float64,
                                  device=solve_device)
            sv = _torch.linalg.svdvals(_c).detach().cpu().numpy()
        else:
            sv = np.linalg.svd(constraint, compute_uv=False)
        record['constraint_rank'] = int(np.sum(sv > 1e-11*sv[0]))
        record['constraint_singular_min'] = float(sv[-1])
        targets, counts, step_scales = {}, {}, {}
        all_eligible, map_residual = True, 0.
        for gi, group in enumerate(current.groups):
            mat = materials[group.name]
            group_scale = wake_step_scale(group.wake_xyz, group.wake_faces, mat)
            step_scales[group.name] = group_scale
            tri, normal, area, _shape, edges = triangle_data(group.wake_xyz, group.wake_faces)
            face_samples = limits(field, tri.mean(axis=1), normal, edges.min(axis=1),
                                  eps_fraction=eps_fraction, ladder=ladder, order=order,
                                  tag="wake_face", audit_quadrature=audit_quadrature)
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
                             scales[free_nodes], eps_fraction=eps_fraction, ladder=ladder, order=order,
                             tag="free_node", audit_quadrature=audit_quadrature)
            velocity = np.zeros_like(group.wake_xyz)
            velocity[free_nodes] = samples['mean']
            finite_limit = ((samples['epsilon_error'] <= epsilon_tol)
                            & (samples['quadrature_error'] <= quadrature_tol))
            fast = np.linalg.norm(samples['mean'], axis=1) > 1e-6
            # Sampling-error thresholds are an audit of the integral, not a
            # selection rule for which nodes may move: freezing a subset of one
            # sheet distorts the wake and changes the loads (measured 1.7-3.9x
            # on ten wings). Production defaults to 'audit'; 'enforce' keeps the
            # legacy freeze for reproducing pre-2026-10-31 readings.
            admitted = finite_limit if gate_mode in ('enforce', 'safe') else fast
            eligible = np.zeros(len(velocity), bool)
            eligible[free_nodes] = admitted & fast
            # Opposed endpoint velocities have an undefined average tangent.
            tangent_rejections = 0
            for node in mat.order:
                pred = int(mat.predecessor[node])
                if (pred >= 0 and mat.predecessor[pred] >= 0 and eligible[node]
                        and eligible[pred] and np.linalg.norm(velocity[node]+velocity[pred]) <= 2e-6):
                    eligible[node] = False
                    tangent_rejections += 1
            all_eligible &= bool(np.all(eligible[free_nodes]))
            target = streamline_target(group.wake_xyz, mat, velocity, eligible,
                                       step_scale=group_scale,
                                       truncation_policy=truncation_policy)
            map_residual = max(map_residual, float(np.max(np.linalg.norm(
                              target[mat.used]-group.wake_xyz[mat.used], axis=1))/mesh.reference_chord))
            targets[group.name] = target
            counts[group.name] = dict(root=len(mat.roots), eligible=int(eligible.sum()),
                                      convergence=int((~finite_limit).sum()),
                                      speed=int((finite_limit & ~fast).sum())+tangent_rejections,
                                      rejected=int((~eligible[free_nodes]).sum()),
                                      gate_mode=gate_mode,
                                      truncation_policy=truncation_policy,
                                      truncation_nodes=int(group_scale['truncation'].sum()),
                                      audit_rejected=int((~finite_limit).sum()),
                                      advanced_despite_audit=int(
                                          np.count_nonzero(admitted & ~finite_limit)),
                                      production_safety=(
                                          'sampling_converged' if gate_mode in ('enforce', 'safe')
                                          else 'sampling_not_gated'))
            diag.update(edge_integration=field.method,
                        edge_quadrature_error_interpretation=(
                            "not_applicable_exact_straight_edge_integrals"
                            if field.method == "analytic_edges" else
                            "projection_fan_base_vs_refined_fan"
                            if field.method == "rm_regularized_p1" else "48_vs_96_gauss"),
                        area_min=float(area.min()), aspect_p90=float(np.percentile(
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
        # Under 'audit' every finite node moves, so all_eligible is trivially
        # true and this reduces to the whole-sheet test; under 'enforce' it also
        # requires that no node was frozen.
        candidate = bool(all_eligible and face_checks
                         and map_residual <= geometry_tol and nonlinear_ok)
        record['kinematic_candidate'] = candidate
        record['state_sha256'] = snapshot_hash(arrays)
        record['outgoing_update'] = None
        stop = iteration == iterations or candidate or not nonlinear_ok
        if candidate:
            status = 'kinematic_candidate_not_physical_acceptance'
        elif not nonlinear_ok:
            status = ("blocked_initial_nonlinear_solve" if iteration == 0
                      else "blocked_updated_nonlinear_solve")
            record["nonlinear_failure_phase"] = ("initial_fixed_wake" if iteration == 0
                                                 else "after_accepted_geometry_update")
        if not stop:
            update = dict(accepted=False, relaxation=None, trial_reasons=[],
                          eligible=counts, displacement_max=0., displacement_rms=0.,
                          control='per_node_step_cap', global_backtracks=0,
                          clip={}, gate=None, control_nodes=None)
            proposed = None
            maximum = max(float(np.max(np.linalg.norm(targets[g.name]-g.wake_xyz, axis=1)))
                          for g in current.groups)
            if maximum <= 1e-15*mesh.reference_chord:
                status, stop = 'blocked_velocity_or_zero_update', True
            else:
                clipped, clipped_ledgers = {}, {}
                for g in current.groups:
                    mat = materials[g.name]
                    caps = gates.max_step_fraction*step_scales[g.name]['scale']
                    clipped[g.name], clipped_ledgers[g.name] = clip_step_update(
                        g.wake_xyz, targets[g.name], caps, roots=mat.roots,
                        reference=mat.reference_xyz)
                scale_maps = {name: s['scale'] for name, s in step_scales.items()}
                update['step_scale'] = {}
                for group_now in current.groups:
                    name = group_now.name
                    s = step_scales[name]
                    node_caps = gates.max_step_fraction*s['scale']
                    requested = np.linalg.norm(targets[name]-group_now.wake_xyz, axis=1)
                    usable = node_caps > 0
                    update['step_scale'][name] = dict(
                        unit='geometric_edge',
                        link_median=(float(np.median(s['link'][s['link'] > 0]))
                                     if np.any(s['link'] > 0) else None),
                        link_max=float(s['link'].max()),
                        geometric_median=float(np.median(s['geometric'][np.isfinite(
                            s['geometric'])])),
                        truncation_nodes=int(s['truncation'].sum()),
                        requested_max=float(requested.max()) if len(requested) else None,
                        requested_over_cap_max=(float(np.max(requested[usable]/node_caps[usable]))
                                                if np.any(usable) else None))
                update['clip'] = {name: dict(ledger, clipped_ids=ledger['clipped_ids'][:20])
                                  for name, ledger in clipped_ledgers.items()}
                mode_none = all(ledger['mode'] == 'none'
                                for ledger in clipped_ledgers.values())
                update['control'] = 'none' if mode_none else 'per_node_step_cap'
                accepted_xyz = None
                audit = geometry_audit(current, clipped, materials, gates, declared=declared,
                                       step_scale=scale_maps)
                update['gate'] = audit['detail']
                if audit['reason'] is None:
                    accepted_xyz, update['relaxation'], update['accepted'] = clipped, 1.0, True
                else:
                    update['trial_reasons'].append(audit['reason'])
                    update['trial_details'] = [dict(backtrack=-1, reason=audit['reason'],
                                                    scale=1.0, gate=audit['detail'])]
                    for backtrack in range(max_backtracks+1):
                        factor = 2.**(-backtrack)
                        trial_xyz = {g.name: g.wake_xyz+factor*(clipped[g.name]-g.wake_xyz)
                                     for g in current.groups}
                        trial = geometry_audit(current, trial_xyz, materials, gates,
                                               declared=declared, step_scale=scale_maps)
                        if trial['reason'] is None:
                            accepted_xyz = trial_xyz
                            update.update(accepted=True, relaxation=factor,
                                          global_backtracks=backtrack,
                                          control=('none+bounded_global_fallback' if mode_none else
                                                   'per_node_step_cap+bounded_global_fallback'),
                                          gate=trial['detail'])
                            break
                        update['trial_reasons'].append(trial['reason'])
                        update['trial_details'].append(dict(
                            backtrack=backtrack, reason=trial['reason'], scale=factor,
                            gate=trial['detail']))
                    else:
                        # Report the LAST refusal, not the first audit, so the
                        # controlling nodes come from the trial that blocked.
                        update['gate'] = update['trial_details'][-1]['gate']
                if accepted_xyz is None:
                    update['control_nodes'] = _controlling_nodes(update['gate'], clipped_ledgers)
                    status, stop = 'blocked_geometry', True
                else:
                    ds = np.concatenate([np.linalg.norm(
                        accepted_xyz[g.name][materials[g.name].used]
                        -g.wake_xyz[materials[g.name].used], axis=1) for g in current.groups])
                    update.update(displacement_max=float(ds.max()),
                                  displacement_rms=float(np.sqrt(np.mean(ds**2))),
                                  moved_nodes=int(np.count_nonzero(ds > 0)))
                    proposed = accepted_xyz
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
                initial_equivalence_method="operator_gate_and_reused_canonical_state",
                seconds=time.perf_counter()-started, converged=False,
                kinematic_candidate=history[-1]['kinematic_candidate'],
                physical_accuracy_accepted=False,
                method='P1-coupled-rooted-material-arclength-Picard',
                gate_mode=gate_mode,
                kernel=("analytic_P1_straight_edges_with_shared_trace_cancellation"
                        if velocity_method == "analytic_edges" else
                        "RM_regularized_P1_edge_plus_interior_gradient"
                        if velocity_method == "rm_regularized_p1" else
                        "legacy_fixed_order_edge_gauss"),
                velocity_method=velocity_method, core_radius=core_radius,
                area_quadrature=area_quadrature, fan_order=fan_order,
                fan_strips=fan_strips,
                strength_solve_kernel="unregularized_P1_BEM_operators")


def replay_snapshot(arrays, expected_sha, *, velocity_method=None, core_radius=None,
                    quadrature_order=None, area_quadrature=None, fan_order=None,
                    fan_strips=None, fan_refinement_order=None,
                    fan_refinement_strips=None, near_ratio=None,
                    adaptive_far_order=None, adaptive_far_levels=None,
                    adaptive_budget=None, adaptive_max_escalations=None,
                    analytic_backend=None, analytic_device=None, analytic_chunk=None):
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
    # Legacy snapshots (without sampling_kernel) are intentionally still
    # replayable using their original 48-point Gauss integration.
    recorded_method = str(arrays['sampling_kernel']) if 'sampling_kernel' in arrays else "legacy_gauss"
    recorded_core = (float(arrays['sampling_core_radius'])
                     if 'sampling_core_radius' in arrays
                     and np.isfinite(arrays['sampling_core_radius']) else None)
    method = recorded_method if velocity_method is None else velocity_method
    selected_core = recorded_core if core_radius is None else float(core_radius)
    selected_order = int(arrays['sampling_order']) if quadrature_order is None else int(quadrature_order)
    recorded_area = str(arrays['sampling_area_quadrature']) if 'sampling_area_quadrature' in arrays else "legacy_duffy"
    recorded_fan_order = int(arrays['sampling_fan_order']) if 'sampling_fan_order' in arrays else 16
    recorded_fan_strips = int(arrays['sampling_fan_strips']) if 'sampling_fan_strips' in arrays else 8
    selected_area = recorded_area if area_quadrature is None else area_quadrature
    selected_fan_order = recorded_fan_order if fan_order is None else int(fan_order)
    selected_fan_strips = recorded_fan_strips if fan_strips is None else int(fan_strips)
    recorded_refine_order = (int(arrays['sampling_fan_refinement_order'])
                             if 'sampling_fan_refinement_order' in arrays
                             and int(arrays['sampling_fan_refinement_order']) > 0 else None)
    recorded_refine_strips = (int(arrays['sampling_fan_refinement_strips'])
                              if 'sampling_fan_refinement_strips' in arrays
                              and int(arrays['sampling_fan_refinement_strips']) > 0 else None)
    selected_refine_order = (recorded_refine_order if fan_refinement_order is None
                             else int(fan_refinement_order))
    selected_refine_strips = (recorded_refine_strips if fan_refinement_strips is None
                              else int(fan_refinement_strips))
    def _recorded(key, fallback):
        if key not in arrays:
            return fallback
        value = float(np.asarray(arrays[key]).reshape(-1)[0])
        return value

    selected_near = (float(near_ratio) if near_ratio is not None
                     else _recorded('sampling_near_ratio', 2.0))
    selected_far_order = (int(adaptive_far_order) if adaptive_far_order is not None
                          else int(_recorded('sampling_adaptive_far_order', 8)))
    selected_far_levels = (int(adaptive_far_levels) if adaptive_far_levels is not None
                           else int(_recorded('sampling_adaptive_far_levels', 1)))
    selected_budget = (float(adaptive_budget) if adaptive_budget is not None
                       else _recorded('sampling_adaptive_budget', 2e-4))
    selected_escalations = (int(adaptive_max_escalations)
                            if adaptive_max_escalations is not None
                            else int(_recorded('sampling_adaptive_max_escalations', 2)))
    # Snapshots written before the analytic backend existed recorded the 'loop' path,
    # which is still the default, so an old snapshot replays bitwise unchanged.
    recorded_backend = (str(arrays['sampling_analytic_backend'])
                        if 'sampling_analytic_backend' in arrays else "loop")
    recorded_device = (str(arrays['sampling_analytic_device'])
                       if 'sampling_analytic_device' in arrays else "cpu")
    recorded_chunk = (int(_recorded('sampling_analytic_chunk', 256)))
    selected_backend = recorded_backend if analytic_backend is None else str(analytic_backend)
    selected_device = recorded_device if analytic_device is None else str(analytic_device)
    selected_chunk = recorded_chunk if analytic_chunk is None else int(analytic_chunk)
    field = TotalVelocity(mesh, dict(mu=arrays['mu'], gamma=arrays['gamma'],
                                   alpha_deg=float(arrays['alpha_deg'])),
                          method=method, core_radius=selected_core,
                          area_quadrature=selected_area, fan_order=selected_fan_order,
                          fan_strips=selected_fan_strips,
                          fan_refinement_order=selected_refine_order,
                          fan_refinement_strips=selected_refine_strips,
                          near_ratio=selected_near, adaptive_far_order=selected_far_order,
                          adaptive_far_levels=selected_far_levels,
                          adaptive_budget=selected_budget,
                          adaptive_max_escalations=selected_escalations,
                          analytic_backend=selected_backend,
                          analytic_device=selected_device,
                          analytic_chunk=selected_chunk)
    plan = field.quadrature_plan(selected_order)
    kwargs = dict(eps_fraction=float(arrays['sampling_eps_fraction']),
                  ladder=arrays['sampling_ladder'], order=selected_order)
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
        samples = dict(face=limits(field, tri.mean(axis=1), normal, edges.min(axis=1),
                                   tag="replay_face", **kwargs),
                       node=limits(field, group.wake_xyz[free], normals[free], scale[free],
                                   tag="replay_node", **kwargs))
        for location, values in samples.items():
            for key, value in values.items():
                name = p+location+'_'+key
                equal &= bool(np.array_equal(value, arrays[name]))
                differences[name] = float(np.max(np.abs(value-arrays[name])))
    return dict(state_sha256=expected_sha, bitwise_equal=equal, max_differences=differences,
                recorded_method=recorded_method, evaluated_method=method,
                recorded_core_radius=recorded_core, evaluated_core_radius=selected_core,
                quadrature_order=selected_order,
                recorded_area_quadrature=recorded_area,
                evaluated_area_quadrature=selected_area,
                fan_order=selected_fan_order, fan_strips=selected_fan_strips,
                fan_refinement_order=selected_refine_order,
                fan_refinement_strips=selected_refine_strips,
                quadrature_plan=plan,
                analytic_backend=selected_backend,
                analytic_device=field.analytic_device,
                analytic_chunk=selected_chunk,
                recorded_analytic_backend=recorded_backend,
                replay_mode=("reproduce_recorded_samples" if method == recorded_method
                             and selected_core == recorded_core
                             and selected_order == int(arrays['sampling_order'])
                             and selected_area == recorded_area
                             and selected_fan_order == recorded_fan_order
                             and selected_fan_strips == recorded_fan_strips
                             and selected_refine_order == recorded_refine_order
                             and selected_refine_strips == recorded_refine_strips
                             and selected_backend == recorded_backend
                             and selected_device == recorded_device
                             and selected_chunk == recorded_chunk
                             else "alternate_velocity_on_frozen_state"),
                evaluation_finite=all(np.isfinite(v) for v in differences.values()),
                strength_solve_kernel="frozen_unregularized_P1_BEM_state",
                physical_accuracy_accepted=False)
