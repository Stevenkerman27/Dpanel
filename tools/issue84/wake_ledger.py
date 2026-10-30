"""Per-iteration node ledger for the G2 wake update (issue #84, stage B).

The B2 solver publishes aggregate counters only
(``wakes[group]["eligibility"]`` with root/eligible/convergence/speed/rejected).
This tool replays the same solve with a save callback and derives a
node-level ledger from the arrays the solver itself stores, so every counter
can be explained by a concrete node ID set instead of a total.

Definitions (copied from panel/p1_coupled_material_wake.py:559-564, where the
counters are incremented, not re-derived here):

* ``root``        -- nodes in ``material.roots`` (fixed to reference geometry).
* ``free``        -- ``used`` nodes that are not roots; only these are sampled.
* ``convergence`` -- free nodes whose finite-difference epsilon error exceeds
                     ``epsilon_tol`` OR whose quadrature-refinement error
                     exceeds ``quadrature_tol``. It is the sampling-rejection
                     set, not a solver-iteration counter.
* ``speed``       -- free nodes with a resolved sample but |v| <= 1e-6, plus
                     nodes rejected because the node and its predecessor have
                     opposed tangents.
* ``rejected``    -- free nodes not eligible = convergence + speed (+tangent).

Rows are emitted per (iteration, group, node) with the gate inputs, the
candidate Picard target and displacement, incident face area/aspect and the
distance to the nearest wall triangle before and after the accepted update.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from panel.p1_coupled_material_wake import (solve_coupled_material_wake, streamline_target,
                                           triangle_data)
from tools.issue84.wake_node_rejection import build_case


def _wall_distance(points, wall_xyz, wall_faces):
    """Exact point-to-wall-triangle distance, vectorized over wall faces."""
    tri = wall_xyz[wall_faces]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    out = np.empty(len(points))
    for index, p in enumerate(points):
        ab, ac, ap = b - a, c - a, p - a
        v = np.einsum("fi,fi->f", ap, ab) / np.maximum(np.einsum("fi,fi->f", ab, ab), 1e-300)
        w = np.einsum("fi,fi->f", ap, ac) / np.maximum(np.einsum("fi,fi->f", ac, ac), 1e-300)
        u = 1.0 - v - w
        d_plane = np.abs(np.einsum("fi,fi->f", p - a, np.cross(ab, ac))) / np.maximum(
            np.linalg.norm(np.cross(ab, ac), axis=1), 1e-300)
        inside = (u >= 0) & (v >= 0) & (w >= 0)
        best = np.where(inside, d_plane, np.inf)
        for p0, p1 in ((a, b), (b, c), (c, a)):
            seg = p1 - p0
            t = np.clip(np.einsum("fi,fi->f", p - p0, seg)
                        / np.maximum(np.einsum("fi,fi->f", seg, seg), 1e-300), 0.0, 1.0)
            closest = p0 + t[:, None] * seg
            best = np.minimum(best, np.linalg.norm(p - closest, axis=1))
        out[index] = float(best.min())
    return out


def _incident(prefix, arrays, faces, group_xyz):
    """Per-node incident face count, minimum area and maximum aspect."""
    tri, _normal, area, _shape, edges = triangle_data(group_xyz, faces)
    aspect = edges.max(axis=1) / np.maximum(edges.min(axis=1), 1e-300)
    count = np.zeros(len(group_xyz), int)
    area_min = np.full(len(group_xyz), np.inf)
    aspect_max = np.zeros(len(group_xyz))
    for k in range(3):
        np.add.at(count, faces[:, k], 1)
        np.minimum.at(area_min, faces[:, k], area)
        np.maximum.at(aspect_max, faces[:, k], aspect)
    return count, area_min, aspect_max


def _ledger_rows(iteration, mesh, record, arrays, wall_xyz, wall_faces, tol):
    rows = []
    sets = {}
    for gi, group in enumerate(mesh.groups):
        p = "g%d_" % gi
        xyz = np.asarray(arrays[p + "xyz"], float)
        faces = np.asarray(arrays[p + "faces"], int)
        roots = np.asarray(arrays[p + "roots"], int)
        used = np.asarray(arrays[p + "used"], int)
        predecessor = np.asarray(arrays[p + "predecessor"], int)
        order = np.asarray(arrays[p + "order"], int)
        reference_xyz = np.asarray(arrays[p + "reference_xyz"], float)
        eligible = np.asarray(arrays[p + "eligible"], bool)
        free = np.setdiff1d(used, roots)
        material = SimpleNamespace(roots=roots, used=used, predecessor=predecessor,
                                   order=order, reference_xyz=reference_xyz)
        velocity = np.zeros_like(xyz)
        # Node samples are stored positionally against the sorted free-node index,
        # not per global node ID.
        mean = np.asarray(arrays[p + "node_mean"], float)
        velocity[free] = mean
        eps_err = np.asarray(arrays[p + "node_epsilon_error"], float)
        quad_err = np.asarray(arrays[p + "node_quadrature_error"], float)
        eps_by_node = np.zeros(len(xyz))
        quad_by_node = np.zeros(len(xyz))
        eps_by_node[free] = eps_err
        quad_by_node[free] = quad_err
        target = streamline_target(xyz, material, velocity, eligible)
        displacement = np.linalg.norm(target - xyz, axis=1)
        count, area_min, aspect_max = _incident(p, arrays, faces, xyz)
        distance = _wall_distance(xyz, wall_xyz, wall_faces)
        finite_limit = ((eps_by_node <= tol["epsilon"])
                        & (quad_by_node <= tol["quadrature"]))
        speed_ok = np.linalg.norm(velocity, axis=1) > 1e-6
        is_free = np.zeros(len(xyz), bool)
        is_free[free] = True
        convergence = set(int(n) for n in free if not finite_limit[n])
        speed = set(int(n) for n in free if finite_limit[n] and not speed_ok[n])
        rejected = set(int(n) for n in free if not eligible[n])
        tangent = rejected - convergence - speed
        sets[group.name] = dict(
            total_nodes=int(len(xyz)), roots=[int(r) for r in roots],
            free=[int(n) for n in free],
            convergence=sorted(convergence), speed=sorted(speed),
            tangent=sorted(tangent), rejected=sorted(rejected),
            eligible=[int(n) for n in free if eligible[n]])
        for node in range(len(xyz)):
            rows.append(dict(
                iteration=int(iteration), group=group.name, node=int(node),
                predecessor=int(predecessor[node]), is_root=bool(node in set(roots.tolist())),
                is_free=bool(is_free[node]), eligible=bool(eligible[node]),
                epsilon_error=float(eps_by_node[node]) if is_free[node] else 0.0,
                quadrature_error=float(quad_by_node[node]) if is_free[node] else 0.0,
                epsilon_ok=bool(finite_limit[node]),
                speed=float(np.linalg.norm(velocity[node])),
                tangent_rejected=bool(node in tangent),
                xyz=[float(x) for x in xyz[node]],
                velocity=[float(x) for x in velocity[node]],
                target=[float(x) for x in target[node]],
                displacement=float(displacement[node]),
                incident_faces=int(count[node]),
                incident_area_min=float(area_min[node]) if np.isfinite(area_min[node]) else None,
                incident_aspect_max=float(aspect_max[node]),
                wall_distance=float(distance[node]),
            ))
    return rows, sets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-half-surface", nargs="+", required=True)
    parser.add_argument("--case", default="g2", choices=("g1", "g2"))
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--kutta", default="pressure", choices=("pressure", "signed_speed"))
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--epsilon-tol", type=float, default=0.02)
    parser.add_argument("--quadrature-tol", type=float, default=0.002)
    parser.add_argument("--boundary-only", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("runs/issue84/wake_ledger_g2.json"))
    args = parser.parse_args(argv)
    started = time.perf_counter()
    mesh, alpha_deg = build_case(args.main_half_surface, case=args.case, alpha_deg=args.alpha)
    collected = []
    tol = dict(epsilon=args.epsilon_tol, quadrature=args.quadrature_tol)

    def callback(iteration, current, state, record, arrays):
        rows, sets = _ledger_rows(iteration, current, record, arrays,
                                  current.xyz, current.faces, tol)
        counters = {}
        for group_name, diagnostics in record.get("wakes", {}).items():
            counters[group_name] = dict(diagnostics["eligibility"])
        collected.append(dict(iteration=int(iteration),
                              status=None, counters=counters,
                              map_residual_chord=record.get("map_residual_chord"),
                              kinematic_candidate=record.get("kinematic_candidate"),
                              outgoing_update=(None if record.get("outgoing_update") is None
                                               else {k: v for k, v in record["outgoing_update"].items()
                                                     if k != "trial_reasons"}),
                              trial_reasons=(record["outgoing_update"]["trial_reasons"]
                                             if record.get("outgoing_update") else []),
                              sets=sets, rows=rows))

    result = solve_coupled_material_wake(mesh, alpha_deg, kutta=args.kutta,
                                         iterations=args.iterations,
                                         max_nfev=args.max_nfev,
                                         epsilon_tol=args.epsilon_tol,
                                         quadrature_tol=args.quadrature_tol,
                                         save_callback=callback)
    seconds = time.perf_counter() - started

    for entry in collected:
        entry["intersections"] = {}
        for name, sets in entry["sets"].items():
            convergence = set(sets["convergence"])
            speed = set(sets["speed"])
            tangent = set(sets["tangent"])
            rejected = set(sets["rejected"])
            roots = set(sets["roots"])
            entry["intersections"][name] = dict(
                convergence_and_rejected=len(convergence & rejected),
                convergence_minus_rejected=len(convergence - rejected),
                rejected_minus_convergence=sorted(rejected - convergence),
                rejected_minus_convergence_count=len(rejected - convergence),
                speed_and_rejected=len(speed & rejected),
                tangent_and_rejected=len(tangent & rejected),
                convergence_and_speed=len(convergence & speed),
                roots_and_convergence=len(roots & convergence),
                eligible=len(sets["eligible"]), rejected=len(rejected),
                free=len(sets["free"]), convergence=len(convergence),
                speed=len(speed), tangent=len(tangent), roots=len(roots),
                total_nodes=sets["total_nodes"])

    payload = dict(schema="dpanel.issue84.wake_ledger.v1",
                   python=platform.python_version(), numpy=np.__version__,
                   case=args.case, alpha_deg=float(alpha_deg), kutta=args.kutta,
                   iterations=args.iterations, tolerances=tol,
                   status=result["status"], converged=bool(result["converged"]),
                   physical_accuracy_accepted=False,
                   wall_faces=int(len(mesh.faces)), seconds=seconds,
                   counter_definitions=dict(
                       root="len(material.roots) fixed to reference geometry",
                       eligible="free nodes passing epsilon, quadrature, speed and tangent gates",
                       convergence="free nodes failing epsilon_tol or quadrature_tol (sampling rejection)",
                       speed="free nodes with |v| <= 1e-6 plus opposed-tangent rejections",
                       rejected="free nodes not eligible = convergence + speed + tangent"),
                   increments_at=dict(
                       counters="panel/p1_coupled_material_wake.py:561-564 (per iteration inside the group loop)",
                       eligible_flag="panel/p1_coupled_material_wake.py:543-556",
                       tangent="panel/p1_coupled_material_wake.py:549-555",
                       update_acceptance="panel/p1_coupled_material_wake.py:604-630",
                       solve_status="panel/p1_coupled_material_wake.py:596-603"),
                   iterations_out=collected)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("status", result["status"], "converged", result["converged"],
          "seconds %.1f" % seconds)
    for entry in collected:
        print("iteration", entry["iteration"], "counters", json.dumps(entry["counters"]))
        for name, info in entry["intersections"].items():
            print("  ", name, json.dumps(dict(convergence=info["convergence"],
                                              rejected=info["rejected"],
                                              overlap=info["convergence_and_rejected"],
                                              extra=info["rejected_minus_convergence_count"])))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
