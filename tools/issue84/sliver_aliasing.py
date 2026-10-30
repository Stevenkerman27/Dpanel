"""Item C3 follow-up: the near-sheet aliasing source is sliver triangles, not *distance*.

Earlier (docs/ISSUE84_RM_BENCH.md section 3) I reported that no single quadrature
order converges near the sheet and that order 32 is the worst.  Re-measuring per
part shows the effect is owned by a handful of 520:1 sliver triangles in the
main-wing wake sheet (edges 0.05 / 26.0 / 26.0).  For them the fixed
vertex-collapsed Duffy rule aliases in (u, v) because the kernel peak is
compressed by the triangle aspect ratio; the wall part and the other wake group
are exact to 1e-17 at the same point.

The cure is anisotropic refinement, i.e. longest-edge bisection until the
sub-triangle diameter is comparable to the core radius, which is cheap because a
sliver only needs ~log2(L/core) bisections along its long direction.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("PANEL_ROOT", Path.cwd()))
sys.path.insert(0, str(ROOT / "runs/pr85_git"))

import numpy as np

from tools.issue84.wake_node_rejection import build_case
from panel.p1_lifting_g2 import assemble_multi, solve_multi
from panel.p1_coupled_material_wake import TotalVelocity
from panel.p1_regularized_sheet_velocity import RegularizedP1SheetField

CORE = 0.01
ORDERS = (8, 16, 32, 48, 96)
REF_ORDER = 192


def quadrature(n):
    q, w = np.polynomial.legendre.leggauss(n)
    q, w = (q + 1.0) / 2.0, w / 2.0
    u, v = np.meshgrid(q, q, indexing="ij")
    weights = (w[:, None] * w[None, :] * (1.0 - u)).reshape(-1)
    bary = np.stack(((1.0 - u) * (1.0 - v), u, (1.0 - u) * v), axis=-1).reshape(-1, 3)
    return bary, weights


def triangle_weights(triangles, area2, point, n, core=CORE):
    """Per-triangle area-integrand contribution, shape (n_tri, 3)."""
    bary, weights = quadrature(n)
    src = np.einsum("qv,fvi->fqi", bary, triangles)
    r = point[None, None, :] - src
    den = (np.einsum("fqi,fqi->fq", r, r) + core ** 2) ** 1.5
    return np.einsum("fq,fqi->fi", weights[None, :] / den, r) * area2[:, None]


def edge_lengths(t):
    return np.sort(np.array([np.linalg.norm(t[1] - t[0]),
                             np.linalg.norm(t[2] - t[1]),
                             np.linalg.norm(t[0] - t[2])]))


def bisect(tri, tol):
    """Longest-edge bisection until every sub-triangle diameter is <= tol."""
    lengths = edge_lengths(tri)
    if lengths[2] <= tol:
        return [tri]
    if np.linalg.norm(tri[1] - tri[0]) == lengths[2]:
        a, b, c = tri[0], tri[1], tri[2]
    elif np.linalg.norm(tri[2] - tri[1]) == lengths[2]:
        a, b, c = tri[1], tri[2], tri[0]
    else:
        a, b, c = tri[2], tri[0], tri[1]
    m = 0.5 * (a + b)
    return bisect(np.stack([a, m, c]), tol) + bisect(np.stack([m, b, c]), tol)


def bisected_contribution(tri, point, tol, sub_order=8, core=CORE):
    subs = bisect(tri, tol)
    total = np.zeros(3)
    bary, weights = quadrature(sub_order)
    for s in subs:
        a2 = np.linalg.norm(np.cross(s[1] - s[0], s[2] - s[0]))
        src = bary @ s
        r = point - src
        den = (np.einsum("qi,qi->q", r, r) + core ** 2) ** 1.5
        total += np.einsum("q,qi->i", weights / den, r) * a2
    return total, len(subs)


def main():
    mesh, alpha = build_case([str(ROOT / "runs/half_cad_wingbody_v1/surface.json")])
    operators = assemble_multi(mesh)
    state = dict(solve_multi(mesh, alpha, kutta="pressure", operators=operators))
    snap = np.load(str(ROOT / "runs/issue84/g2_ref/iteration_000.npz"))
    state["mu"] = snap["mu"]
    state["gamma"] = snap["gamma"]
    field = TotalVelocity(mesh, state, method="rm_regularized_p1", core_radius=CORE)

    parts = [(mesh.xyz, mesh.faces, np.asarray(state["mu"])[mesh.dof_face])]
    offset = 0
    names = []
    for group in mesh.groups:
        ng = group.gamma_hat.shape[1]
        nodal = group.gamma_hat @ np.asarray(state["gamma"])[offset:offset + ng]
        parts.append((group.wake_xyz, group.wake_faces, nodal[group.wake_faces]))
        names.append(group.name)
        offset += ng
    part_fields = [RegularizedP1SheetField([p], core_radius=CORE) for p in parts]

    main = [g for g in mesh.groups if g.name == "main_wing"][0]
    V = np.asarray(main.wake_xyz, float)
    F = np.asarray(main.wake_faces, int)
    tri = V[F]
    centers = tri.mean(axis=1)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]

    out = {"core_radius": CORE, "orders": list(ORDERS), "ref_order": REF_ORDER,
           "metric": "max |dv| / |v| over the 12 sampled wake face centers",
           "bands": {}}
    face_idx = np.linspace(0, len(F) - 1, 12).astype(int)
    worst = (None, None, 0.0)
    for ratio in (0.3, 1.0):
        pts = np.asarray([centers[i] + ratio * CORE * normals[i] for i in face_idx])
        ref = np.stack([np.asarray(field(p, order=REF_ORDER)) for p in pts])
        rows = {}
        for order in ORDERS:
            v = np.stack([np.asarray(field(p, order=order)) for p in pts])
            err = np.linalg.norm(v - ref, axis=1) / np.linalg.norm(ref, axis=1)
            rows["order%d" % order] = {"max_rel": float(err.max()),
                                       "per_point": [float(x) for x in err]}
            for k, val in enumerate(err):
                if val > worst[2]:
                    worst = (k, ratio, float(val))
        out["bands"]["r/core=%.1f" % ratio] = rows
        print("r/core %.1f : %s" % (ratio, " ".join(
            "o%d %.2e" % (o, rows["order%d" % o]["max_rel"]) for o in ORDERS)))

    k, ratio, val = worst
    fid = int(face_idx[k])
    point = centers[fid] + ratio * CORE * normals[fid]
    print("\nworst point: band r/core=%.1f, wake face %d, rel %.3e" % (ratio, fid, val))

    decomposition = {}
    for name, pf in zip(["wall"] + names, part_fields):
        vals = {n: pf._area_velocity(point, n) for n in (8, 32, REF_ORDER)}
        decomposition[name] = {
            "abs_area8": float(np.linalg.norm(vals[8])),
            "abs_area32": float(np.linalg.norm(vals[32])),
            "abs_area192": float(np.linalg.norm(vals[REF_ORDER])),
            "abs_diff_32_192": float(np.linalg.norm(vals[32] - vals[REF_ORDER])),
            "abs_edge": float(np.linalg.norm(pf._edge_velocity(point))),
            "rel_8_vs_192": float(np.linalg.norm(vals[8] - vals[REF_ORDER])
                                  / max(np.linalg.norm(vals[REF_ORDER]), 1e-300)),
            "rel_32_vs_192": float(np.linalg.norm(vals[32] - vals[REF_ORDER])
                                   / max(np.linalg.norm(vals[REF_ORDER]), 1e-300)),
        }
        print("  %-8s |area32-192| %.3e |edge| %.3e rel32 %.3e"
              % (name, decomposition[name]["abs_diff_32_192"],
                 decomposition[name]["abs_edge"], decomposition[name]["rel_32_vs_192"]))
    out["decomposition"] = decomposition

    wake_name = names[[g.name for g in mesh.groups].index("main_wing")]
    pf = part_fields[1 + [g.name for g in mesh.groups].index("main_wing")]
    c32 = np.cross(pf.gamma, triangle_weights(pf.triangles, pf._triangle_area2, point, 32))
    c192 = np.cross(pf.gamma, triangle_weights(pf.triangles, pf._triangle_area2, point, REF_ORDER))
    diff = np.linalg.norm(c32 - c192, axis=1)
    top = np.argsort(-diff)[:6]
    sensitive = []
    for fi in top:
        t = pf.triangles[fi]
        L = edge_lengths(t)
        d_plane = abs(float(np.dot(point - t[0], pf.normals[fi])))
        d_edge = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            e = t[b] - t[a]
            w = point - t[a]
            s = float(np.clip(np.dot(w, e) / np.dot(e, e), 0.0, 1.0))
            d_edge.append(float(np.linalg.norm(w - s * e)))
        pieces = len(bisect(t, 4.0 * CORE))
        sensitive.append({
            "triangle": int(fi),
            "edges": [float(x) for x in L],
            "aspect": float(L[2] / L[0]),
            "diameter": float(L[2]),
            "d_plane": d_plane,
            "d_edges": d_edge,
            "abs_c32": float(np.linalg.norm(c32[fi])),
            "abs_c192": float(np.linalg.norm(c192[fi])),
            "ratio_32_over_192": float(np.linalg.norm(c32[fi]) / np.linalg.norm(c192[fi])),
            "bisect4core_pieces": int(pieces),
        })
        print("  tri %4d aspect %6.1f diam %6.3f  |c32|/|c192| %6.2f  bisect(4core) needs %5d pieces"
              % (fi, sensitive[-1]["aspect"], sensitive[-1]["diameter"],
                 sensitive[-1]["ratio_32_over_192"], pieces))
    out["worst_point"] = {"band": ratio, "wake_face": fid, "rel": val,
                          "sensitive_triangles": sensitive}
    total32 = np.sum(c32[top], axis=0)
    total192 = np.sum(c192[top], axis=0)
    out["top6_sum"] = {
        "abs_32": float(np.linalg.norm(total32)),
        "abs_192": float(np.linalg.norm(total192)),
        "ratio_32_over_192": float(np.linalg.norm(total32) / np.linalg.norm(total192)),
        "rel_32_vs_192": float(np.linalg.norm(total32 - total192)
                               / np.linalg.norm(total192)),
    }
    print("  top-6 sum: |32|/|192| = %.3f   rel32 %.3e"
          % (out["top6_sum"]["ratio_32_over_192"], out["top6_sum"]["rel_32_vs_192"]))

    def aspect_stats(triangles):
        t = np.asarray(triangles, float).reshape(-1, 3, 3)
        L = np.stack([np.linalg.norm(t[:, 1] - t[:, 0], axis=1),
                      np.linalg.norm(t[:, 2] - t[:, 1], axis=1),
                      np.linalg.norm(t[:, 0] - t[:, 2], axis=1)], axis=1)
        L.sort(axis=1)
        a = L[:, 2] / L[:, 0]
        return {"n": int(len(a)), "median": float(np.median(a)),
                "p90": float(np.percentile(a, 90)), "max": float(a.max()),
                "gt50": int((a > 50).sum()), "gt200": int((a > 200).sum()),
                "gt500": int((a > 500).sum())}

    out["aspect"] = {"wall": aspect_stats(np.asarray(part_fields[0].triangles))}
    for name, field in zip(names, part_fields[1:]):
        out["aspect"][name] = aspect_stats(np.asarray(field.triangles))
    print("\naspect ratios:", json.dumps(out["aspect"]))

    target = ROOT / "docs/issue84/sliver_aliasing.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2))
    print("wrote", target.relative_to(ROOT))


if __name__ == "__main__":
    main()