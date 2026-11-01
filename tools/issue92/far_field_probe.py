"""Distance-band accuracy of cheap far-field surrogates for the fan area kernel.

The fan is accurate near the sheet and expensive everywhere: it evaluates order^2
nodes on every sub-triangle of every face. A near-field restriction only pays if the
faces outside the restriction can be integrated cheaply without losing the accuracy
the fan was introduced for. This tool measures, per distance band, how far the two
cheap candidates are from a fine fan reference:

* a monopole moment of each face, A (p - c) / (|p - c|^2 + a^2)^(3/2),
* the shipped single-Duffy map at a low order.

Nothing here is wired into production; it decides whether a hybrid evaluator is
worth building, and with which far-field model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from panel.p1_projection_duffy import area_terms, area_velocity
from panel.p1_regularized_sheet_velocity import RegularizedP1SheetField


def _sheet(case):
    """Wake sheet geometry plus a smooth synthetic scalar P1 potential on it."""
    from tools.issue84.wake_node_rejection import build_case

    mesh, _alpha = build_case([case])
    group = mesh.groups[0]
    xyz = np.asarray(group.wake_xyz, float)
    faces = np.asarray(group.wake_faces, int)
    mu = 1e-3 * (np.sin(2.0 * xyz[:, 1]) + np.cos(1.5 * xyz[:, 0]))
    return xyz, faces, mu


def _monopole_terms(tri, point, core):
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    centre = tri.mean(axis=1)
    r = point[None, :] - centre
    d2 = np.einsum("fi,fi->f", r, r) + core * core
    return (area / (d2 * np.sqrt(d2)))[:, None] * r


def _velocity(gamma_faces, terms):
    """Area velocity: sum over faces of gamma_f x (the face's area integral)."""
    return np.sum(np.cross(gamma_faces, terms), axis=0)


def measure(xyz, faces, mu, bands, *, core, reference_order=48, reference_strips=1):
    field = RegularizedP1SheetField([(xyz, faces, mu[faces])], core_radius=core)
    tri = field.triangles
    gamma = field.gamma
    centre = tri.reshape(-1, 3).mean(axis=0)
    extents = np.linalg.norm(tri.reshape(-1, 3) - centre, axis=1)
    records = []
    for ratio in bands:
        distance = ratio * core
        downstream = np.array([tri.reshape(-1, 3)[:, 0].max() + 1.0, centre[1], centre[2]])
        for point, name in ((centre + np.array([0.0, 0.0, distance]), "off_plane"),
                            (centre - np.array([0.0, 0.0, distance]), "below"),
                            (downstream + np.array([0.0, 0.0, distance]), "downstream")):
            point = np.asarray(point, float)
            ref_terms = area_terms(tri, point, core, order=reference_order,
                                   strips=reference_strips)
            ref = _velocity(gamma, ref_terms)
            ref_norm = float(np.linalg.norm(ref))
            row = {"r_over_core": ratio, "direction": name, "distance": float(distance),
                   "reference_velocity": float(ref_norm)}
            for label, terms in (
                ("monopole", _monopole_terms(tri, point, core)),
                ("fan16x1", area_terms(tri, point, core, order=16, strips=1)),
                ("fan24x1", area_terms(tri, point, core, order=24, strips=1)),
                ("fan32x1", area_terms(tri, point, core, order=32, strips=1)),
            ):
                got = _velocity(gamma, terms)
                row[label] = float(np.linalg.norm(got - ref) / max(ref_norm, 1e-300))
            for order in (8, 16, 24):
                got = field._area_velocity(point, order)
                row["legacy_duffy%d" % order] = float(
                    np.linalg.norm(got - ref) / max(ref_norm, 1e-300))
            row["face_extent_median"] = float(np.median(extents))
            # share of faces a near-field restriction would keep: this is what decides
            # whether the hybrid is cheap, not the far-field error alone
            r = point[None, :] - tri.mean(axis=1)
            d = np.linalg.norm(r, axis=1)
            row["faces_within_10_core"] = int(np.count_nonzero(d < 10.0 * core))
            row["faces_within_30_core"] = int(np.count_nonzero(d < 30.0 * core))
            row["face_count"] = int(len(tri))
            records.append(row)
            print("r/core %6.1f %-9s ref %.3e | mono %.2e fan16 %.2e fan32 %.2e legacy8 %.2e legacy16 %.2e"
                  % (ratio, name, ref_norm, row["monopole"], row["fan16x1"],
                     row["fan32x1"], row["legacy_duffy8"], row["legacy_duffy16"]))
    return records


def near_share(case, *, core, ratios=(1.0, 3.0, 10.0, 30.0)):
    """Faces within a multiple of the core radius of the real sampler targets.

    The sampler evaluates at free wake nodes offset by 1e-3 of the node scale, and
    distance is taken to the nearest vertex of each face: a target sits next to its
    own vertex, so its incident faces are near even though their centroids are half
    a face away.
    """
    import numpy as np

    from panel.p1_coupled_material_wake import node_step_scale
    from panel.p1_steady_free_wake import prepare_wake_material

    from tools.issue84.wake_node_rejection import build_case

    mesh, _alpha = build_case([case])
    group = mesh.groups[0]
    xyz = np.asarray(group.wake_xyz, float)
    faces = np.asarray(group.wake_faces, int)
    tri = xyz[faces]
    scale = np.nan_to_num(np.asarray(node_step_scale(xyz, faces), float))
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    node_normal = np.zeros_like(xyz)
    np.add.at(node_normal, faces.ravel(), np.repeat(normals, 3, axis=0))
    node_normal /= np.maximum(np.linalg.norm(node_normal, axis=1)[:, None], 1e-300)
    roots = set(np.asarray(prepare_wake_material(group).roots).tolist())
    free = [i for i in range(len(xyz)) if i not in roots]
    targets = xyz[free] + 1e-3 * scale[free][:, None] * node_normal[free]
    rows = []
    for ratio in ratios:
        distance = np.linalg.norm(targets[:, None, None, :] - tri[None, :, :, :], axis=3)
        near = (distance.min(axis=2) < ratio * core).sum(axis=1)
        row = {"r_over_core": float(ratio),
               "median": int(np.median(near)),
               "p90": int(np.percentile(near, 90)),
               "max": int(near.max()),
               "mean_share": float(near.mean() / len(tri)),
               "face_count": int(len(tri)),
               "targets": int(len(targets))}
        rows.append(row)
        print("R = %5.1f core  faces within: median %4d  p90 %4d  max %4d  (of %d)  mean share %.4f"
              % (ratio, row["median"], row["p90"], row["max"], len(tri), row["mean_share"]))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="runs/half_cad_wingbody_v1/surface.json")
    parser.add_argument("--core", type=float, default=0.01)
    parser.add_argument("--bands", type=float, nargs="+",
                        default=(3.0, 10.0, 30.0, 100.0, 300.0))
    parser.add_argument("--out", type=Path, default=Path("docs/issue92/far_field_bands.json"))
    parser.add_argument("--near-share", action="store_true",
                        help="also report how many faces a near-field restriction keeps")
    args = parser.parse_args(argv)
    from tools.issue92.fixtures import _first_existing

    xyz, faces, mu = _sheet(_first_existing(args.case))
    records = measure(xyz, faces, mu, args.bands, core=args.core)
    payload = {"case": str(args.case), "core_radius": args.core,
               "triangle_count": int(len(faces)), "records": records}
    if args.near_share:
        payload["near_share"] = near_share(_first_existing(args.case), core=args.core)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
