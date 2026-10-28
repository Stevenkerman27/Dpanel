"""Check prescribed wake triangles against the actual piecewise planar wall."""
import numpy as np


def check_wake_geometry(points, faces, wake_faces, require_positive_z=True):
    triangles, wake = points[faces], points[wake_faces]
    cross = np.cross(wake[:, 1] - wake[:, 0], wake[:, 2] - wake[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    assert np.all(area2 > 0), "Degenerate moved wake triangle"
    normals = cross / area2[:, None]
    if require_positive_z:
        assert np.all(normals[:, 2] > 0), "Moved wake leaves the positive-z graph parameterization"
    plane_offset = np.sum(normals * wake[:, 0], axis=1)
    planes, group = np.unique(np.round(np.column_stack((normals, plane_offset)), 11), axis=0, return_inverse=True)
    segment_count = 0
    for number in range(len(planes)):
        selected = np.flatnonzero(group == number)
        n = normals[selected[0]]
        center = wake[selected[0], 0]
        l = wake[selected[0], 1] - center
        l /= np.linalg.norm(l)
        m = np.cross(n, l)
        wall_plane = np.stack(((triangles - center) @ l, (triangles - center) @ m,
                               (triangles - center) @ n), axis=-1)
        sheet = np.stack(((wake[selected] - center) @ l, (wake[selected] - center) @ m,
                          (wake[selected] - center) @ n), axis=-1)
        assert np.max(abs(sheet[:, :, 2])) < 1e-9, "Wake plane grouping is unresolved"
        segments = []
        for tri in wall_plane:
            z = tri[:, 2]
            on_plane = abs(z) <= 1e-12
            assert on_plane.sum() < 3, "Wall patch lies in prescribed wake plane"
            if z.min() > 1e-12 or z.max() < -1e-12:
                continue
            intersection = [point[:2] for point in tri[on_plane]]
            for k, q in ((0, 1), (1, 2), (2, 0)):
                if not on_plane[k] and not on_plane[q] and z[k] * z[q] < 0:
                    intersection.append((tri[k] + z[k] / (z[k] - z[q]) * (tri[q] - tri[k]))[:2])
            if len(intersection) == 2:
                segments.append(intersection)
        segments = np.asarray(segments).reshape(-1, 2, 2)
        segment_count += len(segments)
        a = sheet[:, 0, :2]
        e1, e2 = sheet[:, 1, :2] - a, sheet[:, 2, :2] - a
        den = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        for first in range(0, len(segments), 128):
            delta = segments[first:first + 128, :, None] - a[None, None]
            u = (delta[:, :, :, 0] * e2[None, None, :, 1] - delta[:, :, :, 1] * e2[None, None, :, 0]) / den
            v = (e1[None, None, :, 0] * delta[:, :, :, 1] - e1[None, None, :, 1] * delta[:, :, :, 0]) / den
            bary = np.stack((1 - u - v, u, v), axis=-1)
            start, slope = bary[:, 0], bary[:, 1] - bary[:, 0]
            bounds = np.divide(-start, slope, out=np.zeros_like(slope), where=slope != 0)
            enter = np.maximum(0., np.max(np.where(slope > 0, bounds, -np.inf), axis=-1))
            leave = np.minimum(1., np.min(np.where(slope < 0, bounds, np.inf), axis=-1))
            middle = start + ((enter + leave) / 2)[:, :, None] * slope
            assert not np.any((leave > enter) & np.all(middle > 1e-10, axis=-1)), "Prescribed wake crosses solid wall"
    winding_max = 0.
    for first in range(0, len(wake), 128):
        r = triangles[None] - wake.mean(axis=1)[first:first + 128, None, None]
        rad = np.linalg.norm(r, axis=-1)
        r0, r1, r2 = r[:, :, 0], r[:, :, 1], r[:, :, 2]
        det = np.sum(r0 * np.cross(r1, r2), axis=-1)
        den = rad.prod(axis=-1) + np.sum(r0 * r1, axis=-1) * rad[:, :, 2]
        den += np.sum(r1 * r2, axis=-1) * rad[:, :, 0] + np.sum(r2 * r0, axis=-1) * rad[:, :, 1]
        winding_max = max(winding_max, float(np.max(abs(np.sum(2 * np.arctan2(det, den), axis=1) / (4 * np.pi)))))
    assert winding_max < 1e-8, "Prescribed wake has face centers inside a solid"
    # Check open-interior crossings of noncoplanar wake triangles. Coplanar
    # overlap and tangential touching are outside this certificate.
    self_crossings = 0
    low, high = wake.min(axis=1), wake.max(axis=1)
    for first in range(len(wake)):
        candidates = np.flatnonzero((np.arange(len(wake)) > first)
                                   & np.all(high >= low[first] - 1e-12, axis=1)
                                   & np.all(low <= high[first] + 1e-12, axis=1))
        candidates = candidates[~np.any(wake_faces[candidates, :, None] == wake_faces[first, None, :], axis=(1, 2))]
        for a, selected in ((wake[first], wake[candidates]),):
            if not len(selected):
                continue
            # Test both directions so a small triangle cutting a large one is detected.
            for reverse in (False, True):
                target = selected if reverse else np.broadcast_to(a, selected.shape)
                edges = np.broadcast_to(a, selected.shape) if reverse else selected
                e1, e2 = target[:, 1] - target[:, 0], target[:, 2] - target[:, 0]
                for k, q in ((0, 1), (1, 2), (2, 0)):
                    direction = edges[:, q] - edges[:, k]
                    p = np.cross(direction, e2)
                    determinant = np.sum(e1 * p, axis=1)
                    active = abs(determinant) > 1e-12 * np.linalg.norm(e1, axis=1) * np.linalg.norm(p, axis=1)
                    delta = edges[:, k] - target[:, 0]
                    inverse = np.divide(1., determinant, out=np.zeros_like(determinant), where=active)
                    u = np.sum(delta * p, axis=1) * inverse
                    cross_delta = np.cross(delta, e1)
                    v = np.sum(direction * cross_delta, axis=1) * inverse
                    t = np.sum(e2 * cross_delta, axis=1) * inverse
                    self_crossings += int(np.count_nonzero(active & (u > 1e-10) & (v > 1e-10)
                                                          & (u + v < 1 - 1e-10) & (t > 1e-10) & (t < 1 - 1e-10)))
    assert not self_crossings, "Nonadjacent wake triangles cross in their open interiors"
    return dict(plane_groups=len(planes), wall_slice_segments=segment_count, intersections=0,
                nonadjacent_noncoplanar_wake_crossings=self_crossings, positive_z_graph_required=require_positive_z,
                maximum_abs_centroid_winding=winding_max,
                scope="Triangle wall crossings and centroid winding; wake transverse crossings only; no coplanar overlap/tangential-touch or wall self-intersection certificate")
