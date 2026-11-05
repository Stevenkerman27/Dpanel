"""Experimental clipping and one-direction-at-a-time refinement of P1 wakes."""
from dataclasses import replace

import numpy as np

from tools.wake_sensitivity_geometry import attachment_rows


def clip_wake(mesh, length):
    group = mesh.groups[0]
    top = attachment_rows(mesh)
    limit = float(group.wake_xyz[top["attached"], 0].max()) + length * mesh.reference_chord
    xyz = list(group.wake_xyz.copy())
    hats = list(group.gamma_hat.copy())
    faces, cuts = [], {}
    for face in group.wake_faces:
        polygon = []
        for a, b in zip(face, np.roll(face, -1)):
            a, b = int(a), int(b)
            inside_a, inside_b = xyz[a][0] <= limit, xyz[b][0] <= limit
            if inside_a:
                polygon.append(a)
            if inside_a != inside_b:
                edge = tuple(sorted((a, b)))
                if edge not in cuts:
                    t = (limit-xyz[a][0]) / (xyz[b][0]-xyz[a][0])
                    cuts[edge] = len(xyz)
                    xyz.append((1-t)*xyz[a]+t*xyz[b])
                    hats.append((1-t)*hats[a]+t*hats[b])
                polygon.append(cuts[edge])
        for k in range(1, len(polygon)-1):
            tri = [polygon[0], polygon[k], polygon[k+1]]
            p = np.asarray([xyz[i] for i in tri])
            if np.linalg.norm(np.cross(p[1]-p[0], p[2]-p[0])) > 1e-14:
                faces.append(tri)
    changed = replace(group, wake_xyz=np.asarray(xyz), gamma_hat=np.asarray(hats),
                      wake_faces=np.asarray(faces, int))
    return replace(mesh, groups=(changed,)), dict(
        length_chord=length, cut_plane_x=limit, new_edge_nodes=len(cuts),
        far_rule="finite P1 sheet ends at constant x=max(initial attachment x)+length*c; zero continuation beyond; inherited linear Gamma hats at cut",
        wall_unchanged=True, original_vertices_unchanged=True,
        gamma_dofs_unchanged=changed.gamma_hat.shape[1] == group.gamma_hat.shape[1])


def refine_wake(mesh, direction):
    group = mesh.groups[0]
    xyz = list(group.wake_xyz.copy())
    hats = list(group.gamma_hat.copy())
    marked = set()
    attached = set(attachment_rows(mesh)["attached"].tolist())
    axis = 0 if direction == "streamwise" else 1
    for face in group.wake_faces:
        edges = [(int(face[k]), int(face[(k+1) % 3])) for k in range(3)]
        candidates = [edge for edge in edges if not set(edge).issubset(attached)]
        if candidates:
            a, b = max(candidates, key=lambda e: abs(xyz[e[1]][axis]-xyz[e[0]][axis]))
            marked.add(tuple(sorted((a, b))))
    midpoint = {}
    for a, b in sorted(marked):
        midpoint[(a, b)] = len(xyz)
        xyz.append((xyz[a]+xyz[b])/2)
        hats.append((hats[a]+hats[b])/2)
    faces = []
    for face in group.wake_faces:
        vertices = [int(n) for n in face]
        mids = [midpoint.get(tuple(sorted((vertices[k], vertices[(k+1) % 3]))))
                for k in range(3)]
        count = sum(m is not None for m in mids)
        if count == 0:
            faces.append(vertices)
        elif count == 1:
            k = next(i for i in range(3) if mids[i] is not None)
            a, b, c = [vertices[(k+i) % 3] for i in range(3)]
            faces.extend(((a, mids[k], c), (mids[k], b, c)))
        elif count == 2:
            k = next(i for i in range(3) if mids[i] is not None and mids[(i+1) % 3] is not None)
            a, b, c = [vertices[(k+i) % 3] for i in range(3)]
            ab, bc = mids[k], mids[(k+1) % 3]
            faces.extend(((ab, b, bc), (a, ab, c), (ab, bc, c)))
        else:
            a, b, c = vertices
            ab, bc, ca = mids
            faces.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
    changed = replace(group, wake_xyz=np.asarray(xyz), gamma_hat=np.asarray(hats),
                      wake_faces=np.asarray(faces, int))
    return replace(mesh, groups=(changed,)), dict(
        direction=direction, levels=1, split_edges=len(midpoint),
        rule="conforming midpoint split of each face's largest projected edge in the selected axis; attachment edges retained",
        off_axis_note="unstructured diagonal edges have both x/y projections; report both spacing distributions rather than claim a tensor-grid refinement",
        original_vertices_unchanged=True, gamma_basis="exact inherited P1 midpoint values",
        gamma_dofs_unchanged=changed.gamma_hat.shape[1] == group.gamma_hat.shape[1])
