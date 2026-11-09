"""Separate-bank potential-gradient recovery at matched trailing-edge points.

Area-weighted one-ring recovery is the production option. The quadratic QR fit
is an explicit research control; paired convergence rejects it on coarse grids.
Both use physical P1 DOFs without crossing the wake cut or an end cap.
"""
import torch


def matched_te_velocity_maps(mesh, xyz=None, *, device="cpu", method="area", max_rings=5):
    """Return upper/lower (station,3,wall_dof) maps and patch rank diagnostics."""
    xyz = torch.as_tensor(mesh.xyz, dtype=torch.float64, device=device) if xyz is None else xyz
    faces = torch.as_tensor(mesh.faces, dtype=torch.long, device=xyz.device)
    dofs = torch.as_tensor(mesh.dof_face, dtype=torch.long, device=xyz.device)
    triangles = xyz[faces]
    area_vector = torch.cross(triangles[:, 1] - triangles[:, 0],
                              triangles[:, 2] - triangles[:, 0], dim=1)
    area2 = torch.linalg.vector_norm(area_vector, dim=1)
    normals = area_vector / area2[:, None]
    g1 = torch.cross(triangles[:, 2] - triangles[:, 0], normals, dim=1) / area2[:, None]
    g2 = torch.cross(normals, triangles[:, 1] - triangles[:, 0], dim=1) / area2[:, None]
    face_gradient = torch.stack((-g1 - g2, g1, g2), dim=1)
    if method not in ("area", "quadratic"):
        raise ValueError("TE recovery method must be area or quadratic")
    dof_to_geo = torch.zeros(mesh.ndof_mu, dtype=torch.long, device=xyz.device)
    dof_to_geo[dofs.reshape(-1)] = faces.reshape(-1)
    dof_xyz = xyz[dof_to_geo]
    maps, receipt = [[], []], []
    for group in mesh.groups:
        for station, pair in enumerate(group.te_pairs):
            upper, lower = map(int, pair)
            point = dof_xyz[upper]
            for side, (bank, anchor) in enumerate(((1, upper), (-1, lower))):
                bank_faces = [int(f) for f in group.face_ids if mesh.face_bank[int(f)] == bank]
                incident = [f for f in bank_faces if anchor in mesh.dof_face[f]]
                if not incident:
                    raise ValueError("TE recovery is missing bank-specific incident faces")
                if method == "area":
                    weights = area2[incident] / area2[incident].sum()
                    contribution = weights[:, None, None] * face_gradient[incident]
                    operator = xyz.new_zeros((3, mesh.ndof_mu)).index_add(
                        1, dofs[incident].reshape(-1), contribution.reshape(-1, 3).T)
                    maps[side].append(operator)
                    receipt.append(dict(group=group.name, station=station, bank=bank,
                                        method=method, faces=len(incident)))
                    continue
                # Only the quadratic fit needs a local span tangent.
                te_points = dof_xyz[torch.as_tensor(group.te_pairs[:, 0], device=xyz.device)]
                if len(te_points) < 2:
                    raise ValueError("Quadratic TE recovery needs two interior TE stations")
                left, right = max(0, station - 1), min(len(te_points) - 1, station + 1)
                span = te_points[right] - te_points[left]
                span = span / torch.linalg.vector_norm(span)
                normal = area_vector[incident].sum(dim=0)
                normal = normal / torch.linalg.vector_norm(normal)
                tangent = span - (span @ normal) * normal
                tangent = tangent / torch.linalg.vector_norm(tangent)
                chord = torch.cross(tangent, normal, dim=0)
                support = {anchor}
                for ring in range(1, max_rings + 1):
                    patch = [f for f in bank_faces if any(int(d) in support for d in mesh.dof_face[f])]
                    support.update(int(d) for f in patch for d in mesh.dof_face[f])
                    ids = torch.tensor(sorted(support), dtype=torch.long, device=xyz.device)
                    delta = dof_xyz[ids] - point
                    u, v = delta @ chord, delta @ tangent
                    hu, hv = u.abs().max(), v.abs().max()
                    if len(ids) < 6 or float(hu.detach()) <= 0 or float(hv.detach()) <= 0:
                        continue
                    u, v = u / hu, v / hv
                    design = torch.stack((torch.ones_like(u), u, v, u*u, u*v, v*v), dim=1)
                    singular = torch.linalg.svdvals(design.detach())
                    ratio = float(singular[-1] / singular[0])
                    if ratio > 1e-8:
                        break
                else:
                    raise ValueError("TE recovery patch is rank deficient after %d rings" % max_rings)
                q, r = torch.linalg.qr(design, mode="reduced")
                fit = torch.linalg.solve_triangular(r, q.T, upper=True)
                gradient = chord[:, None] * fit[1][None, :] / hu \
                    + tangent[:, None] * fit[2][None, :] / hv
                operator = xyz.new_zeros((3, mesh.ndof_mu)).index_add(1, ids, gradient)
                # Fit differences from the TE value so an arbitrary potential
                # gauge cancels before the polynomial solve.
                operator[:, anchor] = operator[:, anchor] - gradient.sum(dim=1)
                maps[side].append(operator)
                receipt.append(dict(
                    group=group.name, station=station, bank=bank, rings=ring,
                    support=len(ids), singular_ratio=ratio,
                    chord_extent=float(hu.detach()), span_extent=float(hv.detach())))
    return torch.stack(maps[0]), torch.stack(maps[1]), receipt
