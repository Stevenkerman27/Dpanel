"""Tensor assembly of the P1 wall/wake influence operators (S4 of the torch port).

This is the assembly half of the Route B port: the wall doublet operator `A`, the wake
doublet contraction `W` and the face gradient map `grad`, evaluated with torch only. The
formulas are a line-by-line transcription of `panel/p1_lifting_g1.py::_doublet_matrix`,
`_gradient_map` and `_mesh_as_kernel` (which `p1_lifting_g2.py::_mesh` duplicates); the only
changes are batching over source triangles and scattering into DOF columns with
`scatter_add_` instead of item assignment. No NumPy/NumPy-LAPACK/SciPy call is allowed in
this module: a torch process that calls into LAPACK afterwards dies with 0xC0000005
(see docs/ISSUE92_PHASE1_WAKE_EVALUATOR.md 6f).

The self-face correction is kept as in production: for a target that is the centroid of the
source triangle itself the row is replaced by -B/2 with B = 1/3 + (X - centroid) . G^T,
which is the principal-value limit of the doublet sheet, not a quadrature rule.
"""
from __future__ import annotations

import torch


def _require_tensor(name, value):
    if not torch.is_tensor(value):
        raise TypeError("%s must be a torch tensor" % name)


def mesh_kernel(xyz, faces, *, device=None, dtype=torch.float64):
    """Triangle kernel of a surface mesh: centroids, normals, areas, basis gradients."""
    xyz = torch.as_tensor(xyz, dtype=dtype, device=device)
    faces = torch.as_tensor(faces, dtype=torch.long, device=device)
    tri = xyz[faces]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    raw = torch.linalg.cross(e1, e2)
    scale = torch.linalg.vector_norm(raw, dim=1)
    if bool((scale <= 0).any()):
        raise ValueError("Degenerate wing triangle")
    normal = raw / scale[:, None]
    g1 = torch.linalg.cross(e2, normal) / scale[:, None]
    g2 = torch.linalg.cross(normal, e1) / scale[:, None]
    gradients = torch.stack((-g1 - g2, g1, g2), dim=1)
    return dict(triangles=tri, centers=tri.mean(dim=1), normals=normal,
                areas=scale / 2, basis_gradients=gradients)


def doublet_matrix(triangles, targets, *, self_faces=None, face_dofs=None, ndof=None,
                   face_chunk=64, device=None, dtype=torch.float64):
    """Analytic P1 doublet operator, batched over source triangles.

    Returns a (len(targets), ndof) tensor whose columns are scattered by `face_dofs`
    (default: three consecutive DOFs per triangle). Equivalent to the NumPy
    `_doublet_matrix` including its near-singular edge guard.
    """
    tri = torch.as_tensor(triangles, dtype=dtype, device=device)
    X = torch.as_tensor(targets, dtype=dtype, device=device)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError("triangles must have shape (n, 3, 3)")
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("targets must have shape (m, 3)")
    n_face = int(tri.shape[0])
    n_target = int(X.shape[0])
    if face_dofs is None:
        dofs = torch.arange(3 * n_face, device=tri.device).reshape(n_face, 3)
    else:
        dofs = torch.as_tensor(face_dofs, dtype=torch.long, device=tri.device)
        if dofs.shape != (n_face, 3):
            raise ValueError("face_dofs must have shape (n, 3)")
    if ndof is None:
        ndof = int(dofs.max()) + 1
    if face_chunk < 1:
        raise ValueError("face_chunk must be positive")
    A = torch.zeros((n_target, int(ndof)), dtype=dtype, device=tri.device)
    self_index = None if self_faces is None else torch.as_tensor(
        self_faces, dtype=torch.long, device=tri.device)
    four_pi = 4.0 * torch.pi
    arange_all = torch.arange(n_face, device=tri.device)
    for f0 in range(0, n_face, face_chunk):
        f1 = min(n_face, f0 + face_chunk)
        t = tri[f0:f1]
        e1 = t[:, 1] - t[:, 0]
        e2 = t[:, 2] - t[:, 0]
        raw = torch.linalg.cross(e1, e2)
        scale = torch.linalg.vector_norm(raw, dim=1)
        if bool((scale <= 0).any()):
            raise ValueError("Zero-area source triangle")
        normal = raw / scale[:, None]
        g1 = torch.linalg.cross(e2, normal) / scale[:, None]
        g2 = torch.linalg.cross(normal, e1) / scale[:, None]
        G = torch.stack((-g1 - g2, g1, g2), dim=1)
        ctr = t.mean(dim=1)
        r = t[None, :, :, :] - X[:, None, None, :]
        L = torch.linalg.vector_norm(r, dim=3)
        if bool((L == 0).any()):
            raise ValueError("Source vertex target")
        a, b, c = r[:, :, 0], r[:, :, 1], r[:, :, 2]
        den = (L[:, :, 0] * L[:, :, 1] * L[:, :, 2]
               + (a * b).sum(dim=2) * L[:, :, 2]
               + (b * c).sum(dim=2) * L[:, :, 0]
               + (c * a).sum(dim=2) * L[:, :, 1])
        D = -2.0 * torch.atan2((a * torch.linalg.cross(b, c)).sum(dim=2), den) / four_pi
        h = ((X[:, None, :] - t[None, :, 0, :]) * normal[None]).sum(dim=2)
        gd = torch.zeros_like(r[:, :, 0])
        for k, l in ((0, 1), (1, 2), (2, 0)):
            edge = t[:, l] - t[:, k]
            ll = torch.linalg.vector_norm(edge, dim=1)
            out = torch.linalg.cross(edge / ll[:, None], normal)
            rk, rl = r[:, :, k], r[:, :, l]
            dotr = (rk * rl).sum(dim=2)
            product = L[:, :, k] * L[:, :, l]
            op = dotr < 0
            cr = torch.linalg.cross(rk, rl)
            cr2 = (cr * cr).sum(dim=2)
            plus = torch.where(op, cr2 / torch.where(op, product - dotr,
                                                    torch.ones_like(product)),
                               product + dotr)
            gap = 2.0 * plus / (L[:, :, k] + L[:, :, l] + ll[None])
            if bool((gap <= 0).any()):
                raise FloatingPointError("Near-singular edge geometry")
            gd = gd - out[None] * torch.log1p(2.0 * ll[None] / gap)[:, :, None]
        M = (X[:, None, :] - ctr[None]) * D[:, :, None] + h[:, :, None] * gd / four_pi
        rows = D[:, :, None] / 3.0 + torch.einsum("tfi,fki->tfk", M, G)
        if self_index is not None:
            mask = self_index[:, None] == arange_all[None, f0:f1]
            B = 1.0 / 3.0 + torch.einsum(
                "tfk,fik->tfi", X[:, None, :] - ctr[None], G)
            rows = torch.where(mask[:, :, None], -0.5 * B, rows)
        block = dofs[f0:f1]
        for k in range(3):
            A.scatter_add_(1, block[:, k][None, :].expand(n_target, f1 - f0), rows[:, :, k])
    return A


def gradient_map(faces, dof_face, basis_gradients, *, ndof=None, device=None,
                 dtype=torch.float64):
    """Face gradient operator: (n_face, 3, ndof) with mode i holding d/dx_i on face f."""
    dofs = torch.as_tensor(dof_face, dtype=torch.long, device=device)
    bg = torch.as_tensor(basis_gradients, dtype=dtype, device=device)
    n_face = int(dofs.shape[0])
    if bg.shape[:2] != (n_face, 3):
        raise ValueError("basis_gradients must have shape (n_face, 3, 3)")
    if ndof is None:
        ndof = int(dofs.max()) + 1
    elif int(ndof) < int(dofs.max()) + 1:
        raise ValueError("ndof is smaller than the largest face DOF")
    grad = torch.zeros((n_face, 3, ndof), dtype=dtype, device=bg.device)
    for k in range(3):
        grad.scatter_add_(2, dofs[:, k][:, None, None].expand(n_face, 3, 1),
                          bg[:, k][:, :, None])
    return grad


def wake_operator(wake_triangles, targets, gamma, *, mode="per_face", face_dofs=None,
                  ndof=None, face_chunk=64, device=None, dtype=torch.float64):
    """Contract the wake doublet operator with the wake strength.

    Both production solvers are nodal: G1 sums `D_j @ gamma_hat[wake_faces[j]]` over wake
    triangles with shared wake-node strengths, and G2 contracts one nodal operator per group.
    So `mode="nodal"` is the production path and `mode="per_face"` (three private DOFs per
    triangle) exists only to unit-check the per-triangle doublet call. The result is
    (len(targets), n_gamma).
    """
    gamma = torch.as_tensor(gamma, dtype=dtype, device=device)
    if gamma.ndim != 2:
        raise ValueError("gamma must have shape (dofs, n_gamma)")
    if mode == "per_face":
        tri = torch.as_tensor(wake_triangles, dtype=dtype, device=device)
        n_face = int(tri.shape[0])
        if gamma.shape[0] != 3 * n_face:
            raise ValueError("per_face wake gamma must have 3 rows per triangle")
        dofs = torch.arange(3 * n_face, device=tri.device).reshape(n_face, 3)
        A = doublet_matrix(tri, targets, face_dofs=dofs, ndof=3 * n_face,
                           face_chunk=face_chunk, dtype=dtype)
        return A @ gamma
    if mode == "nodal":
        if face_dofs is None:
            raise ValueError("nodal mode needs face_dofs")
        A = doublet_matrix(wake_triangles, targets, face_dofs=face_dofs, ndof=ndof,
                           face_chunk=face_chunk, dtype=dtype)
        return A @ gamma
    raise ValueError("mode must be 'per_face' or 'nodal'")


def loads(*, grad, mu, areas, normals, centers, reference_point, reference_area,
          reference_chord, alpha_deg, group_gamma=(), dtype=torch.float64):
    """Cp, CL, CL_gamma, CD_pressure and CM_y on tensors (S4 post-processing).

    `group_gamma` holds (values, spanwise stations) pairs: G1 passes the whole spanwise Gamma
    with zero end points, G2 passes one pair per group. CL_gamma is the sum over the pairs,
    which is how both production load functions reduce it.
    """
    velocity = torch.einsum("fij,j->fi", grad, mu)
    cp = 1.0 - (velocity ** 2).sum(dim=1)
    face_forces = -(cp * areas)[:, None] * normals
    net = face_forces.sum(dim=0)
    moment = torch.linalg.cross(centers - reference_point, face_forces).sum(dim=0)
    alpha = torch.as_tensor(alpha_deg, dtype=dtype) * torch.pi / 180.0
    U = torch.stack((torch.cos(alpha), torch.zeros_like(alpha), torch.sin(alpha)))
    lift = torch.stack((-torch.sin(alpha), torch.zeros_like(alpha), torch.cos(alpha)))
    total = torch.zeros((), dtype=dtype)
    for values, stations in group_gamma:
        total = total + _trapezoid(values, stations, dtype=dtype)
    return dict(CL=float(net @ lift / reference_area),
                CL_gamma=float(2.0 * total / reference_area),
                CD_pressure=float(net @ U / reference_area),
                CM_y=float(moment[1] / (reference_area * reference_chord)),
                cp=cp)


def _trapezoid(values, stations, *, dtype):
    v = torch.as_tensor(values, dtype=dtype)
    s = torch.as_tensor(stations, dtype=dtype)
    if v.ndim != 1 or s.shape != v.shape:
        raise ValueError("trapezoid needs matching 1-D values and stations")
    return ((v[1:] + v[:-1]) * 0.5 * (s[1:] - s[:-1])).sum()

