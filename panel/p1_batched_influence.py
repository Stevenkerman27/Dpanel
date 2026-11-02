"""Batched float64 influence kernel for the P1 doublet system (issue 92, Phase 1).

The reference implementation in panel/p1_lifting_g1.py (_doublet_matrix) walks the
source triangles in Python and vectorises only over targets, so a 3220-face wing costs
one Python iteration per wake triangle: 1528 calls and 12.21 s of the elliptic run. This
module evaluates the identical closed-form kernel over target x source blocks, which
turns the source loop into tensor work and lets the same code run on CUDA unchanged.

The kernel is the exact single-triangle analytic doublet plus the analytic edge-log
gradient term:

    D    = -2 atan2(a.(bxc), L0 L1 L2 + (a.b)L2 + (b.c)L0 + (c.a)L1) / (4 pi)
    gd   = -sum over edges  out_k * log1p(2 ll / gap)
    rows = D/3 + [ (X - ctr) D + h gd/(4 pi) ] G^T ,   h = (X - t0).n

with r_k the vectors from the target to vertex k, out_k the in-plane outward normal of
edge k, gap the reference triangle-edge split measure and G the row-stacked vertex
gradients of the linear basis. This is the same expression as
panel/p1_closed_body.py influence_block: nothing is approximated and no quadrature is
introduced, so parity against the reference is a tolerance check rather than a modelling
decision.

Conventions that must not drift:

* float64 everywhere, on CPU and CUDA;
* the self-panel row is replaced by -0.5 (1/3 + (X - ctr) G^T) exactly where the target
  face index equals the source face index, as in the reference;
* face-to-DOF accumulation uses the same face_dofs scatter, so mirror winding and the
  bank/cut conventions are inherited from the caller rather than re-derived here;
* blocks are bounded (target_block, source_block); an [N_target, N_source, 3] tensor is
  never materialised for the largest fixture.

Torch is imported lazily so a NumPy-only interpreter can still import the module and
inspect the block sizes.
"""
from __future__ import annotations

import numpy as np

# Sized from the elliptic block sweep (tools/issue92/block_sweep.py): 512x256 is the
# fastest configuration on this host at 8-16 torch threads, and the block only changes
# the traversal order, never the arithmetic. The peak temporary is a (512, 256, 3, 3)
# float64 tensor, about 9 MB, so an [N_target, N_source, 3] tensor is never formed.
DEFAULT_TARGET_BLOCK = 512
DEFAULT_SOURCE_BLOCK = 256


def _torch():
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the interpreter
        raise ModuleNotFoundError(
            "panel.p1_batched_influence needs PyTorch; install the CPU build or add the "
            "torch site directory to PYTHONPATH") from exc
    return torch


def _as_float64(t, torch):
    return t if t.dtype == torch.float64 else t.to(torch.float64)


def source_invariants(tris, *, device="cpu", dtype=None):
    """Precompute every source-only quantity once per assembly.

    tris is (n_source, 3, 3) in an order that does not depend on the targets. The
    returned dict holds the outside normals, vertex gradients, centroids, edge
    directions and lengths, so the block loop only reads them.
    """
    torch = _torch()
    dtype = torch.float64 if dtype is None else dtype
    tri = _as_float64(torch.as_tensor(np.asarray(tris, dtype=np.float64), device=device), torch)
    if tri.ndim != 3 or tuple(tri.shape[1:]) != (3, 3):
        raise ValueError("tris must be (n_source, 3, 3)")
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    normal = torch.cross(e1, e2, dim=-1)
    a2 = torch.linalg.norm(normal, dim=-1)
    if bool((a2 <= 0).any()):
        raise ValueError("Zero-area source triangle")
    n_hat = normal / a2[:, None]
    g1 = torch.cross(e2, n_hat, dim=-1) / a2[:, None]
    g2 = torch.cross(n_hat, e1, dim=-1) / a2[:, None]
    gradients = torch.stack((-g1 - g2, g1, g2), dim=1)  # (S, 3 vertices, 3 coords)
    edges = torch.stack((tri[:, 1] - tri[:, 0],
                         tri[:, 2] - tri[:, 1],
                         tri[:, 0] - tri[:, 2]), dim=1)  # (S, 3 edges, 3 coords)
    lengths = torch.linalg.norm(edges, dim=-1)
    outward = torch.cross(edges / lengths[..., None], n_hat[:, None, :], dim=-1)
    return {
        "tri": tri,
        "normal": n_hat,
        "gradients": gradients,
        "centroid": tri.mean(dim=1),
        "lengths": lengths,
        "outward": outward,
    }


def block_rows(targets, src, index, *, torch):
    """Rows (Bt, Bs, 3) of the doublet kernel for one target/source block."""
    tri = src["tri"][index]                      # (Bs, 3, 3)
    n_hat = src["normal"][index]                 # (Bs, 3)
    centroid = src["centroid"][index]
    gradients = src["gradients"][index]          # (Bs, 3, 3)
    lengths = src["lengths"][index]              # (Bs, 3)
    outward = src["outward"][index]              # (Bs, 3, 3)

    r = tri[None, :, :, :] - targets[:, None, None, :]   # (Bt, Bs, 3 vertices, 3 coords)
    distance = torch.linalg.norm(r, dim=-1)              # (Bt, Bs, 3)
    if bool((distance == 0).any()):
        raise ValueError("Source vertex coincides with a target point")

    a, b, c = r[..., 0, :], r[..., 1, :], r[..., 2, :]
    l0, l1, l2 = distance[..., 0], distance[..., 1], distance[..., 2]
    den = (l0 * l1 * l2
           + (a * b).sum(-1) * l2
           + (b * c).sum(-1) * l0
           + (c * a).sum(-1) * l1)
    triple = (torch.cross(a, b, dim=-1) * c).sum(-1)
    solid = -2.0 * torch.atan2(triple, den) / (4.0 * np.pi)
    height = ((targets[:, None, :] - tri[None, :, 0, :]) * n_hat[None]).sum(-1)

    gd = torch.zeros_like(distance)  # (Bt, Bs, 3 edges), accumulated per edge index
    for k, l in ((0, 1), (1, 2), (2, 0)):
        rk, rl = r[..., k, :], r[..., l, :]
        dot = (rk * rl).sum(-1)
        product = distance[..., k] * distance[..., l]
        cross_norm2 = (torch.cross(rk, rl, dim=-1) ** 2).sum(-1)
        opposed = dot < 0
        denominator = torch.where(opposed, product - dot, torch.ones_like(product))
        plus = torch.where(opposed, cross_norm2 / denominator, product + dot)
        gap = 2.0 * plus / (distance[..., k] + distance[..., l] + lengths[:, k][None])
        if bool((gap <= 0).any()):
            raise FloatingPointError("Near-singular edge geometry")
        gd = gd - outward[:, k][None] * torch.log1p(2.0 * lengths[:, k][None] / gap)[..., None]
    moment = ((targets[:, None, :] - centroid[None]) * solid[..., None]
              + height[..., None] * gd / (4.0 * np.pi))
    return solid[..., None] / 3.0 + torch.einsum("bsc,skc->bsk", moment, gradients)


def influence_matrix(tris, targets, *, self_faces=None, face_dofs, ndof=None,
                     target_block=DEFAULT_TARGET_BLOCK, source_block=DEFAULT_SOURCE_BLOCK,
                     device="cpu", dtype=None, src=None, return_tensor=False):
    """Scattered doublet influence of tris on targets, as an (n_target, ndof) array.

    Each source vertex k contributes rows[:, s, k] to column face_dofs[s, k], which is
    exactly the reference accumulation. Where self_faces[t] == s the row is replaced by
    the reference self-panel expression.

    src is an optional dict from source_invariants; passing it (already moved to the
    target device) separates source transfer from kernel time, which is what the CUDA
    measurement needs.
    """
    torch = _torch()
    dtype = torch.float64 if dtype is None else dtype
    tri_np = np.asarray(tris, dtype=np.float64)
    target_np = np.asarray(targets, dtype=np.float64)
    if tri_np.ndim != 3 or tri_np.shape[1:] != (3, 3):
        raise ValueError("tris must be (n_source, 3, 3)")
    if target_np.ndim != 2 or target_np.shape[1] != 3:
        raise ValueError("targets must be (n_target, 3)")
    if int(target_block) <= 0 or int(source_block) <= 0:
        raise ValueError("target_block and source_block must be positive")
    face_dofs = np.asarray(face_dofs, dtype=np.int64)
    if face_dofs.shape != (len(tri_np), 3):
        raise ValueError("face_dofs must be (n_source, 3)")
    if face_dofs.min() < 0:
        raise ValueError("face_dofs must be non-negative")
    if ndof is None:
        ndof = int(face_dofs.max()) + 1
    ndof = int(ndof)
    if face_dofs.max() >= ndof:
        raise ValueError("face_dofs must index [0, ndof)")

    if src is None:
        src = source_invariants(tri_np, device=device, dtype=dtype)
    elif int(src["tri"].shape[0]) != len(tri_np):
        raise ValueError("precomputed source invariants do not match tris")
    targets_t = _as_float64(torch.as_tensor(target_np, device=device), torch)
    n_source, n_target = len(tri_np), len(target_np)
    out = torch.zeros((n_target, ndof), dtype=dtype, device=device)
    if self_faces is not None:
        self_faces = np.asarray(self_faces, dtype=np.int64).reshape(-1)
        if len(self_faces) != n_target:
            raise ValueError("self_faces must be (n_target,)")

    for t0 in range(0, n_target, int(target_block)):
        t1 = min(n_target, t0 + int(target_block))
        for s0 in range(0, n_source, int(source_block)):
            s1 = min(n_source, s0 + int(source_block))
            index = torch.arange(s0, s1, device=device)
            rows = block_rows(targets_t[t0:t1], src, index, torch=torch)
            if self_faces is not None:
                same = torch.as_tensor(self_faces[t0:t1][:, None] == np.arange(s0, s1)[None, :],
                                       device=device)
                if bool(same.any()):
                    projection = targets_t[t0:t1][:, None, :] - src["centroid"][index][None]
                    base = 1.0 / 3.0 + torch.einsum("bsc,skc->bsk", projection,
                                                      src["gradients"][index])
                    rows = torch.where(same[..., None], -0.5 * base, rows)
            dofs = torch.as_tensor(face_dofs[s0:s1].reshape(-1), device=device)
            out[t0:t1].index_add_(1, dofs, rows.reshape(t1 - t0, -1))
    if return_tensor:
        # A device-side result lets a CUDA caller contract with gamma_hat on the GPU.
        # Round-tripping through NumPy would put a BLAS call in the same process as a
        # live CUDA context, which crashes on this host.
        return out
    return out.detach().cpu().numpy()


def wake_matrix(wake_xyz, wake_faces, targets, *, target_block=DEFAULT_TARGET_BLOCK,
                source_block=DEFAULT_SOURCE_BLOCK, device="cpu", dtype=None, src=None,
                return_tensor=False):
    """Wake-sheet coefficients (n_target, n_wake_nodes) for a triangulated sheet.

    This is the batched form of the reference loop that calls the kernel once per wake
    triangle; multiplying the result by gamma_hat gives the same W.
    """
    xyz = np.asarray(wake_xyz, dtype=np.float64)
    faces = np.asarray(wake_faces, dtype=np.int64)
    tris = xyz[faces]
    return influence_matrix(tris, targets, face_dofs=faces, ndof=len(xyz),
                            target_block=target_block, source_block=source_block,
                            device=device, dtype=dtype, src=src,
                            return_tensor=return_tensor)
