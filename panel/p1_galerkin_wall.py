"""P1 Galerkin wall operator with batched target-face assembly.

The source-triangle P1 double-layer integral uses the analytic kernel.
Target test integration uses a Duffy-transformed tensor Gauss rule. Target
faces are batched so the kernel traverses source faces once per batch rather
than once per target face. The collocation assembler remains available for A/B.
"""
from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss

from .p1_lifting_g1 import _doublet_matrix, _mesh_as_kernel


def _duffy_rule(order):
    if not isinstance(order, int) or order < 2:
        raise ValueError("Duffy quadrature order must be an integer >= 2")
    q,w=leggauss(order);q,w=(q+1.)/2.,w/2.
    u,v=np.meshgrid(q,q,indexing="ij")
    wu,wv=np.meshgrid(w,w,indexing="ij")
    weights=(wu*wv*(1.-u)).reshape(-1)
    bary=np.stack(((1.-u)*(1.-v),u,(1.-u)*v),axis=-1).reshape(-1,3)
    return bary,weights


def p1_mass_matrix(wing):
    """Exact assembled surface P1 mass matrix using physical potential DOFs."""
    mesh=_mesh_as_kernel(wing)
    M=np.zeros((wing.ndof_mu,wing.ndof_mu))
    local=np.array([[2.,1.,1.],[1.,2.,1.],[1.,1.,2.]])
    for f,dofs in enumerate(wing.dof_face):
        M[np.ix_(dofs,dofs)]+=(mesh.areas[f]/12.)*local
    return M


def _assemble_batched(tris, dof_face, areas, centers, xyz, ndof, U,
                      bary, qweights, *, wake_tris=None, wake_dofs=None,
                      wake_basis=None, nwake=0, target_face_batch=64,
                      kernel_block=2048):
    """Shared target-batched P1 weak assembly; source integration is analytic."""
    nface=len(tris);nq=len(qweights)
    A=np.zeros((ndof,ndof))
    n_gamma=0 if wake_basis is None else wake_basis.shape[1]
    W=np.zeros((ndof,n_gamma))
    b=np.zeros((ndof,3))
    for start in range(0,nface,target_face_batch):
        stop=min(start+target_face_batch,nface)
        face_ids=np.arange(start,stop)
        points=np.einsum("qk,fkj->fqj",bary,xyz[face_ids])
        weights=2.*areas[face_ids,None]*qweights[None,:]
        flat=points.reshape(-1,3)
        self_faces=np.repeat(face_ids,nq)
        Dw=_doublet_matrix(tris,flat,self_faces=self_faces,
                           face_dofs=dof_face,ndof=ndof,block=kernel_block)
        Dw=Dw.reshape(stop-start,nq,ndof)
        Dgamma=None
        if wake_tris is not None and n_gamma:
            Dnodes=_doublet_matrix(wake_tris,flat,face_dofs=wake_dofs,
                                   ndof=nwake,block=kernel_block)
            Dgamma=(Dnodes@wake_basis).reshape(stop-start,nq,n_gamma)
        for k in range(3):
            test_weights=weights*bary[None,:,k]
            contrib=np.einsum("fq,fqj->fj",test_weights,Dw,optimize=True)
            np.add.at(A,dof_face[face_ids,k],contrib)
            bcontrib=-np.einsum("fq,fqj->fj",test_weights,points,optimize=True)
            np.add.at(b,dof_face[face_ids,k],bcontrib)
            if Dgamma is not None:
                gcontrib=np.einsum("fq,fqg->fg",test_weights,Dgamma,optimize=True)
                np.add.at(W,dof_face[face_ids,k],gcontrib)
    return A,W,b


def assemble_galerkin_wall(wing,freestream=(1.,0.,0.),order=3,
                           block=2048,include_wake=True,target_face_batch=64):
    """Assemble weak P1 wall A, wake W, RHS b, and exact mass matrix M.

    block controls target points per source-kernel chunk; target_face_batch
    controls the number of target triangles assembled at once.
    """
    U=np.asarray(freestream,dtype=float).reshape(3)
    if not np.isfinite(U).all():raise ValueError("freestream must be finite")
    if not isinstance(block,int) or block<1:raise ValueError("block must be positive")
    if not isinstance(target_face_batch,int) or target_face_batch<1:
        raise ValueError("target_face_batch must be positive")
    bary,qweights=_duffy_rule(order)
    mesh=_mesh_as_kernel(wing);tris=mesh.triangles
    if include_wake:
        wake_tris=np.asarray(wing.wake_xyz)[np.asarray(wing.wake_faces,int)]
        wake_dofs=np.asarray(wing.wake_faces,int);nwake=len(wing.wake_xyz)
        H=np.asarray(wing.gamma_hat,float)
        if H.shape[0]!=nwake:raise ValueError("wake Gamma interpolation shape mismatch")
    else:
        wake_tris=wake_dofs=H=None;nwake=0
    A,W,bxyz=_assemble_batched(
        tris,np.asarray(wing.dof_face,int),mesh.areas,mesh.centers,
        tris,int(wing.ndof_mu),U,bary,qweights,
        wake_tris=wake_tris,wake_dofs=wake_dofs,wake_basis=H,nwake=nwake,
        target_face_batch=target_face_batch,kernel_block=block)
    M=p1_mass_matrix(wing)
    b=bxyz@U
    return dict(A=A,W=W,b=b,M=M,mesh=mesh,order=order,
        formulation="P1 Galerkin target-test projection (batched)",
        self_jump="exact -1/2 local P1 mass contribution",
        includes_wake=bool(include_wake),target_face_batch=target_face_batch,
        kernel_block=block,physical_accuracy_accepted=False)
