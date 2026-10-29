"""Experimental P1 Galerkin wall operator for DPanel.

The source-triangle linear P1 double-layer integrals use the existing analytic
kernel. The target-face test-function integral is performed by Duffy quadrature.
This module does not replace the production collocation solver and deliberately
does not yet assemble the TE jump or pressure-Kutta closure.
"""
from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss

from .p1_lifting_g1 import _doublet_matrix, _mesh_as_kernel


def _duffy_rule(order):
    if not isinstance(order, int) or order < 2:
        raise ValueError("Duffy quadrature order must be an integer >= 2")
    q, w = leggauss(order)
    q, w = (q+1.)/2., w/2.
    u, v = np.meshgrid(q, q, indexing="ij")
    wu, wv = np.meshgrid(w, w, indexing="ij")
    weights = wu*wv*(1.-u)
    bary = np.stack(((1.-u)*(1.-v), u, (1.-u)*v), axis=-1).reshape(-1,3)
    weights = weights.reshape(-1)
    # Barycentric mapping has Jacobian |(b-a)x(c-a)|*(1-u) = 2A*(1-u).
    return bary, weights


def p1_mass_matrix(wing):
    """Exact assembled surface P1 mass matrix using the actual potential DOFs."""
    mesh = _mesh_as_kernel(wing)
    M = np.zeros((wing.ndof_mu, wing.ndof_mu))
    local_factor = mesh.areas/12.
    local = np.array([[2.,1.,1.],[1.,2.,1.],[1.,1.,2.]])
    for f, dofs in enumerate(wing.dof_face):
        M[np.ix_(dofs,dofs)] += local_factor[f]*local
    return M


def assemble_galerkin_wall(wing, freestream=(1.,0.,0.), order=3,
                           block=32, include_wake=True):
    """Assemble the weak P1 projection of the current interior potential trace.

    A_G[i,j] = int_S N_i(x) [D_S^- N_j](x) dS_x.
    W_G[i,g] = int_S N_i(x) [D_W H_g](x) dS_x.
    b_G[i]   = -int_S N_i(x) Phi_inf(x) dS_x.

    Here D^- uses the sign convention of the existing _doublet_matrix,
    including its exact own-face -1/2 P1 trace. The source-face integral is
    analytic; target test integration uses the specified Duffy rule.
    """
    U = np.asarray(freestream, dtype=float).reshape(3)
    if not np.isfinite(U).all():
        raise ValueError("freestream must be finite")
    if not isinstance(block,int) or block < 1:
        raise ValueError("block must be a positive integer")
    bary, qweights = _duffy_rule(order)
    mesh = _mesh_as_kernel(wing)
    tris = mesh.triangles
    nface, ndof = len(wing.faces), int(wing.ndof_mu)
    ngfull = int(wing.gamma_hat.shape[1])
    A = np.zeros((ndof, ndof))
    W = np.zeros((ndof, max(0,ngfull-2)))
    b = np.zeros(ndof)
    M = p1_mass_matrix(wing)
    if include_wake:
        wake_tris = np.asarray(wing.wake_xyz)[np.asarray(wing.wake_faces,dtype=int)]
        wake_dofs = np.asarray(wing.wake_faces,dtype=int)
        nwake = len(wing.wake_xyz)
        Hfull = np.asarray(wing.gamma_hat,dtype=float)
        if Hfull.shape[0] != nwake or Hfull.shape[1] != ngfull:
            raise ValueError("wake Gamma interpolation shape mismatch")
        H = Hfull[:,1:-1]
    else:
        wake_tris = None
        wake_dofs = None
        nwake = 0
        H = None
    for f, face in enumerate(wing.faces):
        tri = tris[f]
        points = bary @ tri
        weights = qweights * (2.*mesh.areas[f])
        # Every target point lies strictly inside the target triangle.
        # The source kernel supplies the exact interior trace for source face f.
        Dwall = _doublet_matrix(tris, points, self_faces=np.full(len(points),f),
                                face_dofs=wing.dof_face, ndof=ndof, block=block)
        Dwake = None
        if include_wake and ngfull > 2:
            Dw_nodes = _doublet_matrix(wake_tris, points, face_dofs=wake_dofs,
                                       ndof=nwake, block=block)
            Dwake = Dw_nodes @ H
        for k, row_dof in enumerate(wing.dof_face[f]):
            wk = weights*bary[:,k]
            A[row_dof] += wk @ Dwall
            if Dwake is not None:
                W[row_dof] += wk @ Dwake
            b[row_dof] -= np.dot(wk, points @ U)
    return dict(A=A,W=W,b=b,M=M,mesh=mesh,order=order,
                formulation="P1 Galerkin target test / analytic source P1 doublet",
                self_jump="exact -1/2 local P1 mass contribution",
                includes_wake=bool(include_wake),
                physical_accuracy_accepted=False)
