"""P1 Galerkin lifting solver for G1/G2 meshes.

Uses the shared geometry, wake interpolation, TE jump and Kutta definitions.
The wall residual is reported in both raw weak form and the dual mass norm.
The collocation path remains available for regression and A/B comparisons.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import cholesky, solve_triangular, lstsq, null_space
from scipy.optimize import least_squares

from .p1_lifting_g1 import _doublet_matrix
from .p1_lifting_g2 import _mesh, assemble_multi
from .p1_galerkin_wall import _duffy_rule, _assemble_batched


def _mass_from_arrays(dof_face, areas, ndof):
    M=np.zeros((ndof,ndof))
    local=np.array([[2.,1.,1.],[1.,2.,1.],[1.,1.,2.]])
    for f,dofs in enumerate(np.asarray(dof_face,int)):
        M[np.ix_(dofs,dofs)] += (areas[f]/12.)*local
    return M


def assemble_galerkin_multi(mesh, *, order=3, block=2048, target_face_batch=64):
    """Assemble weak wall A_G, wake W_G, exact mass M and RHS moments.

    Target faces are batched; all lifting-group wakes are concatenated into one
    source sheet with block-diagonal Gamma interpolation, avoiding one analytic
    source-kernel traversal per wake group and target face.
    """
    if not isinstance(block,int) or block<1:
        raise ValueError("block must be a positive integer")
    if not isinstance(target_face_batch,int) or target_face_batch<1:
        raise ValueError("target_face_batch must be a positive integer")
    bary,qweights=_duffy_rule(order)
    base=assemble_multi(mesh)
    ms=base["mesh"];tris=ms.triangles
    nm=int(mesh.ndof_mu);groups=mesh.groups
    ngs=[int(g.gamma_hat.shape[1]) for g in groups]
    ng=sum(ngs);slices={};offset=0
    for g,n in zip(groups,ngs):
        slices[g.name]=slice(offset,offset+n);offset+=n
    M=_mass_from_arrays(mesh.dof_face,ms.areas,nm)
    wake_tris_list=[];wake_dofs_list=[];wake_nodes_total=0
    H=np.zeros((sum(len(g.wake_xyz) for g in groups),ng))
    node_offset=0
    for g,n in zip(groups,ngs):
        xyz_w=np.asarray(g.wake_xyz,float)
        faces_w=np.asarray(g.wake_faces,int)
        wake_tris_list.append(xyz_w[faces_w])
        wake_dofs_list.append(faces_w+node_offset)
        H[node_offset:node_offset+len(xyz_w),slices[g.name]]=np.asarray(g.gamma_hat,float)
        node_offset+=len(xyz_w)
    wake_tris=np.concatenate(wake_tris_list,axis=0)
    wake_dofs=np.concatenate(wake_dofs_list,axis=0)
    A,W,bxyz=_assemble_batched(
        tris,np.asarray(mesh.dof_face,int),ms.areas,ms.centers,tris,nm,
        np.zeros(3),bary,qweights,wake_tris=wake_tris,wake_dofs=wake_dofs,
        wake_basis=H,nwake=node_offset,target_face_batch=target_face_batch,
        kernel_block=block)
    return dict(A=A,W=W,M=M,b_xyz=bxyz,mesh=ms,base_mesh=mesh,
        J=base["J"],K=base["K"],grad=base["grad"],slices=slices,
        te_face_pairs=[p for group_pairs in base["te_face_pairs"] for p in group_pairs],
        group_info=base["group_info"],order=order,block=block,
        target_face_batch=target_face_batch,
        formulation="batched P1 Galerkin target-test projection",
        self_jump="exact full P1 mass contribution -0.5*M_local",
        physical_accuracy_accepted=False)


def _pressure_residual(x,ops,nm):
    v=np.einsum("fij,j->fi",ops["grad"],x[:nm])
    return np.asarray([
        np.mean(np.sum(v[top]**2,axis=1))-np.mean(np.sum(v[bot]**2,axis=1))
        for top,bot in ops["te_face_pairs"]],float)


def _pressure_jacobian(x,ops,nm,ng):
    v=np.einsum("fij,j->fi",ops["grad"],x[:nm])
    rows=[]
    for top,bot in ops["te_face_pairs"]:
        dt=2*np.mean(np.einsum("fi,fij->fj",v[top],ops["grad"][top]),axis=0)
        db=2*np.mean(np.einsum("fi,fij->fj",v[bot],ops["grad"][bot]),axis=0)
        rows.append(np.r_[dt-db,np.zeros(ng)])
    return np.asarray(rows)


def solve_galerkin_multi(mesh, alpha_deg=5., *, kutta="signed_speed",
                         pressure_weight=5., max_nfev=100, operators=None,
                         order=3):
    """Solve full multi-wing Galerkin wall system with existing TE closures."""
    ops=assemble_galerkin_multi(mesh,order=order) if operators is None else operators
    A,W,M,bxyz=ops["A"],ops["W"],ops["M"],ops["b_xyz"]
    J,K,grad,ms=ops["J"],ops["K"],ops["grad"],ops["mesh"]
    nm,ng=A.shape[1],W.shape[1]
    alpha=np.deg2rad(float(alpha_deg))
    U=np.array([np.cos(alpha),0.,np.sin(alpha)])
    b=bxyz@U
    # Weak residual is in the dual of the P1 test space: use M^{-1/2} scaling.
    L=cholesky(M,lower=True,check_finite=True)
    B=solve_triangular(L,np.column_stack((A,W)),lower=True)
    bw=solve_triangular(L,b,lower=True)
    def linear_fit(C):
        N=null_space(C,rcond=1e-11)
        if N.shape[1]==0:
            raise ValueError("constraints consume all Galerkin state degrees of freedom")
        z,_,rank,sv=lstsq(B@N,bw,lapack_driver="gelsd")
        return N@z,N,int(rank),sv
    if kutta=="signed_speed":
        x,N,rank,sv=linear_fit(np.vstack((J,K)))
        nonlinear=None
    elif kutta=="pressure":
        x_seed,_,_,_=linear_fit(np.vstack((J,K)))
        N=null_space(J,rcond=1e-11)
        z0=N.T@x_seed
        def fun(z):
            x_=N@z
            return np.r_[B@x_-bw,float(pressure_weight)*_pressure_residual(x_,ops,nm)]
        def jac(z):
            x_=N@z
            return np.vstack((B,float(pressure_weight)*_pressure_jacobian(x_,ops,nm,ng)))@N
        fit=least_squares(fun,z0,jac=jac,max_nfev=max_nfev,
                          xtol=1e-11,ftol=1e-11,gtol=1e-11)
        x=N@fit.x
        rank=int(np.linalg.matrix_rank(B@N))
        sv=np.linalg.svd(B@N,compute_uv=False)
        nonlinear=dict(success=bool(fit.success),message=str(fit.message),nfev=int(fit.nfev))
    else:
        raise ValueError("kutta must be signed_speed or pressure")
    mu=x[:nm];gamma=x[nm:]
    v=np.einsum("fij,j->fi",grad,mu)
    cp=1.-np.sum(v*v,axis=1)
    faceforces=-(cp*ms.areas)[:,None]*ms.normals
    force=faceforces.sum(axis=0)
    moment=np.cross(ms.centers-mesh.reference_point,faceforces).sum(axis=0)
    lift=np.array([-np.sin(alpha),0.,np.cos(alpha)])
    group_results=[]
    for g in mesh.groups:
        fg=faceforces[g.face_ids].sum(axis=0)
        G=gamma[ops["slices"][g.name]]
        Gfull=np.r_[0.,G,0.]
        group_results.append(dict(name=g.name,CL_pressure=float(fg@lift/mesh.reference_area),
            CL_gamma=float(2*np.trapezoid(Gfull,g.gamma_y)/mesh.reference_area),
            Gamma_integral=float(np.trapezoid(Gfull,g.gamma_y)),
            gamma=G.tolist(),gamma_y=g.gamma_y.tolist()))
    raw=A@mu+W@gamma-b
    dual=solve_triangular(L,raw,lower=True)
    pr=_pressure_residual(x,ops,nm)
    singular_values=np.asarray(sv,float)
    sv_max=float(singular_values[0]) if len(singular_values) else 0.
    sv_min=float(singular_values[-1]) if len(singular_values) else 0.
    constrained_condition=float(sv_max/sv_min) if sv_min>0 else float("inf")
    gamma_col_norms=np.linalg.norm(W,axis=0) if W.shape[1] else np.zeros(0)
    mu_norm=float(np.linalg.norm(mu));gamma_norm=float(np.linalg.norm(gamma))
    cl_value=float(force@lift/mesh.reference_area)
    cd_value=float(force@U/mesh.reference_area)
    implausible=[]
    if not np.isfinite([cl_value,cd_value,mu_norm,gamma_norm,constrained_condition]).all():
        implausible.append("nonfinite_solution_or_condition")
    if abs(cl_value)>20.:implausible.append("abs_CL_gt_20")
    if abs(cd_value)>1.:implausible.append("abs_CD_pressure_gt_1")
    if gamma_norm>50.*max(mu_norm,1e-30):implausible.append("gamma_norm_gt_50_mu_norm")
    if constrained_condition>1e6:implausible.append("constrained_condition_gt_1e6")
    if gamma_col_norms.size and float(gamma_col_norms.min()) < 1e-4*max(float(np.median(gamma_col_norms)),1e-30):
        implausible.append("near_zero_support_gamma_column")
    solution_status="implausible_solution" if implausible else (
        "solver_success" if nonlinear is None or nonlinear["success"] else "nonlinear_not_converged")
    return dict(alpha_deg=float(alpha_deg),kutta=kutta,
        CL=cl_value,
        CL_gamma=float(sum(g["CL_gamma"] for g in group_results)),
        CD_pressure=cd_value,
        CM_y=float(moment[1]/(mesh.reference_area*mesh.reference_chord)),
        force=force.tolist(),mu=mu,cp=cp,gamma=gamma,group_results=group_results,
        jump_rms=float(np.linalg.norm(J@x)/np.sqrt(max(1,J.shape[0]))),
        signed_kutta_rms=float(np.linalg.norm(K@x)/np.sqrt(max(1,K.shape[0]))),
        pressure_kutta_rms=float(np.linalg.norm(pr)/np.sqrt(max(1,len(pr)))),
        wall_rms=float(np.linalg.norm(raw)/np.sqrt(len(raw))),
        wall_dual_rms=float(np.linalg.norm(dual)/np.sqrt(len(dual))),
        linear_constrained_rank=int(rank),
        linear_singular_min=sv_min,
        linear_singular_max=sv_max,
        constrained_condition=constrained_condition,
        mu_norm=mu_norm,gamma_norm=gamma_norm,
        gamma_column_norm_min=float(gamma_col_norms.min()) if gamma_col_norms.size else 0.,
        gamma_column_norm_median=float(np.median(gamma_col_norms)) if gamma_col_norms.size else 0.,
        solution_status=solution_status,implausible_reasons=implausible,
        nonlinear=nonlinear,order=int(ops["order"]),
        physical_accuracy_accepted=False)
