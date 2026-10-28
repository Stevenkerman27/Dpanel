"""Independent comparisons of P1 surface Cp and total off-surface potential velocity."""
import numpy as np


def _a(name, x):
    x = np.asarray(x, float)
    if not np.isfinite(x).all():
        raise ValueError(name + " contains nonfinite values")
    return x


def p1_surface_gradient(xyz, faces, dof_face, mu):
    """Per-triangle P1 gradient plus aspect/conditioning diagnostics."""
    x, f, df, m = _a("xyz", xyz), np.asarray(faces, int), np.asarray(dof_face, int), _a("mu", mu).reshape(-1)
    if x.ndim != 2 or x.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3 or df.shape != f.shape:
        raise ValueError("expected xyz=(N,3), faces/dof_face=(F,3)")
    if len(f) == 0 or np.any(f < 0) or np.any(f >= len(x)) or np.any(df < 0) or np.any(df >= len(m)):
        raise ValueError("invalid faces or DOF mapping")
    tri = x[f]; e1 = tri[:,1]-tri[:,0]; e2 = tri[:,2]-tri[:,0]
    av = np.cross(e1,e2); twice_area = np.linalg.norm(av,axis=1)
    scale = np.maximum(np.linalg.norm(e1,axis=1)*np.linalg.norm(e2,axis=1),1e-300)
    if np.any(twice_area <= 1e-14*scale): raise ValueError("degenerate triangle")
    n = av/twice_area[:,None]
    g1 = np.cross(e2,n)/twice_area[:,None]; g2 = np.cross(n,e1)/twice_area[:,None]
    bg = np.stack((-g1-g2,g1,g2),axis=1)
    grad = np.einsum("fa,fai->fi",m[df],bg)
    edges = np.stack((np.linalg.norm(e1,axis=1),np.linalg.norm(e2,axis=1),
                      np.linalg.norm(tri[:,2]-tri[:,1],axis=1)),axis=1)
    sv = np.linalg.svd(np.stack((e1,e2),axis=1),compute_uv=False)
    q = dict(area=twice_area/2,normal=n,centroid=tri.mean(axis=1),
             edge_min=edges.min(axis=1),edge_max=edges.max(axis=1),
             aspect_ratio=edges.max(axis=1)/edges.min(axis=1),
             edge_condition=sv[:,0]/sv[:,-1],
             gradient_operator_norm=np.max(np.linalg.norm(bg,axis=2),axis=1))
    return grad,q


def cp_from_velocity(v, U=1.):
    v=_a("velocity",v); U=float(U)
    if v.shape[-1]!=3 or not np.isfinite(U) or U<=0: raise ValueError("invalid velocity/U")
    return 1-np.sum(v*v,axis=-1)/U**2


def potential_gradient_fd4(potential, points, step):
    """Fourth-order central gradient of scalar total potential; step may be per-point."""
    p=_a("points",points); h=_a("step",step)
    if h.ndim==0: h=np.full(len(p),float(h))
    else: h=h.reshape(-1)
    if p.ndim!=2 or p.shape[1]!=3 or h.shape!=(len(p),) or np.any(h<=0):
        raise ValueError("points=(N,3), step positive scalar or (N,)")
    out=np.empty_like(p); eye=np.eye(3)
    for k in range(3):
        d=h[:,None]*eye[k]
        v=np.asarray(potential(np.concatenate((p-2*d,p-d,p+d,p+2*d))),float).reshape(-1)
        if v.size!=4*len(p) or not np.isfinite(v).all(): raise ValueError("potential must return 4N finite scalars")
        fm2,fm1,fp1,fp2=np.split(v,4)
        out[:,k]=(fm2-8*fm1+8*fp1-fp2)/(12*h)
    return out


def off_surface_cp_ladder(potential, centers, normals, freestream, offsets,
                          step_fraction=.2, offset_sign=1., local_scale=None):
    """Independent total-potential gradient at one-sided normal offsets."""
    c,n=_a("centers",centers),_a("normals",normals); U=_a("freestream",freestream).reshape(3)
    levels=_a("offsets",offsets).reshape(-1)
    scale=None if local_scale is None else _a("local_scale",local_scale).reshape(-1)
    if c.ndim!=2 or c.shape[1]!=3 or n.shape!=c.shape or len(levels)<2: raise ValueError("bad points/normals/offset ladder")
    if scale is not None and (scale.shape!=(len(c),) or np.any(scale<=0)): raise ValueError("bad local scale")
    if np.any(levels<=0) or not 0<step_fraction<.25 or offset_sign not in (-1.,1.): raise ValueError("bad offset controls")
    nn=np.linalg.norm(n,axis=1)
    if np.any(nn==0): raise ValueError("zero normal")
    n=n/nn[:,None]; result=[]
    for lev in levels:
        h=np.full(len(c),float(lev)) if scale is None else float(lev)*scale
        step=np.minimum(h*step_fraction,h/5)
        p=c+offset_sign*h[:,None]*n
        v=potential_gradient_fd4(potential,p,step)
        vn=np.einsum("fi,fi->f",v,n); vt=v-vn[:,None]*n
        result.append(dict(offset_level=float(lev),offset_min=float(h.min()),offset_median=float(np.median(h)),
                           offset_max=float(h.max()),fd_step_median=float(np.median(step)),
                           velocity_total=v,velocity_normal=vn,velocity_tangent=vt,
                           cp_full_speed=cp_from_velocity(v,np.linalg.norm(U)),
                           cp_tangent_speed=cp_from_velocity(vt,np.linalg.norm(U))))
    for i,r in enumerate(result):
        r["cp_tangent_delta_previous"]=None if i==0 else r["cp_tangent_speed"]-result[i-1]["cp_tangent_speed"]
    return result


def compare_p1_to_offsurface(xyz,faces,dof_face,mu,potential,freestream,
                             offsets=(.2,.1,.05),step_fraction=.2,
                             offset_sign=1.,local_scale="edge_min"):
    """Report both grad(mu) and U_t+grad(mu) conventions against total-potential FD."""
    grad,q=p1_surface_gradient(xyz,faces,dof_face,mu); U=_a("freestream",freestream).reshape(3)
    n=q["normal"]; ut=U-np.einsum("fi,i->f",n,U)[:,None]*n
    if local_scale is None: scale=None
    elif isinstance(local_scale,str) and local_scale=="edge_min": scale=q["edge_min"]
    elif isinstance(local_scale,str) and local_scale=="sqrt_area": scale=np.sqrt(q["area"])
    else: scale=_a("local_scale",local_scale)
    ladder=off_surface_cp_ladder(potential,q["centroid"],n,U,offsets,step_fraction,offset_sign,scale)
    cp0=cp_from_velocity(grad,np.linalg.norm(U)); cp1=cp_from_velocity(grad+ut,np.linalg.norm(U))
    for r in ladder:
        r["delta_cp_vs_grad_mu"]=r["cp_tangent_speed"]-cp0
        r["delta_cp_vs_Ut_plus_grad_mu"]=r["cp_tangent_speed"]-cp1
        r["delta_vt_vs_grad_mu"]=np.linalg.norm(r["velocity_tangent"]-grad,axis=1)
        r["delta_vt_vs_Ut_plus_grad_mu"]=np.linalg.norm(r["velocity_tangent"]-grad-ut,axis=1)
    return dict(p1_gradient=grad,p1_cp_current_grad_mu=cp0,p1_cp_candidate_Ut_plus_grad_mu=cp1,
                quality=q,offsurface_ladder=ladder,
                summary=dict(n_faces=len(faces),
                    aspect_ratio_p50_p90_max=np.percentile(q["aspect_ratio"],[50,90,100]).tolist(),
                    edge_condition_p50_p90_max=np.percentile(q["edge_condition"],[50,90,100]).tolist(),
                    gradient_operator_norm_p50_p90_max=np.percentile(q["gradient_operator_norm"],[50,90,100]).tolist()))


def kutta_pressure_diagnostic(cp,areas,upper_faces,lower_faces,weighting="equal"):
    """Per-TE-station Cp jump. Equal weighting matches current G1/G2 residual; no smoothing."""
    cp,a=_a("cp",cp).reshape(-1),_a("areas",areas).reshape(-1)
    up,lo=np.asarray(upper_faces,int),np.asarray(lower_faces,int)
    if up.ndim==1: up=up[:,None]
    if lo.ndim==1: lo=lo[:,None]
    if up.ndim!=2 or lo.ndim!=2 or up.shape[0]!=lo.shape[0]: raise ValueError("bad TE face groups")
    if np.any(up<0) or np.any(lo<0) or np.any(up>=len(cp)) or np.any(lo>=len(cp)): raise ValueError("TE face index out of range")
    if weighting not in ("equal","area"): raise ValueError("weighting must be equal or area")
    def mean(idx):
        return np.mean(cp[idx],axis=1) if weighting=="equal" else np.sum(cp[idx]*a[idx],axis=1)/np.sum(a[idx],axis=1)
    cu,cl=mean(up),mean(lo)
    return dict(weighting=weighting,cp_upper=cu,cp_lower=cl,cp_jump_upper_minus_lower=cu-cl,
                speed2_jump_over_U2_lower_minus_upper=cu-cl)


def pressure_load_diagnostic(cp,areas,normals,centers,reference_area,drag_axis,lift_axis,
                             reference_point=(0.,0.,0.),reference_chord=1.):
    """Unclipped pressure-force integration with signed drag cancellation budget."""
    cp,a=_a("cp",cp).reshape(-1),_a("areas",areas).reshape(-1)
    n,c=_a("normals",normals),_a("centers",centers)
    d,l=_a("drag_axis",drag_axis).reshape(3),_a("lift_axis",lift_axis).reshape(3)
    rp=_a("reference_point",reference_point).reshape(3); sref,cref=float(reference_area),float(reference_chord)
    if n.shape!=c.shape or n.shape!=(len(cp),3) or len(a)!=len(cp) or sref<=0 or cref<=0: raise ValueError("bad load geometry/reference")
    dn,ln=np.linalg.norm(d),np.linalg.norm(l)
    if dn<=0 or ln<=0: raise ValueError("drag/lift axes must be nonzero")
    d,l=d/dn,l/ln
    ff=-(cp*a)[:,None]*n; cd=ff@d/sref; cl=ff@l/sref
    force=ff.sum(axis=0); moment=np.cross(c-rp,ff).sum(axis=0); net=float(cd.sum()); abs_sum=float(np.abs(cd).sum())
    return dict(face_cd_contributions=cd,face_cl_contributions=cl,CL_pressure=float(cl.sum()),CD_pressure=net,
                drag_positive_sum=float(cd[cd>0].sum()),drag_negative_sum=float(cd[cd<0].sum()),
                drag_abs_sum=abs_sum,drag_cancellation_ratio=abs_sum/abs(net) if abs(net)>1e-15 else float("inf"),
                CM_vector=moment/(sref*cref),force=force,moment=moment)


def diagnose_g2_state(mesh,state,offsets=(.2,.1,.05),step_fraction=.2,offset_sign=1.,local_scale="edge_min"):
    """Adapter for solve_multi output; potential includes freestream, wall and every wake."""
    from .p1_steady_free_wake import _potentials_at
    alpha=np.deg2rad(float(state["alpha_deg"])); U=np.array([np.cos(alpha),0.,np.sin(alpha]))
    phi=lambda points: _potentials_at(mesh,state,points)
    report=compare_p1_to_offsurface(mesh.xyz,mesh.faces,mesh.dof_face,state["mu"],phi,U,
                                   offsets,step_fraction,offset_sign,local_scale)
    cp=_a("solver cp",state["cp"]).reshape(-1)
    if cp.shape!=report["p1_cp_current_grad_mu"].shape: raise ValueError("Cp face count mismatch")
    report["cp_solver"]=cp
    delta=report["p1_cp_current_grad_mu"]-cp
    report["delta_cp_recomputed_p1_minus_solver"]=delta
    report["summary"]["solver_cp_recompute_max_abs"]=float(np.max(np.abs(delta)))
    report["summary"]["solver_cp_recompute_rms"]=float(np.sqrt(np.mean(delta*delta)))
    for r in report["offsurface_ladder"]:
        d=r["cp_tangent_speed"]-cp
        r["delta_cp_offsurface_tangent_minus_solver"]=d
        r["cp_offsurface_vs_solver_rms"]=float(np.sqrt(np.mean(d*d)))
        r["cp_offsurface_vs_solver_max_abs"]=float(np.max(np.abs(d)))
    return report
