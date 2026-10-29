"""Exact TE potential-jump compatibility audit, distinct from pressure Kutta."""
from __future__ import annotations
import numpy as np


def _v(name,x):
    a=np.asarray(x,float).reshape(-1)
    if not np.isfinite(a).all(): raise ValueError(f"{name} must be finite")
    return a


def audit_te_jump_integral(te_y,mu_jump,gamma_y,gamma_values,*,atol=1e-10):
    """Exact integral/L2/max residual on the common refinement of both P1 traces."""
    y,j,gy,g=_v("te_y",te_y),_v("mu_jump",mu_jump),_v("gamma_y",gamma_y),_v("gamma_values",gamma_values)
    atol=float(atol)
    if (len(y)<2 or j.shape!=y.shape or len(gy)<2 or g.shape!=gy.shape
            or np.any(np.diff(y)<=0) or np.any(np.diff(gy)<=0)
            or gy[0]>y[0]+atol or gy[-1]<y[-1]-atol or not np.isfinite(atol) or atol<=0):
        raise ValueError("invalid TE/Gamma knots or arrays")
    interior=gy[(gy>y[0]+atol)&(gy<y[-1]-atol)]
    aligned=all(np.min(np.abs(y-k))<=atol for k in interior)
    common=np.unique(np.r_[y,interior])
    jj=np.interp(common,y,j); gg=np.interp(common,gy,g); r=jj-gg; h=np.diff(common)
    integ=float(np.sum(h*(r[:-1]+r[1:])/2))
    sq=float(np.sum(h*(r[:-1]**2+r[:-1]*r[1:]+r[1:]**2)/3))
    moments=np.zeros(len(common))
    for i,dx in enumerate(h):
        moments[i]+=dx*(2*r[i]+r[i+1])/6
        moments[i+1]+=dx*(r[i]+2*r[i+1])/6
    scale=max(1.,float(np.max(np.abs(jj))),float(np.max(np.abs(gg))))
    edges=[]
    for a,b in zip(y[:-1],y[1:]):
        mask=(common>=a-atol)&(common<=b+atol); yy=common[mask]; rr=r[mask]; hh=np.diff(yy)
        ei=float(np.sum(hh*(rr[:-1]+rr[1:])/2))
        es=float(np.sum(hh*(rr[:-1]**2+rr[:-1]*rr[1:]+rr[1:]**2)/3))
        edges.append(dict(y0=float(a),y1=float(b),integral_residual=ei,
            l2_rms=float(np.sqrt(max(es,0.)/(b-a))),max_abs_residual=float(np.max(np.abs(rr))),
            gamma_knot_inside=bool(np.any((interior>a+atol)&(interior<b-atol)))))
    return dict(status="pass" if np.max(np.abs(r))<=atol*scale else "jump_mismatch",
        condition="potential_jump_compatibility_only_not_pressure_kutta",
        partition_aligned=bool(aligned),gamma_knots_inside_te_edges=interior.tolist(),
        nodal_residual=(j-np.interp(y,gy,g)).tolist(),
        nodal_residual_max=float(np.max(np.abs(j-np.interp(y,gy,g)))),
        jump_integral=float(np.trapezoid(j,y)),gamma_integral=float(np.trapezoid(np.interp(y,gy,g),y)),
        integral_residual=integ,integral_residual_normalized=integ/(scale*(y[-1]-y[0])),
        l2_residual=float(np.sqrt(max(sq,0.))),l2_rms=float(np.sqrt(max(sq,0.)/(y[-1]-y[0]))),
        max_abs_residual=float(np.max(np.abs(r))),weak_moment_linf=float(np.max(np.abs(moments))),
        integral_alone_can_hide_cancellation=bool(abs(integ)<=atol*scale*(y[-1]-y[0]) and
                                                    np.max(np.abs(r))>10*atol*scale),
        edges=edges,physical_accuracy_accepted=False)


def audit_g1_te_jump(wing,mu,gamma,*,atol=1e-10):
    mu,gamma=_v("mu",mu),_v("gamma",gamma)
    up,lo=np.asarray(wing.te_up,int),np.asarray(wing.te_down,int)
    y=np.asarray(wing.xyz,float)[np.asarray(wing.te_geo,int),1]
    order=np.argsort(y); y,up,lo=y[order],up[order],lo[order]
    if len(mu)!=wing.ndof_mu or len(gamma)!=len(wing.gamma_y)-2: raise ValueError("state size mismatch")
    return audit_te_jump_integral(y,mu[up]-mu[lo],wing.gamma_y,np.r_[0.,gamma,0.],atol=atol)


def audit_g2_group_te_jump(mesh,state,operators,group_name,*,atol=1e-10):
    groups={g.name:g for g in mesh.groups}
    if group_name not in groups: raise ValueError(f"unknown group {group_name}")
    g=groups[group_name]; pairs=np.asarray(g.te_pairs,int)
    mu,gamma=_v("mu",state["mu"]),_v("gamma",state["gamma"])
    G=np.r_[0.,gamma[operators["slices"][group_name]],0.]
    jumps=mu[pairs[:,0]]-mu[pairs[:,1]]
    y=np.r_[g.gamma_y[0],g.te_pair_y,g.gamma_y[-1]]
    dmu=np.r_[0.,jumps,0.]; order=np.argsort(y)
    return audit_te_jump_integral(y[order],dmu[order],g.gamma_y,G,atol=atol)
