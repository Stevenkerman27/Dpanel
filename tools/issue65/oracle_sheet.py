import numpy as np
from panel.wake_velocity import (sheet_velocity, sheet_velocity_pv, triangle_normal,
                                 solid_angle, doublet_potential, segment_velocity,
                                 triangle_ring_velocity)
TRI=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]); N=triangle_normal(TRI)
def mu(p): return 0.3+0.7*p[...,1]
GRAD=np.array([0.,0.7,0.]); MU0=mu(TRI.mean(axis=0))
print("mu0 at centroid", MU0)
def midpoints(n):
    P=[];W=[]
    for i in range(n):
        for j in range(n-i):
            P.append(((i+1/3)/n,(j+1/3)/n)); W.append(0.5/n/n)
            if i+j<n-1: P.append(((i+2/3)/n,(j+2/3)/n)); W.append(0.5/n/n)
    b=np.array(P); a=b[:,0]; c=b[:,1]
    return np.stack([1-a-c,a,c],axis=1)@TRI, np.array(W)
QP,QW=midpoints(80)
def phi(p):
    r=p[None,:]-QP; R=np.linalg.norm(r,axis=1)
    return -(1/(4*np.pi))*np.sum(QW*mu(QP)*(r@N)/R**3)
def fd(f,p,h=1e-6):
    g=np.zeros(3)
    for i in range(3):
        e=np.zeros(3); e[i]=h; g[i]=(f(p+e)-f(p-e))/(2*h)
    return g
print("== off-sheet: module vs FD of area-quadrature potential (independent oracle) ==")
for q in (np.array([0.3,0.3,0.8]), np.array([-0.4,0.2,0.3]), np.array([0.2,0.7,0.12]), np.array([0.5,0.5,2.0])):
    u=sheet_velocity(q,TRI,MU0,GRAD); uq=fd(phi,q)
    print("  q",np.round(q,3),"module",np.round(u,8),"oracle",np.round(uq,8),
          "rel",f"{np.linalg.norm(u-uq)/np.linalg.norm(uq):.2e}")
print("== PV / jump at centroid (mu linear) ==")
c=TRI.mean(axis=0); gamma=np.cross(GRAD,N)
res=sheet_velocity_pv(c,TRI,MU0,GRAD)
print("  pv",np.round(res["principal_value"],8),"conv",f"{res[chr(99)+chr(111)+chr(110)+chr(118)+chr(101)+chr(114)+chr(103)+chr(101)+chr(110)+chr(99)+chr(101)]:.2e}")
print("  jump",np.round(res["jump"],8),"  gamma x n",np.round(np.cross(gamma,N),8),"  normal jump",f"{res[chr(110)+chr(111)+chr(114)+chr(109)+chr(97)+chr(108)+chr(95)+chr(106)+chr(117)+chr(109)+chr(112)]:.2e}")
print("== const mu: ring closed form vs exact solid-angle gradient ==")
for q in (np.array([0.3,0.3,0.8]), np.array([-0.4,0.2,0.3])):
    u=sheet_velocity(q,TRI,1.0,np.zeros(3)); uk=fd(lambda s: doublet_potential(s,TRI,1.0),q)
    print("  q",np.round(q,3),"rel",f"{np.linalg.norm(u-uk)/np.linalg.norm(uk):.2e}", "ring-rel",
          f"{np.linalg.norm(triangle_ring_velocity(q,TRI,1.0)-uk)/np.linalg.norm(uk):.2e}")
