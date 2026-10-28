import numpy as np
from panel.wake_velocity import triangle_normal, solid_angle, _edge_quadrature
TRI=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]); N=triangle_normal(TRI)
def midpoints(n):
    P=[];W=[]
    for i in range(n):
        for j in range(n-i):
            P.append(((i+1/3)/n,(j+1/3)/n)); W.append(0.5/n/n)
            if i+j<n-1: P.append(((i+2/3)/n,(j+2/3)/n)); W.append(0.5/n/n)
    b=np.array(P); a=b[:,0]; c=b[:,1]
    return np.stack([1-a-c,a,c],axis=1)@TRI, np.array(W)
QP,QW=midpoints(80)
print("identity  int_S r/|r|^3 dS  vs  closed_int (1/|r|)(t x n) dl - Omega n")
for p in (np.array([0.3,0.3,0.8]), np.array([-0.4,0.2,0.3]), np.array([0.5,0.5,2.0]), np.array([0.2,0.7,0.12])):
    r=p[None,:]-QP; R=np.linalg.norm(r,axis=1)
    area=np.sum(QW[:,None]*r/R[:,None]**3,axis=0)
    bnd=np.zeros(3)
    for k in range(3):
        a,b=TRI[k],TRI[(k+1)%3]; t=(b-a)/np.linalg.norm(b-a)
        pts,w=_edge_quadrature(a,b); rel=p[None,:]-pts; Rr=np.linalg.norm(rel,axis=1)
        bnd+=np.cross(t,N)*np.sum(w/Rr)
    red=bnd-solid_angle(p,TRI)*N
    print("  p",np.round(p,3),"rel",f"{np.linalg.norm(area-red)/np.linalg.norm(area):.2e}")
