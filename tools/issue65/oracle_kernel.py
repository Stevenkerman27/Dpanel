import numpy as np, os
os.environ.setdefault("JAX_ENABLE_X64","1")
import jax.numpy as jnp
from panel.kernel import triangle_potential
from panel.wake_velocity import sheet_velocity, triangle_normal, solid_angle
TRI=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]); N=triangle_normal(TRI)
MUY=0.7; MU0=0.3+MUY/3.0; GRAD=np.array([0.,MUY,0.])
def kernel_pair(p):
    d,m=triangle_potential(jnp.asarray(p[jnp.newaxis,:]), jnp.asarray(TRI[jnp.newaxis,:]), linear_y=True)
    return float(d[0,0]), float(m[0,0])
def fd(f,p,h=1e-5):
    g=np.zeros(3)
    for i in range(3):
        e=np.zeros(3); e[i]=h; g[i]=(f(p+e)-f(p-e))/(2*h)
    return g
print("kernel D vs exact -Omega/(4pi):")
for q in (np.array([0.3,0.3,0.8]), np.array([-0.4,0.2,0.3])):
    d,m=kernel_pair(q)
    print("  q",np.round(q,3),"D",d,"-Om/4pi",-solid_angle(q,TRI)/(4*np.pi),"M",m)
print("compare gradients: kernel phi_k = MU0*D + MUY*M  (convention A) and MUY*(M - M(yc-zero)) etc")
for q in (np.array([0.3,0.3,0.8]), np.array([-0.4,0.2,0.3]), np.array([0.5,0.5,2.0])):
    u=sheet_velocity(q,TRI,MU0,GRAD)
    gk=fd(lambda s: sum(c*float(v) for c,v in zip((MU0,MUY),kernel_pair(s))), q)
    print("  q",np.round(q,3),"mine",np.round(u,7),"grad(kernelA)",np.round(gk,7),
          "rel(mine,-grad)",f"{np.linalg.norm(u+gk)/np.linalg.norm(u):.2e}")
