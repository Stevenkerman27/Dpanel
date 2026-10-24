"""Independent surface quadrature and potential-difference checks of the kernels."""
import json
from pathlib import Path

import numpy as np
from numba import set_num_threads
from numpy.polynomial.legendre import leggauss

from triangle_kernel import influence

set_num_threads(8)
root = Path(__file__).resolve().parent
vertices = np.array([[0.,0.,0.],[1.1,.1,.03],[.2,.9,-.07]])
normal = np.cross(vertices[1]-vertices[0],vertices[2]-vertices[0])
twice_area = np.linalg.norm(normal)
normal /= twice_area
center = vertices.mean(axis=0)
targets = np.array([center+.4*normal,center-.3*normal,[2.,-.7,.8]])
point, weight = leggauss(64)
u,v = np.meshgrid((point+1)/2,(point+1)/2,indexing="ij")
weight2 = np.outer(weight/2,weight/2)*(1-u)*twice_area
quadrature = (vertices[0]+u[...,None]*(vertices[1]-vertices[0])
              +((1-u)*v)[...,None]*(vertices[2]-vertices[0])).reshape(-1,3)
weight2 = weight2.ravel()
records = []
for kind in [0,1]:
    potential, velocity = influence(targets,vertices[None],normal[None],np.full(len(targets),-1),kind)
    delta = targets[:,None,:]-quadrature[None,:,:]
    distance = np.linalg.norm(delta,axis=2)
    if kind == 0:
        p_ref = (weight2[None,:]/distance).sum(axis=1)/(4*np.pi)
        v_ref = -(weight2[None,:,None]*delta/distance[...,None]**3).sum(axis=1)/(4*np.pi)
    else:
        height = delta @ normal
        p_ref = (weight2[None,:]*height/distance**3).sum(axis=1)/(4*np.pi)
        v_ref = (weight2[None,:,None]*(normal/distance[...,None]**3
                 -3*height[...,None]*delta/distance[...,None]**5)).sum(axis=1)/(4*np.pi)
    error_p = float(np.max(np.abs(potential[:,0]-p_ref)))
    error_v = float(np.max(np.abs(velocity[:,0]-v_ref)))
    records.append(dict(test=f"kind{kind}_independent_quadrature",potential_error=error_p,
                        velocity_error=error_v,passed=error_p<1e-11 and error_v<1e-11))
    fd = np.zeros_like(velocity[:,0])
    for axis in range(3):
        offset = np.eye(3)[axis]*1e-5
        plus,_ = influence(targets+offset,vertices[None],normal[None],np.full(len(targets),-1),kind)
        minus,_ = influence(targets-offset,vertices[None],normal[None],np.full(len(targets),-1),kind)
        fd[:,axis] = (plus[:,0]-minus[:,0])/(2e-5)
    error = float(np.max(np.abs(velocity[:,0]-fd)))
    records.append(dict(test=f"kind{kind}_potential_derivative",error=error,passed=error<1e-8))

_,exterior = influence(center[None],vertices[None],normal[None],np.array([0]),0)
_,above = influence((center+1e-8*normal)[None],vertices[None],normal[None],np.array([-1]),0)
_,below = influence((center-1e-8*normal)[None],vertices[None],normal[None],np.array([-1]),0)
jump = float((above[0,0]-below[0,0]) @ normal)
records.append(dict(test="source_jump",measured=jump,expected=-1.,
                    exterior_trace=float(exterior[0,0] @ normal),passed=abs(jump+1)<1e-6))

mesh = np.load(root/"reference/sphere_L2.npz")
panels = mesh["xyz"][mesh["tri"]]
normals = np.cross(panels[:,1]-panels[:,0],panels[:,2]-panels[:,0])
normals /= np.linalg.norm(normals,axis=1)[:,None]
points = np.array([[0.,0.,0.],[3.,0.,0.]])
doublet,gradient = influence(points,panels,normals,np.full(2,-1),1)
error = float(max(np.max(np.abs(doublet.sum(axis=1)-[-1.,0.])),
                  np.max(np.abs(gradient.sum(axis=1)))))
records.append(dict(test="closed_doublet_gauss_identity",error=error,passed=error<1e-12))
(root/"kernel_tests.json").write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2),flush=True)
assert all(row["passed"] for row in records)
