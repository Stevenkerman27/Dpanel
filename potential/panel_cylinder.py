"""Independent 2D constant-line-source cylinder reference; exterior trace, no circulation."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

parser = argparse.ArgumentParser()
parser.add_argument("--panels",type=int,default=128)
parser.add_argument("--out",type=Path,required=True)
args = parser.parse_args()
threadpool_limits(8)
started = time.perf_counter()
args.out.mkdir(parents=True,exist_ok=True)
angle = np.arange(args.panels+1)*2.*np.pi/args.panels
vertices = np.column_stack([np.cos(angle),np.sin(angle)])
first, last = vertices[:-1],vertices[1:]
edge = last-first
length = np.linalg.norm(edge,axis=1)
tangent = edge/length[:,None]
normal = np.column_stack([tangent[:,1],-tangent[:,0]])
center = (first+last)/2.
delta = center[:,None]-first[None]
s = np.einsum("ijk,jk->ij",delta,tangent)
h = np.einsum("ijk,jk->ij",delta,normal)
vt = -np.log((s*s+h*h)/((s-length)**2+h*h))/(4.*np.pi)
vn = -np.arctan2(h*length,h*h+s*(s-length))/(2.*np.pi)
np.fill_diagonal(vt,0.)
np.fill_diagonal(vn,-.5)
kernel = vt[:,:,None]*tangent[None]+vn[:,:,None]*normal[None]
matrix = np.einsum("ijk,ik->ij",kernel,normal)
flow = np.array([1.,0.])
sigma = solve(matrix,-normal@flow)
velocity = flow+np.einsum("ijk,j->ik",kernel,sigma)
cp = 1.-np.sum(velocity**2,axis=1)
exact_cp = 1.-4.*normal[:,1]**2
residual = np.einsum("ij,ij->i",velocity,normal)
force = -np.sum(cp[:,None]*normal*length[:,None],axis=0)/2.
result = dict(method="2D constant-line source, analytic gradient, exterior Neumann collocation",
              scope="Unbounded incompressible circular polygon; no circulation",
              panels=args.panels,cp_rmse=float(np.sqrt(np.average((cp-exact_cp)**2,weights=length))),
              normal_rms=float(np.sqrt(np.average(residual**2,weights=length))),
              normalized_source_flux=float(np.sum(sigma*length)/length.sum()),
              pressure_force_coefficient=force.tolist(),condition_1=float(np.linalg.cond(matrix,1)),
              elapsed_seconds=time.perf_counter()-started,
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
np.savez_compressed(args.out/"surface.npz",center=center,normal=normal,velocity=velocity,cp=cp,
                    exact_cp=exact_cp,sigma=sigma)
(args.out/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
print(json.dumps(result,indent=2))
