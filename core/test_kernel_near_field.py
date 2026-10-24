"""Near-face limits, physical edge logarithm and coordinate similarity of the kernel."""
import json
from pathlib import Path

import numpy as np
from numba import set_num_threads

from triangle_kernel import influence

set_num_threads(8)
root = Path(__file__).resolve().parent
panel = np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
normal = np.array([[0.,0.,1.]])
distances = 10.**(-np.arange(2,13,dtype=float))
targets = np.column_stack([np.full(len(distances),1./3.),np.full(len(distances),1./3.),distances])
sp, sv = influence(targets,panel,normal,np.full(len(targets),-1,dtype=np.int64),0)
dp, dv = influence(targets,panel,normal,np.full(len(targets),-1,dtype=np.int64),1)
ps, vs = influence(np.array([[1./3.,1./3.,0.]]),panel,normal,np.array([0]),0)
pd, vd = influence(np.array([[1./3.,1./3.,0.]]),panel,normal,np.array([0]),1)
records = [dict(test="near_face_source_trace",last_height=float(distances[-1]),
                max_gradient_error=float(np.max(abs(sv[-1,0]-vs[0,0]))),
                potential_error=float(abs(sp[-1,0]-ps[0,0])),
                passed=bool(np.max(abs(sv[-1,0]-vs[0,0])) < 1e-10
                            and abs(sp[-1,0]-ps[0,0]) < 1e-10)),
           dict(test="near_face_doublet_trace",
                potential_error=float(abs(dp[-1,0]-pd[0,0])),
                max_gradient_error=float(np.max(abs(dv[-1,0]-vd[0,0]))),
                passed=bool(abs(dp[-1,0]-pd[0,0]) < 1e-10
                            and np.max(abs(dv[-1,0]-vd[0,0])) < 1e-10))]
edge_targets = np.column_stack([np.full(len(distances),.5),distances,np.zeros(len(distances))])
ep, ev = influence(edge_targets,panel,normal,np.zeros(len(distances),dtype=np.int64),0)
increments = np.diff(ev[:,0,1])
expected = np.log(10.)/(2.*np.pi)
records.append(dict(test="physical_edge_logarithm",distances=distances.tolist(),
                    tangential_gradient=ev[:,0,1].tolist(),expected_increment=float(expected),
                    last_increment=float(increments[-1]),
                    passed=bool(np.isfinite(ev).all() and abs(increments[-1]-expected) < 1e-8)))
field = np.array([[.27,.32,.11],[1.2,-.6,.2]])
for kind in [0,1]:
    p, v = influence(field,panel,normal,np.full(2,-1,dtype=np.int64),kind)
    for scale in [.001,1000.]:
        pp, vv = influence(field*scale,panel*scale,normal,np.full(2,-1,dtype=np.int64),kind)
        ep = float(np.max(abs(pp/scale**(1-kind)-p)))
        eg = float(np.max(abs(vv*scale**kind-v)))
        records.append(dict(test=f"kind{kind}_scale_{scale}",potential_error=ep,gradient_error=eg,
                            passed=bool(max(ep,eg) < 1e-12)))
(root/"near_field_tests.json").write_text(json.dumps(records,indent=2,allow_nan=False),encoding="utf-8")
print(json.dumps(records,indent=2))
assert all(row["passed"] for row in records)
