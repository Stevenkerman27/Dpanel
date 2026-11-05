"""Collect completed material-map checks and audit wake self-intersections."""
import argparse
import json
from pathlib import Path

import numpy as np

from panel.p1_geometry_contact import _clip_polygon
from panel.shed_groups import _segment_triangle_t
from tools.issue92.b3_dpanel import _wing
from panel.p1_lifting_g2 import combine_wings
from tools.wake_mesh_interventions import refine_wake


def overlap_area(p,q,tol):
    normal=np.cross(p[1]-p[0],p[2]-p[0])
    normal/=np.linalg.norm(normal)
    other=np.cross(q[1]-q[0],q[2]-q[0])
    other/=np.linalg.norm(other)
    if abs(normal@other)<1.-1e-9 or np.max(abs((q-p[0])@normal))>tol:
        return 0.
    axis=int(np.argmax(abs(normal)))
    a=np.delete(p-p[0],axis,axis=1)
    b=np.delete(q-p[0],axis,axis=1)
    if (a[1,0]-a[0,0])*(a[2,1]-a[0,1])-(a[1,1]-a[0,1])*(a[2,0]-a[0,0])<0:
        a=a[::-1]
    clipped=_clip_polygon(b,a)
    if len(clipped)<3:
        return 0.
    xy=clipped-clipped[0]
    return abs(float(np.sum(xy[:,0]*np.roll(xy[:,1],-1)-xy[:,1]*np.roll(xy[:,0],-1))))/(2*abs(normal[axis]))

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--runs",type=Path,required=True)
parser.add_argument("--out",type=Path,required=True)
parser.add_argument("--name-suffix",default="")
args=parser.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)
root=Path(__file__).resolve().parents[1]
specs=json.loads((root/"docs/issue92/b3_cases.json").read_text())["cases"]
report=dict(runs={},physical_accuracy_accepted=False)
for folder in sorted(args.runs.iterdir()):
    if not folder.is_dir() or not (folder/"history.json").exists():
        continue
    if not folder.name.endswith(args.name_suffix):
        continue
    history=json.loads((folder/"history.json").read_text())
    rows=history["rows"]
    report["runs"][folder.name]=dict(status=history["status"],solver_status=history.get("solver_status"),
        converged=history.get("converged"),seconds=history["seconds"],states=len(rows),
        initial_CL=rows[0]["CL"],final_CL=rows[-1]["CL"],final_CL_gamma=rows[-1]["CL_gamma"],
        max_declared_attachment=max(r["declared_attachment_max"] for r in rows),
        max_wall_contact=max(r["wall_contact_max"] for r in rows),
        minimum_signed_jacobian=min(r["min_signed_jacobian"] for r in rows),
        original_inner_failures=sum(not r["nonlinear"]["success"] for r in rows),
        strict_floor_failures=sum(r["nonlinear"]["wall_over_floor"]>1.05 for r in rows),
        final_normal_slip=max(v["normal_slip_rms"] for v in rows[-1]["wakes"].values()),
        final_raw_map_residual=next((r["raw_map_residual"] for r in reversed(rows) if "raw_map_residual" in r),None),
        backtracked_updates=sum(r.get("orientation_backtracking",{}).get("factor",1.)<1. for r in rows),
        checkpoints={})
    wing=_wing(specs[history["case"]])
    mesh=combine_wings([wing])
    for receipt in history.get("refinements",[]):
        mesh,actual=refine_wake(mesh,"streamwise",max_projected_edge=2.*wing.chord)
        assert actual["split_edges"]==receipt["split_edges"]
    faces=mesh.groups[0].wake_faces
    xyz=mesh.groups[0].wake_xyz
    report["runs"][folder.name]["wake_faces"]=len(faces)
    report["runs"][folder.name]["max_reference_dx"]=float(np.max(abs(xyz[faces[:,[1,2,0]],0]-xyz[faces,0])))
    tol=1e-10*float(wing.chord)
    for row in rows:
        if "geometry" not in row:
            continue
        with np.load(folder/row["snapshot"]) as data:
            tri=data["wake_xyz"][faces]
        lower,upper=tri.min(axis=1),tri.max(axis=1)
        violations=[];roundoff_pairs=0
        for i in range(len(faces)):
            ids=np.flatnonzero(np.all(lower[i]<=upper[:i]+tol,axis=1)
                               &np.all(lower[:i]<=upper[i]+tol,axis=1))
            for j in ids:
                common=np.intersect1d(faces[i],faces[j])
                overlap=overlap_area(tri[i],tri[j],tol)
                area_roundoff=64*np.finfo(float).eps*max(float(np.ptp(np.vstack((tri[i],tri[j])),axis=0).max()),1e-12)**2
                if overlap>area_roundoff:
                    violations.append([i,int(j),"coplanar_overlap"])
                    continue
                if overlap>0:
                    roundoff_pairs+=1
                # Adjacent triangles can meet only at their shared simplex.
                points=[]
                for a,b in ((tri[i],tri[j]),(tri[j],tri[i])):
                    for start,end in zip(a,np.roll(a,-1,axis=0)):
                        t=_segment_triangle_t(start,end,b,tol)
                        if t is not None:
                            points.append(start+np.clip(t,0.,1.)*(end-start))
                if not points:
                    continue
                if len(common):
                    shared=tri[i][np.isin(faces[i],common)]
                    p=np.asarray(points)
                    distance=np.linalg.norm(p[:,None]-shared[None],axis=2).min(axis=1)
                    if len(shared)==2:
                        edge=shared[1]-shared[0]
                        t=np.clip((p-shared[0])@edge/(edge@edge),0.,1.)
                        distance=np.linalg.norm(p-shared[0]-t[:,None]*edge,axis=1)
                    points=p[distance>tol].tolist()
                if points:
                    violations.append([i,int(j),"intersection"])
        quality=row["geometry"]
        report["runs"][folder.name]["checkpoints"][str(row["iteration"])]=dict(
            CL=row["CL"],signed=row["min_signed_jacobian"],
            new_wall_violations=quality["new_or_deeper_violation_count"],
            self_intersection_pairs=len(violations),first_self_pairs=violations[:20],
            coplanar_roundoff_pairs=roundoff_pairs,
            snapshot_sha256=row["snapshot_sha256"])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(folder.name,"collected",len(rows),"states",flush=True)
