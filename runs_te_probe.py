
import numpy as np
from panel.p1_geometry_contact import classify_swept_wall, _plane_distance, _point_inside_triangle_plane
wall = np.array([[1.5,0.,0.],[1.47092,0.05947,0.],[1.26848,0.,0.16346]])
wake_old = np.array([[1.5,0.,0.],[1.47,-0.059,0.],[1.70,-0.1,0.]])
wake_new = wake_old + np.array([0.002,-0.001,0.])
hull = np.concatenate((wake_old,wake_new))
d,n = _plane_distance(hull, wall)
print("signed distances", np.round(d,6))
print("normal", np.round(n,4))
for p in hull:
    print("inside footprint?", _point_inside_triangle_plane(p, wall, 1e-9), np.round(p,4))
print("no surface:", classify_swept_wall(hull, wall, tol=1e-9, declared='vertex', declared_region=wall[:1], graze_tol=0.02*0.2))
print("surface:   ", classify_swept_wall(hull, wall, tol=1e-9, declared='vertex', declared_region=wall[:1], graze_tol=0.02*0.2, wall_surface=wall[None]))
