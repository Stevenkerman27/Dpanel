
import numpy as np
from panel.p1_geometry_contact import classify_swept_wall
TOL = 1e-6
WALL = np.array([[-.5, 0., 0.], [1.5, 0., 0.], [.5, .8, 0.]])
SHARED = np.array([[0., 0., 0.], [1., 0., 0.]])
old = np.array([[0., 0., 0.], [1., 0., 0.], [.3, .02, .02]])
new = np.array([[0., 0., 0.], [1., 0., 0.], [.3, -.03, -.6]])
hull = np.concatenate((old, new), 0)
for g in (0.05, 0.):
    c = classify_swept_wall(hull, WALL, tol=TOL, declared='edge', declared_region=SHARED, graze_tol=g)
    print("graze", g, c)
deep_inside = np.array([[0., 0., 0.], [1., 0., 0.], [.3, .02, -.6]])
hull2 = np.concatenate((old, deep_inside), 0)
print("deep inside:", classify_swept_wall(hull2, WALL, tol=TOL, declared='edge', declared_region=SHARED, graze_tol=0.05))
