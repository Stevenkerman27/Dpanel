
import numpy as np
from panel.p1_geometry_contact import _surface_distance

def seg_dist(p, a, b):
    ab = b-a; t = np.dot(p-a, ab)/max(np.dot(ab, ab), 1e-300)
    t = min(max(t, 0.), 1.)
    return np.linalg.norm(p-(a+t*ab))

def exact(p, tri):
    a, b, c = tri
    n = np.cross(b-a, c-a); nn = np.linalg.norm(n)
    best = min(np.linalg.norm(p-a), np.linalg.norm(p-b), np.linalg.norm(p-c),
               seg_dist(p,a,b), seg_dist(p,b,c), seg_dist(p,c,a))
    if nn > 1e-12:
        n = n/nn
        d = np.dot(p-a, n)
        foot = p-d*n
        # barycentric inside test
        v0, v1, v2 = b-a, c-a, foot-a
        d00, d01, d11 = v0@v0, v0@v1, v1@v1
        d20, d21 = v2@v0, v2@v1
        den = d00*d11-d01*d01
        if abs(den) > 1e-300:
            u = (d11*d20-d01*d21)/den; w = (d00*d21-d01*d20)/den
            if u >= -1e-12 and w >= -1e-12 and u+w <= 1+1e-12:
                best = min(best, abs(d))
    return best

rng = np.random.default_rng(1)
bad = 0
for _ in range(200):
    tri = rng.normal(size=(3,3))
    p = rng.normal(size=3)*2
    d = _surface_distance(np.array([p]), tri)[0]
    e = exact(p, tri)
    if not np.isclose(d, e, rtol=1e-6, atol=1e-9):
        bad += 1
        if bad <= 4:
            print("MISMATCH d=%.6f exact=%.6f" % (d, e))
print("bad:", bad, "/200")
