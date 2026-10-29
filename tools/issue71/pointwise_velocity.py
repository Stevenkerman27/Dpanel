
"""Pointwise convergence of the P1 surface speed at fixed physical probe points.

A consistent discretisation must give a Cauchy sequence of |V| at a FIXED point on the
surface as the mesh is refined.  We evaluate |V| at the face that contains the probe
point (the P1 velocity is constant per face), so each refinement samples the field at
the same physical location with a smaller stencil.
"""
import numpy as np
from panel.p1_lifting_g1 import make_wing
from panel.p1_lifting_g2 import combine_wings, assemble_multi, solve_multi

ALPHA = 5.0
SPAN = 8


def build(c):
    w = make_wing(chord_panels=c, span_panels=SPAN, camber=0.02)
    mesh = combine_wings([w], names=["main_wing"], origins=[(0.0, 0.0, 0.0)],
                         reference_chord=w.chord, reference_point=w.reference_point)
    return mesh, assemble_multi(mesh)


def bary(tri_f, p):
    a, b, c = tri_f
    v0 = b - a; v1 = c - a; v2 = p - a
    d00 = v0 @ v0; d01 = v0 @ v1; d11 = v1 @ v1
    d20 = v2 @ v0; d21 = v2 @ v1
    den = d00 * d11 - d01 * d01
    if abs(den) < 1e-300:
        return None
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return np.array([1.0 - v - w, v, w])


def probe(mesh, p, bank):
    tri = np.asarray(mesh.xyz)[np.asarray(mesh.faces)]
    cen = tri.mean(axis=1)
    sel = np.where((mesh.face_bank == bank) & (np.abs(cen[:, 1] - p[1]) < 0.6 * SPAN))[0]
    d = np.linalg.norm(cen[sel] - p, axis=1)
    order = sel[np.argsort(d)][:40]
    best = None
    for f in order:
        w = bary(tri[f], p)
        if w is None:
            continue
        if np.all(w > -1e-9):
            return f, w
        m = np.min(w)
        if best is None or m > best[1]:
            best = (f, m)
    return best[0], None


def main():
    probes = [("upper", 0.02), ("upper", 0.10), ("upper", 0.30), ("upper", 0.60),
              ("upper", 0.90), ("upper", 0.98), ("lower", 0.30), ("lower", 0.90)]
    print("pointwise |V| at fixed probe points, mid-span, alpha=5deg, pressure Kutta")
    series = {}
    for c in (4, 6, 8, 12, 16, 24):
        mesh, ops = build(c)
        sol = solve_multi(mesh, ALPHA, kutta="pressure", operators=ops, max_nfev=200)
        mu = np.asarray(sol["mu"])
        v = np.einsum("fij,j->fi", ops["grad"], mu)
        sp = np.linalg.norm(v, axis=1)
        cen = np.asarray(ops["mesh"].centers)
        xs = (cen[:, 0] - cen[:, 0].min()) / (cen[:, 0].max() - cen[:, 0].min())
        banks = sorted(set(int(b) for b in mesh.face_bank))
        mez = {b: float(np.mean(np.asarray(ops["mesh"].centers)[mesh.face_bank == b, 2])) for b in banks}
        upper = max(banks, key=lambda b: mez[b])
        lower = min(banks, key=lambda b: mez[b])
        line = []
        for bank_name, xc in probes:
            bank = upper if bank_name == "upper" else lower
            y = cen[np.argmax(np.abs(cen[:, 1]))][1]
            yb = y * 0.5 if bank_name == "upper" else -y * 0.5
            p = np.array([xs.min() + xc * (xs.max() - xs.min()), yb, 0.0])
            # project onto the surface: use the nearest face centroid's z
            tri = np.asarray(mesh.xyz)[np.asarray(mesh.faces)]
            c2 = tri.mean(axis=1)
            cand = np.where((mesh.face_bank == bank) & (np.abs(c2[:, 1] - p[1]) < 0.5))[0]
            zc = c2[cand[np.argmin(np.abs((c2[cand, 0] - c2[:, 0].min()) / (c2[:, 0].max() - c2[:, 0].min()) - xc))], 2]
            p[2] = zc
            f, w = probe(mesh, p, bank)
            line.append(float(sp[f]))
        series[c] = (line, float(sol["CL"]), float(sp.max()))
        print("c=%2d CL=%9.6f max|V|=%6.3f  probes=%s"
              % (c, sol["CL"], sp.max(), " ".join("%6.3f" % x for x in line)))
    keys = list(series)
    print("\nrelative change between successive refinements (negative = diverging sign/magnitude)")
    for a, b in zip(keys, keys[1:]):
        rel = [(series[b][0][i] - series[a][0][i]) / max(abs(series[a][0][i]), 1e-9) for i in range(len(probes))]
        print("c=%2d -> %2d : %s" % (a, b, " ".join("%+7.2f%%" % (100 * r) for r in rel)))


if __name__ == "__main__":
    main()
