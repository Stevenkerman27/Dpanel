
"""Bisect the P1 inconsistency: closed wing with no lift, then TE cut with Gamma=0, then Gamma!=0."""
import numpy as np
from scipy.linalg import null_space

from panel.p1_lifting_g1 import make_wing
from panel.p1_lifting_g2 import combine_wings, assemble_multi

ALPHA = 5.0
SPAN = 8
PROBES = [0.02, 0.10, 0.30, 0.60, 0.90, 0.98]


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
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return np.array([1.0 - v - w, v, w])


def probe_face(mesh, ops, p, bank):
    tri = np.asarray(mesh.xyz)[np.asarray(mesh.faces)]
    cen = np.asarray(ops["mesh"].centers)
    sel = np.where((mesh.face_bank == bank) & (np.abs(cen[:, 1] - p[1]) < 0.6 * SPAN))[0]
    order = sel[np.argsort(np.linalg.norm(cen[sel] - p, axis=1))][:40]
    best = None
    for f in order:
        w = bary(tri[f], p)
        if np.all(w > -1e-9):
            return f
        m = float(np.min(w))
        if best is None or m > best[1]:
            best = (f, m)
    return best[0]


def run(c, stage):
    mesh, ops = build(c)
    A, W, J = ops["A"], ops["W"], ops["J"]
    ms = ops["mesh"]; nm = A.shape[1]; ng = W.shape[1]
    alpha = np.deg2rad(ALPHA)
    U = np.array([np.cos(alpha), 0.0, np.sin(alpha)])
    rhs = -ms.centers @ U
    weights = np.sqrt(ms.areas / ms.areas.mean())
    B = weights[:, None] * np.column_stack((A, W))
    bw = weights * rhs
    g = mesh.groups[0]
    y = np.asarray(g.gamma_y)
    half = 0.5 * np.max(np.abs(y))
    shape = np.sqrt(np.clip(1.0 - (y[1:-1] / half) ** 2, 0.0, None))
    gamma = np.zeros(ng) if stage == "cut0" else 0.60 * shape
    if stage == "nocut":
        x, *_ = np.linalg.lstsq(B, bw, rcond=None)
        resid = float(np.linalg.norm(B @ x - bw) / np.sqrt(len(bw)))
    else:
        target = np.r_[np.zeros(nm), gamma]
        N = null_space(J, rcond=1e-11)
        z, *_ = np.linalg.lstsq(B @ N, bw - B @ target, rcond=None)
        x = target + N @ z
        resid = float(np.linalg.norm(B @ x - bw) / np.sqrt(len(bw)))
    v = np.einsum("fij,j->fi", ops["grad"], x[:nm])
    sp = np.linalg.norm(v, axis=1)
    cen = np.asarray(ms.centers)
    banks = sorted(set(int(b) for b in mesh.face_bank))
    mez = {b: float(np.mean(cen[mesh.face_bank == b, 2])) for b in banks}
    upper = max(banks, key=lambda b: mez[b])
    xs = (cen[:, 0] - cen[:, 0].min()) / (cen[:, 0].max() - cen[:, 0].min())
    yb = 0.5 * np.max(np.abs(cen[:, 1]))
    out = []
    for xc in PROBES:
        cand = np.where((mesh.face_bank == upper) & (np.abs(cen[:, 1] - yb) < 0.6))[0]
        zc = float(np.mean(cen[cand[np.argsort(np.abs(xs[cand] - xc))][:3], 2]))
        p = np.array([xs.min() + xc * (xs.max() - xs.min()), yb, zc])
        f = probe_face(mesh, ops, p, upper)
        out.append(float(sp[f]))
    return out, resid


def main():
    for stage in ("nocut", "cut0", "cutg"):
        print("\n== %s" % stage)
        seq = {}
        for c in (4, 6, 8, 12, 16, 24):
            vals, resid = run(c, stage)
            seq[c] = vals
            print("  c=%2d resid=%.2e  %s" % (c, resid, " ".join("%6.3f" % v for v in vals)))
        ks = list(seq)
        for a, b in zip(ks, ks[1:]):
            rel = [(seq[b][i] - seq[a][i]) / max(abs(seq[a][i]), 1e-9) for i in range(len(PROBES))]
            print("   %2d->%2d %s" % (a, b, " ".join("%+7.2f%%" % (100 * r) for r in rel)))


if __name__ == "__main__":
    main()
