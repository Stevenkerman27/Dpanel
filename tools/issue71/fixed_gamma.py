
"""Two isolating experiments for issue #71.

A) Prescribed circulation: with Gamma fixed analytically (same for every mesh), does the
   P1 surface speed at fixed points converge?  This separates "inconsistent discretisation"
   from "the Kutta selection picks a different branch".

B) Branch multiplicity: the production pressure-Kutta least squares is started at the
   signed-Kutta solution.  If the answer moves with the Kutta weight or the iteration cap,
   the residual landscape has several roots and the solver is not selecting a branch.
"""
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import least_squares

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


def probe_face(mesh, ops, p, bank):
    tri = np.asarray(mesh.xyz)[np.asarray(mesh.faces)]
    cen = np.asarray(ops["mesh"].centers)
    sel = np.where((mesh.face_bank == bank) & (np.abs(cen[:, 1] - p[1]) < 0.6 * SPAN))[0]
    order = sel[np.argsort(np.linalg.norm(cen[sel] - p, axis=1))][:40]
    best = None
    for f in order:
        w = bary(tri[f], p)
        if w is None:
            continue
        if np.all(w > -1e-9):
            return f
        m = float(np.min(w))
        if best is None or m > best[1]:
            best = (f, m)
    return best[0]


def solve_fixed_gamma(mesh, ops, gamma_fixed):
    A, W, J = ops["A"], ops["W"], ops["J"]
    ms = ops["mesh"]; nm = A.shape[1]
    alpha = np.deg2rad(ALPHA)
    U = np.array([np.cos(alpha), 0.0, np.sin(alpha)])
    rhs = -ms.centers @ U
    weights = np.sqrt(ms.areas / ms.areas.mean())
    B = weights[:, None] * np.column_stack((A, W))
    bw = weights * rhs
    target = np.r_[np.zeros(nm), gamma_fixed]
    N = null_space(J, rcond=1e-11)
    z, *_ = np.linalg.lstsq(B @ N, bw - B @ target, rcond=None)
    state = target + N @ z
    return state


def part_a():
    probes = [("u", 0.02), ("u", 0.10), ("u", 0.30), ("u", 0.60), ("u", 0.90), ("u", 0.98)]
    print("A) prescribed elliptical Gamma, pointwise |V| at fixed mid-span points")
    seq = {}
    for c in (4, 6, 8, 12, 16, 24):
        mesh, ops = build(c)
        g = mesh.groups[0]
        y = np.asarray(g.gamma_y)
        half = 0.5 * np.max(np.abs(y))
        shape = np.sqrt(np.clip(1.0 - (y[1:-1] / half) ** 2, 0.0, None))
        gamma = 0.60 * shape
        state = solve_fixed_gamma(mesh, ops, gamma)
        v = np.einsum("fij,j->fi", ops["grad"], state[: ops["A"].shape[1]])
        sp = np.linalg.norm(v, axis=1)
        cen = np.asarray(ops["mesh"].centers)
        banks = sorted(set(int(b) for b in mesh.face_bank))
        mez = {b: float(np.mean(cen[mesh.face_bank == b, 2])) for b in banks}
        upper = max(banks, key=lambda b: mez[b])
        xs = (cen[:, 0] - cen[:, 0].min()) / (cen[:, 0].max() - cen[:, 0].min())
        yb = 0.5 * np.max(np.abs(cen[:, 1]))
        line = []
        for _n, xc in probes:
            cand = np.where((mesh.face_bank == upper) & (np.abs(cen[:, 1] - yb) < 0.6))[0]
            zc = float(np.mean(cen[cand[np.argsort(np.abs(xs[cand] - xc))][:3], 2]))
            p = np.array([xs.min() + xc * (xs.max() - xs.min()), yb, zc])
            f = probe_face(mesh, ops, p, upper)
            line.append(float(sp[f]))
        seq[c] = line
        print("  c=%2d  %s" % (c, " ".join("%6.3f" % x for x in line)))
    ks = list(seq)
    print("  successive relative change:")
    for a, b in zip(ks, ks[1:]):
        rel = [(seq[b][i] - seq[a][i]) / max(abs(seq[a][i]), 1e-9) for i in range(len(probes))]
        print("   %2d->%2d %s" % (a, b, " ".join("%+7.2f%%" % (100 * r) for r in rel)))


def part_b():
    print("\nB) pressure-Kutta branch dependence (production solver)")
    for c in (8, 12):
        mesh, ops = build(c)
        for w in (1.0, 5.0, 50.0):
            for nfev in (80, 400):
                s = solve_multi(mesh, ALPHA, kutta="pressure", operators=ops, kut_weight=w, max_nfev=nfev)
                print("  c=%2d kut_weight=%5.1f max_nfev=%3d  CL=%+9.6f sum(Gamma)=%+8.4f nfev=%3d pk_rms=%.2e"
                      % (c, w, nfev, s["CL"], float(np.sum(s["gamma"])), s["nonlinear"]["nfev"],
                         s["pressure_kutta_rms"]))


if __name__ == "__main__":
    part_a()
    part_b()
