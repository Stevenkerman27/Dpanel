
"""Does the Kutta selection survive if the last chordwise cell is excluded?

For a prescribed circulation the wall P1 solve is a linear least-squares problem with
the algebraic TE jump imposed exactly.  We therefore treat Gamma as the scalar/vector
unknown of a small nonlinear problem whose residual is a Kutta estimator, and compare
estimators that do NOT evaluate |V| on the single cell that touches the trailing edge.
"""
import sys
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import least_squares

from panel.p1_lifting_g1 import make_wing
from panel.p1_lifting_g2 import combine_wings, assemble_multi

ALPHA = 5.0


def build(c, span=8, camber=0.02):
    w = make_wing(chord_panels=c, span_panels=span, camber=camber)
    mesh = combine_wings([w], names=["main_wing"], origins=[(0.0, 0.0, 0.0)],
                         reference_chord=w.chord, reference_point=w.reference_point)
    ops = assemble_multi(mesh)
    return mesh, ops


def picks(mesh, ops, skip=0, band=2):
    """Per TE pair, faces on each bank ordered by distance to the TE point."""
    ms = ops["mesh"]
    out = []
    for g in mesh.groups:
        for i, (up, down) in enumerate(g.te_pairs):
            y = float(g.te_pair_y[i]); pos = g.te_xyz[i]
            sides = []
            for bank, _dof in ((1, up), (-1, down)):
                cand = g.face_ids[mesh.face_bank[g.face_ids] == bank]
                metric = 4 * (ms.centers[cand, 0] - pos[0]) ** 2 + (ms.centers[cand, 1] - y) ** 2
                order = cand[np.argsort(metric)]
                sides.append(order[skip:skip + band])
            out.append(tuple(sides))
    return out


def solve_scaled(ops, weights, gamma_scale, pairs, estimator):
    """Least squares in (mu, gamma) with J exact; gamma = scale * gamma_ref_shape."""
    A, W, J = ops["A"], ops["W"], ops["J"]
    ms = ops["mesh"]; nm = A.shape[1]; ng = W.shape[1]
    alpha = np.deg2rad(ALPHA)
    U = np.array([np.cos(alpha), 0.0, np.sin(alpha)])
    rhs = -ms.centers @ U
    B = weights[:, None] * np.column_stack((A, W))
    bw = weights * rhs
    c = J[:, :nm]; H = -J[:, nm:]
    Nj = null_space(J, rcond=1e-11)

    def build_state(z):
        return Nj @ z

    def fun(z):
        state = build_state(z)
        mu = state[:nm]; gam = state[nm:]
        return np.r_[B @ state - bw, 10.0 * estimator(mu, gam)]

    z0 = Nj.T @ np.linalg.lstsq(np.vstack((J, 1e-3 * B)), np.r_[np.zeros(J.shape[0]), 1e-3 * bw], rcond=None)[0]
    r = least_squares(fun, z0, max_nfev=120, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    state = build_state(r.x)
    return state, float(np.linalg.norm(r.fun[: len(bw)] / weights) / np.sqrt(len(bw))), r.nfev


def main():
    print("variant convergence of CL and sum(gamma) vs chordwise refinement")
    for name in ("baseline", "skip1", "skip2", "xcomp", "band"):
        row = []
        for c in (4, 6, 8, 12, 16):
            mesh, ops = build(c)
            grad = ops["grad"]; ms = ops["mesh"]; nm = ops["A"].shape[1]
            weights = np.sqrt(ms.areas / ms.areas.mean())
            skip = 0 if name == "baseline" else (2 if name == "skip1" else 4)
            pairs = picks(mesh, ops, skip=skip, band=2)
            if name == "band":
                pairs = []
                for g in mesh.groups:
                    for i in range(len(g.te_pairs)):
                        sides = []
                        for bank in (1, -1):
                            cand = g.face_ids[mesh.face_bank[g.face_ids] == bank]
                            sel = cand[(ms.centers[cand, 0] - ms.centers[cand, 0].min())
                                       / (ms.centers[cand, 0].max() - ms.centers[cand, 0].min()) > 0.55]
                            sides.append(sel if len(sel) else cand)
                        pairs.append(tuple(sides))

            def estimator(mu, gam, pairs=pairs, name=name):
                v = np.einsum("fij,j->fi", grad, mu)
                if name == "xcomp":
                    return np.array([np.mean(v[t, 0]) - np.mean(v[b, 0]) for t, b in pairs])
                return np.array([np.mean(np.sum(v[t] ** 2, axis=1)) - np.mean(np.sum(v[b] ** 2, axis=1))
                                 for t, b in pairs])

            z0 = None
            state, wrms, nfev = solve_scaled(ops, weights, 1.0, pairs, estimator)
            mu = state[:nm]; gam = state[nm:]
            v = np.einsum("fij,j->fi", grad, mu)
            cp = 1 - np.sum(v ** 2, axis=1)
            ff = -(cp * ms.areas)[:, None] * ms.normals
            net = ff.sum(axis=0)
            alpha = np.deg2rad(ALPHA)
            lift = np.array([-np.sin(alpha), 0.0, np.cos(alpha)])
            cl = float(net @ lift / mesh.reference_area)
            row.append((c, cl, float(gam.sum()), wrms, nfev))
        print("\n== %s" % name)
        for c, cl, sg, wrms, nfev in row:
            print("  c=%2d CL=%9.6f sum(Gamma)=%9.4f wallrms=%.2e nfev=%d" % (c, cl, sg, wrms, nfev))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
