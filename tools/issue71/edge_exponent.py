"""Measure the sharp-edge exponent in the actual closed-body P1 solution (issue #71, PR 74).

Prediction (docs/ISSUE71_EDGE_SINGULARITY_DERIVATION.md, panel/p1_edge_singularity.py):
a straight edge with fluid wedge angle Omega carries phi ~ r**lambda cos(lambda*theta),
lambda = pi/Omega, so the antisymmetric trace pair behaves like +/- r**lambda and the
velocity like r**(lambda-1). For the zero-thickness LE/TE cusp Omega -> 2*pi, lambda = 1/2.

This script does not enrich anything. It measures, in the *existing* P1 solution:
  (a) the potential jump across the LE and the TE at mid span, as a function of the
      physical distance to the edge, and the fitted exponent;
  (b) the near-TE velocity slope for two different fit windows, to show that a velocity
      exponent fit is window dependent and must not be used as an acceptance quantity.

Run from a worktree that contains panel/:
    python tools/issue71/edge_exponent.py
"""
from __future__ import annotations
import numpy as np

from panel.p1_lifting_g1 import make_wing
from panel.p1_lifting_g2 import combine_wings, assemble_multi

ALPHA = 5.0
SPAN = 8
CHORDS = (8, 12, 16, 24)


def build(chord_panels):
    wing = make_wing(chord_panels=chord_panels, span_panels=SPAN, camber=0.02)
    mesh = combine_wings([wing], names=["main_wing"], origins=[(0.0, 0.0, 0.0)],
                         reference_chord=wing.chord, reference_point=wing.reference_point)
    return wing, mesh, assemble_multi(mesh)


def solve_closed_body(ops):
    """Closed body: no TE cut, no wake, Gamma=0 -> pure Dirichlet trace least squares."""
    A, W = ops["A"], ops["W"]
    surface = ops["mesh"]
    nm = A.shape[1]
    alpha = np.deg2rad(ALPHA)
    U = np.array([np.cos(alpha), 0.0, np.sin(alpha)])
    weight = np.sqrt(surface.areas / surface.areas.mean())
    B = weight[:, None] * np.column_stack((A, W))
    rhs = weight * (-surface.centers @ U)
    x, *_ = np.linalg.lstsq(B, rhs, rcond=None)
    residual = float(np.linalg.norm(B @ x - rhs) / np.sqrt(len(rhs)))
    return x[:nm], residual


def midspan_traces(wing, mesh, mu):
    """Upper/lower nodal potential at one mid-span station, ordered by chordwise x."""
    xyz = np.asarray(mesh.xyz)
    dof = np.asarray(mesh.dof_face, int)
    bank = np.asarray(mesh.face_bank, int)
    # duplicated TE potential dofs live on the same geometric TE nodes
    coords = np.vstack([xyz, xyz[np.asarray(wing.te_geo)[1:-1]]])
    y = 0.25 * (np.max(xyz[:, 1]) + np.min(xyz[:, 1]))
    out = {}
    for b in (1, -1):
        ids = np.unique(dof[bank == b].ravel())
        ids = [i for i in ids if abs(coords[i, 1] - y) < 1e-12]
        ids.sort(key=lambda i: coords[i, 0])
        out[b] = (np.array([coords[i, 0] for i in ids]), np.array([mu[i] for i in ids]))
    n = min(len(out[1][0]), len(out[-1][0]))
    return out[1][0][:n], out[1][1][:n] - out[-1][1][:n]


def fit(s, q, lo, hi, power):
    m = (s >= lo) & (s <= hi) & (s > 0)
    if m.sum() < 2:
        return None
    slope = float(np.polyfit(np.log(s[m]), np.log(np.abs(q[m])), 1)[0])
    amp = float(np.exp(np.mean(np.log(np.abs(q[m])) - power * np.log(s[m]))))
    return slope, amp, int(m.sum())


def main():
    print("closed-body P1 solution, camber 0.02, alpha 5 deg, mid-span station")
    print("lambda = pi/Omega = 1/2 for the zero-thickness LE/TE cusp\n")
    header = "c    residual   LE slope/amp   TE dmu at 1 cell   |V| slope (s<0.2)  |V| slope (0.02..0.15)"
    print(header)
    for c in CHORDS:
        wing, mesh, ops = build(c)
        mu, residual = solve_closed_body(ops)
        xs, dmu = midspan_traces(wing, mesh, mu)
        s_te, s_le = 1.0 - xs, np.maximum(xs, 0.0)
        le = fit(s_le, dmu, 0.005, 0.05, 0.5)
        near_te = dmu[-2] if len(dmu) > 2 else np.nan
        centres = np.asarray(ops["mesh"].centers)
        velocity = np.linalg.norm(np.einsum("fij,j->fi", ops["grad"], mu), axis=1)
        bank = np.asarray(mesh.face_bank, int)
        y = 0.25 * (np.max(np.asarray(mesh.xyz)[:, 1]) + np.min(np.asarray(mesh.xyz)[:, 1]))
        base = (np.abs(centres[:, 1] - y) < 0.6) & (bank != 0)
        s_face = 1.0 - centres[:, 0]
        slopes = []
        for lo, hi in ((0.0, 0.2), (0.02, 0.15)):
            m = base & (s_face >= lo) & (s_face <= hi)
            slopes.append(float(np.polyfit(np.log(s_face[m]), np.log(velocity[m]), 1)[0])
                          if m.sum() >= 3 else np.nan)
        print("c=%2d %9.3e  %s  %s   %+.3f              %+.3f" % (
            c, residual,
            ("%+.3f/%.3f" % (le[0], le[1])) if le else "n<2",
            "%+.3f (s=%.4f)" % (near_te, s_te[-2]) if len(dmu) > 2 else "-",
            slopes[0], slopes[1]))


if __name__ == "__main__":
    main()
