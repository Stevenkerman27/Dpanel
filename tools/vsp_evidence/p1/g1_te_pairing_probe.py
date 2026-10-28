"""Diagnostic probe for issue #58 / PR #59: rebuild the half-CAD adapter with correct TE pairing.

Run with PYTHONPATH pointing at the PR #59 worktree (module `panel.p1_lifting_g1`):

    set PYTHONPATH=D:\3D\Projects\panel\runs\pr59_git
    python tools/vsp_evidence/p1/g1_te_pairing_probe.py

It is a probe, not a fix: the TE pairs are reconstructed from upper_te_faces/lower_te_faces
(the real cross-TE pairing) instead of trailing_edges (the TE edge chain), and the lower-bank
potential trace is duplicated at each shared TE node. See docs/ISSUE58_G1_TE_PAIRING_DEFECT.md.
"""
"""Probe: corrected DPanel half-surface adapter for PR59 G1 (TE pairs from upper/lower faces)."""
import json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, r"D:/3D/Projects/panel/runs/pr59_git")
import panel.p1_lifting_g1 as M

def from_half_fixed(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    xyz = np.asarray(data["coordinates"], float); faces = np.asarray(data["faces"], int)
    wake = np.asarray(data["wake_faces"], int)
    up_f = np.asarray(data["upper_te_faces"], int); lo_f = np.asarray(data["lower_te_faces"], int)
    half_y = np.asarray(data["gamma_stations"], float)
    tol = 1e-10 * max(1.0, float(np.ptp(xyz, axis=0).max()))
    mirror_map = np.arange(len(xyz), dtype=int)
    others = np.flatnonzero(xyz[:, 1] > tol)
    mirror_map[others] = len(xyz) + np.arange(len(others))
    reflected = xyz[others] * np.array([1., -1., 1.])
    vertices = np.vstack([xyz, reflected])
    full_faces = np.asarray(np.vstack([faces, mirror_map[faces][:, ::-1]]), dtype=int)
    full_wake = np.asarray(np.vstack([wake, mirror_map[wake][:, ::-1]]), dtype=int)
    tri = vertices[full_faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    face_bank = np.where(normal[:, 2] > 1e-12, 1, np.where(normal[:, 2] < -1e-12, -1, 0))
    pairs_up = np.concatenate([up_f, mirror_map[up_f]])
    pairs_lo = np.concatenate([lo_f, mirror_map[lo_f]])
    ys = np.r_[-half_y[:0:-1], half_y]
    used = np.unique(full_faces)
    compact = np.full(len(vertices), -1, dtype=int)
    compact[used] = np.arange(len(used))
    dface = compact[full_faces].copy(); nd = len(used)
    tips = [float(ys.min()), float(ys.max())]
    duplicated = {}
    rows = []  # (global vertex, upper dof, lower dof, y)
    for fu, fl in zip(pairs_up, pairs_lo):
        shared = sorted(set(full_faces[fu].tolist()) & set(full_faces[fl].tolist()))
        if not shared:
            continue
        for v in shared:
            y = float(vertices[v, 1])
            if y <= tips[0] + tol or y >= tips[1] - tol:
                continue
            if v not in duplicated:
                duplicated[v] = nd; nd += 1
            rows.append((v, compact[v], duplicated[v], y))
    for f in range(len(full_faces)):
        if face_bank[f] < 0:
            dface[f] = [duplicated.get(int(v), int(compact[v])) for v in full_faces[f]]
    rows.sort(key=lambda r: (r[3], r[0]))
    te_geo = np.asarray([r[0] for r in rows], int)
    p1_te = np.asarray([[r[1], r[2]] for r in rows], int)
    te_y = np.asarray([r[3] for r in rows], float)
    H = M._interpolation_basis(vertices[:, 1][np.unique(full_wake)], ys)
    wids = np.unique(full_wake)
    Ghat = np.zeros((len(vertices), len(ys) - 2)); Ghat[wids] = H[:, 1:-1]
    ref_area = float(data.get("reference_area", (ys[-1] - ys[0]) * data["reference_chord"]))
    chord = float(data.get("reference_chord", 1.))
    rp = np.asarray(data.get("reference_point", [0., 0., 0.]), float)
    return M.Wing(vertices, full_faces, face_bank, te_geo, p1_te[:, 0], p1_te[:, 1], p1_te, te_y,
                  dface, nd, vertices, full_wake, ys, Ghat, chord, float(ys[-1] - ys[0]),
                  "half CAD mirrored (fixed TE pairing)", ref_area, rp)

if __name__ == "__main__":
    path = r"D:/3D/Projects/panel/runs/half_cad_wingbody_v1/surface.json"
    wing = from_half_fixed(path)
    print("faces", len(wing.faces), "dofs", wing.ndof_mu, "te rows", len(wing.te_pairs), "gamma", wing.gamma_hat.shape[1])
    print("te y range %.3f..%.3f" % (wing.te_pair_y.min(), wing.te_pair_y.max()))
    for method in ("signed_speed", "pressure"):
        for alpha in (0.0, 5.0):
            out = M.solve(wing, alpha_deg=alpha, kutta=method)
            print("%-12s a=%4.1f CL=%+.6f CLg=%+.6f CDp=%+.6f CMy=%+.6f wall=%.2e jump=%.2e kuttaP=%s kuttaS=%.2e ngam=%.4f" % (
                method, alpha, out["CL"], out["CL_gamma"], out["CD_pressure"], out["CM_y"], out["wall_rms"], out["jump_rms"],
                ("%.2e" % out["pressure_kutta_rms"]) if out["pressure_kutta_rms"] is not None else "-", out["signed_kutta_rms"],
                float(np.max(np.abs(out["gamma"])))))