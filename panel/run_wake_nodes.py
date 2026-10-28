"""Phase B1 evidence: material-node velocity and advancement on a synthetic sheet.

Run:  python -m panel.run_wake_nodes [--nx 6] [--ny 6] [--steps 6] [--json out.json]

The case is analytic: a regular wake strip with an elliptic spanwise doublet
strength (zero at the free lateral edges) and a fixed TE attachment row.  It
carries a known vortex sheet, so the two velocity estimators -- the PV+local node
velocity of Phase B and the two-sided face-centroid probe of the old P1 free wake
-- can be compared on one and the same strength field.  That is the iteration-0
comparison issue #65 asks for; the coupled re-solve of mu and Gamma is Phase B2.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .wake_nodes import (StepGates, advance, boundary_nodes, grid_sheet,
                         node_velocity, probe_velocity_face_centers,
                         sheet_diagnostics)


def _bucket_report(sheet, vel, nx, ny, span):
    faces = np.asarray(sheet.faces, np.int64)
    i = np.arange(sheet.n_nodes) % (nx + 1)
    j = np.arange(sheet.n_nodes) // (nx + 1)
    eta = np.abs(sheet.vertices[:, 1]) / span
    core = (i > 0) & (i < nx) & (eta < 0.8)
    tip = eta >= 0.8
    conv = vel.convergence_array
    out = dict(
        core=dict(nodes=int(core.sum()), median=float(np.median(conv[core])),
                  max=float(np.max(conv[core]))),
        near_tip=dict(nodes=int(tip.sum()), median=float(np.median(conv[tip])),
                      max=float(np.max(conv[tip]))),
        te_row=dict(nodes=int((i == 0).sum()), median=float(np.median(conv[i == 0])),
                    max=float(np.max(conv[i == 0]))),
        boundary_nodes=int(boundary_nodes(sheet).sum()),
        speed_core_median=float(np.median(np.linalg.norm(vel.pv[core], axis=1))),
        speed_max=float(np.linalg.norm(vel.pv, axis=1).max()),
    )
    face_core = core[faces].all(axis=1)
    probe = probe_velocity_face_centers(sheet)
    pv_face = vel.pv[faces].mean(axis=1)
    uref = float(np.linalg.norm(sheet.freestream))
    out["estimator"] = dict(
        faces_core=int(face_core.sum()),
        core_max_abs=float(np.max(np.linalg.norm(
            probe[face_core] - pv_face[face_core], axis=1))) if face_core.any() else None,
        core_rel_to_uref=float(np.max(np.linalg.norm(
            probe[face_core] - pv_face[face_core], axis=1)) / uref) if face_core.any() else None,
        all_max_abs=float(np.max(np.linalg.norm(probe - pv_face, axis=1))),
    )
    return out, core


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nx", type=int, default=6)
    ap.add_argument("--ny", type=int, default=6)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--ds", type=float, default=0.08)
    ap.add_argument("--eps-convergence", type=float, default=2e-2)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)

    sheet = grid_sheet(args.nx, args.ny, loading="elliptic", mu0=0.3)
    span = float(np.max(np.abs(sheet.vertices[:, 1])))
    vel = node_velocity(sheet)
    buckets, core = _bucket_report(sheet, vel, args.nx, args.ny, span)
    diag = sheet_diagnostics(sheet, vel, bands=3)

    gates = StepGates(max_eps_convergence=args.eps_convergence)
    history = []
    current = sheet
    for k in range(args.steps):
        v = node_velocity(current)
        res = advance(current, v.pv, args.ds, gates=gates,
                      eps_convergence=v.convergence_array)
        v_new = node_velocity(res.sheet)
        d = sheet_diagnostics(res.sheet, v_new, bands=3)
        face_core = core[np.asarray(sheet.faces, np.int64)].all(axis=1)
        history.append(dict(
            step=k + 1,
            accepted=int(res.accepted.sum()),
            accepted_core=int((res.accepted & core).sum()),
            reasons=res.reasons,
            slip_rms=d["overall"]["slip_rms"],
            slip_max=d["overall"]["slip_max"],
            dcp_rms=d["overall"]["dcp_rms"],
            eps_convergence=d["eps_convergence"],
            area_min=d["quality"]["area_min"],
            aspect_p90=d["quality"]["aspect_p90"],
            step_max=res.stats["step_max"],
        ))
        current = res.sheet

    v0 = node_velocity(sheet)
    arc = advance(sheet, v0.pv, args.ds, mode="arc_length", gates=gates,
                  eps_convergence=v0.convergence_array)
    ptime = advance(sheet, v0.pv, args.ds, mode="pseudo_time", gates=gates,
                    eps_convergence=v0.convergence_array)
    moved = arc.accepted & ptime.accepted
    def _angle(a, b):
        na = np.linalg.norm(a, axis=1)
        nb = np.linalg.norm(b, axis=1)
        ok = (na > 0) & (nb > 0)
        if not ok.any():
            return 0.0
        cos = np.einsum("ij,ij->i", a[ok], b[ok]) / (na[ok] * nb[ok])
        return float(np.degrees(np.max(np.arccos(np.clip(cos, -1.0, 1.0)))))

    modes = dict(
        moved=int(moved.sum()),
        direction_angle_max_deg=_angle(
            arc.sheet.vertices[moved] - sheet.vertices[moved],
            ptime.sheet.vertices[moved] - sheet.vertices[moved]),
        arc_step_max=arc.stats["step_max"],
        pseudo_step_max=ptime.stats["step_max"],
    )

    report = dict(
        case=dict(nodes=sheet.n_nodes, faces=sheet.n_faces, nx=args.nx, ny=args.ny,
                  span=span, freestream=list(map(float, sheet.freestream)),
                  loading="elliptic", mu0=0.3),
        iteration0=dict(**buckets, diagnostics=diag),
        march=history,
        advance_modes=modes,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
