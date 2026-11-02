"""Reporting helpers that used to live in tools/issue71/galerkin_cad_sweep.py.

The sweep driver itself was retired with the Galerkin line (see
docs/RETIRED_RESEARCH.md), but three of its functions are generic: they turn a
solved state into receipts. They moved here so the live tests that assert those
receipts keep running against live code.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def clean(value):
    if isinstance(value, np.ndarray):
        return [clean(item) for item in value.tolist()]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def trace_audit_summary(mesh):
    """Keep topology acceptance evidence beside every CAD solve receipt."""
    fields = ("wall_geometric_nodes", "wall_potential_dofs", "split_nodes",
              "te_jump_rows", "contact_jump_rows", "unsplit_zero_gamma_tips",
              "contact_edges", "trailing_edges", "contact_constraints_are_not_kutta")
    out = {}
    for group in mesh.groups:
        audit = getattr(group, "trace_audit", None)
        if audit is not None:
            out[group.name] = {key: clean(audit[key]) for key in fields if key in audit}
    return out


def contact_jump_diagnostics(mesh, state, ops):
    """Evaluate the wall trace jump for either Galerkin or collocation states."""
    from .p1_wake_contact import contact_jump_report
    return contact_jump_report(mesh, state["mu"], state["gamma"], ops["slices"])

def component_pressure_loads(path, state, mesh, geometry, alpha_deg):
    """Split the pressure force by exported CAD physical component tag."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    half = np.asarray(raw.get("component_ids", []), int)
    if not len(half):
        return {}
    ids = np.r_[half, half]
    if len(ids) != len(mesh.triangles) or len(state.get("cp", [])) != len(mesh.triangles):
        return {"error": "component ID count does not match mirrored wall faces"}
    role_by_tag = {}
    provenance = raw.get("provenance", {})
    for entry in provenance.get("CAD_passes", []):
        for item in entry.get("component_classification", []):
            tag = item.get("physical_tag")
            role = item.get("selected_role")
            if tag is not None and role:
                role_by_tag[int(tag)] = str(role)
    cp = np.asarray(state["cp"], float)
    face_force = -(cp * mesh.areas)[:, None] * mesh.normals
    alpha = np.deg2rad(float(alpha_deg))
    U = np.array([np.cos(alpha), 0.0, np.sin(alpha)])
    lift = np.array([-np.sin(alpha), 0.0, np.cos(alpha)])
    cd_face = face_force @ U / geometry.reference_area
    result = {}
    for tag in np.unique(ids):
        mask = ids == tag
        force = face_force[mask].sum(axis=0)
        moment = np.cross(mesh.centers[mask] - np.asarray(geometry.reference_point),
                          face_force[mask]).sum(axis=0)
        cdi = cd_face[mask]
        local_ids = np.flatnonzero(mask)
        high_cp = local_ids[np.argsort(np.abs(cp[mask]))[-10:][::-1]]
        high_drag = local_ids[np.argsort(cd_face[mask])[-10:][::-1]]
        total = float(cdi.sum())
        ratio = float(np.abs(cdi).sum() / abs(total)) if abs(total) > 1e-15 else None
        status = ("defined" if abs(total) > 1e-15 else
                  ("no_drag_contributions" if np.abs(cdi).sum() == 0.0 else "complete_cancellation"))
        result[str(role_by_tag.get(int(tag), "component_%d" % int(tag)))] = dict(
            physical_tag=int(tag), faces=int(mask.sum()),
            CL_pressure=float(force @ lift / geometry.reference_area),
            CD_pressure=float(force @ U / geometry.reference_area),
            CM_y=float(moment[1] / (geometry.reference_area * geometry.reference_chord)),
            drag_positive_sum=float(cdi[cdi > 0].sum()),
            drag_negative_sum=float(cdi[cdi < 0].sum()),
            drag_abs_sum=float(np.abs(cdi).sum()),
            drag_cancellation_ratio=ratio,
            drag_cancellation_status=status,
            cp_abs_max=float(np.max(np.abs(cp[mask]))),
            cp_min=float(np.min(cp[mask])), cp_max=float(np.max(cp[mask])),
            top_abs_cp_faces=[dict(face=int(i), center=mesh.centers[i].tolist(),
                                  cp=float(cp[i]), area=float(mesh.areas[i]),
                                  CD_face=float(cd_face[i])) for i in high_cp],
            top_positive_drag_faces=[dict(face=int(i), center=mesh.centers[i].tolist(),
                                         cp=float(cp[i]), area=float(mesh.areas[i]),
                                         CD_face=float(cd_face[i])) for i in high_drag],
        )
    return result
