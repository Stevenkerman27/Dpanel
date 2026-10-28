"""Reusable two-layer lifting-surface fixture for the v2 shed-group entry.

The fixture is the accepted v1 half wing (cases/panel_half/thick_wing_two_strip.json)
duplicated at a vertical shift, so the two groups are independent closed slabs
with their own horizontal wakes. It is a *numerical* fixture: it does not claim
that a stacked pair is a real aircraft, and it is only used to exercise the
assembly, the rejection gates and the oracle.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .shed_groups import ShedGroup

DEFAULT_CASE = Path(__file__).resolve().parents[1] / "cases" / "panel_half" / "thick_wing_two_strip.json"


def load_case(path=None):
    case = Path(DEFAULT_CASE if path is None else path)
    raw = json.loads(case.read_text(encoding="utf-8"))
    return case, raw


def _group(raw, *, node_offset=0, face_offset=0, dz=0.0, gid="wing_z0",
           root_condition="symmetry_plane", role=None):
    return ShedGroup(
        group_id=gid,
        wall_faces=np.asarray(raw["faces"], dtype=np.int32) + face_offset,
        wake_faces=np.asarray(raw["wake_faces"], dtype=np.int32) + node_offset,
        shed_edges=np.asarray(raw["trailing_edges"], dtype=np.int32) + node_offset,
        upper_te=np.asarray(raw["upper_te_faces"], dtype=np.int32) + face_offset,
        lower_te=np.asarray(raw["lower_te_faces"], dtype=np.int32) + face_offset,
        stations=np.asarray(raw["gamma_stations"], dtype=float),
        station_dof=np.asarray(raw["gamma_station_dof"], dtype=np.int32),
        role=role,
        root_condition=root_condition,
        wake_z=float(dz),
    )


def two_layer_case(path=None, shift=0.4, *, groups=2):
    """Return (case_path, raw, vertices, faces, component_ids, group_list).

    shift is the vertical offset of the second surface. groups=1 keeps only the
    z=0 surface (a v1-compatible single group); groups=0 has no shed group.
    """
    case, raw = load_case(path)
    x = np.asarray(raw["coordinates"], dtype=float)
    f = np.asarray(raw["faces"], dtype=np.int32)
    tags = np.asarray(raw["component_ids"], dtype=np.int32)
    if groups == 0:
        return case, raw, x, f, tags, []
    if groups == 1:
        return case, raw, x, f, tags, [_group(raw, gid="wing_z0")]
    n = len(x)
    second = x.copy()
    second[:, 2] += float(shift)
    vertices = np.concatenate((x, second), axis=0)
    faces = np.concatenate((f, f + n), axis=0)
    component_ids = np.concatenate((tags, tags), axis=0)
    return (case, raw, vertices, faces, component_ids,
            [_group(raw, gid="wing_low"),
             _group(raw, node_offset=n, face_offset=len(f), dz=shift,
                    gid="wing_high")])


def as_surface_json(raw, vertices, faces, component_ids, groups):
    """Write-friendly v1-shaped surface dict describing the grouped fixture."""
    return dict(
        schema="dpanel.half_surface.v1",
        coordinates=np.asarray(vertices, dtype=float).tolist(),
        faces=np.asarray(faces, dtype=np.int32).tolist(),
        wake_faces=np.concatenate([np.asarray(g.wake_faces, dtype=np.int32)
                                   for g in groups], axis=0).tolist() if groups else [],
        component_ids=np.asarray(component_ids, dtype=np.int32).tolist(),
        trailing_edges=np.concatenate([np.asarray(g.shed_edges, dtype=np.int32)
                                       for g in groups], axis=0).tolist() if groups else [],
        upper_te_faces=np.concatenate([np.asarray(g.upper_te, dtype=np.int32)
                                       for g in groups], axis=0).tolist() if groups else [],
        lower_te_faces=np.concatenate([np.asarray(g.lower_te, dtype=np.int32)
                                       for g in groups], axis=0).tolist() if groups else [],
        gamma_stations=np.asarray(groups[0].stations, dtype=float).tolist() if groups else [0.0, 1.0],
        gamma_station_dof=np.asarray(groups[0].station_dof, dtype=np.int32).tolist() if groups else [0, -1],
        reference_area=float(raw["reference_area"]),
        reference_chord=float(raw["reference_chord"]),
        moment_reference=[float(v) for v in raw["moment_reference"]],
        freestream=[float(v) for v in raw["freestream"]],
        density=float(raw["density"]),
        audit=dict(fixture="thick_wing_two_strip", groups=[g.group_id for g in groups],
                   physical_accuracy_accepted=False),
        provenance=dict(type="stacked two-layer numerical fixture",
                        shift=float(groups[1].wake_z) if len(groups) > 1 else 0.0,
                        physical_accuracy_accepted=False),
    )


def group_specs(groups):
    """JSON-able group declarations (schema dpanel.half_shed_groups.v2)."""
    return dict(
        schema="dpanel.half_shed_groups.v2",
        wall_source="surface.json",
        groups=[dict(group_id=g.group_id,
                     wall_faces=np.asarray(g.wall_faces, dtype=np.int32).tolist(),
                     wake_faces=np.asarray(g.wake_faces, dtype=np.int32).tolist(),
                     shed_edges=np.asarray(g.shed_edges, dtype=np.int32).tolist(),
                     upper_te=np.asarray(g.upper_te, dtype=np.int32).tolist(),
                     lower_te=np.asarray(g.lower_te, dtype=np.int32).tolist(),
                     stations=np.asarray(g.stations, dtype=float).tolist(),
                     station_dof=np.asarray(g.station_dof, dtype=np.int32).tolist(),
                     kind=g.kind, role=g.role, root_condition=g.root_condition,
                     source_component_ids=list(g.source_component_ids or ()),
                     wake_z=g.wake_z) for g in groups])
