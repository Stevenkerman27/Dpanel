"""Persistent fixed-topology GRASS mesh shared by Web, P1 solve and gradients."""
import json
import math
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from .p1_cst_gradient import grass_section_geometry
from .p1_lifting_g1 import Wing


def build_grass_mesh(sections, size, chord_panels=None, span_panels_per_segment=None):
    if not math.isfinite(size) or size <= 0:
        raise ValueError("网格尺寸必须为正且有限")
    leading = np.asarray(sections["leading_edge_xyz"], dtype=float)
    chord = np.asarray(sections["chord"], dtype=float)
    chord_panels = (max(3, int(math.ceil(float(chord.max()) / size)))
                    if chord_panels is None else int(chord_panels))
    span_panels = (max(2, int(math.ceil(float(np.linalg.norm(np.diff(leading, axis=0), axis=1).max()) / size)))
                   if span_panels_per_segment is None else int(span_panels_per_segment))
    wing, _, _, _ = grass_section_geometry(
        sections, chord_panels=chord_panels, span_panels_per_segment=span_panels)
    values = {}
    for field in fields(Wing):
        value = getattr(wing, field.name)
        values[field.name] = value.tolist() if isinstance(value, np.ndarray) else value
    return dict(schema="dpanel.grass_fixed_topology.v1", sections=sections,
                chord_panels=chord_panels, span_panels_per_segment=span_panels,
                wake_panels=2, wake_length=10., wing=values,
                coordinates=wing.xyz.tolist(), faces=wing.faces.tolist(),
                wake_faces=wing.wake_faces.tolist(), mirror=False)


def load_grass_wing(data):
    if data.get("schema") != "dpanel.grass_fixed_topology.v1":
        raise ValueError("Not a GRASS fixed-topology mesh")
    values = dict(data["wing"])
    integer = {"faces", "face_bank", "te_geo", "te_up", "te_down", "te_pairs",
               "dof_face", "wake_faces"}
    floating = {"xyz", "te_pair_y", "wake_xyz", "gamma_y", "gamma_hat", "reference_point"}
    for name in integer | floating:
        values[name] = np.asarray(values[name], dtype=int if name in integer else float)
    wing = Wing(**values)
    if (not np.array_equal(np.asarray(data["coordinates"]), wing.xyz)
            or not np.array_equal(np.asarray(data["faces"]), wing.faces)
            or not np.array_equal(np.asarray(data["wake_faces"]), wing.wake_faces)):
        raise ValueError("显示网格与正式求解网格不一致")
    return wing


def read_grass_mesh(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data, load_grass_wing(data)


def reconstruct_grass_mesh(data):
    geometry = {name: torch.tensor(data["sections"][name], dtype=torch.float64, requires_grad=True)
                for name in ("leading_edge_xyz", "chord", "twist_radians", "cst")}
    wing, xyz, wake, area = grass_section_geometry(
        geometry, chord_panels=data["chord_panels"],
        span_panels_per_segment=data["span_panels_per_segment"],
        wake_panels=data["wake_panels"], wake_length=data["wake_length"])
    saved = load_grass_wing(data)
    for name in ("xyz", "faces", "dof_face", "wake_xyz", "wake_faces",
                 "gamma_y", "gamma_hat", "te_pairs", "te_geo", "te_pair_y"):
        if not np.array_equal(getattr(wing, name), getattr(saved, name)):
            raise ValueError("保存网格与 GRASS 计算图不一致: " + name)
    if wing.reference_area != saved.reference_area:
        raise ValueError("保存网格与 GRASS 参考面积不一致")
    return geometry, saved, xyz, wake, area
