"""Report raw wall fields by surface tag and upper/lower bank."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--mesh", type=Path)
parser.add_argument("--contact-distance", type=float, default=0.1,
                    help="Contact neighborhood radius in chord lengths")
args = parser.parse_args()

run = args.run.resolve()
mesh_path = args.mesh.resolve() if args.mesh else (run.parent / "mesh" / "mesh.npz").resolve()
field_path = run / "field.npz"
result_path = run / "result.json"
mesh_metadata_path = mesh_path.with_name("mesh.json")
field = np.load(field_path, allow_pickle=False)
mesh = np.load(mesh_path, allow_pickle=False)
result = json.loads(result_path.read_text(encoding="utf-8"))
mesh_metadata = json.loads(mesh_metadata_path.read_text(encoding="utf-8"))

points = field["surface_points"]
velocity = field["surface_velocity"]
normal = field["surface_normal"]
vn = field["surface_normal_velocity"]
cp = field["cp"]
weight = field["surface_weight"]
tags = field["tags"]
faces = field["faces"]
mach = float(result["mach"])
beta = 1.0 + 0.2 * mach**2 * (1.0 - np.sum(velocity**2, axis=2))
density_ratio = beta**2.5
raw_mass_flux = density_ratio * vn * weight

physical_tags = [tag for tag in [10, 20, 30] if np.any(tags == tag)]
wall = np.isin(tags, physical_tags)
body_nodes = np.unique(faces[tags == 20]) if np.any(tags == 20) else np.array([], dtype=int)
cut_nodes = np.unique(np.r_[mesh["cut_upper_nodes"], mesh["cut_lower_nodes"]])
contact_nodes = np.intersect1d(body_nodes, cut_nodes)
chord = float(mesh_metadata["chord"])
contact_radius = args.contact_distance * chord
if len(contact_nodes):
    contact_tree = cKDTree(field["points"][contact_nodes])
    contact_distance, _ = contact_tree.query(points.reshape(-1, 3), k=1)
    contact_region = contact_distance.reshape(points.shape[:2]) < contact_radius
else:
    contact_region = np.zeros(points.shape[:2], dtype=bool)

audit = {
    "status": result.get("status", "unknown"),
    "run_directory": str(run),
    "mesh_path": str(mesh_path),
    "inputs_sha256": {
        "field_npz": hashlib.sha256(field_path.read_bytes()).hexdigest(),
        "result_json": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "mesh_npz": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        "mesh_json": hashlib.sha256(mesh_metadata_path.read_bytes()).hexdigest(),
    },
    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "ncpu_from_result": result.get("ncpu_requested"),
    "contact_definition": {
        "body_tag": 20,
        "cut_node_set": "union of mesh cut_upper_nodes and cut_lower_nodes",
        "shared_node_count": int(len(contact_nodes)),
        "distance_limit_chords": float(args.contact_distance),
        "distance_limit": float(contact_radius),
        "sample_selection": "Surface quadrature points within the distance limit of shared body/cut nodes",
    },
    "nonfinite_totals": {
        "vn": int((~np.isfinite(vn[wall])).sum()),
        "cp": int((~np.isfinite(cp[wall])).sum()),
        "surface_weight": int((~np.isfinite(weight[wall])).sum()),
        "density_ratio": int((~np.isfinite(density_ratio[wall])).sum()),
    },
    "z_zero_wall_quadrature_count": int((wall[:, None] & (points[:, :, 2] == 0.0)).sum()),
    "banks": {},
    "interpretation": (
        "Upper and lower banks are reported separately. Signed mass fluxes are not combined; "
        "these are raw field diagnostics, not an accuracy acceptance gate."
    ),
}

bank_masks = {
    "upper_z_positive": points[:, :, 2] > 0.0,
    "lower_z_negative": points[:, :, 2] < 0.0,
}
for bank_name, bank_mask in bank_masks.items():
    audit["banks"][bank_name] = {}
    for tag in physical_tags:
        tag_mask = tags == tag
        selections = {
            "whole_bank": tag_mask[:, None] & bank_mask,
            "contact_neighborhood": tag_mask[:, None] & bank_mask & contact_region,
        }
        audit["banks"][bank_name][str(tag)] = {}
        for region_name, selected in selections.items():
            sample_count = int(selected.sum())
            finite_vn = selected & np.isfinite(vn)
            finite_cp = selected & np.isfinite(cp)
            finite_weight = selected & np.isfinite(weight)
            finite_flux = selected & np.isfinite(raw_mass_flux)
            audit["banks"][bank_name][str(tag)][region_name] = {
                "quadrature_sample_count": sample_count,
                "area": float(np.sum(weight[finite_weight])),
                "area_with_finite_vn": float(np.sum(weight[finite_vn])),
                "vn_finite_count": int(finite_vn.sum()),
                "vn_nonfinite_count": int((selected & ~np.isfinite(vn)).sum()),
                "raw_vn_rms_area_weighted": (
                    float(np.sqrt(np.average(vn[finite_vn]**2, weights=weight[finite_vn])))
                    if finite_vn.any() and np.sum(weight[finite_vn]) > 0.0 else None
                ),
                "raw_vn_min_finite": float(np.min(vn[finite_vn])) if finite_vn.any() else None,
                "raw_vn_max_finite": float(np.max(vn[finite_vn])) if finite_vn.any() else None,
                "raw_vn_abs_max_finite": float(np.max(np.abs(vn[finite_vn]))) if finite_vn.any() else None,
                "mass_flux_finite_sample_sum": float(np.sum(raw_mass_flux[finite_flux])),
                "mass_flux_absolute_finite_sample_sum": float(np.sum(np.abs(raw_mass_flux[finite_flux]))),
                "mass_flux_finite_sample_count": int(finite_flux.sum()),
                "mass_flux_nonfinite_count": int((selected & ~np.isfinite(raw_mass_flux)).sum()),
                "cp_finite_count": int(finite_cp.sum()),
                "cp_nonfinite_count": int((selected & ~np.isfinite(cp)).sum()),
                "cp_min_finite": float(np.min(cp[finite_cp])) if finite_cp.any() else None,
                "cp_max_finite": float(np.max(cp[finite_cp])) if finite_cp.any() else None,
            }

output_path = run / "wall_bank_audit.json"
output_path.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
(run / "audit_wall_banks.py.snapshot").write_bytes(Path(__file__).read_bytes())
print(json.dumps(audit, indent=2, allow_nan=False))
