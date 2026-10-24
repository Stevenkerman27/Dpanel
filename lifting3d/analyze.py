"""Independent diagnostics of saved raw surface fields; leave solver acceptance and original data unchanged."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--mesh", type=Path, required=True)
args = parser.parse_args()
field = np.load(args.run/"field.npz")
mesh = np.load(args.mesh)
metadata = json.loads(args.mesh.with_name("mesh.json").read_text())
result = json.loads((args.run/"result.json").read_text())
length = float(metadata["chord"])
assert "surface_points" in field, "This analysis requires saved physical surface quadrature points"
coordinates = field["surface_points"]
velocity = field["surface_velocity"]
normal = field["surface_normal"]
weight = field["surface_weight"]
cp = field["cp"]
tags = field["tags"]
wall = np.isin(tags, [10, 20, 30])
up, lower = tags == 201, tags == 202
normal_velocity = np.sum(velocity*normal, axis=2)
beta = 1.+.2*result["mach"]**2*(1.-np.sum(velocity**2, axis=2))
assert np.all(beta > 0.)
rho = beta**2.5
normal_flux = rho*normal_velocity
pair_distance = np.linalg.norm(coordinates[up, :, None]-coordinates[lower, None, :], axis=3)
paired = pair_distance.argmin(axis=2)
assert pair_distance.min(axis=2).max()/length < 1e-10
jump = normal_flux[up]+np.take_along_axis(normal_flux[lower], paired, axis=1)
cp_jump = cp[up]-np.take_along_axis(cp[lower], paired, axis=1)
analysis = dict(
    result_sha256=hashlib.sha256((args.run/"result.json").read_bytes()).hexdigest(),
    field_sha256=hashlib.sha256((args.run/"field.npz").read_bytes()).hexdigest(),
    mesh_sha256=hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    raw_wake_flux_jump_rms=float(np.sqrt(np.average(jump**2, weights=weight[up]))),
    raw_wake_flux_jump_max=float(abs(jump).max()),
    raw_wake_pressure_jump_rms=float(np.sqrt(np.average(cp_jump**2, weights=weight[up]))),
    wake_upper_normal_velocity_rms=float(np.sqrt(np.average(normal_velocity[up]**2, weights=weight[up]))),
    raw_mass_by_tag={str(tag): float(np.sum(normal_flux[tags == tag]*weight[tags == tag]))
                     for tag in np.unique(tags)},
    regions={})
seam_nodes = []
edge_components = {}
for face, tag in zip(field["faces"][wall], tags[wall]):
    for i, j, mid in [(0, 1, 3), (1, 2, 4), (2, 0, 5)]:
        edge_components.setdefault(tuple(sorted(face[[i, j]])), []).append((int(tag), int(face[mid])))
for pair, sides in edge_components.items():
    if {side[0] for side in sides} == {20, 30}:
        seam_nodes += list(pair)+[sides[0][1]]
wp, wc, wv, ww = coordinates[wall], cp[wall], normal_velocity[wall], weight[wall]
span = metadata["span"]
origin = np.array(metadata.get("normalization_origin", [0., 0., 0.]))
local_wp = wp-origin
if metadata.get("geometry_family") == "imported_cad_planar_te":
    tip_y = mesh["gamma_stations"][[0, -1]]
    region = np.full(wc.shape, "surface", dtype="<U12")
    tip_distance = np.minimum(abs(wp[:, :, 1]-tip_y[0]), abs(wp[:, :, 1]-tip_y[1]))
    region[tip_distance < .1*length] = "tip"
    analysis["region_definition"] = "Declared TE tip span neighborhoods within 0.1 reference chord; no inferred CAD chordwise regions"
else:
    span_fraction = np.minimum(abs(2.*local_wp[:, :, 1]/span), 1.)
    chord = length*(1.+(metadata.get("taper_ratio", 1.)-1.)*span_fraction)
    leading_x = (.5*span*span_fraction*np.tan(np.deg2rad(metadata.get("quarter_chord_sweep_deg", 0.)))
                 +.25*(length-chord))
    chord_coordinate = (local_wp[:, :, 0]-leading_x)/chord
    region = np.full(wc.shape, "interior", dtype="<U12")
    region[chord_coordinate < .1] = "leading"
    region[chord_coordinate > .95] = "trailing"
    region[abs(local_wp[:, :, 1]) > span/2.-.1*length] = "tip"
if seam_nodes:
    seam_xyz = field["points"][np.unique(seam_nodes)]
    distance, _ = cKDTree(seam_xyz).query(wp.reshape(-1, 3))
    region[distance.reshape(wc.shape) < .1*length] = "junction"
    analysis["junction_definition"] = "Distance to shared body/wing seam P2 nodes <0.1 chord; complete-cover region takes priority"
for name in np.unique(region):
    selected = region == name
    analysis["regions"][str(name)] = dict(area=float(ww[selected].sum()),
        normal_rms=float(np.sqrt(np.average(wv[selected]**2, weights=ww[selected]))),
        normal_max=float(abs(wv[selected]).max()), cp_min=float(wc[selected].min()),
        cp_max=float(wc[selected].max()))
maximum_index = np.unravel_index(np.argmax(abs(wv)), wv.shape)
minimum_index = np.unravel_index(np.argmin(wc), wc.shape)
analysis["global_normal_max"] = dict(value=float(wv[maximum_index]), location=wp[maximum_index].tolist(),
                                     region=str(region[maximum_index]))
analysis["global_cp_min"] = dict(value=float(wc[minimum_index]), location=wp[minimum_index].tolist(),
                                region=str(region[minimum_index]))
component_sum = np.sum([entry["force_coefficient"] for entry in result["component_loads"].values()], axis=0)
analysis["component_sum_error"] = float(np.max(abs(component_sum-result["pressure_force_coefficient"])))
analysis["pressure_circulation_lift_difference"] = result["cl_pressure"]-result["cl_jump"]
analysis["note"] = "All raw samples retained. Weak nodal balance, raw flux jumps, wall error and frozen-wake model error are distinct."
(args.run/"analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
(args.run/"analyze.py.snapshot").write_bytes(Path(__file__).read_bytes())
fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
axes[0].scatter(wp[:, :, 0].ravel(), wp[:, :, 1].ravel(), c=wc.ravel(), s=1, cmap="coolwarm")
axes[0].set(xlabel="x", ylabel="y", title="Raw wall Cp, all samples")
image = axes[1].scatter(wp[:, :, 0].ravel(), wp[:, :, 2].ravel(), c=wc.ravel(), s=1, cmap="coolwarm")
axes[1].set(xlabel="x", ylabel="z", title="Side projection")
fig.colorbar(image, ax=axes[:2], label="Cp")
axes[2].plot(result["gamma_stations"], result["potential_jump"])
axes[2].set(xlabel="y", ylabel="Potential jump", title="Solved wake jump")
fig.savefig(args.run/"surface_diagnostics.png", dpi=170)
plt.close(fig)
print(json.dumps(analysis, indent=2))
