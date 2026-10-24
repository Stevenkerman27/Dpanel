"""Change only the fixed wake orientation on an existing mesh, preserving solid nodes and topology."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent/"volume"))
import p2

parser = argparse.ArgumentParser()
parser.add_argument("--mesh", type=Path, required=True)
parser.add_argument("--angle", type=float, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
assert not args.out.exists() or not any(args.out.iterdir()), "Preserve existing output"
assert np.isfinite(args.angle) and abs(args.angle) < 30.
metadata = json.loads(args.mesh.with_name("mesh.json").read_text())
mesh = dict(np.load(args.mesh))
assert "reference_points" in mesh or metadata.get("wake_tilt_deg", 0.) == 0., "Need an undeformed reference mesh"
reference_points = mesh.get("reference_points", mesh["points"]).copy()
length = float(metadata["chord"])
origin = np.array(metadata.get("normalization_origin", [0., 0., 0.]))
points = (reference_points-origin)/length
radius = metadata["far_radius"]/length
span = metadata["span"]/length
local_fraction = np.minimum(abs(2.*points[:, 1]/span), 1.)
local_chord = 1.+(metadata.get("taper_ratio", 1.)-1.)*local_fraction
local_te = (.25+.5*span*local_fraction*np.tan(np.deg2rad(metadata.get("quarter_chord_sweep_deg", 0.)))
            +.75*local_chord)
downstream = np.maximum(points[:, 0]-local_te, 0.)
span_decay = np.maximum((abs(points[:, 1])-span/2.)/(radius-span/2.), 0.)
displacement = downstream*np.tan(np.deg2rad(args.angle))*(1.-abs(points[:, 2])/radius)**2*(1.-span_decay)**2
wall = np.unique(mesh["boundary_triangles"][mesh["boundary_tags"] != 100])
assert np.max(abs(displacement[wall])) < 1e-12, "Solid geometry would move"
points[:, 2] += displacement
mesh["points"] = points*length+origin
mesh["reference_points"] = reference_points
barycentric = np.array([[i, j, k, 8-i-j-k] for i in range(9)
                        for j in range(9-i) for k in range(9-i-j)])/8.
_, derivative = p2.p2_shape(barycentric)
minimum = np.inf
for first in range(0, len(mesh["tets"]), 256):
    jacobian = np.einsum("eia,qib->eqab", points[mesh["tets"][first:first+256]], derivative)
    minimum = min(minimum, float(np.linalg.det(jacobian).min()))
args.out.mkdir(parents=True, exist_ok=True)
metadata.update(wake_tilt_deg=args.angle, min_normalized_dense_jacobian=minimum,
                dense_jacobian_samples_per_element=len(barycentric),
                tilt_source_mesh_sha256=hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
                geometry_builder_sha256=metadata.get("source_sha256"),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
if minimum <= 0.:
    metadata["status"] = "rejected_nonpositive_deformed_jacobian"
    (args.out/"failure.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    raise SystemExit(2)
np.savez_compressed(args.out/"mesh.npz", **mesh)
metadata["mesh_sha256"] = hashlib.sha256((args.out/"mesh.npz").read_bytes()).hexdigest()
metadata["status"] = "mesh_geometry_checks_passed"
(args.out/"mesh.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
(args.out/"tilt_mesh.py.snapshot").write_bytes(Path(__file__).read_bytes())
print(json.dumps(metadata, indent=2))
