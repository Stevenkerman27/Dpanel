"""Locate actual curved-mesh distortion relative to its unchanged reference connectivity."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent/"volume"))
import p2

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mesh", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
assert not args.out.exists(), "Preserve previous audits"
args.out.mkdir(parents=True)
mesh = np.load(args.mesh)
points, reference, tets = mesh["points"], mesh["reference_points"], mesh["tets"]
lattice = np.array([[i, j, k, 8-i-j-k] for i in range(9)
                    for j in range(9-i) for k in range(9-i-j)])/8.
_, derivative = p2.p2_shape(lattice)
minimum, ratios = [], []
for first in range(0, len(tets), 256):
    nodes = tets[first:first+256]
    current_jacobian = np.einsum("eia,qib->eqab", points[nodes], derivative)
    reference_jacobian = np.einsum("eia,qib->eqab", reference[nodes], derivative)
    determinant = np.linalg.det(current_jacobian)
    initial_determinant = np.linalg.det(reference_jacobian)
    assert np.min(initial_determinant) > 0.
    minimum.extend(determinant.min(axis=1))
    ratios.extend((determinant/initial_determinant).min(axis=1))
minimum, ratios = np.array(minimum), np.array(ratios)
worst = np.argsort(ratios)[:20]
wake_adjacent = np.unique(np.r_[mesh["wake_upper_tet"], mesh["wake_lower_tet"]])
wall_adjacent = np.unique(mesh["boundary_tet"][mesh["boundary_tags"] != 100])
records = [dict(element=int(element), min_dense_jacobian=float(minimum[element]),
                min_dense_ratio=float(ratios[element]),
                reference_center=reference[tets[element, :4]].mean(axis=0).tolist(),
                deformed_center=points[tets[element, :4]].mean(axis=0).tolist(),
                shares_wake_face=bool(element in wake_adjacent),
                shares_wall_face=bool(element in wall_adjacent)) for element in worst]
summary = dict(mesh=str(args.mesh.resolve()), samples_per_element=len(lattice),
               nonpositive_elements=int((minimum <= 0.).sum()),
               minimum_dense_ratio=float(ratios.min()), worst=records,
               note="Relative geometric distortion, not a flow condition number or pressure error bound")
(args.out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
np.savez_compressed(args.out/"mesh_distortion.npz", min_dense_jacobian=minimum, min_dense_ratio=ratios)
(args.out/"audit.py.snapshot").write_bytes(Path(__file__).read_bytes())
print(json.dumps(summary, indent=2))
