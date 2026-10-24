"""Independent P2 polynomial/gradient checks and strict mesh topology validation."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from p2 import shape_functions

parser = argparse.ArgumentParser()
parser.add_argument("--mesh", type=Path)
args = parser.parse_args()
records = []
reference_nodes = np.array([[0., 0.], [1., 0.], [0., 1.], [.5, 0.], [.5, .5], [0., .5]])
values, derivative = shape_functions(reference_nodes)
error = float(np.max(abs(values-np.eye(6))))
records.append(dict(test="P2_Kronecker_nodal_values", error=error, passed=error < 1e-13))
reference = np.array([[i/20., j/20.] for i in range(21) for j in range(21-i)])
values, derivative = shape_functions(reference)
error = float(max(np.max(abs(values.sum(axis=1)-1.)), np.max(abs(derivative.sum(axis=1)))))
records.append(dict(test="P2_partition_and_gradient_sum", error=error, passed=error < 1e-13))
corners = np.array([[.2, -.1], [1.5, .1], [0., .8]])
nodes = np.vstack([corners, (corners[0]+corners[1])/2.,
                    (corners[1]+corners[2])/2., (corners[2]+corners[0])/2.])
jacobian = nodes.T@derivative
gradient = np.einsum("qib,qba->qia", derivative, np.linalg.inv(jacobian))
position = values@nodes
x, y = nodes.T
nodal_polynomial = 2.+x-3.*y+.7*x*x-.4*x*y+1.1*y*y
x, y = position.T
exact = 2.+x-3.*y+.7*x*x-.4*x*y+1.1*y*y
exact_gradient = np.column_stack([1.+1.4*x-.4*y, -3.-.4*x+2.2*y])
error = float(max(np.max(abs(values@nodal_polynomial-exact)),
                   np.max(abs(np.einsum("qia,i->qa", gradient, nodal_polynomial)-exact_gradient))))
records.append(dict(test="P2_physical_quadratic_patch", error=error, passed=error < 1e-12))
if args.mesh is not None:
    f = np.load(args.mesh)
    points, triangles, edges, adjacent = [f[key] for key in
        ["points", "triangles", "edges", "edge_triangle"]]
    face_map = {}
    for element, tri in enumerate(triangles):
        for left, right, mid in [(0, 1, 3), (1, 2, 4), (2, 0, 5)]:
            face_map.setdefault(tuple(sorted(tri[[left, right]])), []).append((element, tri[mid]))
    boundary_valid = len(edges) == sum(len(v) == 1 for v in face_map.values())
    for edge, element in zip(edges, adjacent):
        matches = face_map[tuple(sorted(edge[:2]))]
        boundary_valid &= len(matches) == 1 and matches[0] == (element, edge[2])
    records.append(dict(test="P2_boundary_adjacency_and_midnodes", passed=bool(boundary_valid)))
    jacobian = np.einsum("eia,qib->eqab", points[triangles], derivative)
    det = np.linalg.det(jacobian)
    records.append(dict(test="P2_dense_reference_Jacobian", min_det=float(det.min()),
                        samples_per_element=len(reference), passed=bool(np.all(det > 0.))))
    records.append(dict(test="P2_all_nodes_used", passed=bool(len(np.unique(triangles)) == len(points))))
    metadata = json.loads(args.mesh.with_name("mesh.json").read_text())
    if metadata.get("mirror_mesh"):
        origin = np.array(metadata.get("normalization_origin", metadata["farfield_center"]))
        distance, reflected = cKDTree(points).query((points-origin)*np.array([1., -1.])+origin)
        elements = {tuple(sorted(row)) for row in triangles}
        mirror_valid = all(tuple(sorted(row)) in elements for row in reflected[triangles])
        records.append(dict(test="P2_node_and_element_mirror_pairing",
                            max_node_error=float(distance.max()),
                            passed=bool(distance.max() < 1e-10*metadata["chord"] and mirror_valid)))
target = args.mesh.parent/"tests.json" if args.mesh else Path(__file__).with_name("patch_tests.json")
target.write_text(json.dumps(records, indent=2), encoding="utf-8")
print(json.dumps(records, indent=2))
assert all(row["passed"] for row in records)
