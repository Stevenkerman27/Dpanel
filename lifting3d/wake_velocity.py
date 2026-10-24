"""Recover paired nodal wake velocity for geometry prediction, preserving raw solver output."""
import numpy as np
from scipy.interpolate import LinearNDInterpolator

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/"volume"))
import p2


def sample_velocity(mesh, field, x, eta, origin=None, length=1.):
    origin = np.zeros(3) if origin is None else np.asarray(origin)
    points = (mesh["points"]-origin)/length
    reference = (mesh.get("reference_points", mesh["points"])-origin)/length
    tets, phi = mesh["tets"], field["phi"]/length
    reference_nodes = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.],
                                [.5, .5, 0., 0.], [0., .5, .5, 0.], [.5, 0., .5, 0.],
                                [.5, 0., 0., .5], [0., .5, 0., .5], [0., 0., .5, .5]])
    _, derivative = p2.p2_shape(reference_nodes)
    paired_face_velocity = []
    for side in ["upper", "lower"]:
        faces, adjacent = mesh["wake_"+side], mesh["wake_"+side+"_tet"]
        element, inverse = np.unique(adjacent, return_inverse=True)
        jacobian = np.einsum("eia,qib->eqab", points[tets[element]], derivative)
        assert np.min(np.linalg.det(jacobian)) > 0., "Wake predictor encountered a nonpositive nodal Jacobian"
        gradient = np.einsum("qia,eqab->eqib", derivative, np.linalg.inv(jacobian))
        velocity = np.einsum("eqib,ei->eqb", gradient, phi[tets[element]])
        local_nodes = np.argmax(faces[:, :, None] == tets[adjacent][:, None, :], axis=2)
        assert np.array_equal(np.take_along_axis(tets[adjacent], local_nodes, axis=1), faces)
        paired_face_velocity.append(velocity[inverse[:, None], local_nodes])
    faces = mesh["wake_upper"]
    assert np.max(abs(points[faces]-points[mesh["wake_lower"]])) < 1e-12, "Predictor requires identically ordered paired wake faces"
    paired_velocity = .5*(paired_face_velocity[0]+paired_face_velocity[1])
    face_area = field["surface_weight"][field["tags"] == 201].sum(axis=1)/length**2
    recovered, weights = np.zeros_like(points), np.zeros(len(points))
    np.add.at(recovered, faces.ravel(), (paired_velocity*face_area[:, None, None]).reshape(-1, 3))
    np.add.at(weights, faces.ravel(), np.broadcast_to(face_area[:, None], faces.shape).ravel())
    nodes = np.unique(faces)
    recovered[nodes] /= weights[nodes, None]
    interpolation = LinearNDInterpolator(reference[nodes, :2], recovered[nodes])
    xx, ee = np.meshgrid(x, eta, indexing="ij")
    velocity_grid = interpolation(xx, ee)
    assert np.isfinite(velocity_grid).all(), "Predictor grid lies outside the reference wake interpolation hull"
    diagnostics = dict(recovery="Area-weighted paired raw nodal gradients; only used to predict wake geometry",
                       nodes=len(nodes), grid_shape=list(velocity_grid.shape),
                       min_recovered_qx=float(recovered[nodes, 0].min()),
                       min_grid_qx=float(velocity_grid[:, :, 0].min()),
                       max_grid_speed=float(np.linalg.norm(velocity_grid, axis=2).max()))
    return velocity_grid, diagnostics
