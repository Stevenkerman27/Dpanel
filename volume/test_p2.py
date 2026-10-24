import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2


def reference_tet(curved=False):
    corners = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    coordinates = [*corners]
    for i, j in p2.TET_EDGES:
        midpoint = 0.5 * (corners[i] + corners[j])
        if curved and (i, j) == (0, 1):
            midpoint = midpoint + np.array([0.0, 0.0, 0.04])
        if curved and (i, j) == (1, 2):
            midpoint = midpoint + np.array([0.03, 0.0, 0.0])
        coordinates.append(midpoint)
    return np.asarray(coordinates), np.arange(10, dtype=np.int64)[None, :]


def mapping_and_gradient(points, tets, barycentric):
    shape, reference_gradient = p2.p2_shape(barycentric)
    coordinates = points[tets[0]]
    jacobian = np.einsum("ik,qia->qka", coordinates, reference_gradient)
    physical_gradient = np.einsum("qia,qak->qik", reference_gradient, np.linalg.inv(jacobian))
    physical_coordinates = shape @ coordinates
    return shape, physical_gradient, physical_coordinates


def two_tet_patch():
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    corner_tets = [[0, 1, 2, 3], [0, 2, 1, 4]]
    points = list(corners)
    edge_node = {}
    tets = []
    for corner_tet in corner_tets:
        tet = list(corner_tet)
        for i, j in p2.TET_EDGES:
            key = tuple(sorted((corner_tet[i], corner_tet[j])))
            if key not in edge_node:
                edge_node[key] = len(points)
                points.append(0.5 * (corners[key[0]] + corners[key[1]]))
            tet.append(edge_node[key])
        tets.append(tet)
    return np.asarray(points), np.asarray(tets, dtype=np.int64), edge_node


def two_tet_boundary(tets, edge_node):
    face_map = {}
    for tet_index, tet in enumerate(tets):
        for omitted in range(4):
            corner_positions = [i for i in range(4) if i != omitted]
            corners = tet[corner_positions]
            key = tuple(sorted(corners.tolist()))
            edge_ids = [
                edge_node[tuple(sorted((int(corners[0]), int(corners[1]))))],
                edge_node[tuple(sorted((int(corners[1]), int(corners[2]))))],
                edge_node[tuple(sorted((int(corners[2]), int(corners[0]))))],
            ]
            face_map.setdefault(key, []).append(np.r_[corners, edge_ids])
    boundary = [face_nodes[0] for face_nodes in face_map.values() if len(face_nodes) == 1]
    return np.asarray(boundary, dtype=np.int64)


class P2CurvedTetTests(unittest.TestCase):
    def test_shape_partition_and_duffy_measure_orders(self):
        for order in [3, 4]:
            tetra_bary, tetra_weights = p2.tetra_quadrature(order)
            tetra_shape, tetra_gradient = p2.p2_shape(tetra_bary)
            np.testing.assert_allclose(np.sum(tetra_shape, axis=1), 1.0, atol=2.0e-15)
            np.testing.assert_allclose(np.sum(tetra_gradient, axis=1), 0.0, atol=2.0e-15)
            self.assertAlmostEqual(float(np.sum(tetra_weights)), 1.0 / 6.0, places=14)
            triangle_bary, triangle_weights = p2.triangle_quadrature(order)
            triangle_shape, triangle_gradient = p2.triangle_shape(triangle_bary)
            np.testing.assert_allclose(np.sum(triangle_shape, axis=1), 1.0, atol=2.0e-15)
            np.testing.assert_allclose(np.sum(triangle_gradient, axis=1), 0.0, atol=2.0e-15)
            self.assertAlmostEqual(float(np.sum(triangle_weights)), 0.5, places=14)

    def test_quadrature_is_invariant_to_barycentric_columns_one_and_two(self):
        for order in [3, 4]:
            tetra_bary, tetra_weights = p2.tetra_quadrature(order)
            tetra_probe = (
                np.exp(0.37 * tetra_bary[:, 0] - 0.21 * tetra_bary[:, 1] * tetra_bary[:, 3])
                + tetra_bary[:, 2] ** 2 * np.sin(tetra_bary[:, 1] + 0.4)
            )
            tetra_swapped_probe = (
                np.exp(0.37 * tetra_bary[:, 0] - 0.21 * tetra_bary[:, 2] * tetra_bary[:, 3])
                + tetra_bary[:, 1] ** 2 * np.sin(tetra_bary[:, 2] + 0.4)
            )
            self.assertAlmostEqual(float(tetra_weights @ tetra_probe),
                                   float(tetra_weights @ tetra_swapped_probe), places=14)

            triangle_bary, triangle_weights = p2.triangle_quadrature(order)
            triangle_probe = np.exp(0.23 * triangle_bary[:, 0] - 0.31 * triangle_bary[:, 1] * triangle_bary[:, 2])
            triangle_swapped_probe = np.exp(0.23 * triangle_bary[:, 0] - 0.31 * triangle_bary[:, 2] * triangle_bary[:, 1])
            self.assertAlmostEqual(float(triangle_weights @ triangle_probe),
                                   float(triangle_weights @ triangle_swapped_probe), places=14)

    def test_internal_face_flux_uses_two_raw_traces_and_closes_area(self):
        points, tets, edge_node = two_tet_patch()
        phi = points @ np.array([0.7, -0.2, 0.4])
        affine_audit = p2.internal_face_flux_jump(points, tets, phi, 0.2, 3)
        self.assertEqual(affine_audit["status"], "measured")
        self.assertEqual(affine_audit["internal_face_count"], 1)
        self.assertLess(affine_audit["area_weighted_flux_jump_rms"], 2.0e-14)
        self.assertLess(affine_audit["area_relative_integral_gap"], 2.0e-14)
        self.assertLess(affine_audit["max_point_coordinate_gap"], 2.0e-14)
        self.assertLess(affine_audit["max_relative_outward_area_vector_sum"], 2.0e-14)

        phi[edge_node[(0, 3)]] += 0.2
        discontinuous_audit = p2.internal_face_flux_jump(points, tets, phi, 0.2, 3)
        self.assertGreater(discontinuous_audit["area_weighted_flux_jump_rms"], 1.0e-3)
        self.assertLess(discontinuous_audit["area_relative_integral_gap"], 2.0e-14)

    def test_mesh_rejects_point_nodes_unused_by_tetrahedra(self):
        points, tets, edge_node = two_tet_patch()
        boundary = two_tet_boundary(tets, edge_node)
        points_with_orphan = np.vstack((points, [[3.0, 4.0, 5.0]]))
        with tempfile.TemporaryDirectory() as directory:
            mesh_path = Path(directory) / "orphan.npz"
            np.savez(
                mesh_path,
                points=points_with_orphan,
                tets=tets,
                boundary_triangles=boundary,
                boundary_tags=np.full(len(boundary), 100, dtype=np.int64),
            )
            with self.assertRaisesRegex(ValueError, "point nodes unused"):
                p2.load_mesh(mesh_path)

    def test_physical_quadratic_patch_on_affine_tet(self):
        points, tets = reference_tet()
        barycentric, _ = p2.tetra_quadrature(4)
        shape, gradient, physical = mapping_and_gradient(points, tets, barycentric)
        linear = np.array([0.4, -0.7, 1.1])
        hessian = np.array([[0.8, 0.2, -0.1], [0.2, -0.4, 0.3], [-0.1, 0.3, 0.6]])
        nodal_value = points @ linear + 0.5 * np.einsum("ni,ij,nj->n", points, hessian, points)
        exact_value = physical @ linear + 0.5 * np.einsum("qi,ij,qj->q", physical, hessian, physical)
        exact_gradient = linear[None, :] + physical @ hessian
        recovered_value = shape @ nodal_value[tets[0]]
        recovered_gradient = np.einsum("qik,i->qk", gradient, nodal_value[tets[0]])
        np.testing.assert_allclose(recovered_value, exact_value, atol=2.0e-14)
        np.testing.assert_allclose(recovered_gradient, exact_gradient, atol=3.0e-14)

    def test_affine_physical_field_on_curved_isoparametric_tet(self):
        points, tets = reference_tet(curved=True)
        barycentric, _ = p2.tetra_quadrature(3)
        _, gradient, physical = mapping_and_gradient(points, tets, barycentric)
        affine_gradient = np.array([0.31, -0.82, 1.27])
        nodal_value = points @ affine_gradient + 0.9
        recovered_gradient = np.einsum("qik,i->qk", gradient, nodal_value[tets[0]])
        recovered_value = p2.p2_shape(barycentric)[0] @ nodal_value[tets[0]]
        np.testing.assert_allclose(recovered_gradient, np.broadcast_to(affine_gradient, recovered_gradient.shape), atol=3.0e-14)
        np.testing.assert_allclose(recovered_value, physical @ affine_gradient + 0.9, atol=2.0e-14)

    def test_curved_volume_quadrature_order_sensitivity(self):
        points, tets = reference_tet(curved=True)
        phi = points @ np.array([0.4, -0.2, 0.7])
        energies = [p2.evaluate(points, tets, phi, 0.0, order, False)[4] for order in [3, 4]]
        self.assertAlmostEqual(energies[0], energies[1], places=13)

    def test_jacobian_matches_independent_residual_difference(self):
        points, tets = reference_tet(curved=True)
        rng = np.random.default_rng(20261023)
        phi = points @ np.array([0.12, -0.07, 0.09]) + 0.008 * rng.standard_normal(len(points))
        direction = rng.standard_normal(len(points))
        residual, jacobian, _, _, _ = p2.evaluate(points, tets, phi, 0.35, 3, True)
        step = 1.0e-6
        plus = p2.evaluate(points, tets, phi + step * direction, 0.35, 3, False)[0]
        minus = p2.evaluate(points, tets, phi - step * direction, 0.35, 3, False)[0]
        np.testing.assert_allclose(jacobian @ direction, (plus - minus) / (2.0 * step), rtol=5.0e-8, atol=3.0e-9)

    def test_energy_directional_derivative_matches_weak_residual(self):
        points, tets = reference_tet(curved=True)
        rng = np.random.default_rng(87)
        phi = points @ np.array([0.09, 0.11, -0.04]) + 0.006 * rng.standard_normal(len(points))
        direction = rng.standard_normal(len(points))
        residual = p2.evaluate(points, tets, phi, 0.3, 4, False)[0]
        step = 1.0e-6
        e_plus = p2.evaluate(points, tets, phi + step * direction, 0.3, 4, False)[4]
        e_minus = p2.evaluate(points, tets, phi - step * direction, 0.3, 4, False)[4]
        self.assertAlmostEqual((e_plus - e_minus) / (2.0 * step), float(residual @ direction), delta=3.0e-9)

    def test_low_mach_cp_and_energy_limits(self):
        q2 = np.array([0.2, 1.0, 1.8])
        expected_cp = 1.0 - q2
        volumes = np.array([0.2, 0.3, 0.5])
        velocity = np.array([[0.2, 0.4, 0.1], [1.0, 0.1, 0.0], [0.6, 0.2, 0.3]])
        expected_energy = 0.5 * np.dot(volumes, np.einsum("ei,ei->e", velocity, velocity))
        for mach in [1.0e-5, 1.0e-8]:
            cp = p2.pressure_coefficient(q2, mach)
            self.assertTrue(np.all(np.isfinite(cp)))
            np.testing.assert_allclose(cp, expected_cp, rtol=0.0, atol=2.0e-10)
            self.assertAlmostEqual(float(np.dot(volumes, p2.energy_primitive(np.einsum("ei,ei->e", velocity, velocity), mach))), expected_energy, delta=1.0e-10)


if __name__ == "__main__":
    unittest.main()
