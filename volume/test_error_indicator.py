import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import error_indicator as indicator


def tet_nodes(curved=False):
    corners = np.array([
        [0., 0., 0.],
        [1.2, 0., 0.],
        [0.1, 0.9, 0.],
        [0.2, 0.1, 1.1],
    ])
    nodes = list(corners)
    for edge, (i, j) in enumerate(indicator.TET_EDGES):
        midpoint = 0.5 * (corners[i] + corners[j])
        if curved and edge == 0:
            midpoint += np.array([0.0, 0.0, 0.025])
        if curved and edge == 4:
            midpoint += np.array([0.015, -0.005, 0.0])
        nodes.append(midpoint)
    return np.asarray(nodes), np.arange(10, dtype=np.int64)


def quadratic_value(x, linear, hessian, constant=0.0):
    return constant + x @ linear + 0.5 * np.einsum("...i,ij,...j->...", x, hessian, x)


def two_tet_patch():
    corners = np.array([
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0., 0., -1.],
    ])
    corner_tets = ([0, 1, 2, 3], [0, 2, 1, 4])
    nodes = list(corners)
    edge_ids = {}
    tets = []
    for corners_in_tet in corner_tets:
        tet = list(corners_in_tet)
        for i, j in indicator.TET_EDGES:
            edge = tuple(sorted((corners_in_tet[i], corners_in_tet[j])))
            if edge not in edge_ids:
                edge_ids[edge] = len(nodes)
                nodes.append(0.5 * (corners[edge[0]] + corners[edge[1]]))
            tet.append(edge_ids[edge])
        tets.append(tet)
    face_map = {}
    for element, tet in enumerate(tets):
        for omitted in range(4):
            local_corners = [i for i in range(4) if i != omitted]
            face_corners = [tet[i] for i in local_corners]
            face = np.r_[face_corners, [
                edge_ids[tuple(sorted((face_corners[0], face_corners[1])))],
                edge_ids[tuple(sorted((face_corners[1], face_corners[2])))],
                edge_ids[tuple(sorted((face_corners[2], face_corners[0])))],
            ]]
            face_map.setdefault(tuple(sorted(face_corners)), []).append((element, face))
    boundary, adjacent = [], []
    for records in face_map.values():
        if len(records) == 1:
            adjacent.append(records[0][0])
            boundary.append(records[0][1])
    return (
        np.asarray(nodes),
        np.asarray(tets, dtype=np.int64),
        np.asarray(boundary, dtype=np.int64),
        np.asarray(adjacent, dtype=np.int64),
        edge_ids,
    )


class P2ErrorIndicatorTests(unittest.TestCase):
    def test_shapes_partition_and_quadrature_measure(self):
        for order in (3, 4):
            bary, weights = indicator.tetra_rule(order)
            shape, gradient, hessian = indicator.tetra_shape(bary)
            np.testing.assert_allclose(shape.sum(axis=1), 1.0, atol=3.e-15)
            np.testing.assert_allclose(gradient.sum(axis=1), 0.0, atol=3.e-15)
            self.assertAlmostEqual(float(weights.sum()), 1.0 / 6.0, places=14)
            tri, tri_weights = indicator.triangle_rule(order)
            tri_shape, tri_gradient = indicator.triangle_shape(tri)
            np.testing.assert_allclose(tri_shape.sum(axis=1), 1.0, atol=3.e-15)
            np.testing.assert_allclose(tri_gradient.sum(axis=1), 0.0, atol=3.e-15)
            self.assertAlmostEqual(float(tri_weights.sum()), 0.5, places=14)
            self.assertEqual(hessian.shape, (10, 3, 3))

    def test_affine_tet_physical_quadratic_exact_hessian(self):
        points, tet = tet_nodes()
        bary, _ = indicator.tetra_rule(4)
        linear = np.array([0.32, -0.71, 0.48])
        hessian_exact = np.array([[0.6, 0.1, -0.08], [0.1, -0.3, 0.04], [-0.08, 0.04, 0.25]])
        phi = quadratic_value(points, linear, hessian_exact)
        _, _, velocity, hessian, density, divergence = indicator.flux_state(
            points, tet, phi[tet], bary, 0.4
        )
        physical = indicator.tetra_shape(bary)[0] @ points[tet]
        np.testing.assert_allclose(velocity, linear + physical @ hessian_exact, atol=8.e-14)
        np.testing.assert_allclose(hessian, np.broadcast_to(hessian_exact, hessian.shape), atol=1.e-13)
        speed2 = np.einsum("qi,qi->q", velocity, velocity)
        beta = 1.0 + 0.2 * 0.4**2 * (1.0 - speed2)
        expected = beta**2.5 * (
            np.trace(hessian_exact) - 0.4**2 / beta
            * np.einsum("qi,ij,qj->q", velocity, hessian_exact, velocity)
        )
        np.testing.assert_allclose(density, beta**2.5, rtol=2.e-14, atol=2.e-14)
        np.testing.assert_allclose(divergence, expected, rtol=2.e-13, atol=2.e-13)

    def test_curved_isoparametric_coordinate_potential_has_zero_hessian(self):
        points, tet = tet_nodes(curved=True)
        bary, _ = indicator.tetra_rule(4)
        phi = points[:, 1].copy()
        _, _, velocity, hessian, _, divergence = indicator.flux_state(
            points, tet, phi[tet], bary, 0.2
        )
        np.testing.assert_allclose(velocity, np.broadcast_to([0., 1., 0.], velocity.shape), atol=1.e-13)
        np.testing.assert_allclose(hessian, 0.0, atol=2.e-13)
        np.testing.assert_allclose(divergence, 0.0, atol=5.e-13)

    def test_physical_hessian_and_compressible_divergence_match_finite_difference(self):
        points, tet = tet_nodes()
        linear = np.array([0.21, -0.37, 0.14])
        hessian_exact = np.array([[0.43, -0.06, 0.08], [-0.06, 0.29, 0.03], [0.08, 0.03, -0.17]])
        phi = quadratic_value(points, linear, hessian_exact)
        bary0 = np.array([[0.31, 0.24, 0.19, 0.26]])
        _, _, velocity0, hessian0, _, divergence0 = indicator.flux_state(
            points, tet, phi[tet], bary0, 0.35
        )
        corner_j = np.column_stack((points[tet[1]] - points[tet[0]],
                                    points[tet[2]] - points[tet[0]],
                                    points[tet[3]] - points[tet[0]]))
        inverse_corner_j = np.linalg.inv(corner_j)
        step = 2.e-5
        gradient_fd = np.empty((3, 3))
        flux_div_fd = 0.0
        for axis in range(3):
            physical_step = np.eye(3)[axis] * step
            ref_step = inverse_corner_j @ physical_step
            bary_step = np.array([-ref_step.sum(), *ref_step])
            plus = bary0 + bary_step[None, :]
            minus = bary0 - bary_step[None, :]
            _, _, q_plus, _, rho_plus, _ = indicator.flux_state(points, tet, phi[tet], plus, 0.35)
            _, _, q_minus, _, rho_minus, _ = indicator.flux_state(points, tet, phi[tet], minus, 0.35)
            gradient_fd[:, axis] = ((q_plus - q_minus) / (2.0 * step))[0]
            flux_plus = rho_plus[0] * q_plus[0, axis]
            flux_minus = rho_minus[0] * q_minus[0, axis]
            flux_div_fd += (flux_plus - flux_minus) / (2.0 * step)
        np.testing.assert_allclose(gradient_fd, hessian0[0], rtol=2.e-9, atol=2.e-10)
        self.assertAlmostEqual(float(divergence0[0]), float(flux_div_fd), delta=2.e-9)

    def test_isoparametric_hessian_formula_with_curved_mapping_by_difference(self):
        points, tet = tet_nodes(curved=True)
        # A quadratic in reference coordinates is represented exactly by this P2 field.
        bary_nodes = np.vstack((np.eye(4), np.array([
            0.5 * (np.eye(4)[i] + np.eye(4)[j]) for i, j in indicator.TET_EDGES
        ])))
        lam = bary_nodes
        phi = 0.2 + 0.31 * lam[:, 1] - 0.11 * lam[:, 2]**2 + 0.07 * lam[:, 1] * lam[:, 3]
        bary0 = np.array([[0.28, 0.27, 0.19, 0.26]])
        _, _, q0, h0, _, _ = indicator.flux_state(points, tet, phi[tet], bary0, 0.0)
        j0, _, _, _, _, _ = indicator.flux_state(points, tet, phi[tet], bary0, 0.0)
        h = 2.e-5
        q_fd = np.empty((3, 3))
        for axis in range(3):
            dxi = np.linalg.solve(j0[0], np.eye(3)[axis]) * h
            dbary = np.r_[-dxi.sum(), dxi]
            _, _, qp, _, _, _ = indicator.flux_state(points, tet, phi[tet], bary0 + dbary, 0.0)
            _, _, qm, _, _, _ = indicator.flux_state(points, tet, phi[tet], bary0 - dbary, 0.0)
            q_fd[:, axis] = ((qp - qm) / (2.0 * h))[0]
        np.testing.assert_allclose(q_fd, h0[0], rtol=2.e-7, atol=2.e-8)

    def test_dorfler_marks_smallest_stable_prefix(self):
        values = np.array([1.0, 4.0, 2.0, 3.0])
        marked, order, cumulative, total = indicator.mark_dorfler(values, 0.5)
        self.assertEqual(order.tolist(), [1, 3, 2, 0])
        self.assertAlmostEqual(total, 10.0)
        self.assertEqual(np.flatnonzero(marked).tolist(), [1, 3])
        self.assertGreaterEqual(cumulative[1], 0.5)
        self.assertLess(cumulative[0], 0.5)

    def test_two_tet_raw_flux_jump_and_per_cell_contribution(self):
        points, tets, faces, adjacent, edge_ids = two_tet_patch()
        mesh = {"boundary_tet": adjacent}
        tags = np.full(len(faces), 100, dtype=np.int64)
        phi = points @ np.array([0.7, -0.2, 0.4])
        affine = indicator.compute_indicator(mesh, points, tets, phi, faces, tags, 0.2, 3)
        self.assertEqual(affine["internal_face_count"], 1)
        self.assertLess(affine["bulk_total"], 1.e-25)
        self.assertLess(affine["internal_total"], 1.e-25)
        self.assertGreaterEqual(affine["marked_contribution_fraction"], 0.5)

        phi[edge_ids[(0, 2)]] += 0.2
        perturbed = indicator.compute_indicator(mesh, points, tets, phi, faces, tags, 0.2, 3)
        self.assertGreater(perturbed["internal_total"], 1.e-5)
        self.assertAlmostEqual(
            float(np.sum(perturbed["cell_total"])),
            perturbed["bulk_total"] + perturbed["internal_total"] + perturbed["wake_total"]
            + perturbed["wall_total"],
            places=13,
        )
        self.assertGreaterEqual(perturbed["marked_contribution_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
