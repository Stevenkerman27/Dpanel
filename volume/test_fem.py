import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solve


def cube_mesh_with_center():
    points = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    boundary_quads = [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]
    tets = []
    triangles = []
    for quad in boundary_quads:
        for face in ([quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]):
            tet = [8, *face]
            matrix = np.column_stack((np.ones(4), points[tet]))
            if np.linalg.det(matrix) < 0.0:
                tet[1], tet[2] = tet[2], tet[1]
            tets.append(tet)
            triangles.append(face)
    tets = np.asarray(tets, dtype=np.int64)
    triangles = np.asarray(triangles, dtype=np.int64)
    for index, triangle in enumerate(triangles):
        center_to_face = points[8] - points[triangle[0]]
        normal = np.cross(points[triangle[1]] - points[triangle[0]], points[triangle[2]] - points[triangle[0]])
        if np.dot(normal, center_to_face) > 0.0:
            triangles[index, 1], triangles[index, 2] = triangles[index, 2], triangles[index, 1]
    return points, tets, triangles


class TetrahedralFemTests(unittest.TestCase):
    def test_affine_element_gradient_is_exact(self):
        points, tets, _ = cube_mesh_with_center()
        gradients, volumes = solve.element_geometry(points, tets)
        exact_gradient = np.array([0.7, -1.2, 2.3])
        phi = points @ exact_gradient + 4.1
        recovered = np.einsum("eij,ei->ej", gradients, phi[tets])
        np.testing.assert_allclose(recovered, np.broadcast_to(exact_gradient, recovered.shape), atol=2.0e-14)
        self.assertAlmostEqual(float(np.sum(volumes)), 8.0)

    def test_affine_patch_and_global_weak_balance(self):
        points, tets, _ = cube_mesh_with_center()
        gradients, volumes = solve.element_geometry(points, tets)
        exact_gradient = np.array([0.7, -1.2, 2.3])
        phi = points @ exact_gradient + 4.1
        residual, _, element_gradient, _, _ = solve.assemble(points, tets, gradients, volumes, phi, 0.0, False)
        np.testing.assert_allclose(element_gradient, np.broadcast_to(exact_gradient, element_gradient.shape), atol=2.0e-14)
        self.assertAlmostEqual(float(residual[8]), 0.0, places=13)
        self.assertAlmostEqual(float(np.sum(residual)), 0.0, places=13)

    def test_mesh_boundary_contract_and_orientations(self):
        points, tets, triangles = cube_mesh_with_center()
        gradients, volumes = solve.element_geometry(points, tets)
        self.assertTrue(np.all(volumes > 0.0))
        for triangle in triangles:
            matching = [tet for tet in tets if set(triangle).issubset(set(tet))]
            self.assertEqual(len(matching), 1)
            opposite = next(node for node in matching[0] if node not in triangle)
            normal = np.cross(points[triangle[1]] - points[triangle[0]], points[triangle[2]] - points[triangle[0]])
            self.assertLess(float(np.dot(normal, points[opposite] - points[triangle[0]])), 0.0)

    def test_jacobian_matches_independent_residual_difference(self):
        points, tets, _ = cube_mesh_with_center()
        gradients, volumes = solve.element_geometry(points, tets)
        rng = np.random.default_rng(731)
        phi = points @ np.array([0.15, -0.1, 0.05]) + 0.01 * rng.standard_normal(len(points))
        direction = rng.standard_normal(len(points))
        residual, jacobian, _, _, _ = solve.assemble(points, tets, gradients, volumes, phi, 0.4, True)
        step = 1.0e-6
        residual_plus = solve.assemble(points, tets, gradients, volumes, phi + step * direction, 0.4, False)[0]
        residual_minus = solve.assemble(points, tets, gradients, volumes, phi - step * direction, 0.4, False)[0]
        finite_difference = (residual_plus - residual_minus) / (2.0 * step)
        np.testing.assert_allclose(jacobian @ direction, finite_difference, rtol=2.0e-8, atol=2.0e-9)

    def test_energy_directional_derivative_matches_weak_residual(self):
        points, tets, _ = cube_mesh_with_center()
        gradients, volumes = solve.element_geometry(points, tets)
        rng = np.random.default_rng(197)
        phi = points @ np.array([0.1, -0.12, 0.08]) + 0.01 * rng.standard_normal(len(points))
        direction = rng.standard_normal(len(points))
        residual = solve.assemble(points, tets, gradients, volumes, phi, 0.35, False)[0]
        step = 1.0e-6
        velocity_plus = np.einsum("eij,ei->ej", gradients, (phi + step * direction)[tets])
        velocity_minus = np.einsum("eij,ei->ej", gradients, (phi - step * direction)[tets])
        finite_difference = (
            solve.energy(volumes, velocity_plus, 0.35) - solve.energy(volumes, velocity_minus, 0.35)
        ) / (2.0 * step)
        self.assertAlmostEqual(finite_difference, float(residual @ direction), delta=2.0e-8)

    def test_low_mach_pressure_coefficient_is_stable(self):
        q2 = np.array([0.2, 1.0, 1.8])
        incompressible = 1.0 - q2
        for mach in [1.0e-2, 1.0e-5, 1.0e-8]:
            cp = solve.pressure_coefficient(q2, mach)
            self.assertTrue(np.all(np.isfinite(cp)))
            np.testing.assert_allclose(cp, incompressible, rtol=0.0, atol=2.0e-5)

    def test_low_mach_energy_has_incompressible_limit(self):
        volumes = np.array([0.2, 0.3, 0.5])
        velocity = np.array([[0.2, 0.4, 0.1], [1.0, 0.1, 0.0], [0.6, 0.2, 0.3]])
        expected = 0.5 * np.dot(volumes, np.einsum("ei,ei->e", velocity, velocity))
        for mach in [1.0e-5, 1.0e-8]:
            self.assertAlmostEqual(solve.energy(volumes, velocity, mach), expected, delta=1.0e-10)


if __name__ == "__main__":
    unittest.main()
