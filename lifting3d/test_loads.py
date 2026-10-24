"""Analytic affine-pressure box and dimensional identities for pressure moments."""
import unittest

import numpy as np

from lifting3d.loads import pressure_loads


class PressureLoadTests(unittest.TestCase):
    def test_affine_pressure_on_translated_box(self):
        center = np.array([2., -3., 1.])
        half_width = np.array([1., 2., 3.])
        gradient = np.array([.2, -.1, .3])
        gauss = np.array([-1., 1.])/np.sqrt(3.)
        coordinates, normals, weights = [], [], []
        for axis in range(3):
            other = [i for i in range(3) if i != axis]
            for sign in [-1., 1.]:
                face = np.tile(center, (4, 1))
                face[:, axis] += sign*half_width[axis]
                face[:, other] += (np.array(np.meshgrid(gauss, gauss)).reshape(2, -1).T
                                   *half_width[other])
                normal = np.zeros((4, 3))
                normal[:, axis] = -sign
                coordinates.append(face)
                normals.append(normal)
                weights.append(np.full(4, np.prod(half_width[other])))
        coordinates, normals, weights = np.array(coordinates), np.array(normals), np.array(weights)
        cp = 1.+coordinates@gradient
        tags = np.array([20, 20, 30, 30, 30, 30])
        reference_area, reference_length = 7., 2.
        moment_center = np.array([.25, .1, -.2])
        loads = pressure_loads(coordinates, cp, normals, weights, tags, moment_center,
                               reference_area, reference_length)
        volume = 8.*np.prod(half_width)
        exact_force = -volume*gradient/reference_area
        exact_moment = np.cross(center-moment_center, exact_force)/reference_length
        np.testing.assert_allclose(loads["force_coefficient"], exact_force, rtol=0., atol=2e-14)
        np.testing.assert_allclose(loads["moment_coefficient"], exact_moment, rtol=0., atol=2e-14)
        displacement = np.array([.7, -.3, .4])
        shifted = pressure_loads(coordinates, cp, normals, weights, tags, moment_center+displacement,
                                 reference_area, reference_length)
        np.testing.assert_allclose(shifted["moment_coefficient"],
            np.array(loads["moment_coefficient"])-np.cross(displacement, exact_force)/reference_length,
            rtol=0., atol=2e-14)
        scale, translation = 1000., np.array([100., -200., 300.])
        scaled = pressure_loads(scale*coordinates+translation, cp, normals, scale**2*weights, tags,
                                scale*moment_center+translation, scale**2*reference_area, scale*reference_length)
        np.testing.assert_allclose(scaled["force_coefficient"], exact_force, rtol=0., atol=2e-14)
        np.testing.assert_allclose(scaled["moment_coefficient"], exact_moment, rtol=0., atol=2e-14)
        np.testing.assert_allclose(sum(np.array(row["moment_coefficient"]) for row in loads["components"].values()),
                                    exact_moment, rtol=0., atol=2e-14)


if __name__ == "__main__":
    unittest.main()
