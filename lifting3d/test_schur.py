"""Check exact block elimination against independently assembled linear systems."""
import unittest

import numpy as np
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import spsolve

from lifting3d.schur import solve_step


class SchurTests(unittest.TestCase):
    def test_exact_nonsymmetric_circulation_closure(self):
        generator = np.random.default_rng(5923)
        raw = generator.normal(size=(20, 20))
        a = csr_matrix(raw.T@raw+np.eye(20))
        b = csr_matrix(generator.normal(size=(20, 3)))
        c = csr_matrix(generator.normal(size=(3, 20)))
        d = csr_matrix(4.*np.eye(3)+generator.normal(size=(3, 3)))
        matrix = bmat([[a, b], [c, d]], format="csc")
        known = generator.normal(size=23)
        residual = -(matrix@known)
        step, diagnostics = solve_step(a, b, c, d, residual)
        reference = spsolve(matrix, -residual)
        np.testing.assert_allclose(step, known, rtol=0., atol=3e-13)
        np.testing.assert_allclose(step, reference, rtol=0., atol=3e-13)
        self.assertLess(diagnostics["linear_relative_residual"], 1e-13)

    def test_singular_circulation_closure_is_not_regularized(self):
        a, b, c = csr_matrix(np.eye(3)), csr_matrix(np.ones((3, 1))), csr_matrix(np.ones((1, 3)))
        d = csr_matrix([[3.]])
        with self.assertRaises(np.linalg.LinAlgError):
            solve_step(a, b, c, d, np.ones(4))


if __name__ == "__main__":
    unittest.main()
