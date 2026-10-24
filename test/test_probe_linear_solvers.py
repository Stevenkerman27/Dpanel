"""Unit-test the GMRES diagnostic itself; not a VortexAD convergence claim."""
import importlib.util
from pathlib import Path

import numpy as np

module_path = Path(__file__).resolve().with_name("probe_linear_solvers.py")
spec = importlib.util.spec_from_file_location("dpanel_linear_solver_probe", module_path)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_primal_transpose_gmres_against_small_nonsymmetric_direct_solve():
    A = np.array([[4., -1., 0.], [0.5, 3., 1.], [0., -0.2, 2.]], dtype=np.float64)
    b = np.array([1., 2., -1.], dtype=np.float64)
    for transpose in (False, True):
        mat = A.T if transpose else A
        reference = np.linalg.solve(mat, b)
        for jacobi in (False, True):
            result = probe.timed_gmres(
                A, b, transpose=transpose,
                rtol=1e-10, restart=10, maxiter=4, jacobi=jacobi,
            )
            assert result["info"] == 0
            assert result["iterations"] > 0
            assert result["true_relative_residual"] < 1e-9
            assert np.allclose(result["_x"], reference, rtol=1e-9, atol=1e-11)
