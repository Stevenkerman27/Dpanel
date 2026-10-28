"""Cross-check extracted Torch panel potentials and spatial gradients.

Independent reference: Dpanel's existing NumPy/Numba analytic triangle kernel.
Both kernels use the same outward orientation and G=1/(4*pi*r); no self-trace
or singularity test is implied here (the separate suites cover those cases).
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numba")

from torch_panel.kernels import triangle_influence


def _reference():
    file = Path(__file__).resolve().parents[1] / "core" / "triangle_kernel.py"
    spec = importlib.util.spec_from_file_location("dpanel_reference_triangle_kernel", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.influence


def test_potentials_and_field_gradients_agree_with_numba():
    face = np.array(
        [[[0., 0., 0.], [1.1, 0.1, 0.03], [0.2, 0.9, -0.07]]],
        dtype=np.float64,
    )
    targets = np.array(
        [[0.31, 0.24, 0.36], [0.45, 0.2, -0.55], [1.4, -0.3, 0.6]],
        dtype=np.float64,
    )
    face_normal = np.cross(face[:, 1] - face[:, 0], face[:, 2] - face[:, 0])
    face_normal /= np.linalg.norm(face_normal, axis=1)[:, None]

    influence = _reference()
    source_ref, source_grad_ref = influence(
        targets, face, face_normal, np.full(len(targets), -1, dtype=np.int64), 0
    )
    doublet_ref, doublet_grad_ref = influence(
        targets, face, face_normal, np.full(len(targets), -1, dtype=np.int64), 1
    )

    x = torch.tensor(targets, dtype=torch.float64, requires_grad=True)
    t = torch.tensor(face, dtype=torch.float64)
    source, doublet = triangle_influence(x, t)
    source_grad = torch.autograd.grad(source.sum(), x, retain_graph=True)[0]
    doublet_grad = torch.autograd.grad(doublet.sum(), x)[0]

    np.testing.assert_allclose(source.detach().numpy(), source_ref, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(doublet.detach().numpy(), doublet_ref, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        source_grad.detach().numpy(), source_grad_ref.sum(axis=1), rtol=1e-7, atol=1e-9
    )
    np.testing.assert_allclose(
        doublet_grad.detach().numpy(), doublet_grad_ref.sum(axis=1), rtol=1e-7, atol=1e-9
    )
