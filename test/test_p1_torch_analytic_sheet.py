"""Parity tests for the torch port of the analytic doublet-sheet velocity.

The port must reproduce the numpy closed-form evaluator to rounding: same edges, same
signs, same branches. These tests build a small folded sheet with a strength
discontinuity across a seam (so the boundary-edge integral is genuinely exercised) and
compare both backends point by point.
"""
import numpy as np
import pytest

pytest.importorskip("torch")

from panel.p1_analytic_sheet_velocity import AnalyticSheetField
from panel.p1_torch_analytic_sheet import TorchAnalyticSheetField


def sheet_parts():
    """Two-part strip with a trace jump along the seam at x = xs[1]."""
    nx, ny = 3, 3
    xs = np.linspace(0.0, 0.6, nx)
    ys = np.linspace(0.0, 0.4, ny)
    xyz = np.array([[x, y, 0.1*x] for x in xs for y in ys], float)

    def vid(i, j):
        return i*ny + j

    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            faces.append([a, b, c])
            faces.append([a, c, d])
    faces = np.array(faces, int)
    rng = np.random.default_rng(92)
    strength = rng.uniform(-1.0, 1.0, size=len(xyz))
    left = faces[np.all(xyz[faces][:, :, 0] <= xs[1] + 1e-12, axis=1)]
    right = faces[np.all(xyz[faces][:, :, 0] >= xs[1] - 1e-12, axis=1)]
    # Per-face vertex traces: the seam vertices carry two different values, which is
    # exactly the topological cut the boundary-edge integral is there for.
    return [(xyz, left, strength[left]), (xyz, right, (strength + 0.35)[right])]


def targets(n=24, seed=7):
    rng = np.random.default_rng(seed)
    points = rng.uniform([-0.2, -0.2, -0.3], [0.8, 0.6, 0.7], size=(n, 3))
    # Keep every target off the sheet plane: the trace is a jump there.
    points[:, 2] = np.where(np.abs(points[:, 2]) < 0.05,
                            np.sign(points[:, 2] + 1e-9)*0.05 + 0.02, points[:, 2])
    return points


def relative_error(a, b):
    scale = np.maximum(np.linalg.norm(b, axis=-1), 1e-12)
    return float(np.max(np.linalg.norm(a - b, axis=-1)/scale))


def test_torch_cpu_matches_numpy():
    parts = sheet_parts()
    reference = AnalyticSheetField(parts, freestream=(0.8, 0.0, 0.6))
    field = TorchAnalyticSheetField(parts, freestream=(0.8, 0.0, 0.6), device="cpu")
    points = targets()
    assert field.base is not None
    expected = np.asarray([reference(p) for p in points])
    got = field.velocity(points)
    assert relative_error(got, expected) <= 1e-12
    assert np.allclose(got, expected, rtol=0, atol=1e-14)


def test_from_arrays_matches_parts():
    parts = sheet_parts()
    base = AnalyticSheetField(parts, freestream=(1.0, 0.0, 0.0))
    arrays = {key: getattr(base, key) for key in
              ("triangles", "normals", "gamma", "face_edge_a", "face_edge_b",
               "edge_a", "edge_b", "edge_mu", "t_cross_n")}
    replayed = AnalyticSheetField.from_arrays(arrays, freestream=(1.0, 0.0, 0.0))
    torch_field = TorchAnalyticSheetField.from_arrays(arrays, device="cpu")
    points = targets()
    expected = np.asarray([replayed(p) for p in points])
    assert relative_error(torch_field.velocity(points), expected) <= 1e-12


def test_single_point_matches_batch_and_chunks():
    parts = sheet_parts()
    field = TorchAnalyticSheetField(parts, device="cpu")
    points = targets(9)
    batch = field.velocity(points, chunk=128)
    single = np.asarray([field(p) for p in points])
    one_at_a_time = field.velocity(points, chunk=1)
    # Chunk size changes the reduction order inside the kernels, so this is a
    # rounding-level check, not a bitwise one.
    assert relative_error(one_at_a_time, batch) <= 1e-15
    assert relative_error(single, batch) <= 1e-13


def test_return_tensor_keeps_device_values():
    import torch
    field = TorchAnalyticSheetField(sheet_parts(), device="cpu")
    points = targets(5)
    tensor = field.velocity(points, return_tensor=True)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float64
    assert relative_error(tensor.numpy(), field.velocity(points)) <= 1e-15


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA device")
def test_cuda_matches_numpy():
    parts = sheet_parts()
    reference = AnalyticSheetField(parts)
    field = TorchAnalyticSheetField(parts, device="cuda")
    points = targets()
    expected = np.asarray([reference(p) for p in points])
    got = field.velocity(points)
    assert relative_error(got, expected) <= 1e-12


@pytest.mark.skipif(__import__("torch").cuda.is_available(),
                    reason="a real device is present, absence cannot be simulated")
def test_cuda_without_device_fails_closed():
    with pytest.raises(RuntimeError):
        TorchAnalyticSheetField(sheet_parts(), device="cuda")


def test_invalid_device_rejected():
    with pytest.raises(ValueError):
        TorchAnalyticSheetField(sheet_parts(), device="mps")
    with pytest.raises(ValueError):
        TorchAnalyticSheetField(sheet_parts(), device=3)


def test_nonfinite_and_misshapen_points_rejected():
    field = TorchAnalyticSheetField(sheet_parts(), device="cpu")
    with pytest.raises(ValueError):
        field.velocity(np.array([[0.1, 0.1, np.nan]]))
    with pytest.raises(ValueError):
        field.velocity(np.zeros((4, 2)))


def test_point_on_source_vertex_fails_closed():
    field = TorchAnalyticSheetField(sheet_parts(), device="cpu")
    vertex = field._g["triangles"][0, 0].numpy()
    with pytest.raises(FloatingPointError):
        field.velocity(vertex)
