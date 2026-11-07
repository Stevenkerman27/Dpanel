import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from panel.p1_cst_gradient import fixed_wake_surface
from panel.p1_lifting_g1 import from_dpanel_half_surface, make_wing
from panel.p1_lifting_g2 import combine_wings, solve_multi


def test_imported_surface_forward_and_fixed_y_gradient(tmp_path):
    source = make_wing(chord_panels=3, span_panels=4, wake_panels=2)
    points = np.vstack((source.xyz, source.wake_xyz))
    wake = source.wake_faces + len(source.xyz)
    wall_half = source.faces[source.xyz[source.faces][:, :, 1].mean(axis=1) > 1e-9]
    wake_half = wake[points[wake][:, :, 1].mean(axis=1) > 1e-9]
    used = np.unique(np.r_[wall_half.ravel(), wake_half.ravel(), source.te_geo[2:]])
    indices = np.full(len(points), -1, dtype=int)
    indices[used] = np.arange(len(used))
    te = np.stack((source.te_geo[2:], source.te_geo[2:]), axis=1)
    path = tmp_path / "surface.json"
    path.write_text(json.dumps(dict(
        coordinates=points[used].tolist(), faces=indices[wall_half].tolist(),
        wake_faces=indices[wake_half].tolist(), trailing_edges=indices[te].tolist(),
        gamma_stations=source.gamma_y[2:].tolist(), reference_chord=source.chord,
        reference_area=source.reference_area, reference_point=[0., 0., 0.])),
        encoding="utf-8")
    wing = from_dpanel_half_surface(path)
    scale = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    wall = torch.as_tensor(wing.xyz[:, [0, 2]]) * scale
    wake_points = torch.as_tensor(wing.wake_xyz[:, [0, 2]]) * scale
    values, info = fixed_wake_surface(path, wall, wake_points, wing.reference_area * scale)
    derivative = torch.stack([torch.autograd.grad(value, scale, retain_graph=True)[0]
                              for value in values]).detach().numpy()
    forward = solve_multi(combine_wings([wing]), 4.)
    np.testing.assert_allclose(values.detach().numpy(), [forward["CL"], forward["CD_pressure"]],
                               atol=2e-9)
    assert info["stationary_verified"]
    results = []
    for factor in (1. - 1e-5, 1. + 1e-5):
        displaced = replace(wing, xyz=wing.xyz * [factor, 1., factor],
                            wake_xyz=wing.wake_xyz * [factor, 1., factor],
                            chord=wing.chord * factor,
                            reference_area=wing.reference_area * factor)
        result = solve_multi(combine_wings([displaced]), 4.)
        results.append([result["CL"], result["CD_pressure"]])
    np.testing.assert_allclose(derivative, (np.asarray(results[1]) - results[0]) / 2e-5,
                               atol=2e-6, rtol=2e-4)
    with pytest.raises(ValueError, match="topology"):
        fixed_wake_surface(path, wall[:-1], wake_points, wing.reference_area)
