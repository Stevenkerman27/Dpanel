"""VortexAD mesh-coordinate VJP test on a small, fixed-topology mesh.

The nodal coordinates are CSDL inputs, while connectivity/TE indices stay fixed.
Checks a local panel-force objective against a centered FD perturbation of one
interior node coordinate. Does not certify Cp without cutoff, nor wing-body CAD.
"""
from pathlib import Path
import numpy as np
import pytest


def _build_case(csdl, vortex, points_value):
    recorder = csdl.Recorder(inline=False)
    recorder.start()
    points = csdl.Variable(value=np.asarray(points_value, dtype=float))
    alpha = csdl.Variable(value=np.array([4.0]))
    # Build topology once on the baseline mesh, then inject differentiable points.
    panel = vortex.PanelMethod({
        "Mach": 0.15,
        "alpha": alpha,
        "mesh_path": str(Path(vortex.SAMPLE_GEOMETRY_PATH) / "pm" / "naca0012_LE_TE_cluster.stl"),
        "ref_area": 10.0,
        "BC": "Dirichlet",
        "drag_type": "Trefftz",
        "Cp cutoff": -5.0,
    })
    # Replace imported mesh with the CSDL variable before pre-processing.
    panel.overwrite_mesh(points)
    panel.compute_cell_adjacency()
    panel.compute_TE_properties(threshold_angle=125)
    panel.declare_outputs(["panel_forces"])
    output = panel.evaluate()
    forces = output["panel_forces"]
    weights = np.zeros(forces.shape)
    weights[..., 2] = 1.0
    objective = csdl.sum(forces * csdl.Variable(value=weights))
    recorder.stop()
    sim = csdl.experimental.JaxSimulator(
        recorder=recorder, additional_inputs=[points, alpha],
        additional_outputs=[objective], gpu=False,
    )
    sim.run()
    return sim, points, alpha, objective


def test_vortexad_mesh_coordinate_vjp_centered_difference():
    csdl = pytest.importorskip("csdl_alpha")
    pytest.importorskip("jax")
    vortex = pytest.importorskip("VortexAD")
    mesh_path = Path(vortex.SAMPLE_GEOMETRY_PATH) / "pm" / "naca0012_LE_TE_cluster.stl"
    assert mesh_path.is_file(), f"Missing upstream mesh: {mesh_path}"

    import meshio
    mesh = meshio.read(mesh_path)
    points0 = np.asarray(mesh.points, dtype=float)
    # The full source mesh is intentionally used first to preserve its exact TE topology.
    sim, points, alpha, objective = _build_case(csdl, vortex, points0)
    totals = sim.compute_totals()
    grad = np.asarray(totals[(objective, points)], dtype=float).reshape(points0.shape)
    assert np.isfinite(grad).all()
    # Select the coordinate with largest finite non-boundary gradient.
    candidate = np.linalg.norm(grad, axis=1)
    idx = int(np.argmax(candidate))
    axis = int(np.argmax(np.abs(grad[idx])))
    assert abs(grad[idx, axis]) > 1e-10

    h = 1e-5
    plus = points0.copy(); plus[idx, axis] += h
    minus = points0.copy(); minus[idx, axis] -= h
    sim_plus, _, _, obj_plus = _build_case(csdl, vortex, plus)
    sim_minus, _, _, obj_minus = _build_case(csdl, vortex, minus)
    fd = (float(np.asarray(sim_plus[obj_plus]).sum()) - float(np.asarray(sim_minus[obj_minus]).sum())) / (2*h)
    ad = float(grad[idx, axis])
    assert np.isclose(ad, fd, rtol=1e-2, atol=1e-5)
