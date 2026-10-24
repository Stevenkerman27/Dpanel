"""VortexAD mesh-coordinate VJP test on a fixed-topology mesh.

Nodal coordinates and alpha are CSDL inputs; connectivity/TE indices are fixed.
Checks a local panel-force objective against a centered FD perturbation of one
interior node. This is not a raw-Cp or wing-body CAD certification.
"""
from pathlib import Path
import numpy as np
import pytest


def _build_case(csdl, vortex, points_value, adjacency, te):
    recorder = csdl.Recorder(inline=False)
    recorder.start()
    points = csdl.Variable(value=np.asarray(points_value, dtype=float))
    alpha = csdl.Variable(value=np.array([4.0]))
    panel = vortex.PanelMethod({
        "Mach": 0.15,
        "alpha": alpha,
        "mesh_path": "unused_topology_is_injected",
        "ref_area": 10.0,
        "BC": "Dirichlet",
        "drag_type": "Trefftz",
        "Cp cutoff": -5.0,
    }, skip_geometry=True)
    panel.points = points
    panel.insert_grid_data(points, adjacency, te)
    panel.setup_flow_properties()
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
    from VortexAD.utils.unstructured_grids.cell_adjacency import find_cell_adjacency
    from VortexAD.utils.unstructured_grids.TE_detection import TE_detection

    raw = meshio.read(mesh_path)
    points0, cells, adjacency, edges2cells, points2cells = find_cell_adjacency(
        raw.points, raw.cells_dict
    )
    te = TE_detection(points0, cells, edges2cells, points2cells, threshold_theta=125)
    adj_data = (points0, cells, adjacency, edges2cells, points2cells)
    sim, points, alpha, objective = _build_case(csdl, vortex, points0, adj_data, te)
    totals = sim.compute_totals()
    grad = np.asarray(totals[(objective, points)], dtype=float).reshape(points0.shape)
    assert np.isfinite(grad).all()

    candidate = np.linalg.norm(grad, axis=1)
    idx = int(np.argmax(candidate))
    axis = int(np.argmax(np.abs(grad[idx])))
    assert abs(grad[idx, axis]) > 1e-10

    h = 1e-5
    plus = points0.copy(); plus[idx, axis] += h
    minus = points0.copy(); minus[idx, axis] -= h
    sim[points] = plus
    sim.run()
    f_plus = float(np.asarray(sim[objective]).sum())
    sim[points] = minus
    sim.run()
    f_minus = float(np.asarray(sim[objective]).sum())
    fd = (f_plus - f_minus) / (2*h)
    ad = float(grad[idx, axis])
    assert np.isclose(ad, fd, rtol=1e-2, atol=1e-5)
