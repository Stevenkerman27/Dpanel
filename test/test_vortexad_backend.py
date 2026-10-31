"""Opt-in VortexAD panel-method forward and angle-gradient smoke test.

Requires isolated VortexAD/CSDL-alpha/JAX. It does not certify raw Cp,
mesh-coordinate VJP, or wing-body coupling. Cp cutoff remains a known limitation.
"""
from pathlib import Path
import numpy as np
import pytest


def test_vortexad_naca0012_forward_and_alpha_vjp():
    csdl = pytest.importorskip("csdl_alpha")
    pytest.importorskip("jax")
    vortex = pytest.importorskip("VortexAD")
    mesh = Path(vortex.SAMPLE_GEOMETRY_PATH) / "pm" / "naca0012_LE_TE_cluster.stl"
    if not mesh.is_file():
        pytest.skip(f"VortexAD sample mesh data missing from this install: {mesh} "
                    "(the wheel ships no STL under core/geometry/sample_meshes/pm; "
                    "use a source checkout on PYTHONPATH or copy the STL files)")

    recorder = csdl.Recorder(inline=False)
    recorder.start()
    alpha = csdl.Variable(value=np.array([5.0]))
    panel = vortex.PanelMethod({
        "Mach": 0.2, "alpha": alpha, "mesh_path": str(mesh),
        "ref_area": 10.0, "BC": "Dirichlet", "drag_type": "Trefftz",
        "Cp cutoff": -5.0,  # upstream clips Cp; raw-pressure gate NOT passed
    })
    panel.declare_outputs(["CL", "CDi", "Cp", "panel_forces"])
    output = panel.evaluate()
    forces = output["panel_forces"]
    weights = np.zeros(forces.shape)
    weights[..., 2] = 1.0
    local_z_sum = csdl.sum(forces * csdl.Variable(value=weights))
    recorder.stop()

    sim = csdl.experimental.JaxSimulator(
        recorder=recorder, additional_inputs=[alpha],
        additional_outputs=[output["CL"], output["CDi"], output["Cp"],
                            output["panel_forces"], local_z_sum],
        gpu=False,
    )
    sim.run()
    cl = np.asarray(sim[output["CL"]], dtype=float)
    cdi = np.asarray(sim[output["CDi"]], dtype=float)
    cp = np.asarray(sim[output["Cp"]], dtype=float)
    panel_force = np.asarray(sim[output["panel_forces"]], dtype=float)
    assert np.isfinite(cl).all() and cl.size > 0 and np.max(cl) > 0
    assert np.isfinite(cdi).all()
    assert cp.size > 0 and np.isfinite(cp).all()
    assert panel_force.shape[-1] == 3 and np.isfinite(panel_force).all()

    totals = sim.compute_totals()
    dcl = np.asarray(totals[(output["CL"], alpha)], dtype=float)
    dlocal = np.asarray(totals[(local_z_sum, alpha)], dtype=float)
    assert np.isfinite(dcl).all() and np.max(np.abs(dcl)) > 0
    assert np.isfinite(dlocal).all() and np.max(np.abs(dlocal)) > 0

    h = 1e-3
    sim[alpha] = np.array([5.0 + h]); sim.run()
    cl_plus = np.asarray(sim[output["CL"]], dtype=float)
    local_plus = float(np.asarray(sim[local_z_sum]).sum())
    sim[alpha] = np.array([5.0 - h]); sim.run()
    cl_minus = np.asarray(sim[output["CL"]], dtype=float)
    local_minus = float(np.asarray(sim[local_z_sum]).sum())
    fd_cl = (cl_plus - cl_minus) / (2*h)
    fd_local = (local_plus - local_minus) / (2*h)
    assert np.allclose(dcl.reshape(fd_cl.shape), fd_cl, rtol=5e-3, atol=1e-5)
    assert np.allclose(dlocal.reshape(()), fd_local, rtol=5e-3, atol=1e-5)
    # Mesh-coordinate VJP, raw Cp and wing-body coupling remain separate gates.
