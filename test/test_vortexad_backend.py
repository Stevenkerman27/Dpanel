"""Opt-in upstream VortexAD panel-method API and angle-gradient smoke test.

Requires the isolated VortexAD/CSDL-alpha/JAX environment. This is not a
production adapter and does not certify raw Cp, mesh-coordinate VJP, or wing-body
coupling. The upstream Cp cutoff remains explicitly recorded as a limitation.
"""
from pathlib import Path

import numpy as np
import pytest


def test_vortexad_naca0012_forward_and_alpha_vjp():
    csdl = pytest.importorskip("csdl_alpha")
    pytest.importorskip("jax")
    vortex = pytest.importorskip("VortexAD")

    mesh = Path(vortex.SAMPLE_GEOMETRY_PATH) / "pm" / "naca0012_LE_TE_cluster.stl"
    assert mesh.is_file(), f"Upstream sample mesh missing: {mesh}"

    recorder = csdl.Recorder(inline=False)
    recorder.start()
    alpha = csdl.Variable(value=np.array([5.0]))
    panel = vortex.PanelMethod({
        "Mach": 0.2,
        "alpha": alpha,
        "mesh_path": str(mesh),
        "ref_area": 10.0,
        "BC": "Dirichlet",
        "drag_type": "Trefftz",
        "Cp cutoff": -5.0,  # upstream path clips Cp; raw-pressure contract NOT passed
    })
    panel.declare_outputs(["CL", "CDi", "Cp", "panel_forces"])
    output = panel.evaluate()
    forces = output["panel_forces"]
    weights = np.zeros(forces.shape)
    weights[..., 2] = 1.0
    weights_var = csdl.Variable(value=weights)
    local_z_sum = csdl.sum(forces * weights_var)
    recorder.stop()

    sim = csdl.experimental.JaxSimulator(
        recorder=recorder,
        additional_inputs=[alpha],
        additional_outputs=[output["CL"], output["CDi"], output["Cp"],
                            output["panel_forces"], local_z_sum],
        gpu=False,
    )
    sim.run()
    cl = np.asarray(sim[output["CL"]], dtype=float)
    cdi = np.asarray(sim[output["CDi"]], dtype=float)
    cp = np.asarray(sim[output["Cp"]], dtype=float)
    panel_force = np.asarray(sim[output["panel_forces"]], dtype=float)
    local_value = float(np.asarray(sim[local_z_sum]).sum())
    assert np.isfinite(cl).all() and cl.size > 0 and np.max(cl) > 0
    assert np.isfinite(cdi).all()
    assert cp.size > 0 and np.isfinite(cp).all()
    assert panel_force.shape[-1] == 3 and np.isfinite(panel_force).all()

    # JAX total derivatives of both a global scalar and a local-force scalar.
    totals = sim.compute_totals()
    dcl = np.asarray(totals[(output["CL"], alpha)], dtype=float)
    dlocal = np.asarray(totals[(local_z_sum, alpha)], dtype=float)
    assert np.isfinite(dcl).all() and np.max(np.abs(dcl)) > 0
    assert np.isfinite(dlocal).all() and np.max(np.abs(dlocal)) > 0

    # Independent centered finite difference at fixed mesh/topology.
    h = 1.0e-3
    sim[alpha] = np.array([5.0 + h])
    sim.run()
    cl_plus = np.asarray(sim[output["CL"]], dtype=float)
    local_plus = float(np.asarray(sim[local_z_sum]).sum())
    sim[alpha] = np.array([5.0 - h])
    sim.run()
    cl_minus = np.asarray(sim[output["CL"]], dtype=float)
    local_minus = float(np.asarray(sim[local_z_sum]).sum())
    fd_cl = (cl_plus - cl_minus) / (2 * h)
    fd_local = (local_plus - local_minus) / (2 * h)
    assert np.allclose(dcl.reshape(fd_cl.shape), fd_cl, rtol=5e-3, atol=1e-5)
    assert np.allclose(dlocal.reshape(()), fd_local, rtol=5e-3, atol=1e-5)

    # Intentionally not asserting Cp is un-clipped: upstream default path has
    # Cp cutoff and must be audited separately before production training.
