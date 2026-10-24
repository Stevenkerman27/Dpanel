"""Compare raw and clipped Cp on an identical fixed VortexAD mesh.

Requires the local diagnostic patch from test/patch_vortexad_raw_cp.py.
Uses Cp_cutoff=0 for an intentionally strong clipping control, and None for
the physical pressure path. This is diagnostic only, not production approval.
"""
from pathlib import Path
import numpy as np
import pytest


def _run(csdl, vortex, cutoff):
    mesh = Path(vortex.SAMPLE_GEOMETRY_PATH) / "pm" / "naca0012_LE_TE_cluster.stl"
    recorder = csdl.Recorder(inline=False)
    recorder.start()
    alpha = csdl.Variable(value=np.array([5.0]))
    panel = vortex.PanelMethod({
        "Mach": 0.2, "alpha": alpha, "mesh_path": str(mesh),
        "ref_area": 10.0, "BC": "Dirichlet", "drag_type": "Trefftz",
        "Cp cutoff": cutoff,
    })
    panel.declare_outputs(["CL", "CDi", "Cp", "Cp_raw", "panel_forces"])
    out = panel.evaluate()
    recorder.stop()
    sim = csdl.experimental.JaxSimulator(
        recorder=recorder, additional_inputs=[alpha],
        additional_outputs=[out["CL"], out["CDi"], out["Cp"], out["Cp_raw"],
                            out["panel_forces"]],
        gpu=False,
    )
    sim.run()
    return {
        "CL": np.asarray(sim[out["CL"]], dtype=float),
        "CDi": np.asarray(sim[out["CDi"]], dtype=float),
        "Cp": np.asarray(sim[out["Cp"]], dtype=float),
        "Cp_raw": np.asarray(sim[out["Cp_raw"]], dtype=float),
        "forces": np.asarray(sim[out["panel_forces"]], dtype=float),
    }


def test_raw_cp_is_preserved_and_cutoff_is_explicit():
    csdl = pytest.importorskip("csdl_alpha")
    pytest.importorskip("jax")
    vortex = pytest.importorskip("VortexAD")
    raw = _run(csdl, vortex, None)
    clipped = _run(csdl, vortex, 0.0)

    assert np.isfinite(raw["Cp_raw"]).all()
    assert np.isfinite(raw["forces"]).all()
    assert np.allclose(raw["Cp"], raw["Cp_raw"], rtol=0, atol=1e-12)
    assert np.allclose(clipped["Cp_raw"], raw["Cp_raw"], rtol=1e-9, atol=1e-9)
    # Smooth maximum uses rho=100; cutoff=0 should visibly alter negative Cp.
    assert np.max(np.abs(clipped["Cp"] - raw["Cp_raw"])) > 1e-2
    assert np.max(np.abs(clipped["forces"] - raw["forces"])) > 1e-2
    # Clipping is an output/force-path choice; raw physical Cp stays untouched.
    assert np.isfinite(raw["CL"]).all() and np.isfinite(raw["CDi"]).all()
