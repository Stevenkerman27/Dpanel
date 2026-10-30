"""Compare raw Cp with VortexAD's default and an intentionally strong cutoff.

Requires the local diagnostic patch from test/patch_vortexad_raw_cp.py.
Records the default Cp_cutoff=-5 effect and uses cutoff=0 as a positive control.
This is diagnostic only, not production approval.
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
    panel.declare_outputs(["CL", "CDi", "CM", "Cp", "Cp_raw", "panel_forces"])
    out = panel.evaluate()
    recorder.stop()
    sim = csdl.experimental.JaxSimulator(
        recorder=recorder, additional_inputs=[alpha],
        additional_outputs=[out["CL"], out["CDi"], out["CM"], out["Cp"],
                            out["Cp_raw"], out["panel_forces"]],
        gpu=False,
    )
    sim.run()
    return {
        "CL": np.asarray(sim[out["CL"]], dtype=float),
        "CDi": np.asarray(sim[out["CDi"]], dtype=float),
        "CM": np.asarray(sim[out["CM"]], dtype=float),
        "Cp": np.asarray(sim[out["Cp"]], dtype=float),
        "Cp_raw": np.asarray(sim[out["Cp_raw"]], dtype=float),
        "forces": np.asarray(sim[out["panel_forces"]], dtype=float),
    }


def _report_delta(label, reference, other):
    print(f"{label}: max|dCp|={np.max(np.abs(other['Cp']-reference['Cp_raw'])):.8e}")
    for key in ("CL", "CDi", "CM", "forces"):
        delta = np.max(np.abs(other[key] - reference[key]))
        scale = max(float(np.max(np.abs(reference[key]))), 1e-12)
        print(f"{label}: {key} max_abs_delta={delta:.8e}, relative_to_raw={delta/scale:.8e}")


def test_raw_cp_is_preserved_and_cutoff_effect_is_measured():
    csdl = pytest.importorskip("csdl_alpha")
    pytest.importorskip("jax")
    vortex = pytest.importorskip("VortexAD")
    mesh = Path(vortex.SAMPLE_GEOMETRY_PATH) / "pm" / "naca0012_LE_TE_cluster.stl"
    if not mesh.is_file():
        pytest.skip(f"VortexAD sample mesh data missing from this install: {mesh} "
                    "(the wheel ships no STL under core/geometry/sample_meshes/pm; "
                    "use a source checkout on PYTHONPATH or copy the STL files)")

    raw = _run(csdl, vortex, None)
    default = _run(csdl, vortex, -5.0)
    clipped = _run(csdl, vortex, 0.0)

    for result in (raw, default, clipped):
        for key in ("CL", "CDi", "CM", "Cp", "Cp_raw", "forces"):
            assert np.isfinite(result[key]).all(), key

    assert np.allclose(raw["Cp"], raw["Cp_raw"], rtol=0, atol=1e-12)
    assert np.allclose(default["Cp_raw"], raw["Cp_raw"], rtol=1e-9, atol=1e-9)
    assert np.allclose(clipped["Cp_raw"], raw["Cp_raw"], rtol=1e-9, atol=1e-9)

    # The strong cutoff is a positive control: it must alter Cp and integrated forces.
    assert np.max(np.abs(clipped["Cp"] - raw["Cp_raw"])) > 1e-2
    assert np.max(np.abs(clipped["forces"] - raw["forces"])) > 1e-2

    # Report the actual production-default (-5) effect without assuming it is active.
    # This is important because an ordinary subsonic NACA0012 case may never reach -5.
    _report_delta("default cutoff=-5", raw, default)
    _report_delta("strong cutoff=0", raw, clipped)
