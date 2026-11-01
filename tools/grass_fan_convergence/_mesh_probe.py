"""Mesh sensitivity of the initial and updated coefficients for one GRASS wing."""
import numpy as np
from panel.p1_coupled_material_wake import solve_coupled_material_wake
from tools.grass_fan_convergence.cases import cases
from tools.grass_fan_convergence.run_sweep import build_mesh

base = [c for c in cases() if c["name"] == "g01_rect_ar6"][0]
kw_fan = dict(velocity_method="rm_regularized_p1", core_radius=0.01,
              area_quadrature="projection_fan", fan_order=8, fan_strips=2,
              fan_refinement_order=16, fan_refinement_strips=2)

for span_panels, chord_panels in [(3, 8), (6, 8), (3, 12), (6, 12)]:
    case = dict(base)
    case["span_panels"] = span_panels
    case["chord_panels"] = chord_panels
    wing, mesh = build_mesh(case)
    for iters in (0, 1):
        res = solve_coupled_material_wake(mesh, 5.0, kutta="pressure",
                                          iterations=iters, order=48, **kw_fan)
        st = res["state"]
        hist = res["history"][0] if res["history"] else {}
        wake = next(iter(hist.get("wakes", {}).values()), {}) if hist else {}
        elig = wake.get("eligibility") or {}
        print("span=%d chord=%d faces=%d wake=%d iters=%d CL=%+.8f CD_p=%+.8f CMy=%+.8f "
              "moved=%s" % (span_panels, chord_panels, len(mesh.faces),
                            len(wing.wake_faces), iters, st["CL"], st["CD_pressure"],
                            st["CM_y"], (hist.get("outgoing_update") or {}).get("moved_nodes")),
              flush=True)
