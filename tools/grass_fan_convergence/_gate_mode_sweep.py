"""Ten geometries x three gate modes, analytic kernel, one geometry update.

Verifies that 'off' reproduces 'audit' bitwise (the refined-fan evaluation is a
pure audit) and quantifies the freezing artifact against the shipped code path
instead of the epsilon_tol=1e3 shortcut.
"""
import json
import time
from pathlib import Path

import numpy as np

from tools.grass_fan_convergence.cases import cases
from tools.grass_fan_convergence.run_sweep import resolve_airfoils
from tools.grass_fan_convergence.wing_builder import build_wing
from panel.p1_lifting_g2 import combine_wings
from panel.p1_coupled_material_wake import solve_coupled_material_wake


def main():
    rows = []
    for case in cases():
        sections = resolve_airfoils([dict(s) for s in case["sections"]])
        wing = build_wing(sections, span_panels=case["span_panels"],
                          chord_panels=case["chord_panels"],
                          wake_panels=case["wake_panels"], wake_length=5.0,
                          name=case["name"])
        mesh = combine_wings([wing], names=[case["name"]], reference_chord=wing.chord,
                             reference_point=np.zeros(3))
        for mode in ("enforce", "audit", "off"):
            started = time.perf_counter()
            result = solve_coupled_material_wake(mesh, 5.0, kutta="pressure",
                                                 iterations=1, gate_mode=mode)
            eligibility = [row["wakes"][g]["eligibility"] for row in result["history"]
                           for g in row["wakes"]]
            row = dict(case=case["name"], gate_mode=mode,
                       seconds=time.perf_counter()-started,
                       status=result["status"],
                       CL=float(result["state"]["CL"]),
                       CD_p=float(result["state"]["CD_pressure"]),
                       CM_y=float(result["state"]["CM_y"]),
                       rejected=[int(e["rejected"]) for e in eligibility],
                       audit_rejected=[int(e["audit_rejected"]) for e in eligibility])
            rows.append(row)
            print("%-20s %-8s %6.2fs CL=%+.8f CD_p=%+.8f CMy=%+.8f rejected=%s" %
                  (case["name"], mode, row["seconds"], row["CL"], row["CD_p"],
                   row["CM_y"], row["rejected"]), flush=True)
    out = Path("docs/issue84/gate_mode_sweep.json")
    out.write_text(json.dumps(dict(note="ten geometries, analytic kernel, one update",
                                   runs=rows), indent=1), encoding="utf-8")
    print("wrote " + str(out))
    # audit vs off must be identical; print the largest deviation per case
    for case in {r["case"] for r in rows}:
        audit = [r for r in rows if r["case"] == case and r["gate_mode"] == "audit"][0]
        off = [r for r in rows if r["case"] == case and r["gate_mode"] == "off"][0]
        print("%-20s audit-vs-off |dCL|=%.3e  time %.2fs vs %.2fs" %
              (case, abs(audit["CL"]-off["CL"]), audit["seconds"], off["seconds"]))


if __name__ == "__main__":
    main()
