
import json, pathlib
base = pathlib.Path(r"D:\3D\Projects\panel\runs\fix_master\docs\issue84")
for name in ["t5_gate_ab_final_advance", "t5_gate_ab_final_frozen", "t5_gate_ab_final_legacy"]:
    p = base / (name + ".json")
    d = json.loads(p.read_text(encoding="utf-8"))
    print("==", name, d["status"], "t=%.1f" % d["seconds"], "trunc", d["truncation_policy"], "step_f", d["step_fraction"])
    for r in d["rows"]:
        c = r.get("contacts") or {}
        mw = ((r.get("clip") or {}).get("main_wing") or {})
        print("  ", r["iteration"], r.get("reason"), "relax=", r.get("relaxation"),
              "dmax=%.3e" % (r.get("displacement_max") or 0.),
              "map=%.4f" % (r["map_residual_chord"] if r.get("map_residual_chord") is not None else float("nan")),
              "CL=", (round(r["CL"], 6) if r.get("CL") is not None else None),
              "CDp=", (round(r["CD_pressure"], 6) if r.get("CD_pressure") is not None else None),
              "viol=", c.get("violations_count"), "pre=", c.get("pre_existing"),
              "clip=", mw.get("clipped"), "capmin=", (round(mw["cap_min"], 5) if mw.get("cap_min") is not None else None))
    print("   trial_reasons:", (d["rows"][-1].get("trial_reasons") or [])[:12])
