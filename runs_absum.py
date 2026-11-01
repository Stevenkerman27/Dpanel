
import json, pathlib
for name in ("advance","frozen","legacy"):
    d = json.loads(pathlib.Path(r"D:\3D\Projects\panel\runs\fix_master\docs\issue84\t5_gate_ab_final_%s.json" % name).read_text(encoding="utf-8"))
    print("==", name)
    for it in d["rows"]:
        c = it.get("contacts") or {}
        clip = it.get("clip") or {}
        print("  it%s acc=%s map=%.6g CDp=%.6g relax=%s clip(mode=%s clipped=%s acc_max=%s cap_min=%s) contacts=%s kinds=%s viol=%s worst=%s" % (
            it["iteration"], it["accepted"], it.get("map_residual_chord") or -1, it.get("CD_pressure") or -1,
            it.get("relaxation"), clip.get("mode"), clip.get("clipped"), clip.get("accepted_max"), clip.get("cap_min"),
            c.get("tested"), c.get("kinds"), c.get("violations_count"), c.get("worst_depth")))
    last = d["rows"][-1]
    print("  last trial_details:", json.dumps(last.get("trial_details"))[:900])
    print("  last reason_counts:", last.get("reason_counts"), "control_nodes", last.get("control_nodes"))
