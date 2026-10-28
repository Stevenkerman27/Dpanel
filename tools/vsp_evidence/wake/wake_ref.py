"""Reference / far-field companions to wake_sweep.py (same 12996-panel mesh).

ref_gui   : the archived GUI-like settings (nodes 8, iters 5, relax 1, FarDist -1)
far5      : nodes 32 / iters 20 / relax 0.5 with a fixed FarDist = 5 (vs auto -1)
"""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wake_sweep as ws


def with_far(txt, far):
    txt = re.sub(r"FarDistToggle = \d+", "FarDistToggle = 1", txt)
    return re.sub(r"FarDist = [0-9.eE+-]+", "FarDist = %g" % far, txt)


def main():
    template = (ws.SRC / "trap.vspaero").read_text()
    cases = [
        ("ref_gui", ws.make_config(template, 8, 5, 1.0)),
        ("far5", with_far(ws.make_config(template, 32, 20, 0.5), 5.0)),
    ]
    out = []
    for tag, txt in cases:
        print("running", tag, flush=True)
        res = ws.run_one(tag, txt)
        print(json.dumps(res), flush=True)
        out.append(res)
    (ws.HERE / "wakesweep" / "results_ref.json").write_text(json.dumps(out, indent=1))
    print("DONE")


if __name__ == "__main__":
    main()