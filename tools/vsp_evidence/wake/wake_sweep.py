"""VSPAERO-side wake sensitivity: NumWakeNodes / WakeIters / WakeRelax / frozen wake.

One fixed thick-panel mesh (the 12996-face level that produced the archived 3.55
panel numbers), one angle of attack (5 deg), only the wake controls change.
"""
import json, re, subprocess, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "api_355_panel"
EXE = Path(r"D:/3D/Projects/OpenVSP-3.55.0-win64/vspaero.exe")


def parse_polar(path):
    text = path.read_text(errors="ignore").splitlines()
    h = next(i for i, l in enumerate(text) if l.strip().startswith("Beta"))
    cols = text[h].split()
    rows = []
    for line in text[h + 1:]:
        f = line.split()
        if len(f) == len(cols):
            try:
                rows.append(dict(zip(cols, (float(v) for v in f))))
            except ValueError:
                pass
    return rows


def make_config(template, nodes, iters, relax, freeze=None, alpha=5.0):
    txt = template
    txt = re.sub(r"NumWakeNodes = \d+", "NumWakeNodes = %d" % nodes, txt)
    txt = re.sub(r"WakeIters = \d+", "WakeIters = %d" % iters, txt)
    txt = re.sub(r"WakeRelax = [0-9.]+", "WakeRelax = %g" % relax, txt)
    txt = re.sub(r"AoA = [0-9., ]+", "AoA = %g" % alpha, txt)
    if freeze is not None:
        txt = re.sub(r"FreezeWakeAtIteration = \d+", "FreezeWakeAtIteration = %d" % freeze, txt)
    return txt


def run_one(tag, txt):
    d = HERE / "wakesweep" / tag
    d.mkdir(parents=True, exist_ok=True)
    for f in ("trap.vspgeom", "trap.vkey"):
        if (SRC / f).exists():
            shutil.copy2(SRC / f, d / f)
    (d / "trap.vspaero").write_text(txt)
    log = open(d / "stdout.log", "w")
    rc = subprocess.call([str(EXE), "-omp", "32", "trap"], cwd=d, stdout=log, stderr=subprocess.STDOUT)
    log.close()
    rows = parse_polar(d / "trap.polar") if (d / "trap.polar").exists() else []
    if not rows:
        return dict(tag=tag, rc=rc, error="no polar rows")
    r = rows[0]
    keep = {k: r.get(k) for k in ("CLi", "CLtot", "CLwtot", "CDi", "CDiw", "CDtot", "CDwtot", "CMytot")}
    return dict(tag=tag, rc=rc, alpha=r.get("AoA"), **keep)


def main():
    template = (SRC / "trap.vspaero").read_text()
    print("template wake:", re.findall(r"(NumWakeNodes|WakeIters|WakeRelax|FreezeWakeAtIteration) = [0-9.]+", template), flush=True)
    cases = []
    for n in (4, 8, 16, 32, 64):
        cases.append(("nodes%02d" % n, make_config(template, n, 20, 0.5)))
    for it in (1, 3, 5, 10, 20, 40):
        cases.append(("iters%02d" % it, make_config(template, 32, it, 0.5)))
    for rl in (0.2, 0.5, 1.0):
        cases.append(("relax%02d" % round(rl * 10), make_config(template, 32, 20, rl)))
    cases.append(("frozen", make_config(template, 32, 20, 0.5, freeze=1)))
    out = []
    for tag, txt in cases:
        print("running", tag, flush=True)
        res = run_one(tag, txt)
        print(json.dumps(res), flush=True)
        out.append(res)
    (HERE / "wakesweep" / "results.json").write_text(json.dumps(out, indent=1))
    print("DONE")


if __name__ == "__main__":
    main()