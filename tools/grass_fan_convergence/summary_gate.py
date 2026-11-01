"""Gate-mode table: freezing versus auditing the sampling error.

Reads docs/issue84/gate_mode_sweep.json (ten geometries, analytic kernel, one
geometry update, gate_mode in enforce/audit/off) and prints markdown.
"""
import argparse
import json
from pathlib import Path

NAMES = {"g01_rect_ar6": "rectangular AR 6", "g02_taper05_ar7": "tapered AR 7",
         "g03_swept30_ar6": "swept 30 deg AR 6", "g04_swept45_taper02": "swept 45 deg taper 0.2",
         "g05_elliptic_ar10": "elliptic AR 10", "g06_glider_ar12": "glider AR 12",
         "g07_delta_ar2p5": "delta AR 2.5", "g08_dihedral_washout": "dihedral + washout",
         "g09_thick0018_ar5": "thick 18% AR 5", "g10_camber4412_ar7": "cambered AR 7"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="docs/issue84/gate_mode_sweep.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    runs = json.loads(Path(args.json).read_text(encoding="utf-8"))["runs"]
    by_case = {}
    for row in runs:
        by_case.setdefault(row["case"], {})[row["gate_mode"]] = row
    lines = ["| geometry | nodes rejected | CL freeze | CL audit | CL off | freeze/audit | max abs(CL audit - CL off) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    worst = 0.
    for case in sorted(by_case):
        rows = by_case[case]
        enforce, audit, off = rows["enforce"], rows["audit"], rows["off"]
        rejected = sum(enforce["rejected"])
        ratio = enforce["CL"]/audit["CL"] if audit["CL"] else float("nan")
        delta = max(abs(audit["CL"]-off["CL"]), abs(audit["CD_p"]-off["CD_p"]),
                    abs(audit["CM_y"]-off["CM_y"]))
        worst = max(worst, ratio if ratio == ratio else 0.)
        lines.append("| %s | %d / %d | %+.5f | %+.5f | %+.5f | %.2fx | %.1e |" %
                     (NAMES.get(case, case), rejected, sum(enforce["audit_rejected"]),
                      enforce["CL"], audit["CL"], off["CL"], ratio, delta))
    lines.append("")
    lines.append("worst freeze/audit ratio: %.2fx" % worst)
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print("wrote " + args.out)


if __name__ == "__main__":
    main()
