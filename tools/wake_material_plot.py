"""Plot frozen B3 ablations and safeguarded normal-slip histories."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--runs",type=Path,required=True)
parser.add_argument("--final",type=Path,required=True)
parser.add_argument("--out",type=Path,required=True)
args=parser.parse_args()
fig,axes=plt.subplots(2,3,figsize=(12,6.5),layout="constrained")
colors={"old":"#c4413a","attachment":"#8661a8","material":"#287db0","both":"#16806a"}
for column,case in enumerate(("rect","swept30","elliptic")):
    for mode in colors:
        data=json.loads((args.runs/(case+"_"+mode+"_raw8")/"history.json").read_text())
        rows=data["rows"]
        axes[0,column].plot([r["iteration"] for r in rows],[r["CL"] for r in rows],
            color=colors[mode],label=mode,linestyle="--" if mode=="material" else "-")
    final=json.loads((args.final/(case+"_both_guarded60")/"history.json").read_text())
    rows=final["rows"]
    axes[1,column].plot([r["iteration"] for r in rows],
        [max(v["normal_slip_rms"] for v in r["wakes"].values()) for r in rows],
        color=colors["both"],marker=".",markersize=3)
    axes[1,column].axhline(.01,color="#777777",linestyle=":",label="slip tolerance")
    axes[0,column].set_title(case)
    axes[0,column].set_xlabel("Unrestricted update")
    axes[1,column].set_xlabel("Safeguarded update")
    axes[1,column].set_title(final.get("solver_status",""),fontsize=10)
    for row in range(2):
        axes[row,column].grid(alpha=.2)
axes[0,0].set_ylabel("CL")
axes[1,0].set_ylabel("Normal slip RMS / U")
axes[0,0].legend(fontsize=8)
axes[1,0].legend(fontsize=8)
fig.suptitle("Material-map repair restores loads; equilibrium remains unaccepted")
args.out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(args.out,dpi=150)
