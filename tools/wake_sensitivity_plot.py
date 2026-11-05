"""Export the observed load and tangency histories as a standalone figure."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
figure, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex="col", constrained_layout=True)
colors = dict(fully_free="black", freeze_1="#0072b2", freeze_2="#009e73",
              freeze_3="#d55e00", smooth="#cc79a7")
for column, case in enumerate(("rect", "swept30", "elliptic")):
    folder = args.out/case
    baseline = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    initial = baseline["rows"][0]["CL"]
    data = {"fully_free": baseline["rows"]}
    for variant in ("freeze_1", "freeze_2", "freeze_3", "smooth"):
        path = folder/("e3_"+variant)/"history.json"
        if path.exists():
            data[variant] = json.loads(path.read_text(encoding="utf-8"))["rows"]
    for variant, rows in data.items():
        iteration = [r["iteration"] for r in rows]
        lift = [r["CL"]/initial for r in rows]
        slip = [max(v["normal_slip_rms"] for v in r["wakes"].values()) for r in rows]
        axes[0, column].plot(iteration, lift, label=variant, color=colors[variant], linewidth=1.25)
        axes[1, column].semilogy(iteration, slip, color=colors[variant], linewidth=1.25)
        invalid = [k for k, r in enumerate(rows) if (
            r["min_reference_signed_jacobian"] if variant == "fully_free"
            else r["geometry"]["min_reference_signed_jacobian"]) <= 0]
        if invalid:
            k = invalid[0]
            axes[0, column].scatter(iteration[k], lift[k], marker="x", color=colors[variant], s=34)
    axes[0, column].set_title(case)
    axes[0, column].legend(fontsize=7, frameon=False)
    axes[1, column].axhline(.01, color="#777777", linestyle="--", linewidth=.8)
    axes[1, column].set_xlabel("Wake update")
    axes[0, column].set_ylim(0., 1.1)
    for row in (0, 1):
        axes[row, column].grid(alpha=.18)
axes[0, 0].set_ylabel("CL / initial pressure-Kutta CL")
axes[1, 0].set_ylabel("Maximum group normal-slip RMS")
figure.suptitle("B3 diagnostic interventions: load retention is not tangency convergence", fontsize=12)
figure.supxlabel("x: first nonpositive reference Jacobian; dashed line: slip target 0.01; physical accuracy not accepted",
                   fontsize=8)
path = args.out/"root_motion_histories.png"
figure.savefig(path, dpi=160)
plt.close(figure)
print(path)
