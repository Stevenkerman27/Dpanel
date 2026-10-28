"""Plot the actual material-wake geometry and retained iteration residuals."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--pairs", type=Path, required=True)
parser.add_argument("--relaxed", type=Path, required=True)
parser.add_argument("--second-angle", type=Path, required=True)
parser.add_argument("--face-lsq", type=Path)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
assert not args.out.exists()
args.out.mkdir(parents=True)
pairs = json.loads((args.pairs / "summary.json").read_text())
relaxed = json.loads((args.relaxed / "summary.json").read_text())
second = json.loads((args.second_angle / "summary.json").read_text())
face = json.loads((args.face_lsq / "summary.json").read_text()) if args.face_lsq else None
fixture = json.loads((args.relaxed / "surface_input.json").read_text())
native = np.asarray(fixture["coordinates"])
wake_nodes = np.unique(fixture["wake_faces"])
mid = wake_nodes[abs(native[wake_nodes, 1]) < 1e-10]
mid = mid[np.argsort(native[mid, 0])]
hinge = wake_nodes[abs(native[wake_nodes, 0] - relaxed["far_hinge_x"]) < 1e-10]
hinge = hinge[np.argsort(native[hinge, 1])]
fixed = next(r for r in pairs["runs"] if r["case"] == "single_wing" and r["alpha"] == 5 and r["mode"] == "fixed")
straight = next(r for r in pairs["runs"] if r["case"] == "single_wing" and r["alpha"] == 5 and r["mode"] == "freestream")
fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
axes = axes.ravel()
curves = [
        ("Fixed, 5 deg", fixed["directory"], "#777777"),
        ("Freestream, 5 deg", straight["directory"], "#bc3e39"),
        ("Material iteration, 5 deg", relaxed["final_directory"], "#23794d"),
        ("Material iteration, 2.5 deg", second["final_directory"], "#266eac")]
if face:
    curves.append(("Face LSQ, 5 deg", face["final_directory"], "#a06a19"))
for label, directory, color in curves:
    data = np.load(Path(directory) / "fields.npz")
    points = data["coordinates"]
    near_mid = mid[native[mid, 0] <= relaxed["far_hinge_x"] + 1e-10]
    axes[0].plot(points[near_mid, 0], points[near_mid, 2], ".-", label=label, color=color)
    axes[1].plot(points[hinge, 1], points[hinge, 2], ".-", color=color)
axes[0].set(xlabel="x / chord", ylabel="z / chord", title="Material line at span label 0")
axes[0].legend(fontsize=8)
axes[1].set(xlabel="y / chord", ylabel="z / chord", title="Actual wake section at x / chord = 5")
for key, label, color in (("near", "Near-wake total normal RMS", "#23794d"),
                          ("far", "Far-wake total normal RMS", "#bc3e39")):
    axes[2].semilogy([r["iteration"] for r in relaxed["history"]],
                     [r["wake_tangency"][key]["total_normal_rms"] for r in relaxed["history"]],
                     ".-", label=label, color=color)
axes[2].axhline(relaxed["tolerance"], color="#777777", linestyle=":", label="Declared tolerance")
axes[2].set(xlabel="Material iteration", ylabel="Normal velocity / U", title="5 deg: tolerance remains unmet")
axes[2].legend(fontsize=8)
if face:
    axes[2].semilogy([r["iteration"] for r in face["history"]],
                     [r["wake_tangency"]["near"]["total_normal_rms"] for r in face["history"]],
                     ".-", color="#a06a19", label="Face LSQ near-wake")
    axes[2].legend(fontsize=8)
    axes[3].plot([r["iteration"] for r in face["history"]],
                 [r["wake_tangency"]["near"]["steady_pressure_jump_rms"] for r in face["history"]],
                 ".-", color="#a06a19", label="Face LSQ near-wake")
    axes[3].axhline(face["tolerance"], color="#777777", linestyle=":", label="Declared tolerance")
    axes[3].set(xlabel="Material iteration", ylabel="RMS of Cp jump", title="5 deg: pressure continuity does not improve")
    axes[3].legend(fontsize=8)
else:
    axes[3].set_visible(False)
for ax in axes:
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
fig.savefig(args.out / "wake_geometry_and_residual.png", dpi=180)
fig.savefig(args.out / "wake_geometry_and_residual.pdf")
plt.close(fig)
evidence = dict(pairs=pairs, relaxed=relaxed, second_angle=second, face_lsq=face,
                interpretation="Geometric and prescribed-wake derivative checks only; no physical accuracy acceptance")
(args.out / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
