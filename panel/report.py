"""Create compact evidence and an unclipped pressure plot from retained runs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

parser = argparse.ArgumentParser()
parser.add_argument("--matrix", type=Path, required=True)
parser.add_argument("--topology", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out = args.out.resolve()
assert not args.out.exists(), "Preserve previous evidence"
args.out.mkdir(parents=True)
matrix = json.loads((args.matrix / "summary.json").read_text())
topology = json.loads((args.topology / "summary.json").read_text())
records = []
for run in matrix["runs"]:
    result = run["result"]
    name = run["name"]
    folder = args.matrix / name
    audit_path = folder / "independent_audit.json"
    records.append(dict(name=name, CL=result["CL"], Cp_min=result["cp_min"], Cp_max=result["cp_max"],
                        wall_panels=result["mesh"]["wall_panels"], wake_panels=result["mesh"]["wake_panels"],
                        warm_seconds=result["timings"]["warm_forward_seconds"],
                        residual=result["linear_true_relative_residual"],
                        condition_1=result["condition_1_estimate"],
                        source_sha256=result["source_sha256"], surface_sha256=result["surface_sha256"],
                        fields_sha256=hashlib.sha256((folder / "fields.npz").read_bytes()).hexdigest(),
                        derivatives_passed=result.get("derivatives_passed"),
                        independent_audit=json.loads(audit_path.read_text()) if audit_path.exists() else None))
evidence = dict(status="paused_before_general_acceptance",
                matrix_checks=matrix["checks"], topology_checks=topology["checks"],
                derivative_scalar_comparisons=sum(row.get("scalar_comparisons", 0) for row in matrix["checks"]),
                runs=records, rejections=matrix["rejections"],
                physical_accuracy_accepted=False,
                interpretation="Geometry gates and discrete derivatives passed; independent physical errors and mesh sensitivity remain.")
(args.out / "summary.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
fig = plt.figure(figsize=(13, 5.4))
for column, name, title in [(1, "thick_body", "Thick NACA4412 / body"),
                            (2, "two_bodies_coarse", "Two bodies crossing the trailing edge")]:
    data = np.load(args.matrix / name / "fields.npz")
    triangles = data["coordinates"][data["faces"]]
    cp = data["Cp_raw"]
    ax = fig.add_subplot(1, 2, column, projection="3d")
    norm = plt.Normalize(cp.min(), cp.max())
    surface = Poly3DCollection(triangles, facecolors=plt.cm.viridis(norm(cp)), edgecolors=(0, 0, 0, .2), linewidths=.2)
    ax.add_collection3d(surface)
    nodes = data["coordinates"][np.unique(data["faces"])]
    for k, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        lo, hi = nodes[:, k].min(), nodes[:, k].max()
        setter(lo - .02 * (hi - lo), hi + .02 * (hi - lo))
    ax.set_box_aspect(np.ptp(nodes, axis=0))
    ax.view_init(elev=27, azim=-125)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(4))
    ax.set(xlabel="x / Cref", ylabel="y / Cref", zlabel="z / Cref", title=title)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, shrink=.55, pad=.08, label="Raw reconstructed Cp")
fig.suptitle("Explicit fluid-cut panel solver: geometry accepted; physical accuracy unaccepted")
fig.subplots_adjust(left=.07, right=.97, bottom=.12, top=.86, wspace=.28)
fig.savefig(args.out / "panel_geometry.png", dpi=180, bbox_inches="tight", pad_inches=.2)
plt.close(fig)
print(json.dumps(dict(matrix_checks=len(matrix["checks"]), topology_checks=len(topology["checks"]),
                      derivatives=evidence["derivative_scalar_comparisons"],
                      warm_seconds_range=[min(row["warm_seconds"] for row in records), max(row["warm_seconds"] for row in records)]),
                 indent=2), flush=True)
