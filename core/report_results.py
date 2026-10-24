"""Render the measured reference convergence and the failed lifting experiment."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
closed = json.loads((root/"results"/"closed_v1"/"summary.json").read_text())
wing = json.loads((root/"results"/"wing_diagnostics_v1"/"summary.json").read_text())
fig, axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for prefix, label, marker in [("sphere","Sphere","o"),("prolate_ellipsoid","Ellipsoid (3,1,1)","s")]:
    rows = [row for row in closed if row["case"] in [f"{prefix}_L{level}" for level in [2,3,4]]]
    axes[0,0].loglog([row["triangles"] for row in rows],[row["cp_rmse"] for row in rows],
                    marker=marker,label=label)
axes[0,0].axhline(.08,color="gray",ls=":",label="L4 gate")
axes[0,0].set(xlabel="Triangle count",ylabel="Area-weighted Cp RMSE",title="Closed bodies: analytic reference")
axes[0,0].legend()
surface = np.load(root/"results"/"closed_v1"/"sphere_L4"/"surface.npz")
cosine = surface["centers"][:,0]/np.linalg.norm(surface["centers"],axis=1)
axes[0,1].scatter(cosine,surface["cp"],s=5,alpha=.35,label="2048 flat triangles")
x = np.linspace(-1,1,300)
axes[0,1].plot(x,1.-2.25*(1.-x*x),color="black",label="Exact smooth sphere")
axes[0,1].set(xlabel="Surface direction dot U",ylabel="Cp",title="No pressure clipping or normal removal")
axes[0,1].legend()
rows = sorted([row for row in wing if row["case"].startswith("wing_t33_")
               and row["mass_mode"] == "constrain" and row["wake_length_cref"] == 50],
              key=lambda row:row["alpha"])
axes[1,0].plot([row["alpha"] for row in rows],[row["pressure_CL"] for row in rows],"o-",label="Pressure integral")
axes[1,0].plot([row["alpha"] for row in rows],[row["wake_CL"] for row in rows],"s--",label="Wake diagnostic")
axes[1,0].axhline(0,color="gray",lw=.7)
axes[1,0].set(xlabel="Alpha (deg)",ylabel="CL",title="Lifting prototype: REJECTED")
axes[1,0].legend()
rows = [row for row in wing if row["case"] in ["wing_t33_a5_w50_constrain","wing_t33_a5_w50_report"]]
for row, marker in zip(rows,["o","s"]):
    axes[1,1].loglog(abs(row["normalized_source_flux"][0]),row["collocation_rms"],marker,
                     ms=8,label=row["mass_mode"])
axes[1,1].axvline(1e-12,color="gray",ls=":")
axes[1,1].axhline(1e-6,color="gray",ls=":")
axes[1,1].set(xlabel="Absolute normalized source flux",ylabel="True normal velocity RMS / U",
              title="Conservation and no-penetration must both pass")
axes[1,1].legend()
for ax in axes.flat:
    ax.grid(True,alpha=.2)
fig.savefig(root/"results"/"validation.png",dpi=180)
fig.savefig(root/"results"/"validation.pdf")
text = ["# Measured Results", "", "## Closed Bodies", "",
        "| Case | N | Cp RMSE | Centroid RMS | Face-interior RMS | Passed |",
        "|---|---:|---:|---:|---:|---|"]
for row in closed:
    interior = f"{row['off_collocation_rms']:.4g}" if row["off_collocation_rms"] is not None else "-"
    text.append(f"| {row['case']} | {row['triangles']} | {row['cp_rmse']:.6g} | "
                f"{row['collocation_rms']:.3g} | {interior} | {row['passed']} |")
text += ["", "Cp reference: radial projection of the flat centroid onto the analytic smooth surface.",
         "", "## Rejected Lifting Prototype", "",
         "| Case | N | CL pressure | CL wake diagnostic | Normal RMS | Source flux | Kutta max |",
         "|---|---:|---:|---:|---:|---:|---:|"]
for row in wing:
    text.append(f"| {row['case']} | {row['triangles']} | {row['pressure_CL']:.6g} | "
                f"{row['wake_CL']:.6g} | {row['collocation_rms']:.4g} | "
                f"{row['normalized_source_flux'][0]:.4g} | {row['kutta_max']:.3g} |")
(root/"RESULTS.md").write_text("\n".join(text)+"\n",encoding="utf-8")
print(root/"RESULTS.md")
print(root/"results"/"validation.png")
