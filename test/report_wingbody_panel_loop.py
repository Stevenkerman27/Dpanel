"""Publish compact evidence and one figure from the completed research runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--base", type=Path, default=Path("runs/wingbody_panel_final"))
parser.add_argument("--out", type=Path, default=Path("docs/wingbody_panel_evidence"))
args = parser.parse_args()
args.out = args.out.resolve()
args.out.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(args.base.resolve() / "mpl")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

run_paths = [args.base, Path("runs/wingbody_panel_alpha0"), Path("runs/wingbody_panel_alpha_minus5"),
             Path("runs/wingbody_panel_thick_wide"), Path("runs/wingbody_panel_thin_narrow"),
             Path("runs/wingbody_panel_refined")]
records = []
for path in run_paths:
    result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    audit_path = path / "independent_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else None
    records.append(dict(run=str(path), result=result, independent_audit=audit,
                        fields_sha256=hashlib.sha256((path / "fields.npz").read_bytes()).hexdigest()))
base, audit = records[0]["result"], records[0]["independent_audit"]
assert base["derivatives_passed"] and audit["stencil_reports"][-1]["rank_deficient_panels"] == 0
assert all(record["result"]["raw_cp_equal_used_cp"] for record in records)
assert all(max(record["result"]["load_closure"].values()) < 1e-10 for record in records)
fields = np.load(args.base / "fields.npz")
failed = np.load("runs/wingbody_panel_upstream/fields.npz")
failure_comparison = dict(
    run="runs/wingbody_panel_upstream",
    nonfinite_cp_panels=int(np.count_nonzero(~np.isfinite(failed["Cp_raw"]))),
    unchanged_global_fields_max_abs={
        name: float(np.max(abs(fields[name] - failed[name])))
        for name in ("AIC_mu", "AIC_sigma", "mu", "sigma")
    },
)
assert max(failure_comparison["unchanged_global_fields_max_abs"].values()) == 0
summary = dict(
    status="first_fixed_topology_wingbody_loop_complete",
    records=records, failure_comparison=failure_comparison,
    scalar_derivative_comparisons=sum(len(row["ad"]) for row in base["geometry_and_angle_derivatives"]),
    physical_accuracy_accepted=False,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
)
(args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
(args.out / "backend_source_sha256.json").write_bytes((args.base / "backend_source_sha256.json").read_bytes())

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
figure = plt.figure(figsize=(13, 8), layout="constrained")
axis = figure.add_subplot(2, 2, 1, projection="3d")
xyz, faces, cp = fields["coordinates"], fields["faces"], fields["Cp_raw"].ravel()
surface = Poly3DCollection(xyz[faces], cmap="coolwarm", edgecolors=(0, 0, 0, .12), linewidths=.15)
surface.set_array(cp)
surface.set_clim(cp.min(), cp.max())
axis.add_collection3d(surface)
axis.auto_scale_xyz(xyz[:, 0], xyz[:, 1], xyz[:, 2])
axis.set_box_aspect(np.ptp(xyz, axis=0))
axis.view_init(elev=27, azim=-48)
axis.set_zticks([])
axis.tick_params(labelsize=8)
axis.set(xlabel="x", ylabel="y", title="CAD union, 696 panels: raw Cp, alpha=5 deg")
figure.colorbar(surface, ax=axis, shrink=.6, pad=.08, label="Cp (no cutoff)")

axis = figure.add_subplot(2, 2, 2)
for parameter, color in zip(("alpha_deg", "central_thickness", "span_factor"), ("#14807a", "#bd483c", "#4858a4")):
    rows = sorted([row for row in base["geometry_and_angle_derivatives"] if row["parameter"] == parameter],
                  key=lambda row: row["step"])
    axis.loglog([row["step"] for row in rows], [max(row["relative_error"]) for row in rows],
                "o-", label=parameter, color=color)
axis.set(xlabel="Centered difference step", ylabel="Max relative error across 5 objectives",
         title="45 scalar AD / finite-difference comparisons")
axis.grid(alpha=.2)
axis.legend(frameon=False)

axis = figure.add_subplot(2, 2, 3)
values = [base["components"]["wing"]["force"][2], base["components"]["body"]["force"][2],
          base["force"][2], base["objective_values"]["seam_Fz"]]
axis.bar(["Wing", "Body", "Total", "Seam subset"], values,
         color=["#14807a", "#bd483c", "#59646a", "#bd9c3e"], width=.6)
for i, value in enumerate(values):
    axis.text(i, value + .025, f"{value:.5f}", ha="center", fontsize=10)
axis.set(ylabel="Fz (q=1, Sref=4)", title="One pressure integration; seam is included in wing/body",
         ylim=(0, max(values) * 1.18))
axis.grid(axis="y", alpha=.15)

axis = figure.add_subplot(2, 2, 4)
labels = ["Base", "alpha=0", "alpha=-5", "b=+.2\ns=1.1", "b=-.2\ns=.9", "2784\npanels"]
axis.bar(labels, [record["result"]["CL"] for record in records],
         color=["#14807a", "#59646a", "#4858a4", "#bd483c", "#bd9c3e", "#59646a"], width=.6)
axis.axhline(0, color="#555555", linewidth=.7)
axis.set(ylabel="CL (fixed Sref=4)", title="Geometry / angle checks; mesh convergence pending")
axis.grid(axis="y", alpha=.15)
figure.savefig(args.out / "wingbody_loop.png", dpi=180)
plt.close(figure)
print(json.dumps(dict(report=str(args.out), comparisons=summary["scalar_derivative_comparisons"],
                      baseline_CL=base["CL"], upstream_nonfinite_cp=failure_comparison["nonfinite_cp_panels"]),
                 indent=2), flush=True)
