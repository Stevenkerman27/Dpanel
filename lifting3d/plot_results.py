"""Plot stored 3D convergence evidence and failed wake iterations without changing their acceptance."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
out = root/"results"/"overview_v1"
assert not out.exists(), "Preserve previous figures"
out.mkdir(parents=True)
sources = [
    root/"results"/"strips_sequence_v1"/"summary.json",
    root/"results"/"farfield_strips_v1"/"summary.json",
    root/"results"/"aspect_ratio_v1"/"summary.json",
    root/"results"/"graph_wake_v1"/"summary.json",
    root/"results"/"graph_wake_harmonic_v1"/"summary.json",
    root/"results"/"weak_kutta_v1"/"s24_a5"/"result.json",
    root/"results"/"weak_kutta_v1"/"s48_a5"/"result.json",
]
strip, far, aspect, wake, harmonic, weak24, weak48 = [json.loads(path.read_text()) for path in sources]
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.), constrained_layout=True)
axes[0, 0].plot([row["strips"] for row in strip["rows"]], [row["cl"] for row in strip["rows"]],
                "o-", color="#276F9C", label="Nodal Kutta")
axes[0, 0].plot([24, 48], [weak24["cl_pressure"], weak48["cl_pressure"]],
                "s-", color="#A0464F", label="Integral Kutta")
axes[0, 0].set(xlabel="Conforming wake strips", ylabel="Pressure CL", title="Strip discretization, AR4, R5")
axes[0, 0].legend()
for mode, marker, color in [("node", "o", "#276F9C"), ("weak", "s", "#A0464F")]:
    rows = [row for row in far["rows"] if row["kutta"] == mode]
    axes[0, 1].plot([row["radius"] for row in rows], [row["cl"] for row in rows], marker+"-", color=color, label=mode)
axes[0, 1].set(xlabel="Farfield radius / chord", ylabel="Pressure CL", title="Fixed outer mesh size, 48 strips")
axes[0, 1].legend()
axes[1, 0].plot([8, 20], [.422169241, .483828951], "^-", color="#6E6B50", label="Lifting line, a0=2pi")
axes[1, 0].plot([8, 20], [.452636769, .523634992], "v-", color="#277C64", label="Lifting line, 2D P2 slope")
aspect_rows = [case for entry in aspect["cases"] for case in entry["cases"]]
for mode, marker, color in [("node", "o", "#276F9C"), ("weak", "s", "#A0464F")]:
    rows = [row for row in aspect_rows if row["kutta"] == mode]
    assert len(rows) == 2, "Aspect-ratio study schema does not contain two cases per Kutta mode"
    axes[1, 0].plot([row["aspect_ratio"] for row in rows], [row["cl_pressure"] for row in rows],
                    marker+"-", color=color, label="P2 "+mode)
axes[1, 0].set(xlabel="Aspect ratio", ylabel="CL", title="Independent lower-order references, h=.2")
axes[1, 0].legend(fontsize=8)
for study, label, marker, color in [(wake, "Analytic mesh blend (failed)", "o", "#276F9C"),
                                   (harmonic, "Harmonic interior (failed)", "s", "#A0464F")]:
    rows = study["history"]
    axes[1, 1].plot([row["iteration"] for row in rows], [row["tangency_rms"] for row in rows],
                    marker+"-", color=color, label=label)
    assert not study["converged"]
axes[1, 1].axhline(.01, color="#277C64", linestyle="--", label="Registered tangency threshold")
axes[1, 1].set(xlabel="Wake geometry iteration", ylabel="Raw paired mean velocity normal RMS",
               title="Graph wake: unsuccessful iterations retained")
axes[1, 1].legend(fontsize=8)
for axis in axes.ravel():
    axis.grid(alpha=.2)
fig.savefig(out/"evidence.png", dpi=180)
manifest = dict(inputs={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                plot_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                note="Finite discretization checks and failed wake experiments; no overall physical acceptance")
(out/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
(out/"plot.py.snapshot").write_bytes(Path(__file__).read_bytes())
print(out/"evidence.png")
