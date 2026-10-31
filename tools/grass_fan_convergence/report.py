"""Render the GRASS-wing fan order/strips convergence report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COEFFS = [("CL", "CL"), ("CL_gamma", r"$C_{L,\Gamma}$"),
          ("CD_pressure", r"$C_{D,p}$"), ("CM_y", r"$C_{M,y}$")]


def load(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = {}
    for r in payload["records"]:
        records.setdefault(r["case"], {})[r["config"]] = r
    return payload, records


def fan_labels(payload):
    return [c for c in payload["configs"] if not c.startswith("analytic")]


def pretty(name):
    """Human label: drop the gNN_ index prefix and the (inaccurate) AR suffix."""
    parts = [p for p in name.split("_")[1:] if not p.startswith("ar")]
    return " ".join(parts) or name


def short(config):
    """Display label of the fan configuration (base order/strips)."""
    return config.split("/")[0].replace(":", "/")


def usable(record):
    """Finite loads: the linear wall solution is usable even if the Kutta polish stalled."""
    cl = record.get("CL")
    return cl is not None and np.isfinite(cl) and abs(cl) < 10.0


def clean(record):
    """The solve reported a normal termination (iteration limit / convergence)."""
    status = str(record.get("status") or "")
    return status.startswith("iteration_limit") or status.startswith("converged")


def valid(record):
    return usable(record)


def rejected(record):
    elig = record.get("eligibility") or {}
    return int(elig.get("rejected") or 0), int(elig.get("eligible") or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="docs/issue84/grass_fan_convergence.json")
    ap.add_argument("--doc", default="docs/issue84/grass_fan_convergence_tables.md")
    ap.add_argument("--figdir", default="docs/issue84/figures")
    ap.add_argument("--open-json", default="docs/issue84/grass_fan_convergence_open.json")
    args = ap.parse_args()

    payload, records = load(args.json)
    records2 = {}
    if args.open_json and Path(args.open_json).exists():
        _, records2 = load(args.open_json)
    fans = fan_labels(payload)
    names = sorted(records)
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    ref = fans[-1]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4), sharex=True)
    for ax, (key, label) in zip(axes.ravel(), COEFFS):
        for name in names:
            series = records[name]
            xs = [i for i, f in enumerate(fans) if f in series and valid(series[f])]
            ys = [series[fans[i]][key] for i in xs]
            if not xs:
                continue
            line, = ax.plot(xs, ys, "o-", lw=1.4, ms=4, label=pretty(name))
            if "analytic" in series and valid(series["analytic"]):
                ax.axhline(series["analytic"][key], color=line.get_color(), ls=":",
                           lw=1.0, alpha=0.85)
            if "analytic_open" in series and valid(series["analytic_open"]):
                ax.axhline(series["analytic_open"][key], color=line.get_color(), ls="--",
                           lw=0.9, alpha=0.65)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.set_xticks(range(len(fans)))
        ax.set_xticklabels([short(f) for f in fans])
    axes[0, 0].legend(fontsize=6.5, ncol=2, loc="best")
    for ax in axes[1]:
        ax.set_xlabel("fan order / strips (dotted: analytic gated, dashed: analytic open)")
    fig.suptitle("Overall coefficients vs projection-fan quadrature (GRASS wings, "
                 r"$\alpha=5^\circ$, 1 geometry update)")
    fig.tight_layout()
    fig.savefig(figdir / "fan_coefficients.png", dpi=160)
    plt.close(fig)

    # Relative deviation of the coarsest fan point against the reference: a bar per
    # geometry, one panel per coefficient, log scale with engineering thresholds.
    coarse = fans[0]
    fig, axes = plt.subplots(1, len(COEFFS), figsize=(16.0, 4.0))
    index = np.arange(len(names))
    for ax, (key, label) in zip(axes, COEFFS):
        ys = []
        for name in names:
            series = records[name]
            base = series[ref][key] if (ref in series and usable(series[ref])) else float("nan")
            rec = series.get(coarse)
            if rec and usable(rec) and np.isfinite(base) and abs(base) > 1e-12:
                ys.append(abs(rec[key] - base) / abs(base))
            else:
                ys.append(np.nan)
        ys = np.asarray(ys, float)
        ax.bar(index, np.where(np.isfinite(ys), ys, 0.0), color="C0", alpha=0.85)
        ax.set_yscale("log")
        ax.set_ylim(1e-8, 1e-1)
        ax.axhline(1e-3, color="0.4", ls="--", lw=0.8)
        ax.axhline(1e-2, color="0.4", ls=":", lw=0.8)
        ax.text(0.02, 1.35e-3, "1e-3", fontsize=6, color="0.3", transform=ax.get_yaxis_transform())
        ax.text(0.02, 1.35e-2, "1e-2", fontsize=6, color="0.3", transform=ax.get_yaxis_transform())
        ax.set_xticks(index)
        ax.set_xticklabels([pretty(n) for n in names], rotation=90, fontsize=6.5)
        ax.set_title("|" + label + "(%s) - %s| / |%s|" % (short(coarse), short(ref), short(ref)),
                     fontsize=8)
        ax.grid(alpha=0.3, axis="y", which="both")
        worst = np.nanmax(ys) if np.any(np.isfinite(ys)) else float("nan")
        ax.set_title(ax.get_title() + "  worst %.1e" % worst, fontsize=8)
    fig.suptitle("Coarsest projection-fan setting %s vs reference %s"
                 % (short(coarse), short(ref)), fontsize=9)
    fig.tight_layout()
    fig.savefig(figdir / "fan_relative_deviation.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name in names:
        series = records[name]
        xs = [i for i, f in enumerate(fans) if f in series and valid(series[f])]
        ys = [series[fans[i]]["seconds"] for i in xs]
        if xs:
            ax.semilogy(xs, ys, "o-", lw=1.3, ms=4, label=pretty(name))
    ax.set_xticks(range(len(fans)))
    ax.set_xticklabels([short(f) for f in fans])
    ax.set_ylabel("wall time per solve [s]")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(figdir / "fan_cost.png", dpi=160)
    plt.close(fig)

    rule = payload.get("refinement_rule", "")
    lines = ["# GRASS-plane wing set: fan order/strips convergence of overall coefficients",
             "",
             "Configs: " + ", ".join(payload["configs"]) + ".",
             "Each fan config carries its own refinement (" + rule + "); 'analytic' is the",
             "fan-independent edge kernel with the production gate, 'analytic_open' the same",
             "kernel with every wake node accepted.",
             "alpha = %.1f deg, iterations = %d, core radius = %.3f, order = %d."
             % (payload["alpha_deg"], payload["iterations"], payload["core_radius"],
                payload["order"]),
             "",
             "Generated by tools/grass_fan_convergence/report.py from " + args.json + ".",
             ""]
    for key, label in COEFFS:
        lines += ["## " + label, "",
                  "| geometry | " + " | ".join(short(f) for f in fans)
                  + " | analytic | analytic open | max rel. dev vs " + short(ref) + " |",
                  "| --- | " + " | ".join("---" for _ in fans) + " | --- | --- | --- |"]
        for name in names:
            series = records[name]
            cells = [("%+.6f%s" % (series[f][key],
                                     "" if clean(series[f]) else " (polish)"))
                     if (f in series and usable(series[f]))
                     else ("invalid" if f in series else "n/a") for f in fans]
            ac = ("%+.6f" % series["analytic"][key]
                  if ("analytic" in series and usable(series["analytic"]))
                  else ("invalid" if "analytic" in series else "n/a"))
            ao = ("%+.6f" % series["analytic_open"][key]
                  if ("analytic_open" in series and usable(series["analytic_open"]))
                  else ("invalid" if "analytic_open" in series else "n/a"))
            base = series[ref][key] if ref in series else float("nan")
            devs = [abs(series[f][key] - base) / abs(base) for f in fans
                    if f in series and usable(series[f]) and abs(base) > 1e-12]
            lines.append("| %s | %s | %s | %s | %s |" % (
                name, " | ".join(cells), ac, ao,
                "%.3e" % max(devs) if devs else "n/a"))
        lines.append("")

    lines += ["## Wake-node gate (rejected / eligible)", "",
              "| geometry | analytic | analytic open | "
              + " | ".join(short(f) for f in fans) + " |",
              "| --- | --- | --- | " + " | ".join("---" for _ in fans) + " |"]
    for name in names:
        series = records[name]
        cells = []
        for f in ["analytic", "analytic_open"]:
            if f in series:
                r, e = rejected(series[f])
                cells.append("%d / %d" % (r, e))
            else:
                cells.append("n/a")
        for f in fans:
            if f in series:
                r, e = rejected(series[f])
                cells.append("%d / %d" % (r, e))
            else:
                cells.append("n/a")
        lines.append("| %s | %s |" % (name, " | ".join(cells)))
    lines.append("")

    lines += ["## Cost", "",
              "| geometry | " + " | ".join(short(f) for f in fans) + " | analytic |",
              "| --- | " + " | ".join("---" for _ in fans) + " | --- |"]
    for name in names:
        series = records[name]
        lines.append("| %s | %s | %s |" % (
            name,
            " | ".join("%.1f s" % series[f]["seconds"] if f in series else "n/a"
                       for f in fans),
            "%.1f s" % series["analytic"]["seconds"] if "analytic" in series else "n/a"))
    lines.append("")
    if records2:
        first = fans[0]
        fig, ax = plt.subplots(figsize=(10.5, 4.6))
        offsets = [-0.30, -0.10, 0.10, 0.30]
        for i, name in enumerate(names):
            series = records[name]
            other = records2.get(name, {})
            other = {first: other[first]} if first in other else {
                first: other[first + "@open"]} if (first + "@open") in other else {}
            points = []
            if "analytic" in series and usable(series["analytic"]):
                points.append((series["analytic"]["CL"], offsets[0], "o"))
            if "analytic_open" in series and usable(series["analytic_open"]):
                points.append((series["analytic_open"]["CL"], offsets[1], "s"))
            if first in series and usable(series[first]):
                points.append((series[first]["CL"], offsets[2], "D"))
            if first in other and usable(other[first]):
                points.append((other[first]["CL"], offsets[3], "v"))
            for y, dx, marker in points:
                ax.plot(i + dx, y, marker, ms=6, color="C%d" % (i % 10))
            if len(points) >= 3:
                ax.vlines(i, min(p[0] for p in points), max(p[0] for p in points),
                          color="C%d" % (i % 10), lw=0.8, alpha=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([pretty(n) for n in names], fontsize=7)
        ax.set_ylabel("CL")
        ax.set_title("CL by kernel and gate: circle analytic gated, square analytic open, "
                     "diamond fan " + short(first) + " gated, triangle fan open", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figdir / "fan_gate_effect.png", dpi=160)
        plt.close(fig)

        lines += ["## Gate effect: same kernel, gate on/off", "",
                  "| geometry | analytic gated | analytic open | fan gated | fan open | "
                  "analytic gated/open | fan gated/open |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for name in names:
            series = records[name]
            other = records2.get(name, {})
            if first + "@open" in other:
                other = {first: other[first + "@open"]}
            def cell(rec, key="CL"):
                return "%+.6f" % rec[key] if rec and usable(rec) else "n/a"
            an = series.get("analytic")
            ao = series.get("analytic_open")
            fg = series.get(first)
            fo = other.get(first)
            ratio1 = (abs(an["CL"] / ao["CL"]) if (an and ao and usable(an) and usable(ao)
                      and abs(ao["CL"]) > 1e-12) else float("nan"))
            ratio2 = (abs(fg["CL"] / fo["CL"]) if (fg and fo and usable(fg) and usable(fo)
                      and abs(fo["CL"]) > 1e-12) else float("nan"))
            lines.append("| %s | %s | %s | %s | %s | %.2fx | %.3fx |" % (
                name, cell(an), cell(ao), cell(fg), cell(fo), ratio1, ratio2))
        lines.append("")
        lines.append("![gate effect](issue84/figures/fan_gate_effect.png)")
        lines.append("")

    lines += ["![coefficients](issue84/figures/fan_coefficients.png)", "",
              "![relative deviation](issue84/figures/fan_relative_deviation.png)", "",
              "![cost](issue84/figures/fan_cost.png)", ""]
    Path(args.doc).write_text("\n".join(lines), encoding="utf-8")
    print("wrote", args.doc)


if __name__ == "__main__":
    main()
