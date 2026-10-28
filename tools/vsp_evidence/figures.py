"""Figures for the DPanel vs VSPAERO trapezoid-wing convergence study."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import tri as mtri
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import NullFormatter, ScalarFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "runs" / "vsp_conv"
FIG = WORK / "figs"
FIG.mkdir(parents=True, exist_ok=True)
DPCOL = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08306b"]
VLMCOL = ["#fee6ce", "#fdae6b", "#fd8d3c", "#a63603"]
DPLEV = ["0.20 (1004)", "0.14 (1822)", "0.10 (3382)", "0.07 (6562)",
         "0.05 (12290)"]
VLMLEV = ["l1 (240)", "l2 (854)", "l3 (3120)", "l4 (12480)"]
X_QC = 0.038888888888888885   # MAC quarter chord, DPanel moment reference
CREF_DP = 0.15555555555555556
CREF_VLM = 0.15


def load():
    summary = json.loads((WORK / "summary.json").read_text(encoding="utf-8"))
    span = np.load(WORK / "span_loading.npz")
    state = np.load(WORK / "dp_s005" / "state.npz")
    surface = json.loads((WORK / "dp_s005" / "mesh" / "surface.json")
                         .read_text(encoding="utf-8"))
    tec = np.load(WORK / "vlm_l4" / "tec" / "span.npz")
    return summary, span, state, surface, tec


def int_axis(ax, ticks):
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(t)) for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(ScalarFormatter())


def fig_convergence(summary):
    dp, vlm = summary["dpanel"], summary["vspaero"]
    n_dp = np.asarray([row["walls"] for row in dp])
    n_vlm = np.asarray([row["panels"] for row in vlm])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))

    ax = axes[0]
    cl_dp = np.asarray([row["cl5"] for row in dp])
    cl_vlm = np.asarray([row["cl5"] for row in vlm])
    ax.semilogx(n_dp, cl_dp, "o-", color=DPCOL[-1], label="DPanel half model (thick)")
    ax.semilogx(n_vlm, cl_vlm, "s--", color=VLMCOL[-1], label="VSPAERO VLM (thin)")
    ax.axhspan(cl_dp[-1] * 0.95, cl_dp[-1] * 1.05, color=DPCOL[-1], alpha=0.10)
    ax.axhspan(cl_vlm[-1] * 0.95, cl_vlm[-1] * 1.05, color=VLMCOL[-1], alpha=0.10)
    ax.set_xlabel("surface panels")
    ax.set_ylabel("CL at 5 deg")
    ax.set_title("lift (bands = 5% of own finest)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    int_axis(ax, [200, 500, 1000, 2000, 5000])

    ax = axes[1]
    ax.semilogx(n_dp, [row["cdt5"] for row in dp], "o-", color=DPCOL[-1],
                label="DPanel Trefftz CD")
    ax.semilogx(n_dp, [row["cdp5"] for row in dp], "^:", color=DPCOL[1],
                label="DPanel integrated Cp drag")
    ax.semilogx(n_vlm, [row["cdi5"] for row in vlm], "s--", color=VLMCOL[-1],
                label="VSPAERO CDi")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("surface panels")
    ax.set_ylabel("CD at 5 deg")
    ax.set_title("drag")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    int_axis(ax, [200, 500, 1000, 2000, 5000])

    ax = axes[2]
    ax.semilogx(n_dp, 100 * np.asarray([row["cl5_rel_to_finest"] for row in dp]),
                "o-", color=DPCOL[-1], label="DPanel CL")
    ax.semilogx(n_vlm, 100 * np.asarray([row["cl5_rel_to_finest"] for row in vlm]),
                "s--", color=VLMCOL[-1], label="VSPAERO CL")
    ax.semilogx(n_dp, 100 * np.asarray([row["cdt5_rel_to_finest"] for row in dp]),
                "o:", color="#41ab5d", label="DPanel Trefftz CD")
    ax.semilogx(n_vlm, 100 * np.asarray([row["cdi5_rel_to_finest"] for row in vlm]),
                "s:", color="#a63603", label="VSPAERO CDi")
    ax.axhspan(-5, 5, color="gray", alpha=0.15)
    ax.set_xlabel("surface panels")
    ax.set_ylabel("deviation from own finest [%]")
    ax.set_title("deviation (shaded = 5% band)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    int_axis(ax, [200, 500, 1000, 2000, 5000])
    fig.suptitle("Trapezoid wing (NACA 2412, taper 0.5, b = 2.0, Sref = 0.30):"
                 " DPanel vs VSPAERO mesh convergence at alpha = 5 deg", fontsize=10)
    fig.tight_layout()
    path = FIG / "fig16_vsp_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_span(span):
    dp_y, dp_cl, dp_g = span["dp_y"], span["dp_cl"], span["dp_gamma"]
    vlm_y, vlm_cl, vlm_g = span["vlm_y"], span["vlm_cl"], span["vlm_gamma"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    ax = axes[0]
    for i in range(len(dp_y)):
        ax.plot(dp_y[i], dp_cl[i], "-", color=DPCOL[i], lw=1.9,
                label="DPanel " + DPLEV[i])
    for i in range(len(vlm_y)):
        ax.plot(vlm_y[i], vlm_cl[i], "--", color=VLMCOL[i], lw=1.4,
                label="VSPAERO " + VLMLEV[i])
    ax.set_xlabel("y / (b/2)")
    ax.set_ylabel("section cl at 5 deg")
    ax.set_title("span loading")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    ref_y, ref_cl = np.asarray(dp_y[-1]), np.asarray(dp_cl[-1])
    keep = ref_cl > 0.3 * ref_cl.max()
    y_keep_max = float(ref_y[keep][-1])
    for i in range(len(dp_y)):
        yy = np.asarray(dp_y[i])
        base = np.interp(yy, ref_y[keep], ref_cl[keep])
        dev = 100 * (np.asarray(dp_cl[i]) / base - 1.0)
        dev = np.where(yy <= y_keep_max + 1e-9, dev, np.nan)
        ax.plot(yy, dev, "o-", ms=3, color=DPCOL[i], label="DPanel " + DPLEV[i])
    ref_vy, ref_vc = np.asarray(vlm_y[-1]), np.asarray(vlm_cl[-1])
    mask = np.isfinite(ref_vc)
    for i in range(len(vlm_y)):
        yy = np.asarray(vlm_y[i])
        cc = np.asarray(vlm_cl[i])
        good = np.isfinite(cc) & (yy <= 0.95)
        base = np.interp(yy[good], ref_vy[mask], ref_vc[mask])
        ax.plot(yy[good], 100 * (cc[good] / base - 1.0), "s--", ms=2.5,
                color=VLMCOL[i], label="VSPAERO " + VLMLEV[i])
    ax.axhspan(-5, 5, color="gray", alpha=0.15)
    ax.set_xlim(0, 1)
    ax.set_ylim(-12, 12)
    ax.set_xlabel("y / (b/2)")
    ax.set_ylabel("cl deviation from own finest [%]")
    ax.set_title("where refinement acts")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[2]
    for i in range(len(dp_y)):
        yy = np.asarray(dp_y[i])
        gg = np.asarray(dp_g[i])
        ax.plot(yy, -np.gradient(gg, yy), "-", color=DPCOL[i], lw=1.9,
                label="DPanel " + DPLEV[i])
    for i in range(len(vlm_y)):
        yy = np.asarray(vlm_y[i])
        gg = np.asarray(vlm_g[i])
        good = np.isfinite(gg)
        ax.plot(yy[good], -np.gradient(gg[good], yy[good]), "--",
                color=VLMCOL[i], lw=1.4, label="VSPAERO " + VLMLEV[i])
    ax.set_xlabel("y / (b/2)")
    ax.set_ylabel("-dGamma/dy  [m/s]")
    ax.set_title("shed vorticity along the span")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 4.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.suptitle("Span loading and implied wake at 5 deg", fontsize=11)
    fig.tight_layout()
    path = FIG / "fig17_vsp_span_loading.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_alpha(summary):
    dp, vlm = summary["dpanel"], summary["vspaero"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    ax = axes[0]
    for i, row in enumerate(dp):
        ax.plot(row["alpha"], row["cl_curve"], "o-", ms=3.5, color=DPCOL[i],
                label="DPanel " + DPLEV[i])
    for i, row in enumerate(vlm):
        ax.plot(row["alpha"], row["cl_curve"], "s--", ms=3.5, color=VLMCOL[i],
                label="VSPAERO " + VLMLEV[i])
    ax.annotate("dCL/dalpha, central at 5 deg\nDPanel %s: %.4f /deg\n"
                "VSPAERO %s: %.4f /deg"
                % (DPLEV[-1].split()[0], dp[-1]["dclda_central"],
                   VLMLEV[-1].split()[0], vlm[-1]["dclda_central"]),
                xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.85, lw=0.5))
    ax.set_xlabel("alpha [deg]")
    ax.set_ylabel("CL")
    ax.set_title("lift curve")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    for i, row in enumerate(dp):
        ax.plot(row["alpha"], row["cmy_curve"], "o-", ms=3.5, color=DPCOL[i],
                label="DPanel " + DPLEV[i])
    for i, row in enumerate(vlm):
        shifted = np.asarray(row["cmy_curve"]) + np.asarray(row["cl_curve"]) \
            * (X_QC / CREF_VLM)
        ax.plot(row["alpha"], shifted, "s--", ms=3.5, color=VLMCOL[i],
                label="VSPAERO " + VLMLEV[i])
    ax.set_xlabel("alpha [deg]")
    ax.set_ylabel("CM at MAC quarter chord")
    ax.set_title("pitching moment (common reference)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.suptitle("Alpha sweep; VSPAERO moments (about X_cg = 0) translated to"
                 " the MAC quarter chord with CMy + CL*x_qc/Cref", fontsize=9)
    fig.tight_layout()
    path = FIG / "fig18_vsp_alpha.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_wake(state, surface):
    x = np.asarray(surface["coordinates"], float)
    wall = np.asarray(surface["faces"], int)
    cp = np.clip(np.asarray(state["cp"]), -1.4, 0.7)
    wake = np.asarray(state["wake"])
    wake_g = np.asarray(state["wake_cl"])
    cmap = plt.get_cmap("coolwarm")

    fig = plt.figure(figsize=(13.2, 5.6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    wing_colors = cmap((cp + 1.4) / 2.1) * np.ones((1, 4))
    wing_colors[:, 3] = 1.0
    ax.add_collection3d(Poly3DCollection(x[wall], facecolors=wing_colors,
                                        linewidths=0))
    gmax = float(np.nanmax(wake_g))
    gcolors = plt.get_cmap("viridis")(0.15 + 0.75 * wake_g / gmax) * np.ones((1, 4))
    gcolors[:, 3] = 0.85
    ax.add_collection3d(Poly3DCollection(wake, facecolors=gcolors, linewidths=0))
    ax.set_xlabel("x", fontsize=8)
    ax.set_ylabel("y", fontsize=8)
    ax.set_zlabel("z", fontsize=8)
    ax.set_xlim(0.0, 0.7)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(-0.12, 0.12)
    ax.view_init(elev=24, azim=-124)
    ax.tick_params(labelsize=7)
    ax.set_title("wing (Cp colours) with the shed wake sheet", fontsize=9)

    ax = fig.add_subplot(1, 2, 2)
    yc = wake[:, :, 1].mean(axis=1)
    xc = wake[:, :, 0].mean(axis=1)
    for target, colour in ((0.21, "#08306b"), (0.40, "#3182bd"),
                           (0.70, "#9ecae1")):
        sel = np.abs(xc - target) < 0.012
        if not np.any(sel):
            continue
        order = np.argsort(yc[sel])
        ax.step(yc[sel][order], wake_g[sel][order], where="mid", color=colour,
                lw=1.6, label="x = %.2f" % target)
    ax.set_xlabel("y / (b/2)")
    ax.set_ylabel("Gamma on the wake sheet [m$^2$/s]")
    ax.set_title("Gamma is frozen downstream (no roll-up)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, title="wake station", title_fontsize=8)
    fig.suptitle("DPanel wake at alpha = 5 deg, %d wall panels: the sheet leaves"
                 " the trailing edge and carries Gamma(y)" % len(wall),
                 fontsize=10)
    fig.tight_layout()
    path = FIG / "fig19_vsp_wake.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_cp(state, surface, tec):
    from scipy.interpolate import griddata

    x = np.asarray(surface["coordinates"], float)
    centers = np.asarray(state["centers"])
    normals = np.asarray(state["normals"])
    cp = np.asarray(state["cp"])
    chord = 0.2 - 0.1 * centers[:, 1]
    xi = centers[:, 0] / chord
    upper = normals[:, 2] > 0.2
    lower = normals[:, 2] < -0.2

    verts = np.asarray(tec["vertices"], float)
    faces = np.asarray(tec["faces"], int)
    tri_v = verts[faces].mean(axis=1)
    cp_v = np.asarray(tec["cp"])
    chord_v = 0.2 - 0.1 * tri_v[:, 1]
    xi_v = tri_v[:, 0] / chord_v

    gx, gy = np.meshgrid(np.linspace(0.02, 0.98, 70), np.linspace(0.02, 0.98, 90))
    gu = griddata((xi[upper], centers[upper, 1]), cp[upper], (gx, gy), method="linear")
    gl = griddata((xi[lower], centers[lower, 1]), cp[lower], (gx, gy), method="linear")
    mean_cp = 0.5 * (gu + gl)
    sheet_cp = griddata((xi_v, tri_v[:, 1]), cp_v, (gx, gy), method="linear")
    print("planform mean Cp: DPanel (upper+lower)/2 %.4f, VSPAERO sheet %.4f"
          % (np.nanmean(mean_cp), np.nanmean(sheet_cp)))

    levels = np.linspace(-1.0, 0.4, 29)
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4))
    for ax, field, title in ((axes[0], mean_cp,
                              "DPanel, thick surface: (Cp_upper+Cp_lower)/2"),
                             (axes[1], sheet_cp, "VSPAERO VLM thin-sheet Cp")):
        artist = ax.contourf(gx, gy, field, levels=levels, cmap="coolwarm",
                             extend="both")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("x / c(y)")
        ax.set_ylabel("y / (b/2)")
        ax.set_title(title, fontsize=9)
        ax.set_aspect(0.7)
        fig.colorbar(artist, ax=ax, shrink=0.85, pad=0.02)
        ax.tick_params(labelsize=7)

    ax = axes[2]
    diff = mean_cp - sheet_cp
    artist = ax.contourf(gx, gy, diff, levels=np.linspace(-0.6, 0.6, 25),
                         cmap="PuOr", extend="both")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x / c(y)")
    ax.set_ylabel("y / (b/2)")
    ax.set_title("DPanel minus VSPAERO", fontsize=9)
    ax.set_aspect(0.7)
    fig.colorbar(artist, ax=ax, shrink=0.85, pad=0.02)
    ax.tick_params(labelsize=7)
    fig.suptitle("Mean-surface pressure coefficient at alpha = 5 deg, mapped to"
                 " x/c(y) so the unswept planform becomes a rectangle", fontsize=10)
    fig.tight_layout()
    path = FIG / "fig20_vsp_cp_maps.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    summary, span, state, surface, tec = load()
    print(fig_convergence(summary))
    print(fig_span(span))
    print(fig_alpha(summary))
    print(fig_wake(state, surface))
    print(fig_cp(state, surface, tec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())