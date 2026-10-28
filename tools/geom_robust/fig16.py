"""fig16: upper-surface Cp on the four geometry-robustness cases (finest mesh)."""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import half_util as H

ORDER = ["swept30", "elliptic", "bwb", "conv"]
TITLE = {"swept30": "swept30 (30 deg LE sweep, NACA 2412)",
         "elliptic": "elliptic (21 cosine stations, NACA 0012)",
         "bwb": "bwb (t/c 0.22 at root, LE kink)",
         "conv": "conv (body + wing + T-tail + fin)"}
VMIN, VMAX = -1.5, 0.5
REFLECT = np.array([1.0, -1.0, 1.0])


def upper_only(nodes, faces, cp, normals):
    """Keep suction-side panels and mirror the right half for a full planform."""
    keep = normals[:, 2] > 0.02
    nodes = np.vstack((nodes, nodes * REFLECT))
    count = len(nodes) // 2
    faces = np.vstack((faces[keep], faces[keep][:, ::-1] + count))
    cp = np.concatenate((cp[keep], cp[keep]))
    return nodes, faces, cp


def main():
    summary = json.load(open(os.path.join(H.DATA, "geom_cases.json")))
    norm = Normalize(vmin=VMIN, vmax=VMAX, clip=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.4))
    for ax, name in zip(axes.ravel(), ORDER):
        data = H.load("geom_" + name)
        nodes, faces, cp = upper_only(data["nodes"], data["faces"], data["cp"],
                                      data["normals"])
        polys, vals = H.view(nodes, faces, cp, plane="xy")
        H.paint(ax, polys, vals, norm, edge="none")
        info = summary[name]
        ax.set_title(TITLE[name], fontsize=9.5, loc="left")
        ax.text(0.0, -0.14,
                "%d wall panels  CL %.4f  CD_p %.5f  Cp_min %.2f"
                % (info["panels"], info["CL"], info["CD"], info["cp_min"]),
                transform=ax.transAxes, fontsize=8.5, color="0.25")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("0.85")
    fig.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.13,
                        wspace=0.06, hspace=0.32)
    cax = fig.add_axes([0.28, 0.055, 0.46, 0.021])
    sm = ScalarMappable(norm=norm, cmap=H.CMAP)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("upper-surface Cp (colour clipped to [%.1f, %.1f]; true minima in the captions)"
                 % (VMIN, VMAX), fontsize=8.5)
    fig.suptitle(H.tr("四种几何的上表面压力（各自最细网格，右半解镜像成整机）",
                      "upper-surface Cp on four geometries (finest mesh, mirrored planform)"),
                 fontsize=10.5)
    H.save(fig, "fig16_geometry_matrix.png")


if __name__ == "__main__":
    main()
