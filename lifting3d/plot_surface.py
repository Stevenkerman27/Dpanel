"""Render saved curved-surface pressure and raw wall errors without modifying the field."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
assert not args.out.exists(), "Preserve previous figures"
field = np.load(args.run/"field.npz", allow_pickle=False)
result = json.loads((args.run/"result.json").read_text())
wall = np.isin(field["tags"], [10, 20, 30])
faces = field["faces"][wall]
coordinates = field["points"][faces]
weight, cp = field["surface_weight"][wall], field["cp"][wall]
vn = field["surface_normal_velocity"][wall]
assert np.isfinite(cp).all() and np.isfinite(vn).all()
face_area = weight.sum(axis=1)
cp_mean = np.sum(cp*weight, axis=1)/face_area
vn_rms = np.sqrt(np.sum(vn**2*weight, axis=1)/face_area)
subtriangles = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]])
triangles = coordinates[:, subtriangles].reshape(-1, 3, 3)
limits = np.array([coordinates.reshape(-1, 3).min(axis=0), coordinates.reshape(-1, 3).max(axis=0)])
fig = plt.figure(figsize=(14, 6), constrained_layout=True)
for index, (values, limits_color, cmap, title, label) in enumerate([
    (cp_mean, [float(cp.min()), float(cp.max())], "coolwarm", "Raw pressure on wing and fuselage", "Area-mean Cp"),
    (vn_rms, [0., float(vn_rms.max())], "viridis", "Raw wall-normal velocity error", "Face RMS |u dot n| / Uinf"),
]):
    axis = fig.add_subplot(1, 2, index+1, projection="3d")
    norm = (TwoSlopeNorm(vmin=limits_color[0], vcenter=0., vmax=limits_color[1])
            if index == 0 and limits_color[0] < 0. < limits_color[1] else Normalize(*limits_color))
    palette = plt.get_cmap(cmap)
    surface = Poly3DCollection(triangles, facecolors=palette(norm(np.repeat(values, 4))),
                               edgecolor="none", zsort="average", rasterized=True)
    axis.add_collection3d(surface)
    axis.set(xlim=limits[:, 0], ylim=limits[:, 1], zlim=limits[:, 2],
             xlabel="x", ylabel="y", zlabel="z", title=title)
    axis.set_box_aspect(np.diff(limits, axis=0)[0])
    axis.set_zticks([limits[0, 2], .5*(limits[0, 2]+limits[1, 2]), limits[1, 2]])
    axis.zaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.view_init(elev=27., azim=-57.)
    axis.grid(False)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=palette), ax=axis,
                 shrink=.55, pad=.09, label=label)
fig.suptitle(f"M={result['mach']:g}, alpha={result['alpha_deg']:g} deg; "
             f"CL={result['cl_pressure']:.5f}; wall RMS={result['wall_normal_rms']:.5f}\n"
             f"Raw Cp range [{cp.min():.3f}, {cp.max():.3f}]; raw |u dot n| max={abs(vn).max():.3f}",
             fontsize=12)
args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=170)
plt.close(fig)
print(args.out)
