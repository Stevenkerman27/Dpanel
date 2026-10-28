"""Dump raw Cp fields of the four geometry-robustness cases for figures."""
import json
import os
import sys

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

ROOT = r"D:\3D\Projects\panel"
sys.path.insert(0, os.path.join(ROOT, "runs", "pr34_git"))

from panel.half_native import solve_half_fields       # noqa: E402
from panel.run_half_native import load_half_surface   # noqa: E402

CASES = [("swept30", "swept30_s075"),
         ("elliptic", "elliptic_s075"),
         ("bwb", "bwb_s055"),
         ("conv", "conv_s075")]
DATA = os.path.join(ROOT, "runs", "viz", "data")


def main():
    os.makedirs(DATA, exist_ok=True)
    summary = {}
    for name, tag in CASES:
        path = os.path.join(ROOT, "runs", "geom_robust", "mesh", tag,
                            "surface.json")
        half = load_half_surface(path)
        out = jax.jit(solve_half_fields)(**half)
        nodes = np.asarray(half["vertices"])
        faces = np.asarray(half["topology"].faces)
        cp = np.asarray(out["Cp_raw_half"])
        np.savez(os.path.join(DATA, "geom_" + name + ".npz"),
                 nodes=nodes, faces=faces, cp=cp,
                 centers=np.asarray(out["panel_centers_half"]),
                 normals=np.asarray(out["panel_normals_half"]),
                 areas=np.asarray(out["panel_areas_half"]),
                 component_ids=np.asarray(half["topology"].component_ids))
        summary[name] = dict(tag=tag, panels=int(len(faces)),
                             CL=float(out["CL"]),
                             CD=float(out["CD_pressure"]),
                             CM_y=float(out["CM"][1]),
                             cp_min=float(cp.min()), cp_max=float(cp.max()))
        print(name, summary[name], flush=True)
    with open(os.path.join(DATA, "geom_cases.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print("OK")


if __name__ == "__main__":
    main()
