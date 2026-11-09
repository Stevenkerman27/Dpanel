"""Compare actual panel planforms with the independent lifting-line reference."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from panel.grass_mesh import build_grass_mesh, load_grass_wing
from panel.p1_lifting_g1 import solve
from panel.te_strip_audit import te_strip_diagnostics
from panel.trefftz_energy_exact import exact_trefftz_piecewise_linear
from tools.lifting_line import lifting_line


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case", choices=("rect", "taper50", "ellipse85", "all"), default="all")
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--span-panels-per-segment", type=int, default=4)
parser.add_argument("--chord-panels", type=int, default=6)
parser.add_argument("--wall", choices=("collocation", "galerkin"), default="galerkin")
parser.add_argument("--kutta", choices=("pressure", "potential_jump"), default="potential_jump")
parser.add_argument("--recovery", choices=("face", "area", "quadratic"), default="face")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
theta = np.linspace(0., .5 * np.pi * .97, 9)
y = 4. * np.sin(theta)
ellipse = 2. * np.cos(theta)
y0, c0 = 4. * .85, 2. * np.sqrt(1. - .85 ** 2)
ramp = y > y0
ellipse[ramp] = c0 + (.24 - c0) * (y[ramp] - y0) / (y[-1] - y0)
cases = dict(rect=np.full_like(y, 1.5), taper50=1.5 - .75 * y / y[-1], ellipse85=ellipse)
code = np.r_[np.full(8, .17), np.full(8, -.17), 0., 0.]
rows = []
for name, chord in cases.items():
    if args.case != "all" and args.case != name:
        continue
    geometry = dict(
        leading_edge_xyz=np.c_[np.zeros(len(y)), y, np.zeros(len(y))],
        chord=chord, twist_radians=np.zeros(len(y)), cst=np.tile(code, (len(y), 1)))
    data = build_grass_mesh(
        geometry, 1., chord_panels=args.chord_panels,
        span_panels_per_segment=args.span_panels_per_segment)
    wing = load_grass_wing(data)
    low = solve(wing, 0., kutta=args.kutta, wall_formulation=args.wall,
                pressure_recovery=args.recovery)
    high = solve(wing, 4., kutta=args.kutta, wall_formulation=args.wall,
                 pressure_recovery=args.recovery)
    gamma = np.r_[0., high["gamma"], 0.]
    drag = exact_trefftz_piecewise_linear(
        wing.gamma_y, gamma, U=1., density=1., reference_area=wing.reference_area)
    ar = np.ptp(wing.xyz[:, 1]) ** 2 / wing.reference_area
    efficiency = drag["CL_circulation"] ** 2 / (np.pi * ar * drag["CDi_ideal"])
    ref = lifting_line(np.r_[-y[::-1], y[1:]], np.r_[chord[::-1], chord[1:]],
                       alpha_deg=4., terms=127)
    slope = (high["CL"] - low["CL"]) / np.deg2rad(4.)
    mesh_audit = te_strip_diagnostics(wing)
    rows.append(dict(
        case=name, wall=args.wall, kutta=args.kutta, recovery=args.recovery,
        chord_panels=args.chord_panels, span_panels_per_segment=args.span_panels_per_segment,
        area=wing.reference_area,
        aspect_ratio=ar, faces=len(wing.faces), cl_pressure=high["CL"],
        cl_gamma=drag["CL_circulation"], cdi=drag["CDi_ideal"], e=efficiency,
        e_using_cl_pressure=high["CL"] ** 2 / (np.pi * ar * drag["CDi_ideal"]),
        ll_cl=ref["cl"], ll_cdi=ref["cdi"], ll_e=ref["e"], ll_slope=ref["cl_alpha"],
        panel_slope=slope, slope_gap=slope / ref["cl_alpha"] - 1.,
        drag_method=drag["method"], mesh_audit=dict(
            ok=mesh_audit["ok"], max_sweep_deg=mesh_audit["max_sweep_deg"],
            max_strip_aspect=mesh_audit["max_strip_aspect"]),
        physical_accuracy_accepted=False))
    np.savez_compressed(
        args.out / ("%s_gamma.npz" % name), y=wing.gamma_y, gamma=gamma,
        ll_y=ref["y"], ll_gamma=ref["gamma"])
    (args.out / "summary.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    print("%s: slope %.6f vs %.6f, e %.6f vs %.6f, CLp %.6f CLg %.6f"
          % (name, slope, ref["cl_alpha"], efficiency, ref["e"],
             high["CL"], drag["CL_circulation"]), flush=True)
