"""Fixed-circulation field convergence and paired lifting-line comparisons.

Run from the repository root with the project's Python environment:
    python -m tools.lifting_line_study --case fixed-gamma --out runs/verification/fields
"""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from panel import p1_lifting_g1 as g1
from panel import p1_lifting_g2 as g2
from panel.p1_galerkin_wall import assemble_galerkin_multi
from panel.p1_potential_jump_torch import affine_state
from panel.p1_pressure_kutta_torch_driver import TorchPressureModel
from panel.trefftz_energy_exact import exact_trefftz_piecewise_linear
from tools.lifting_line import lifting_line


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case", choices=("fixed-gamma", "closures"), required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--chords", nargs="+", type=int, default=[4, 8, 16, 32])
parser.add_argument("--span-panels", type=int, default=12)
parser.add_argument("--span", type=float, default=4.)
parser.add_argument("--thickness", type=float, default=.12)
parser.add_argument("--alpha", type=float, default=4.)
parser.add_argument("--recovery", choices=("face", "area", "quadratic"), default="face")
parser.add_argument("--walls", nargs="+", choices=("collocation", "galerkin"),
                    default=["collocation", "galerkin"])
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
started = time.perf_counter()
alpha = np.deg2rad(args.alpha)
flow = torch.tensor([np.cos(alpha), 0., np.sin(alpha)], dtype=torch.float64)
lift_axis = np.array([-np.sin(alpha), 0., np.cos(alpha)])
rows = []
reference = lifting_line(np.linspace(-args.span / 2., args.span / 2., 2001),
                         np.ones(2001), alpha_deg=args.alpha, terms=127)
metadata = dict(
    case=args.case, python=sys.executable, environment=platform.platform(),
    torch=torch.__version__, threads=torch.get_num_threads(),
    alpha_deg=args.alpha, span=args.span, thickness=args.thickness,
    span_panels=args.span_panels, chords=args.chords, walls=args.walls,
    recovery=args.recovery, fixed_gamma="0.22*sqrt(1-(2*y/span)^2)",
    probes_x=[.05, .2, .5, .8, .95],
    probes_y=.113 * args.span, ll_cl=reference["cl"], ll_e=reference["e"],
    physical_accuracy_accepted=False)

for chord_panels in args.chords:
    wing = g1.make_wing(
        chord_panels=chord_panels, span_panels=args.span_panels,
        thickness=args.thickness, span=args.span, chord=1.,
        wake_length=10., wake_panels=2)
    mesh = g2.combine_wings([wing])
    collocation = g2.assemble_multi(mesh)
    for wall in args.walls:
        ops = collocation if wall == "collocation" else assemble_galerkin_multi(
            mesh, static=None, order=3)
        A = torch.as_tensor(ops["A"], dtype=torch.float64)
        W = torch.as_tensor(ops["W"], dtype=torch.float64)
        J = torch.as_tensor(ops["J"], dtype=torch.float64)
        grad = torch.as_tensor(ops["grad"], dtype=torch.float64)
        ms = ops["mesh"]
        nm, ng = A.shape[1], W.shape[1]
        if wall == "galerkin":
            mass = torch.as_tensor(ops["M"], dtype=torch.float64)
            factor = torch.linalg.cholesky(mass)
            B = torch.linalg.solve_triangular(factor, torch.hstack((A, W)), upper=False)
            wall_rhs = torch.as_tensor(ops["b_xyz"], dtype=torch.float64) @ flow
            rhs = torch.linalg.solve_triangular(factor, wall_rhs[:, None], upper=False)[:, 0]
        else:
            weight = torch.sqrt(torch.as_tensor(ms.areas) / np.mean(ms.areas))
            B = weight[:, None] * torch.hstack((A, W))
            wall_rhs = -(torch.as_tensor(ms.centers) @ flow)
            rhs = weight * wall_rhs
        mu_base, mapping = affine_state(B, rhs, J, nm, ng)
        pairs = [pair for group in ops["te_face_pairs"] for pair in group]
        velocity_maps = None
        if args.recovery != "face":
            from panel.p1_te_recovery import matched_te_velocity_maps
            top, bottom, recovery_receipt = matched_te_velocity_maps(mesh, method=args.recovery)
            velocity_maps = (top, bottom)
        pressure = TorchPressureModel(grad, pairs, nm + ng, velocity_maps=velocity_maps)
        gamma_y = np.asarray(wing.gamma_y)
        gamma_fixed = torch.as_tensor(
            .22 * np.sqrt(np.clip(1. - (2. * gamma_y[1:-1] / args.span) ** 2, 0., None)),
            dtype=torch.float64)
        if args.case == "fixed-gamma":
            states = [("fixed-gamma", torch.cat((mu_base + mapping @ gamma_fixed,
                                                gamma_fixed)), None)]
        else:
            states = []
            for kutta in ("pressure", "potential_jump"):
                try:
                    solved = g2.solve_multi(
                        mesh, args.alpha, operators=ops, wall_formulation=wall, kutta=kutta,
                        pressure_recovery=args.recovery)
                    states.append((kutta, torch.cat((torch.as_tensor(solved["mu"]),
                                                    torch.as_tensor(solved["gamma"]))), solved))
                except RuntimeError as error:
                    rows.append(dict(chord_panels=chord_panels, wall=wall, kutta=kutta,
                                     error=str(error), success=False))
                    print("c%d %s %s refused: %s" % (chord_panels, wall, kutta, error), flush=True)
        for kutta, state, solved in states:
            mu, gamma = state[:nm], state[nm:]
            velocity = (grad @ mu).numpy()
            cp = 1. - np.sum(velocity ** 2, axis=1)
            force = -(cp * ms.areas)[:, None] * ms.normals
            cl = float(force.sum(axis=0) @ lift_axis / mesh.reference_area)
            projected = torch.cat((mu_base + mapping @ gamma, gamma))
            drag = exact_trefftz_piecewise_linear(
                gamma_y, np.r_[0., gamma.numpy(), 0.],
                U=1., density=1., reference_area=mesh.reference_area)
            ar = args.span ** 2 / mesh.reference_area
            efficiency = drag["CL_circulation"] ** 2 / (np.pi * ar * drag["CDi_ideal"])
            probes = []
            for bank in (1, -1):
                faces = np.flatnonzero(mesh.face_bank == bank)
                triangles = ms.triangles[faces]
                # Generated wings are extruded in y. Locate probes in physical
                # x/y, then interpolate z on the containing faceted surface.
                for x_probe in metadata["probes_x"]:
                    edge1 = triangles[:, 1, :2] - triangles[:, 0, :2]
                    edge2 = triangles[:, 2, :2] - triangles[:, 0, :2]
                    delta = np.array([x_probe, metadata["probes_y"]]) - triangles[:, 0, :2]
                    determinant = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
                    second = (delta[:, 0] * edge2[:, 1] - delta[:, 1] * edge2[:, 0]) / determinant
                    third = (edge1[:, 0] * delta[:, 1] - edge1[:, 1] * delta[:, 0]) / determinant
                    bary = np.c_[1. - second - third, second, third]
                    matches = np.flatnonzero(np.min(bary, axis=1) >= -1e-12)
                    if len(matches) == 0:
                        raise RuntimeError("Physical probe outside generated wing")
                    selected = int(matches[0])
                    face = int(faces[selected])
                    potential = float(bary[selected] @ mu.numpy()[mesh.dof_face[face]])
                    probes.append(dict(
                        bank=bank, x=x_probe, y=metadata["probes_y"],
                        z=float(bary[selected] @ triangles[selected, :, 2]),
                        potential=potential, speed=float(np.linalg.norm(velocity[face])),
                        cp=float(cp[face])))
            singular = torch.linalg.svdvals(pressure.jacobian(state)[:, :nm] @ mapping)
            row = dict(
                chord_panels=chord_panels, wall=wall, kutta=kutta, faces=len(mesh.faces),
                success=True if solved is None else bool(solved["nonlinear"]["success"]),
                cl_pressure=cl, cl_gamma=drag["CL_circulation"], cdi=drag["CDi_ideal"],
                e=efficiency, pressure_rms=float(torch.linalg.vector_norm(pressure(state)) / ng ** .5),
                jump_max=float((J @ state).abs().max()),
                gamma_fixed_error=float((gamma - gamma_fixed).abs().max()) if solved is None else None,
                wall_dual_rms=float(torch.linalg.vector_norm(B @ state - rhs) / len(rhs) ** .5),
                conditional_mu_departure=float(torch.linalg.vector_norm(mu - projected[:nm])
                                               / torch.linalg.vector_norm(mu)),
                projected_pressure_rms=float(torch.linalg.vector_norm(pressure(projected)) / ng ** .5),
                condition=float(singular[0] / singular[-1]), sigma_min=float(singular[-1]),
                sigma_max=float(singular[0]), probes=probes)
            previous = [r for r in rows if r.get("wall") == wall and r.get("kutta") == kutta
                        and "probes" in r]
            if previous:
                row["previous_chord_panels"] = previous[-1]["chord_panels"]
                row["potential_change_max"] = max(
                    abs(a["potential"] - b["potential"]) for a, b in zip(probes, previous[-1]["probes"]))
                row["speed_change_max"] = max(
                    abs(a["speed"] - b["speed"]) for a, b in zip(probes, previous[-1]["probes"]))
                row["cp_change_max"] = max(
                    abs(a["cp"] - b["cp"]) for a, b in zip(probes, previous[-1]["probes"]))
            rows.append(row)
            np.savez_compressed(
                args.out / ("%s_%s_c%d.npz" % (wall, kutta, chord_panels)),
                xyz=mesh.xyz, faces=mesh.faces, dof_face=mesh.dof_face,
                gamma_y=gamma_y, gamma=gamma.numpy(), mu=mu.numpy(), velocity=velocity, cp=cp)
            (args.out / "summary.json").write_text(
                json.dumps(dict(metadata=metadata, rows=rows,
                                elapsed_seconds=time.perf_counter() - started),
                           indent=2, allow_nan=False), encoding="utf-8")
            print("c%d %-11s %-14s CLp=%+.7f CLg=%+.7f e=%.5f jump=%.1e speed_change=%s"
                  % (chord_panels, wall, kutta, cl, drag["CL_circulation"], efficiency,
                     row["jump_max"], row.get("speed_change_max")), flush=True)
