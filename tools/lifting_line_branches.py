"""Check roots on the same conditional wall family, with multiple seeds and alpha sweeps.

python -m tools.lifting_line_branches --out runs/verification/lifting_line_fix/branches
"""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from panel.p1_galerkin_wall import assemble_galerkin_multi
from panel.p1_lifting_g1 import make_wing
from panel.p1_lifting_g2 import assemble_multi, combine_wings
from panel.p1_potential_jump_torch import affine_state, potential_jump_newton
from panel.p1_pressure_kutta_torch_driver import TorchPressureModel
from panel.p1_rank_aware_torch import null_space
from panel.p1_te_recovery import matched_te_velocity_maps


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--chords", nargs="+", type=int, default=[4, 8, 16])
parser.add_argument("--alphas", nargs="+", type=float, default=[-4., -2., 0., 2., 4.])
parser.add_argument("--span-panels", type=int, default=12)
parser.add_argument("--wall", choices=("collocation", "galerkin"), default="galerkin")
parser.add_argument("--recovery", choices=("face", "area", "quadratic"), default="area")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
started = time.perf_counter()
rows = []
metadata = dict(python=sys.executable, environment=platform.platform(), torch=torch.__version__,
                wall=args.wall, recovery=args.recovery, alphas=args.alphas,
                span_panels=args.span_panels, chords=args.chords,
                physical_accuracy_accepted=False)

for chord_panels in args.chords:
    wing = make_wing(chord_panels=chord_panels, span_panels=args.span_panels,
                     span=4., chord=1., thickness=.12, wake_length=10., wake_panels=2)
    mesh = combine_wings([wing])
    ops = assemble_galerkin_multi(mesh, static=None, order=3) if args.wall == "galerkin" \
        else assemble_multi(mesh)
    A, W, J, K, grad = (torch.as_tensor(ops[key], dtype=torch.float64)
                        for key in ("A", "W", "J", "K", "grad"))
    nm, ng = A.shape[1], W.shape[1]
    if args.wall == "galerkin":
        factor = torch.linalg.cholesky(torch.as_tensor(ops["M"], dtype=torch.float64))
        B = torch.linalg.solve_triangular(factor, torch.hstack((A, W)), upper=False)
        rhs_xyz = torch.linalg.solve_triangular(
            factor, torch.as_tensor(ops["b_xyz"], dtype=torch.float64), upper=False)
    else:
        weights = torch.sqrt(torch.as_tensor(ops["mesh"].areas) / np.mean(ops["mesh"].areas))
        B = weights[:, None] * torch.hstack((A, W))
        rhs_xyz = -weights[:, None] * torch.as_tensor(ops["mesh"].centers)
    # Geometry is fixed: build the affine map once, with zero flow.
    _, mapping = affine_state(B, torch.zeros(B.shape[0], dtype=B.dtype), J, nm, ng)
    mu_basis = null_space(J[:, :nm])
    base_xyz = mu_basis @ torch.linalg.lstsq(B[:, :nm] @ mu_basis, rhs_xyz).solution
    signed_basis = null_space(torch.vstack((J, K)), rcond=1e-11)
    signed_xyz = signed_basis @ torch.linalg.lstsq(B @ signed_basis, rhs_xyz).solution
    pairs = [pair for group in ops["te_face_pairs"] for pair in group]
    velocity_maps = None
    if args.recovery != "face":
        top, bottom, _ = matched_te_velocity_maps(mesh, method=args.recovery)
        velocity_maps = (top, bottom)
    pressure = TorchPressureModel(grad, pairs, nm + ng, velocity_maps=velocity_maps)
    roots = {}
    for seed in ("signed", "zero", "reverse", "large", "continuation_up", "continuation_down"):
        previous = None
        alphas = sorted(args.alphas, reverse=seed == "continuation_down")
        for alpha_deg in alphas:
            alpha = np.deg2rad(alpha_deg)
            flow = torch.tensor([np.cos(alpha), 0., np.sin(alpha)], dtype=torch.float64)
            base = base_xyz @ flow
            signed = (signed_xyz @ flow)[nm:]
            initial = signed.clone()
            if seed == "zero":
                initial = torch.zeros_like(signed)
            elif seed == "reverse":
                initial = -signed
            elif seed == "large":
                initial = 5. * signed
            elif previous is not None and seed.startswith("continuation"):
                initial = previous.clone()
            trace = {}
            gamma, state, p, evaluations, scale, relative, condition = potential_jump_newton(
                pressure, base, mapping, initial, nm, max_nfev=30, diagnostics=trace)
            converged = bool(torch.isfinite(p).all()) and relative <= 1e-10 \
                and trace["correction_relative"] <= 1e-10
            accepted = converged and condition is not None and condition <= 1e6
            top, bottom = pressure.paired_velocity(state)
            full = torch.cat((gamma.new_zeros(1), gamma, gamma.new_zeros(1)))
            active = gamma[gamma.abs() > max(float(gamma.abs().max()) * 1e-8, 1e-12)]
            roughness = float(torch.linalg.vector_norm(torch.diff(full, n=2))
                              / torch.linalg.vector_norm(full).clamp_min(1e-12))
            disagreement = None
            if accepted:
                if alpha_deg in roots:
                    disagreement = float((gamma - roots[alpha_deg]).abs().max())
                else:
                    roots[alpha_deg] = gamma.clone()
                previous = gamma.clone()
            row = dict(chord_panels=chord_panels, alpha_deg=alpha_deg, seed=seed,
                       converged=converged, accepted=accepted, evaluations=evaluations,
                       condition=condition, pressure_scale=scale,
                       jump_max=float((J @ state).abs().max()),
                       gamma=gamma.tolist(), gamma_sign_changes=int(
                           (active[1:] * active[:-1] < 0).sum()),
                       gamma_second_difference_relative=roughness,
                       gamma_delta_from_first_root=disagreement,
                       top_streamwise_min=float((top @ flow).min()),
                       bottom_streamwise_min=float((bottom @ flow).min()),
                       paired_dot_min=float((top * bottom).sum(dim=1).min()), **trace)
            # Singular/nonfinite runs remain evidence, represented by JSON null.
            for key in ("condition", "condition_scaled", "correction_relative",
                        "relative_residual", "pressure_scale"):
                if row[key] is not None and not np.isfinite(row[key]):
                    row[key] = None
            for item in row["history"]:
                for key, value in item.items():
                    if value is not None and not np.isfinite(value):
                        item[key] = None
            rows.append(row)
            (args.out / "summary.json").write_text(
                json.dumps(dict(metadata=metadata, rows=rows,
                                elapsed_seconds=time.perf_counter() - started),
                           indent=2, allow_nan=False), encoding="utf-8")
    selected = [row for row in rows if row["chord_panels"] == chord_panels]
    print("c%d: accepted %d/%d, max root delta %.3g, min paired dot %.3g"
          % (chord_panels, sum(row["accepted"] for row in selected), len(selected),
             max((row["gamma_delta_from_first_root"] or 0.) for row in selected),
             min(row["paired_dot_min"] for row in selected)), flush=True)
