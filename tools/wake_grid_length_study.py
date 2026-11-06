"""Joint wall/Gamma refinement study of a fixed-wake rectangular wing."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import torch

from panel.p1_lifting_g1 import make_wing
from panel.p1_lifting_g2 import assemble_multi, combine_wings, solve_multi
from panel.p1_rank_aware_torch import null_space
from tools.issue92.trefftz_full import trefftz_cd_full


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--levels", default="4x6,6x8,8x10,10x12,12x14,14x16")
parser.add_argument("--length", type=float, default=5.0)
parser.add_argument("--mode", choices=("grid",), default="grid")
parser.add_argument("--wall-method", choices=("collocation", "galerkin"), default="galerkin")
parser.add_argument("--order", type=int, default=3)
parser.add_argument("--condition", action="store_true")
args = parser.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)
args.out.parent.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
report = dict(command=[sys.executable, *sys.argv], platform=platform.platform(),
              python=sys.version, numpy=np.__version__, torch=torch.__version__,
              cuda=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              mode=args.mode, alpha_deg=4.0, kutta="pressure",
              wall_method=args.wall_method, quadrature_order=args.order if args.wall_method == "galerkin" else None,
              chord_spacing="cosine; clustered at both leading and trailing edges",
              span_spacing="uniform", wake_panels=2,
              comparison="Successive relative change <= 3% for CL, CL_gamma, CD_pressure, CD_trefftz and CM_y; if both absolute values < 1e-3 require absolute change <= 3e-5. Two successive transitions required. This is joint wall/Gamma refinement, not physical validation.",
              records=[])
args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
levels = [tuple(map(int, item.split("x"))) for item in args.levels.split(",")]
lengths = [args.length]
for nc, ns in levels:
    for length in lengths:
        started = time.perf_counter()
        row = dict(chord_panels=nc, span_panels=ns, length_c=length, status="running")
        report["records"].append(row)
        try:
            wing = make_wing(chord_panels=nc, span_panels=ns, wake_length=length, wake_panels=2)
            mesh = combine_wings([wing], reference_chord=1., reference_area=4.,
                                 reference_point=np.array([.25, 0., 0.]))
            row.update(wall_faces=len(wing.faces), wall_dof=wing.ndof_mu,
                       gamma_dof=wing.gamma_hat.shape[1], wake_faces=len(wing.wake_faces))
            if args.condition and args.wall_method != "collocation":
                raise ValueError("--condition currently diagnoses collocation only")
            if args.condition:
                ops = assemble_multi(mesh)
                weights = torch.as_tensor(np.sqrt(ops["mesh"].areas / ops["mesh"].areas.mean()), dtype=torch.float64)
                wall = weights[:, None] * torch.as_tensor(np.hstack((ops["A"], ops["W"])), dtype=torch.float64)
                jump = torch.as_tensor(ops["J"], dtype=torch.float64)
                signed = torch.as_tensor(ops["K"], dtype=torch.float64)
                for label, constraint in (("jump", jump), ("jump_signed", torch.vstack((jump, signed)))):
                    basis = null_space(constraint, rcond=1e-11, device="cpu")
                    singular = torch.linalg.svdvals(wall @ basis)
                    row[label+"_wall_singular_max"] = float(singular[0])
                    row[label+"_wall_singular_min"] = float(singular[-1])
                    row[label+"_wall_condition"] = float(singular[0] / singular[-1])
            if args.wall_method == "galerkin":
                from panel.p1_galerkin_solver import solve_galerkin_multi
                state = solve_galerkin_multi(mesh, 4., kutta="pressure", order=args.order)
            else:
                state = solve_multi(mesh, 4., kutta="pressure",wall_formulation="collocation")
            cd, _ = trefftz_cd_full(np.asarray(wing.gamma_y), np.r_[0., state["gamma"], 0.], area=4.)
            row.update(status="completed", CL=float(state["CL"]), CL_gamma=float(state["CL_gamma"]),
                       CD_pressure=float(state["CD_pressure"]), CD_trefftz=float(cd),
                       CM_y=float(state["CM_y"]), wall_rms=float(state["wall_rms"]),
                       jump_rms=float(state["jump_rms"]),
                       pressure_kutta_rms=float(state["pressure_kutta_rms"]),
                       linear_singular_min=float(state["linear_singular_min"]),
                       wall_dual_rms=state.get("wall_dual_rms"),
                       solution_status=state.get("solution_status"),
                       nonlinear=state.get("nonlinear"))
        except Exception:
            row.update(status="failed", traceback=traceback.format_exc())
        row["seconds"] = time.perf_counter()-started
        args.out.write_text(json.dumps(report, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x.item()), encoding="utf-8")
        print(json.dumps({k: row.get(k) for k in ("chord_panels", "span_panels", "length_c", "status", "solution_status", "CL", "CL_gamma", "CD_pressure", "CD_trefftz", "CM_y", "seconds")}), flush=True)
