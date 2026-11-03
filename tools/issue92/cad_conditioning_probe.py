"""Why the g1_wingbody_cad pressure derivative cannot be certified (Route B, slice S8).

Reads the pickled system that `python -m tools.issue92.s8_adjoint --stage prepare` writes and
answers two questions that section 17.4 of docs/ISSUE92_TORCH_PORT_MAP.md quotes:

  stationarity  how far the forward solve is from a stationary point of the quartic
                objective, for several iteration caps;
  branch        whether a +-1e-4 degree angle change lands on the same solution branch and
                whether the difference quotient is stable at that scale.

Only torch is needed; nothing here touches NumPy linear algebra after torch has run.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(path):
    import torch

    from panel.p1_implicit_adjoint import AdjointProblem
    from panel.p1_pressure_kutta_torch_driver import TorchPressureModel

    record = pickle.loads(Path(path).read_bytes())
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float64)
    operator = tensor(record["B"])
    areas = tensor(record["areas"])
    pairs = [(torch.as_tensor(a, dtype=torch.long), torch.as_tensor(b, dtype=torch.long))
             for a, b in record["pairs"]]
    model = TorchPressureModel(tensor(record["grad"]), pairs, int(operator.shape[1]))
    problem = AdjointProblem(
        wall_operator=operator, kutta_operator=tensor(record["kutta"]),
        centers=tensor(record["centers"]), weights=torch.sqrt(areas/areas.mean()),
        model=model, te_face_pairs=pairs, weight=5.0, max_nfev=400)
    return record, model, problem, torch


def _stationarity(record, model, problem, torch, caps):
    import panel.p1_pressure_kutta_torch_driver as driver

    alpha = torch.tensor(record["alpha_deg"], dtype=torch.float64)
    for cap in caps:
        solve = driver.solve_pressure_kutta_torch(
            wall_operator=problem.wall_operator, wall_rhs=problem.rhs(alpha),
            kutta_operator=problem.kutta_operator, grad=model.grad,
            te_face_pairs=problem.te_face_pairs, x0=problem.linear_state(alpha),
            weight=5.0, max_nfev=cap)
        state = solve.state
        stationarity = float(problem.stationarity(state, alpha))
        condition = float(torch.linalg.cond(problem.reduced_hessian(state)))
        print("cap %4d used %3d status %d stationarity %.3e cond %.3e pressure_rms %.3e"
              % (cap, solve.result.nfev, solve.result.status, stationarity, condition,
                 float(model(state).abs().max())))


def _branch(record, problem, torch, deltas):
    base = None
    for delta in deltas:
        angle = torch.tensor(record["alpha_deg"]+delta, dtype=torch.float64)
        state = problem.state(angle).detach()
        stationarity = float(problem.stationarity(state, angle))
        if base is None:
            base = state
        scale = max(float(state.abs().max()), 1e-30)
        print("alpha%+9.1e nfev %3d status %d stationarity %.3e distance %.4f"
              % (delta, problem.last_pressure_solve.result.nfev,
                 problem.last_pressure_solve.result.status, stationarity,
                 float((state-base).abs().max())/scale))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", choices=("stationarity", "branch"), default="stationarity")
    parser.add_argument("--artifact", type=Path,
                        default=ROOT/"runs/scratch/issue92_loose/s8/g1_wingbody_cad.pkl")
    args = parser.parse_args(argv)
    record, model, problem, torch = _load(args.artifact)
    if args.probe == "stationarity":
        _stationarity(record, model, problem, torch, (80, 200, 400, 800))
    else:
        _branch(record, problem, torch, (0.0, 1e-4, -1e-4, 1e-5, -1e-5, 1e-3, -1e-3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
