"""S8: implicit adjoint of the constrained solve, against central differences.

Two processes, because the reference solve and torch cannot share one on this host:

* stage prepare (NumPy environment) - rebuild each fixture, run the production reference solve
  and pickle the system (operators, geometry, reference quantities, solved state).
* stage adjoint (torch environment) - load the pickles, build the implicit adjoint for the
  linear and the pressure-Gauss-Newton systems, and compare with central differences.

The split is not cosmetic: running the production NumPy solve in the torch environment dies with
0xC0000005 inside scipy.linalg.lstsq, and calling NumPy LAPACK after a torch kernel does the same
(section 6f of the port map).

The load function carries its own angle dependence (the lift direction rotates with alpha).
Autograd therefore sees the angle twice - inside the loads and through the implicit state - so
the gradient returned here is already the total derivative.

Central differences are taken at h and h/10 and Richardson-extrapolated, so the comparison is not
limited by the O(h^2) truncation of a single step.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("PANEL_ROOT", Path.cwd()))

CASES = ("g1_wing_rect", "g1_swept30", "g1_wingbody_cad", "g2_conventional")
ALPHA = 4.0
STEP = 1e-4
MAX_NFEV = 400
THRESHOLDS = dict(declared_before_measurement=True, linear_relative=1e-6,
                  pressure_relative=2e-2, pressure_floor=1e-4, fd_stability=2e-2)


def _prepare(name):
    from tools.issue92.s3_pressure_parity import _build, _manifest
    import panel.p1_lifting_g1 as g1
    import panel.p1_lifting_g2 as g2

    entry = _manifest()["fixtures"][name]
    system = _build(name, entry)
    started = time.perf_counter()
    if system["kind"] == "g2":
        out = g2.solve_multi(system["mesh"], ALPHA, kutta="pressure",
                             operators=system["ops"], pressure_solver="reference")
    else:
        out = g1.solve(system["wing"], ALPHA, kutta="pressure", pressure_solver="reference")
    seconds = time.perf_counter()-started
    state = np.r_[out["mu"], out["gamma"]]
    source = system["mesh"] if system["kind"] == "g2" else system["wing"]
    chord = getattr(source, "reference_chord", None)
    if chord is None:
        chord = source.chord
    mesh = system["ops"]["mesh"]
    return dict(
        name=name, kind=system["kind"], alpha_deg=ALPHA,
        B=np.ascontiguousarray(system["B"]), kutta=np.ascontiguousarray(system["kutta"]),
        centers=np.ascontiguousarray(mesh.centers), areas=np.ascontiguousarray(mesh.areas),
        normals=np.ascontiguousarray(mesh.normals), grad=np.ascontiguousarray(system["grad"]),
        reference_area=float(source.reference_area), reference_chord=float(chord),
        reference_point=np.ascontiguousarray(source.reference_point),
        pairs=[[np.asarray(a, dtype=int), np.asarray(b, dtype=int)] for a, b in system["pairs"]],
        state=state, n_wall=int(out["mu"].shape[0]),
        loads=dict(CL=float(out["CL"]), CL_gamma=float(out["CL_gamma"]),
                   CD_pressure=float(out["CD_pressure"]), CM_y=float(out["CM_y"])),
        pressure_rms=float(out.get("pressure_kutta_rms", float("nan"))),
        pressure_status=out.get("pressure_status"), pressure_seconds=seconds)


def _adjoint(record):
    import torch

    from panel.p1_implicit_adjoint import AdjointProblem, implicit_state
    from panel.p1_pressure_kutta_torch_driver import TorchPressureModel

    order = ("CL", "CD_pressure", "CM_y")
    wall_operator = torch.as_tensor(record["B"], dtype=torch.float64)
    kutta = torch.as_tensor(record["kutta"], dtype=torch.float64)
    centers = torch.as_tensor(record["centers"], dtype=torch.float64)
    areas = torch.as_tensor(record["areas"], dtype=torch.float64)
    normals = torch.as_tensor(record["normals"], dtype=torch.float64)
    grad = torch.as_tensor(record["grad"], dtype=torch.float64)
    reference_point = torch.as_tensor(record["reference_point"], dtype=torch.float64)
    weights = torch.sqrt(areas/areas.mean())
    pairs = [(torch.as_tensor(a, dtype=torch.long), torch.as_tensor(b, dtype=torch.long))
             for a, b in record["pairs"]]
    model = TorchPressureModel(grad, pairs, int(record["B"].shape[1]))
    area = record["reference_area"]

    def loads(alpha, state):
        mu = state[: int(record["n_wall"])]
        velocity = torch.einsum("fij,j->fi", grad, mu)
        cp = 1.0-(velocity*velocity).sum(dim=1)
        forces = -(cp*areas).unsqueeze(-1)*normals
        net = forces.sum(dim=0)
        moment = torch.linalg.cross(centers-reference_point, forces).sum(dim=0)
        radians = torch.deg2rad(alpha)
        zero = torch.zeros_like(radians)
        U = torch.stack((torch.cos(radians), zero, torch.sin(radians)))
        lift = torch.stack((-torch.sin(radians), zero, torch.cos(radians)))
        return dict(CL=net @ lift/area, CD_pressure=net @ U/area,
                    CM_y=moment[1]/(area*record["reference_chord"]))

    entries = {}
    production = torch.as_tensor(record["state"], dtype=torch.float64)
    for level in ("linear", "pressure"):
        variants = (("gauss_newton", "exact") if level == "pressure" else ("exact",))
        problems = {}
        started = time.perf_counter()
        for variant in variants:
            problems[variant] = AdjointProblem(
                wall_operator=wall_operator, kutta_operator=kutta, centers=centers,
                weights=weights, model=(model if level == "pressure" else None),
                te_face_pairs=(pairs if level == "pressure" else None), weight=5.0,
                hessian=variant, max_nfev=MAX_NFEV)
        problem = problems["exact"]
        alpha = torch.tensor(record["alpha_deg"], dtype=torch.float64)
        state = problem.state(alpha).detach()
        forward_seconds = time.perf_counter()-started
        values = loads(alpha, state)
        state_error = float((state-production).abs().max())

        analytic, adjoint_seconds = {}, {}
        for variant in variants:
            differentiable_alpha = alpha.clone().requires_grad_(True)
            started = time.perf_counter()
            differentiable = implicit_state(differentiable_alpha, problems[variant])
            outputs = loads(differentiable_alpha, differentiable)
            analytic[variant] = {}
            for key in order:
                (gradient,) = torch.autograd.grad(outputs[key], differentiable_alpha,
                                                  retain_graph=True)
                analytic[variant][key] = float(gradient)
            adjoint_seconds[variant] = time.perf_counter()-started

        def value_at(offset):
            angle = torch.tensor(record["alpha_deg"]+offset, dtype=torch.float64)
            return loads(angle, problem.state(angle).detach())

        comparisons = {}
        limit = (THRESHOLDS["linear_relative"] if level == "linear"
                 else THRESHOLDS["pressure_relative"])
        for key in order:
            big = float(value_at(STEP)[key]-value_at(-STEP)[key])/(2.0*STEP)
            small = float(value_at(STEP/10.0)[key]-value_at(-STEP/10.0)[key])/(2.0*STEP/10.0)
            richardson = (4.0*small-big)/3.0
            scale = max(abs(richardson), THRESHOLDS["pressure_floor"])
            stability = abs(float(big)-float(small))/scale
            entry = dict(fd_step=float(big), fd_small=float(small), richardson=richardson,
                         scale=scale, fd_stability=stability,
                         oracle_stable=bool(stability <= THRESHOLDS["fd_stability"]))
            for variant in variants:
                total = analytic[variant][key]
                error = abs(total-richardson)
                entry[variant] = dict(
                    adjoint=total, absolute=error, relative=error/scale,
                    passed=bool(error <= limit*scale))
            comparisons[key] = entry
        diagnostics = problem.conditioning(state, alpha)
        entries[level] = dict(
            diagnostic=diagnostics,
            state_error=state_error,
            state_relative=state_error/max(float(production.abs().max()), 1e-30),
            pressure_rms=(record["pressure_rms"] if level == "pressure" else None),
            loads={key: float(values[key]) for key in order},
            comparisons=comparisons, forward_seconds=forward_seconds,
            adjoint_seconds=adjoint_seconds,
            oracle_stable=bool(all(item["oracle_stable"] for item in comparisons.values())),
            passed=bool(all(item["exact"]["passed"] and item["oracle_stable"]
                           for item in comparisons.values())))

    ineligible = [level for level, entry in entries.items() if not entry["oracle_stable"]]
    return dict(name=record["name"], kind=record["kind"], ineligible=ineligible,
                production_loads=record["loads"],
                pressure_status=record["pressure_status"],
                production_seconds=record["pressure_seconds"],
                levels=entries,
                passed=bool(all(entry["passed"] for entry in entries.values())))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "adjoint"), default="adjoint")
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--artifacts", type=Path, default=ROOT/"runs/issue92/s8")
    parser.add_argument("--out", type=Path, default=ROOT/"docs/issue92/s8_adjoint.json")
    args = parser.parse_args(argv)

    if args.stage == "prepare":
        args.artifacts.mkdir(parents=True, exist_ok=True)
        for name in args.cases:
            record = _prepare(name)
            path = args.artifacts/(name + ".pkl")
            path.write_bytes(pickle.dumps(record))
            print("prepared", name, path, path.stat().st_size)
        return 0

    rows = []
    for name in args.cases:
        path = args.artifacts/(name + ".pkl")
        if not path.exists():
            raise SystemExit("missing artifact: %s" % path)
        rows.append(_adjoint(pickle.loads(path.read_bytes())))
        print("adjoint", name, rows[-1]["passed"])

    report = dict(thresholds=THRESHOLDS, alpha_deg=ALPHA, step=STEP,
                  cases={row["name"]: row for row in rows},
                  certified=[row["name"] for row in rows if row["passed"]],
                  unstable={row["name"]: row["ineligible"] for row in rows if row["ineligible"]},
                  passed=bool(all(row["passed"] for row in rows)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    summary = {row["name"]: {level: {key: dict(exact=item["exact"]["relative"],
                                               gauss_newton=item.get("gauss_newton", {}).get(
                                                   "relative"))
                                     for key, item in entry["comparisons"].items()}
                             for level, entry in row["levels"].items()}
               for row in rows}
    print(json.dumps(dict(passed=report["passed"], certified=report["certified"],
                          unstable=report["unstable"], relative=summary), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
