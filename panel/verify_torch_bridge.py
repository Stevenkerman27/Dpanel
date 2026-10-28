"""Verify the CPU Torch bridge on an existing, immutable panel solve directory.

Example:
 python -m panel.verify_torch_bridge --run runs/EXISTING/solve \
     --out runs/NEW_TORCH_BRIDGE --derivatives

This is an integration/AD test, not a new physical-accuracy acceptance. The
parameterization below is a small compatible Torch deformation, not GRASS CAD.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-panels", type=int, default=1000)
    parser.add_argument("--derivatives", action="store_true")
    args = parser.parse_args()
    if args.max_panels <= 0:
        raise ValueError("max-panels must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "source_run": str(args.run.resolve()),
              "physical_accuracy_accepted": False,
              "scope": "CPU float64; fixed topology; first-order Torch/JAX integration only"}
    try:
        import jax
        import jax.numpy as jnp
        import torch
        from .differentiable import load_prepared_run, solve_fields
        from .torch_bridge import TorchPanelBridge, INPUT_NAMES, OUTPUT_NAMES

        # Fail before loading any dense state or compiling an oversized solve.
        fixture = json.loads((args.run / "surface_input.json").read_text(encoding="utf-8"))
        count = len(fixture["faces"])
        if count > args.max_panels:
            raise MemoryError(f"{count} panels exceed diagnostic max-panels={args.max_panels}")
        jax.config.update("jax_enable_x64", True)
        source, metadata = load_prepared_run(args.run)
        if not np.isclose(float(source["freestream"][1]), 0.0):
            raise ValueError("This verification deformation is defined for longitudinal flow")
        bridge = TorchPanelBridge(source["topology"])
        baseline = tuple(torch.tensor(np.array(source[k]), dtype=torch.float64) for k in INPUT_NAMES)
        start = time.perf_counter()
        actual = bridge(*baseline)
        cold = time.perf_counter() - start
        start = time.perf_counter()
        actual = bridge(*baseline)
        warm = time.perf_counter() - start

        def native_selected(*values):
            out = solve_fields(topology=source["topology"], **dict(zip(INPUT_NAMES, values)))
            return tuple(out[k] for k in OUTPUT_NAMES)

        expected = jax.jit(native_selected)(*(source[k] for k in INPUT_NAMES))
        comparisons = {}
        for name, a, b in zip(OUTPUT_NAMES, actual, expected):
            value, reference = a.detach().numpy(), np.asarray(b)
            error = value - reference
            comparisons[name] = dict(
                max_abs=float(np.max(np.abs(error))),
                relative_L2=float(np.linalg.norm(error)/max(np.linalg.norm(reference), 1e-300)),
                passed=bool(np.allclose(value, reference, rtol=2e-10, atol=2e-10)),
            )
        if not all(x["passed"] for x in comparisons.values()):
            raise AssertionError("Torch bridge forward differs from native JAX")
        report.update(
            panels=count, field_regression=comparisons,
            timing_seconds={"bridge_cold_forward": cold, "bridge_warm_forward": warm},
            environment=dict(python=platform.python_version(), jax=jax.__version__,
                             torch=torch.__version__, numpy=np.__version__, device="cpu", dtype="float64"),
            source_physical_accuracy_accepted=metadata["result"].get("physical_accuracy_accepted", False),
            source_surface_sha256=hashlib.sha256((args.run / "surface_input.json").read_bytes()).hexdigest(),
        )
        if args.derivatives:
            origin = torch.tensor(fixture["origin"], dtype=torch.float64)
            x, y, flow, rho, area, chord, ref = baseline
            relative = x - origin
            alpha0 = torch.atan2(flow[2], flow[0])
            speed0 = torch.linalg.vector_norm(flow)
            rng = np.random.default_rng(419)
            seeds = tuple(torch.tensor(rng.normal(size=tuple(v.shape))/np.sqrt(max(v.numel(), 1)),
                                       dtype=torch.float64) for v in actual)

            def objective(theta):
                sx, sy = 1 + .03*theta[0], 1 + .04*theta[1]
                changed = origin + torch.stack((
                    relative[:, 0]*sx,
                    relative[:, 1]*sy,
                    relative[:, 2]*(1+.04*theta[2]*torch.exp(-(relative[:, 1]/chord)**2)),
                ), dim=1)
                stations = origin[1] + (y-origin[1])*sy
                alpha = alpha0 + .01*theta[3]
                speed = speed0*torch.exp(.02*theta[4])
                velocity = speed*torch.stack((torch.cos(alpha), torch.zeros_like(alpha), torch.sin(alpha)))
                density = rho*torch.exp(.03*theta[5])
                point = ref + torch.stack((.02*chord*theta[6], 0*theta[6], 0*theta[6]))
                out = bridge(changed, stations, velocity, density, area*sx*sy, chord*sx, point)
                return sum((v*w).sum() for v, w in zip(out, seeds))

            theta = torch.zeros(7, dtype=torch.float64, requires_grad=True)
            value = objective(theta)
            start = time.perf_counter()
            grad, = torch.autograd.grad(value, theta)
            report["timing_seconds"]["backward_compile_and_first"] = time.perf_counter()-start
            # Recompute a distinct graph: do not retain a JAX pullback in Torch ctx.
            warm_value = objective(theta)
            start = time.perf_counter()
            warm_grad, = torch.autograd.grad(warm_value, theta)
            report["timing_seconds"]["backward_warm_recompute"] = time.perf_counter()-start
            torch.testing.assert_close(grad, warm_grad, rtol=1e-11, atol=1e-11)
            if not torch.isfinite(grad).all() or grad.norm().item() == 0:
                raise AssertionError("Nonfinite or zero end-to-end parameter gradient")
            checks = []
            for direction in (np.array([.3,-.2,.4,.1,-.2,.1,.4]),
                              np.array([-.1,.3,.2,-.4,.2,.1,-.2])):
                direction = torch.tensor(direction, dtype=torch.float64)
                ad = float((grad*direction).sum())
                for h in (1e-3, 1e-4, 1e-5):
                    with torch.no_grad():
                        fd = float((objective(theta.detach()+h*direction)-objective(theta.detach()-h*direction))/(2*h))
                    checks.append(dict(step=h, ad=ad, fd=fd, absolute_error=abs(ad-fd),
                                       relative_error=abs(ad-fd)/max(abs(ad), 1e-12),
                                       passed=bool(abs(ad-fd) <= 1e-7 + 2e-5*abs(ad))))
            report["parameter_gradient"] = grad.tolist()
            report["directional_checks"] = checks
            if not all(c["passed"] for c in checks):
                raise AssertionError("Torch parameter chain failed centered-difference checks")
        report["status"] = "bridge_checks_passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (args.out / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
