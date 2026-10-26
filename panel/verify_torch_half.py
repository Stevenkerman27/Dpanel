"""Float64 JAX vs native PyTorch CPU/CUDA on EXACTLY the same half mesh.

 python -m panel.verify_torch_half --surface runs/HALF/surface.json \
   --out runs/NEW_TORCH_HALF/report.json --device cuda --derivatives

Prepares the validated cut-aware stencil topology once on the host; then
compares both dense half lifting solvers with frozen identical connectivity.
Does not rerun CAD or infer a physical-accuracy acceptance from matching data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import torch

from .run_half_native import load_half_surface
from .half_native import solve_half_fields
from torch_panel.half_lifting import topology_to_torch, solve_half_torch

FIELDS = (
    "matrix_half", "rhs_half", "mu_half", "gamma_half", "sigma_half",
    "wake_basis_half", "wake_slope_half", "te_basis_half",
    "Cp_raw_half", "panel_forces_half", "panel_moments_half",
    "panel_centers_half", "panel_areas_half", "panel_normals_half",
    "surface_velocity_half", "tangent_determinant", "force", "moment",
    "component_forces", "component_moments", "CL", "CD_pressure", "CM",
)


def load_torch_inputs(jax_input, device):
    result = dict(topology=topology_to_torch(jax_input["topology"], device=device))
    for key in ("vertices", "station_y", "freestream", "density",
                "reference_area", "reference_chord", "reference_point"):
        result[key] = torch.tensor(np.asarray(jax_input[key]),
                                   device=device, dtype=torch.float64)
    return result


def objective(out):
    # A mixed total/local-load objective exercises both implicit adjoint
    # and explicit derivatives of panel areas/normals/pressure moment arms.
    return (out["CL"] + .37*out["CM"][1]
            + .01*out["panel_forces_half"][:, 0].sum())


def _compare(name, torch_tensor, jax_tensor, *, rtol, atol):
    a = torch_tensor.detach().cpu().numpy()
    b = np.asarray(jax_tensor)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError(f"{name}: shape/nonfinite mismatch {a.shape} != {b.shape}")
    delta = a-b
    per_entry_ok = np.all(np.abs(delta) <= atol+rtol*np.abs(b))
    return dict(max_abs=float(np.max(np.abs(delta))),
                relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(b), 1e-300)),
                within_tolerance=bool(per_entry_ok))


def verify(surface, *, device, block_size, max_panels, rtol, atol,
           derivatives, stencil_rings=2, max_stencil_rings=6,
           checkpoint_kernel=False):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but torch.cuda.is_available() is false")
    if device not in ("cpu", "cuda"):
        raise ValueError("Explicitly select cpu or cuda; no silent fallback")
    reference_input = load_half_surface(
        surface, stencil_rings=stencil_rings,
        max_stencil_rings=max_stencil_rings)
    nf = len(reference_input["topology"].faces)
    if nf > max_panels:
        raise MemoryError(f"Refusing half mesh {nf} > max_panels={max_panels}")
    ti = load_torch_inputs(reference_input, device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    reference = jax.jit(solve_half_fields)(**reference_input)
    reference["CL"].block_until_ready()
    jax_sec = time.perf_counter()-start

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    start = time.perf_counter()
    actual = solve_half_torch(**ti, block_size=block_size)
    sync()
    torch_cold_sec = time.perf_counter()-start
    compare = {key: _compare(key, actual[key], reference[key],
                             rtol=rtol, atol=atol) for key in FIELDS}
    start = time.perf_counter()
    actual2 = solve_half_torch(**ti, block_size=block_size)
    sync()
    torch_hot_sec = time.perf_counter()-start
    total_state = torch.cat((actual2["mu_half"], actual2["gamma_half"]))
    real_residual = torch.linalg.vector_norm(
        actual2["matrix_half"]@total_state-actual2["rhs_half"])
    rhs_norm = torch.linalg.vector_norm(actual2["rhs_half"])
    residual = float((real_residual/torch.clamp(rhs_norm, min=1e-300)).item())
    results = dict(
        device=device, torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
        jax_version=jax.__version__, float64=True,
        half_wall_panels=nf, unknowns=len(total_state),
        source_sha256=hashlib.sha256(Path(surface).read_bytes()).hexdigest(),
        jax_cpu_reference_seconds=jax_sec, torch_first_seconds=torch_cold_sec,
        torch_hot_seconds=torch_hot_sec, torch_true_relative_residual=residual,
        fields=compare,
        passed_fields=sum(v["within_tolerance"] for v in compare.values()),
        compared_fields=len(compare), physical_accuracy_accepted=False,
    )
    if device == "cuda":
        results.update(cuda_name=torch.cuda.get_device_name(),
                       cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    if derivatives:
        x0, y0 = ti["vertices"], ti["station_y"]
        dx = torch.stack((.02*x0[:, 0], .01*x0[:, 1], .02*x0[:, 2]), dim=1)
        dy = .01*y0
        theta = torch.zeros((), device=device, dtype=torch.float64, requires_grad=True)

        def torch_fun(a):
            return objective(solve_half_torch(
                **{**ti, "vertices": x0+a*dx,
                   "station_y": y0+a*dy}, block_size=block_size))

        t0 = time.perf_counter()
        torch_ad = float(torch.autograd.grad(torch_fun(theta), theta)[0].item())
        sync()
        torch_grad_seconds = time.perf_counter()-t0
        xj, yj = reference_input["vertices"], reference_input["station_y"]
        dxj = jnp.asarray(dx.detach().cpu().numpy())
        dyj = jnp.asarray(dy.detach().cpu().numpy())

        def jax_fun(a):
            return objective(solve_half_fields(**{
                **reference_input, "vertices": xj+a*dxj,
                "station_y": yj+a*dyj}))

        jax_ad = float(jax.grad(jax_fun)(jnp.array(0.)))
        finite_differences = []
        for h in (1e-3, 1e-4, 1e-5):
            fd = float(((torch_fun(h)-torch_fun(-h))/(2*h)).item())
            finite_differences.append(dict(
                h=h, central=fd, torch_abs_error=abs(torch_ad-fd),
                jax_abs_error=abs(jax_ad-fd)))
        results["geometry_station_vjp"] = dict(
            torch=torch_ad, jax=jax_ad, abs_delta=abs(torch_ad-jax_ad),
            fd=finite_differences, torch_backward_seconds=torch_grad_seconds)
        results["gradient_pass"] = bool(
            np.isclose(torch_ad, jax_ad, rtol=1e-4, atol=1e-7)
            and all(np.isclose(torch_ad, z["central"], rtol=1e-4, atol=1e-7)
                    for z in finite_differences))
    else:
        results["gradient_pass"] = None

    # Preserve individual mismatches in the JSON even when an acceptance
    # check fails; errors are never silently downgraded to a pass.
    errors = []
    if residual >= 1e-10:
        errors.append(f"True LU residual {residual} exceeded 1e-10")
    if results["passed_fields"] != len(FIELDS):
        bad = [key for key, value in compare.items() if not value["within_tolerance"]]
        errors.append(f"Torch/JAX numerical mismatch in {bad}")
    if results["gradient_pass"] is False:
        errors.append("Torch/JAX/FD geometry gradient mismatch")
    results["failure_reasons"] = errors
    results["status"] = ("torch_half_numerical_and_gradient_parity_passed"
                         if not errors else "torch_half_parity_failed")
    if device == "cuda":
        # Include the backward peak, not only the two forward calls.
        results["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        results["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--surface", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    p.add_argument("--block", type=int, default=16)
    p.add_argument("--max-panels", type=int, default=1000)
    p.add_argument("--rtol", type=float, default=2e-5)
    p.add_argument("--atol", type=float, default=1e-7)
    p.add_argument("--rings", type=int, default=2)
    p.add_argument("--max-stencil-rings", type=int, default=6)
    p.add_argument("--derivatives", action="store_true")
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("Output exists; do not overwrite archived runs")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status="started", physical_accuracy_accepted=False,
                  source=str(args.surface), device=args.device)
    try:
        report.update(verify(args.surface, device=args.device,
                             block_size=args.block, max_panels=args.max_panels,
                             rtol=args.rtol, atol=args.atol,
                             derivatives=args.derivatives,
                             stencil_rings=args.rings,
                             max_stencil_rings=args.max_stencil_rings))
    except BaseException as exc:
        report.update(status="failed", exception=type(exc).__name__, error=str(exc))
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n",
                            encoding="utf-8")
        raise
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n",
                        encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    if report["status"] != "torch_half_numerical_and_gradient_parity_passed":
        raise RuntimeError("Torch/JAX parity failed; full field diagnostics are in report")


if __name__ == "__main__":
    main()
