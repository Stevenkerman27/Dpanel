"""Opt-in half-native panel CLI for a conforming positive-y surface fixture.

  python -m panel.run_half_native --surface cases/HALF.json \
    --out runs/NEW_HALF_RUN --mode lu
  python -m panel.run_half_native --surface cases/HALF.json \
    --out runs/NEW_HALF_GMRES --mode gmres --near 16

There is no OpenVSP/Gmsh CAD Boolean inside the numerical solver; the source
half-mesh must be a conforming, already-audited exterior. No production path
or accuracy flag is changed.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .half_native import (
    prepare_half_topology, validate_half_configuration, solve_half_fields,
    HalfPanelOperator, _half_loads,
)
from .half_native_gmres import HalfNearFieldPreconditioner, solve_half_gmres


def load_half_surface(path, *, stencil_rings=2,
                      max_stencil_rings=6, return_stencil_audit=False):
    fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    if fixture.get("schema") != "dpanel.half_planar_cut.v1":
        raise ValueError("Expected dpanel.half_planar_cut.v1")
    vertices = np.asarray(fixture["coordinates"], dtype=np.float64)
    station_y = np.asarray(fixture["gamma_stations"], dtype=np.float64)
    built = prepare_half_topology(
        vertices, fixture["faces"], fixture["wake_faces"],
        fixture["trailing_edges"], fixture["upper_te_faces"],
        fixture["lower_te_faces"], station_y,
        fixture["gamma_station_dof"], fixture["component_ids"],
        stencil_rings=stencil_rings, cut_edges=fixture["cut_edges"],
        max_stencil_rings=max_stencil_rings,
        return_stencil_audit=return_stencil_audit,
    )
    topology, stencil_audit = built if return_stencil_audit else (built, None)
    point = np.asarray(fixture["moment_reference"], dtype=np.float64)
    freestream = np.asarray(fixture["freestream"], dtype=np.float64)
    validate_half_configuration(vertices, station_y, topology, freestream, point)
    area = float(fixture["reference_area"])
    chord = float(fixture["reference_chord"])
    density = float(fixture["density"])
    if (min(area, chord, density) <= 0 or
            not np.isfinite([area, chord, density]).all()):
        raise ValueError("Positive SI reference values and density required")
    fields = dict(
        vertices=jnp.asarray(vertices),
        station_y=jnp.asarray(station_y),
        topology=topology,
        freestream=jnp.asarray(freestream),
        density=jnp.asarray(density),
        reference_area=jnp.asarray(area),
        reference_chord=jnp.asarray(chord),
        reference_point=jnp.asarray(point),
    )
    return (fields, stencil_audit) if return_stencil_audit else fields


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--surface", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=("lu", "gmres"), required=True)
    p.add_argument("--block", type=int, default=32)
    p.add_argument("--near", type=int, default=32)
    p.add_argument("--rings", type=int, default=2)
    p.add_argument("--max-stencil-rings", type=int, default=6)
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--restart", type=int, default=40)
    p.add_argument("--cycles", type=int, default=20)
    p.add_argument("--max-panels", type=int, default=1500)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a new report path; preserve existing data")
    args.out.mkdir(parents=True)
    jax.config.update("jax_enable_x64", True)
    report = dict(
        status="started", mode=args.mode, source_surface=str(args.surface),
        JAX=jax.__version__, platform=platform.platform(),
        physical_accuracy_accepted=False)
    try:
        inputs, stencil_audit = load_half_surface(
            args.surface, stencil_rings=args.rings,
            max_stencil_rings=args.max_stencil_rings,
            return_stencil_audit=True)
        report["pressure_stencil_audit"] = stencil_audit
        nf = len(inputs["topology"].faces)
        report["right_wall_panels"] = nf
        report["right_wake_panels"] = len(inputs["topology"].wake_faces)
        report["right_circulation_unknowns"] = inputs["topology"].hat_map.shape[1]
        report["independent_state_unknowns"] = (
            nf + inputs["topology"].hat_map.shape[1])
        if nf > args.max_panels:
            raise MemoryError("Refusing oversized experimental half panel solve")
        t0 = time.perf_counter()
        if args.mode == "lu":
            solve = jax.jit(solve_half_fields)
            out = solve(**inputs)
            out["CL"].block_until_ready()
            report["first_seconds"] = time.perf_counter()-t0
            t0 = time.perf_counter()
            out = solve(**inputs)
            out["CL"].block_until_ready()
            report["hot_seconds"] = time.perf_counter()-t0
            residual = np.asarray(out["linear_residual_half"])
            rhs = np.asarray(out["rhs_half"])
            norm_residual = float(np.linalg.norm(residual) /
                                  max(np.linalg.norm(rhs), 1e-300))
        else:
            op = HalfPanelOperator(
                inputs["vertices"], inputs["station_y"], inputs["topology"],
                inputs["freestream"], block_size=args.block)
            pc = HalfNearFieldPreconditioner(op, max_block_size=args.near)
            solution, gmres_meta = solve_half_gmres(
                op, tolerance=args.tol, restart=args.restart,
                max_cycles=args.cycles, preconditioner=pc)
            mu = jnp.asarray(solution[:nf])
            out = _half_loads(
                inputs["vertices"], inputs["topology"], inputs["freestream"],
                inputs["density"], inputs["reference_area"],
                inputs["reference_chord"], inputs["reference_point"], mu)
            report["first_seconds"] = time.perf_counter()-t0
            t0 = time.perf_counter()
            _, warmed = solve_half_gmres(
                op, tolerance=args.tol, restart=args.restart,
                max_cycles=args.cycles, preconditioner=pc)
            report["hot_seconds"] = time.perf_counter()-t0
            report["gmres"] = gmres_meta
            report["gmres_warm"] = warmed
            norm_residual = gmres_meta["true_relative_residual"]
        report.update(
            CL=float(np.asarray(out["CL"])),
            CD_pressure=float(np.asarray(out["CD_pressure"])),
            CM=np.asarray(out["CM"]).tolist(),
            force_N=np.asarray(out["force"]).tolist(),
            moment_Nm=np.asarray(out["moment"]).tolist(),
            Cp_raw_half_min=float(np.min(np.asarray(out["Cp_raw_half"]))),
            Cp_raw_half_max=float(np.max(np.asarray(out["Cp_raw_half"]))),
            true_relative_residual=norm_residual,
        )
        report["status"] = "numerical_solve_completed_not_physical_validation"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        (args.out/"report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        raise
    (args.out/"report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
