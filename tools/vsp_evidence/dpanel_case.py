"""One DPanel half-model case for the trapezoidal wing: mesh + alpha sweep."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "runs" / "pr49_git"
GMSH = ROOT / "audit" / "runtime" / "python"
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(ROOT / "runs"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
from panel.half_native import solve_half_fields  # noqa: E402
from panel.run_half_native import load_half_surface  # noqa: E402
from trefftz_drag import trefftz_cd  # noqa: E402

CAD = ROOT / "runs" / "vsp_conv" / "cad_trap"


def build_mesh(size, out_dir, alpha_deg, ncpu=8, radius=20.0, far_size=4.0):
    cmd = [sys.executable, "-m", "panel.cad_half",
           "--wing-cad", str(CAD / "wing.step"),
           "--trailing-edge", str(CAD / "trailing_edge.json"),
           "--size", str(size), "--far-size", str(far_size),
           "--radius", str(radius),
           "--alpha-deg", str(alpha_deg), "--ncpu", str(ncpu),
           "--out", str(out_dir)]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(WORK) + os.pathsep + str(GMSH)
    env["TMP"] = env["TEMP"] = env["TMPDIR"] = str(ROOT / "runs" / "pipwork")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(WORK), env=env, capture_output=True,
                          text=True)
    secs = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError("mesh failed (%d):\n%s\n%s"
                           % (proc.returncode, proc.stdout[-2000:],
                              proc.stderr[-2500:]))
    return secs, proc.stdout[-800:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=float, required=True)
    ap.add_argument("--alphas", type=float, nargs="*",
                    default=[0.0, 2.5, 5.0, 7.5, 10.0])
    ap.add_argument("--speed", type=float, default=20.0)
    ap.add_argument("--density", type=float, default=1.225)
    ap.add_argument("--ncpu", type=int, default=8)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--radius", type=float, default=20.0)
    ap.add_argument("--far-size", type=float, default=4.0)
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    mesh_dir = out / "mesh"
    surface = mesh_dir / "surface.json"
    if args.reuse and surface.exists():
        mesh_seconds, tail = None, "reused"
    else:
        mesh_seconds, tail = build_mesh(args.size, mesh_dir, 0.0, args.ncpu,
                                       args.radius, args.far_size)

    raw = json.loads(surface.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    inputs, stencil = load_half_surface(surface, stencil_rings=2,
                                        max_stencil_rings=6,
                                        return_stencil_audit=True)
    load_seconds = time.perf_counter() - t0

    speed = float(args.speed)
    stations = np.asarray(raw["gamma_stations"], float)
    dof = np.asarray(raw["gamma_station_dof"], int)
    ref_area = float(inputs["reference_area"])
    ref_chord = float(inputs["reference_chord"])
    rho = float(args.density)
    q = 0.5 * rho * speed * speed
    span = 2.0 * float(stations[-1])

    solve = jax.jit(solve_half_fields)
    rows = []
    t_all = time.perf_counter()
    for alpha in sorted(args.alphas):
        a = np.radians(float(alpha))
        inp = dict(inputs)
        inp["freestream"] = np.asarray(
            [speed * np.cos(a), 0.0, speed * np.sin(a)], float)
        t1 = time.perf_counter()
        res = solve(**inp)
        secs = time.perf_counter() - t1
        res = {k: np.asarray(v) for k, v in res.items() if hasattr(v, "__len__")}
        gamma = np.asarray(res["gamma_half"], float)
        g_station = np.where(dof >= 0, gamma[np.clip(dof, 0, None)], 0.0)
        # trefftz_cd normalises by span*chord unless the true wing area is
        # supplied; for a tapered wing span*MAC is 3.7% larger than Sref.
        cd_t, cd_info = trefftz_cd(stations, g_station, span, ref_chord,
                                   U=speed, rho=rho, area=ref_area)
        lift = 2.0 * rho * speed * np.sum(
            0.5 * (g_station[:-1] + g_station[1:]) * np.diff(stations))
        cp = np.asarray(res["Cp_raw_half"], float)
        lin = float(np.linalg.norm(np.asarray(res["linear_residual_half"]))
                    / np.linalg.norm(np.asarray(res["rhs_half"])))
        rows.append(dict(
            alpha_deg=float(alpha),
            CL_pressure=float(res["CL"]),
            CL_circulation=float(lift / (q * ref_area)),
            CD_pressure=float(res["CD_pressure"]),
            CD_trefftz=float(cd_t),
            ratio_pressure_trefftz=float(res["CD_pressure"] / cd_t) if cd_t else None,
            CM_y=float(np.asarray(res["CM"])[1]),
            span_efficiency=float((lift / (q * ref_area)) ** 2
                                  / (np.pi * span * span / ref_area) / cd_t)
            if cd_t else None,
            cp_min=float(cp.min()), cp_max=float(cp.max()),
            linear_residual=lin,
            gamma_sum=float(g_station.sum()),
            gamma_max=float(g_station.max()),
            gamma_stations=g_station.tolist(),
            solve_seconds=secs))
        print("alpha %.2f  CL_p %.6f  CL_G %.6f  CD_p %.6f  CD_T %.6f  CMy %.6f  %.2f s"
              % (alpha, rows[-1]["CL_pressure"], rows[-1]["CL_circulation"],
                 rows[-1]["CD_pressure"], rows[-1]["CD_trefftz"],
                 rows[-1]["CM_y"], secs))
    total = time.perf_counter() - t_all

    faces = np.asarray(raw["faces"], np.int64)
    wake = np.asarray(raw["wake_faces"], np.int64)
    info = dict(
        status="half_surface_created_not_physical_accuracy",
        physical_accuracy_accepted=False,
        size=args.size, mesh_seconds=mesh_seconds, mesh_log_tail=tail,
        load_seconds=load_seconds,
        wall_panels=int(len(faces)), wake_panels=int(len(wake)),
        vertices=int(len(raw["coordinates"])),
        unknowns=int(len(np.asarray(inputs["mu_half"] if "mu_half" in inputs else []))),
        gamma_stations=int(len(stations)),
        gamma_dof=int(len(np.unique(dof[dof >= 0]))),
        reference_area=ref_area, reference_chord=ref_chord,
        speed=speed, density=rho, span=span,
        stencil_rings=int(stencil.get("rings", 0)) if isinstance(stencil, dict) else None,
        stencil_audit=stencil if isinstance(stencil, dict) else None,
        solve_seconds_total=total,
        freestream_normals=float(np.abs(np.asarray(raw.get("normals", [[0, 0, 1]])[:, 2])).mean())
        if "normals" in raw else None,
        rows=rows)
    (out / "case.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print("WALL", info["wall_panels"], "WAKE", info["wake_panels"],
          "UNKNOWNS", info["unknowns"], "TOTAL %.1f s" % total)


if __name__ == "__main__":
    main()
