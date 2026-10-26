"""Build one mesh (optionally feature-sized) and solve it: one row of the study."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\3D\Projects\panel")
PR43 = ROOT / "runs" / "pr43_git"
GMSH = ROOT / "audit" / "runtime" / "python"
sys.path.insert(0, str(PR43))
sys.path.insert(0, str(ROOT / "runs"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
from panel.half_native import solve_half_fields  # noqa: E402
from panel.run_half_native import load_half_surface  # noqa: E402
from trefftz_drag import trefftz_cd  # noqa: E402

HALF, ROOT_CHORD, TIP_CHORD = 2.0, 1.0, 0.05


def mesh_metrics(surface_path):
    data = json.loads(Path(surface_path).read_text(encoding="utf-8"))
    x = np.asarray(data["coordinates"], float)
    f = np.asarray(data["faces"], np.int64)
    w = np.asarray(data["wake_faces"], np.int64)
    tri = x[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    cent = tri.mean(axis=1)
    chord = TIP_CHORD + (ROOT_CHORD - TIP_CHORD) * np.sqrt(
        np.clip(1.0 - (np.abs(cent[:, 1]) / HALF) ** 2, 0.0, None)
    )
    le = 0.25 * (ROOT_CHORD - chord)
    le_mask = np.abs(cent[:, 0] - le) <= 0.05
    te_mask = np.abs(cent[:, 0] - (le + chord)) <= 0.05
    edges = np.vstack((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    uniq = np.unique(np.sort(edges, axis=1), axis=0)
    length = np.linalg.norm(x[uniq[:, 0]] - x[uniq[:, 1]], axis=1)
    mid = 0.5 * (x[uniq[:, 0]] + x[uniq[:, 1]])
    mid_chord = TIP_CHORD + (ROOT_CHORD - TIP_CHORD) * np.sqrt(
        np.clip(1.0 - (np.abs(mid[:, 1]) / HALF) ** 2, 0.0, None)
    )
    mid_le = 0.25 * (ROOT_CHORD - mid_chord)
    in_le = np.abs(mid[:, 0] - mid_le) <= 0.05
    in_te = np.abs(mid[:, 0] - (mid_le + mid_chord)) <= 0.05
    e0 = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
    e1 = np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1)
    e2 = np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)
    aspect = np.maximum(np.maximum(e0, e1), e2) / np.maximum(np.minimum(np.minimum(e0, e1), e2), 1e-30)

    def angle(opp, a, b):
        cos = np.clip((a * a + b * b - opp * opp) / (2 * a * b + 1e-300), -1.0, 1.0)
        return np.degrees(np.arccos(cos))

    ang = np.minimum(np.minimum(angle(e0, e1, e2), angle(e1, e2, e0)), angle(e2, e0, e1))
    wake_edges = {tuple(map(int, e)) for e in np.sort(np.vstack((w[:, [0, 1]], w[:, [1, 2]], w[:, [2, 0]])), axis=1)}
    wall_edges = {tuple(map(int, e)) for e in uniq}
    te_pairs = {tuple(map(int, e)) for e in np.sort(np.asarray(data["trailing_edges"], np.int64), axis=1)}
    return dict(
        wall_faces=int(len(f)),
        wake_faces=int(len(w)),
        vertices=int(len(x)),
        wall_edges=int(len(uniq)),
        wall_area=float(area.sum() * 2.0),
        le_band_faces=int(le_mask.sum()),
        le_band_area_fraction=float(area[le_mask].sum() / area.sum()),
        le_band_edge_mean_c=float(length[in_le].mean()) if in_le.any() else None,
        le_band_edge_min_c=float(length[in_le].min()) if in_le.any() else None,
        te_band_faces=int(te_mask.sum()),
        te_band_area_fraction=float(area[te_mask].sum() / area.sum()),
        te_band_edge_mean_c=float(length[in_te].mean()) if in_te.any() else None,
        min_angle_deg=float(ang.min()),
        p01_angle_deg=float(np.percentile(ang, 1)),
        max_aspect=float(aspect.max()),
        p99_aspect=float(np.percentile(aspect, 99)),
        te_edges=int(len(te_pairs)),
        te_edges_shared_with_wake=int(len(te_pairs & wake_edges)),
        wake_edges_shared_with_wall=int(len(wake_edges & wall_edges)),
    )


def build_mesh(case, size, features, out_dir):
    cad = ROOT / "runs" / "geom_robust" / "cad" / case
    cmd = [
        sys.executable, "-m", "panel.cad_half",
        "--wing-cad", str(cad / "wing.step"),
        "--trailing-edge", str(cad / "trailing_edge.json"),
        "--size", str(size), "--far-size", "1.5", "--radius", "5.0",
        "--alpha-deg", "5.0", "--ncpu", "8", "--out", str(out_dir),
    ]
    if features:
        cmd += ["--mesh-features", str(features)]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PR43) + os.pathsep + str(GMSH)
    env["TMP"] = env["TEMP"] = env["TMPDIR"] = str(ROOT / "runs" / "pipwork")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(PR43), env=env, capture_output=True, text=True)
    secs = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError("mesh failed (%d): %s" % (proc.returncode, proc.stderr[-2500:]))
    return secs, proc.stdout[-1500:]


def solve_case(surface_path):
    t0 = time.perf_counter()
    inputs, stencil = load_half_surface(
        surface_path, stencil_rings=2, max_stencil_rings=6, return_stencil_audit=True
    )
    out = jax.jit(solve_half_fields)(**inputs)
    linear = float(np.linalg.norm(np.asarray(out["linear_residual_half"]))
                   / np.linalg.norm(np.asarray(out["rhs_half"])))
    secs = time.perf_counter() - t0
    raw = json.loads(Path(surface_path).read_text(encoding="utf-8"))
    stations = np.asarray(raw["gamma_stations"], float)
    dof = np.asarray(raw["gamma_station_dof"], int)
    gamma = np.asarray(out["gamma_half"], float)
    g_station = np.where(dof >= 0, gamma[np.clip(dof, 0, None)], 0.0)
    flow = np.asarray(inputs["freestream"], float)
    speed = float(np.linalg.norm(flow))
    rho = float(inputs["density"])
    chord = float(inputs["reference_chord"])
    span = float(2.0 * stations[-1])
    cd_t, _ = trefftz_cd(stations, g_station, span, chord, U=speed, rho=rho)
    q = 0.5 * rho * speed * speed
    lift = 2.0 * rho * speed * np.sum(0.5 * (g_station[:-1] + g_station[1:]) * np.diff(stations))
    cp = np.asarray(out["Cp_raw_half"], float)
    return dict(
        unknowns=int(len(gamma) + len(raw["faces"])),
        gamma_stations=int(len(stations)),
        gamma_dof=int(len(np.unique(dof[dof >= 0]))),
        CL_pressure=float(out["CL"]),
        CL_circulation=float(lift / (q * inputs["reference_area"])),
        CD_pressure=float(out["CD_pressure"]),
        CD_trefftz=float(cd_t),
        ratio_pressure_trefftz=float(out["CD_pressure"] / cd_t) if cd_t else None,
        CM_y=float(np.asarray(out["CM"])[1]),
        cp_min=float(cp.min()),
        cp_max=float(cp.max()),
        linear_residual_relative=linear,
        solve_seconds=round(secs, 3),
        pressure_stencil=dict(
            initial_rank_deficient_faces=[int(i) for i in (stencil or {}).get("initial_rank_deficient_faces", [])],
            expanded_faces=[int(i) for i in (stencil or {}).get("expanded_faces", [])],
        ),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--case", default="elliptic")
    ap.add_argument("--size", type=float, required=True)
    ap.add_argument("--features", default=None)
    ap.add_argument("--direction", default=None)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    mesh_dir = out_root / args.tag
    row = dict(tag=args.tag, case=args.case, size=args.size, direction=args.direction,
               features=str(args.features) if args.features else None)
    if (mesh_dir / "surface.json").is_file():
        row["mesh_seconds"] = None
        row["mesh_reused"] = True
    else:
        secs, tail = build_mesh(args.case, args.size, args.features, mesh_dir)
        row["mesh_seconds"] = round(secs, 2)
        row["mesh_reused"] = False
        row["mesh_log_tail"] = tail
    surface = mesh_dir / "surface.json"
    row["mesh"] = mesh_metrics(surface)
    man = mesh_dir / "manifest.json"
    if man.is_file():
        manj = json.loads(man.read_text(encoding="utf-8"))
        prov = manj.get("provenance", {})
        row["feature_sizing"] = prov.get("feature_sizing")
        row["surface_sha256"] = manj.get("surface_sha256") or manj.get("sha256")
    row["solve"] = solve_case(surface)
    (mesh_dir / "case.json").write_text(json.dumps(row, indent=1) + "\n", encoding="utf-8")
    print("TAG=%s walls=%d CL=%.6f CDp=%.6f CDt=%.6f CM=%.6f" % (
        args.tag, row["mesh"]["wall_faces"], row["solve"]["CL_pressure"],
        row["solve"]["CD_pressure"], row["solve"]["CD_trefftz"], row["solve"]["CM_y"]))


if __name__ == "__main__":
    main()
