"""Compare the production fixed-wake solver with VSPAERO thick-panel results."""

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np

from panel.p1_galerkin_wall import assemble_galerkin_multi
from panel.p1_lifting_g2 import combine_wings, solve_multi
from tools.issue92 import b3_dpanel, b3_vspaero
from vspapi.results import read_lod

ROOT = Path(__file__).resolve().parents[1]
CASE_FILE = ROOT / "verification" / "issue92" / "b3_cases.json"


def main(argv=None):
    manifest = json.loads(CASE_FILE.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=manifest["cases"])
    parser.add_argument("--out", type=Path, default=ROOT / "runs/verification/vspaero")
    parser.add_argument("--vsp-root", type=Path, required=True)
    args = parser.parse_args(argv)

    vsp_root = args.vsp_root.resolve()
    exe = vsp_root / "vspaero.exe"
    if not exe.is_file():
        parser.error(f"VSPAERO executable not found: {exe}")
    for package in ("openvsp", "openvsp_config", "degen_geom", "utilities"):
        sys.path.insert(0, str(vsp_root / "python" / package))
    import openvsp

    workdir = args.out.resolve() / f"vsp_{uuid4().hex[:12]}"
    workdir.mkdir(parents=True)
    alphas = manifest["alphas_deg"]
    selected_cases = args.case or ["rect", "elliptic"]
    report = dict(openvsp=openvsp.GetVSPVersion(), executable=str(exe),
                  angles_deg=alphas, mach=0., beta_deg=0., wake_iters=1, wake_nodes=16,
                  vspaero_mode="panel", reference_geometry=str(CASE_FILE.relative_to(ROOT)),
                  selected_cases=selected_cases, cases={})
    for name in selected_cases:
        spec = manifest["cases"][name]
        if "surface" in spec:
            spec = dict(spec, surface=str(ROOT / spec["surface"]))
        wing = b3_dpanel._wing(spec)
        mesh = combine_wings([wing])
        operators = assemble_galerkin_multi(mesh)
        area_scale = mesh.reference_area / spec["sref"]
        dp_rows = []
        for alpha in alphas:
            state = solve_multi(mesh, alpha, operators=operators)
            mode = (state["wall_formulation"], state["kutta_effective"], state["pressure_recovery"])
            if mode != ("galerkin", "pressure", "area") or not state["nonlinear"]["success"]:
                raise RuntimeError(f"{name} alpha={alpha}: production solve did not converge ({mode})")
            dp_rows.append(dict(alpha_deg=alpha, CL=float(state["CL"]) * area_scale,
                                CL_gamma=float(state["CL_gamma"]) * area_scale,
                                gamma=state["gamma"].tolist()))
        print(f"{name}: DPanel solved {len(dp_rows)} angles", flush=True)

        vsp_rows = b3_vspaero.run_case(name, spec, alphas=alphas, levels=("coarse", "fine"),
                                      modes=("panel",), out_dir=workdir, exe=str(exe), ncpu=2)
        dp_slope = float(np.polyfit(alphas, [row["CL"] for row in dp_rows], 1)[0])
        dp_gamma = np.asarray(dp_rows[alphas.index(4.)]["gamma"])
        dp_y = np.asarray(wing.gamma_y)[1:-1]
        station_y = [station["y"] for station in spec["stations"]]
        station_chord = [station["chord"] for station in spec["stations"]]
        comparisons = []
        for row in vsp_rows:
            polar = row.get("polar", [])
            if row["exit"] != 0 or [point["AoA"] for point in polar] != alphas:
                raise RuntimeError(f"{name} {row['level']}: incomplete VSPAERO polar in {row['workdir']}")
            vsp_cl = [point["CLwtot"] for point in polar]
            vsp_slope = float(np.polyfit(alphas, vsp_cl, 1)[0])
            sections = read_lod(Path(row["workdir"]) / f"{name}.lod", n_alpha=len(alphas))
            strips = sections[alphas.index(4.)]
            vsp_y = strips["y"]
            vsp_gamma = .5 * strips["cl"] * np.interp(np.abs(vsp_y), station_y, station_chord)
            y_min = max(dp_y.min(), vsp_y.min())
            y_max = min(dp_y.max(), vsp_y.max())
            if y_min >= y_max or np.max(np.abs(vsp_gamma)) == 0. or np.max(np.abs(dp_gamma)) == 0.:
                raise ValueError(f"{name} {row['level']}: missing common span or circulation at 4 deg")
            common = np.linspace(y_min, y_max, 41)
            order = np.argsort(vsp_y)
            reference = np.interp(common, vsp_y[order], (vsp_gamma / np.max(np.abs(vsp_gamma)))[order])
            measured = np.interp(common, dp_y, dp_gamma / np.max(np.abs(dp_gamma)))
            shape_rms = float(np.sqrt(np.mean((measured - reference) ** 2)))
            comparisons.append(dict(level=row["level"], panels=row["mesh"]["panels"],
                                    CLwtot=vsp_cl, slope_per_deg=vsp_slope,
                                    slope_relative=dp_slope / vsp_slope - 1.,
                                    gamma_shape_rms_at_4_deg=shape_rms, workdir=row["workdir"]))
        matched_camber = name in ("rect", "elliptic")
        report["cases"][name] = dict(
            reference_area=spec["sref"], reference_chord=spec["cref"],
            reference_span=2. * spec["half_span"], moment_reference=spec["moment_reference"],
            geometry_sha256=spec["geometry_sha256"], matched_camber=matched_camber,
            dpanel_reference_area=float(mesh.reference_area), dpanel_faces=len(mesh.faces),
            dpanel_mode=list(mode), dpanel_slope_per_deg=dp_slope,
            dpanel_stations=dp_y.tolist(), dpanel=dp_rows, comparisons=comparisons)
        fine = comparisons[-1]
        print(f"{name}: fine lift slope difference {fine['slope_relative']:+.1%}, "
              f"normalized circulation RMS {fine['gamma_shape_rms_at_4_deg']:.3f}", flush=True)
    pending = workdir / "summary.json"
    pending.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(args.out / "summary.json")
    print(f"wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
