"""Standard-model calibration suite for issue #115 (unsteady-free, inviscid panel).

Each case reports differences against a frozen reference, never a pass/fail
threshold: the analytic rows are exact, the NACA 0012 rows are experimental and
carry their Reynolds/Mach/transition conditions. Run one case with

    python -m tools.calibration --case C02 --out runs/verification/calibration

and every case with

    python -m tools.calibration --out runs/verification/calibration

A case that the current code cannot run yet is reported as blocked with the
reason and its reference table, so the gap is visible instead of hidden.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "benchmarks" / "calibration" / "reference.json"

ALPHA_DEG = 4.0
SPAN = 8.0
TWO_PI = 2.0 * np.pi


def _reference():
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def _environment():
    import scipy
    import torch

    return dict(commit=_git_commit(), dirty=_git_dirty(), platform=platform.platform(),
                python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
                torch=torch.__version__, torch_threads=torch.get_num_threads())


def _git_commit():
    head = ROOT / ".git" / "HEAD"
    try:
        text = head.read_text(encoding="utf-8").strip()
        if text.startswith("ref:"):
            return (ROOT / ".git" / text.split(":", 1)[1].strip()).read_text(
                encoding="utf-8").strip()[:12]
        return text[:12]
    except OSError:
        return "unknown"


def _git_dirty():
    import subprocess

    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                             text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(out.stdout.strip()) if out.returncode == 0 else None


def _summary(case_id, *, status, geometry_source="", solver_mode="", alpha_deg=None,
             samples=None, differences=None, notes=None, levels=None):
    """Per-run record following the field template of issue #115 section 7.2."""
    record = dict(
        case_id=case_id,
        geometry_source=geometry_source,
        geometry_sha256="",
        solver_commit=_git_commit(),
        solver_mode=solver_mode,
        U_inf=1.0, rho_inf=1.0,
        mach=None, reynolds_chord=None, alpha_deg=alpha_deg,
        transition="not applicable (inviscid)",
        reference_area="per case",
        mesh=samples or {},
        algebraic_residual=None,
        max_normal_velocity_ratio=None,
        cp_l2_error=None, cp_linf_error=None,
        cl_pressure=None, cl_circulation=None,
        cd_pressure=None, cdi_trefftz=None,
        moment_quarter_chord=None,
        physical_acceptance="not_assessed",
        status=status,
        differences=differences or {},
        levels=levels or [],
        notes=notes or [],
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
        environment=_environment(),
    )
    return record


# --------------------------------------------------------------------------- C01
def case_c01():
    """2D cylinder: no 2D panel path exists in the current chain."""
    ref = _reference()["tables"]["C01_cylinder_cp"]
    return _summary(
        "C01_cylinder_A0", status="blocked",
        geometry_source="analytic circle, radius 1, N=40..320 panels",
        solver_mode="none: the repository has no 2D panel solver",
        alpha_deg=0.0,
        differences={"reference_table": ref},
        notes=["Issue #115 requires the exact 2D cylinder Cp = 1 - 4 sin^2(theta) and zero "
               "force. The current code is a 3D surface solver (P1 doublet, wing or closed "
               "body) and has no 2D mode, so this case cannot be run without a new 2D path. "
               "The issue explicitly forbids substituting a long capped cylinder mid-section "
               "for the 2D analytic answer.",
               "Reference table is kept frozen in benchmarks/calibration/reference.json so the "
               "case runs the moment a 2D path exists."])


# --------------------------------------------------------------------------- C02
def case_c02(levels=(1, 2, 3)):
    """3D sphere: the first genuinely three-dimensional analytic body."""
    from panel.p1_closed_body import solve, surface_values
    from panel.run_p1_closed import icosphere

    flow = np.array([1.0, 0.0, 0.0])
    rows, states = [], []
    for level in levels:
        vertices, faces = icosphere(level=level, radius=1.0)
        started = time.perf_counter()
        state = solve(vertices, faces, flow)
        elapsed = time.perf_counter() - started
        values = surface_values(state)
        mesh = state["mesh"]
        radius = np.linalg.norm(mesh.centers, axis=1)
        sin2 = 1.0 - (mesh.centers[:, 0] / radius) ** 2
        exact = 1.0 - 2.25 * sin2
        error = values["cp"] - exact
        weights = mesh.areas
        rows.append(dict(
            level=int(level), faces=int(len(faces)), nodes=int(len(state["mu"])),
            cp_l2=float(np.sqrt(np.sum(weights * error ** 2) / np.sum(weights))),
            cp_linf=float(np.max(np.abs(error))),
            cp_biased_mean=float(np.sum(weights * error) / np.sum(weights)),
            force_coefficient=[float(x) for x in values["force_coefficient"]],
            moment=[float(x) for x in values["moment"]],
            max_radius_error=float(np.max(np.abs(radius - 1.0))),
            seconds=elapsed))
        states.append(state)
    for previous, current in zip(rows, rows[1:]):
        current["cp_l2_ratio_to_previous"] = current["cp_l2"] / previous["cp_l2"]

    finest = rows[-1]
    return _summary(
        "C02_sphere_A0", status="ran",
        geometry_source="icosphere level 1..3 (80/320/1280 faces), radius 1",
        solver_mode="P1 continuous doublet, area-weighted QR, no prescribed analytic mu",
        alpha_deg=0.0,
        samples=dict(levels=[r["level"] for r in rows], surface_faces=finest["faces"]),
        differences=dict(
            cp_exact="Cp = 1 - 9/4 sin^2(theta) at the face-centre polar angle",
            cp_l2_relative_to_dynamic_head=finest["cp_l2"],
            cp_linf=finest["cp_linf"],
            force_coefficient_norm=float(np.linalg.norm(finest["force_coefficient"])),
            ideal_force="zero net pressure force on a closed body in uniform inviscid flow"),
        levels=rows,
        notes=["Cp is compared at the face centre projected onto the exact radius; the panel "
               "surface is faceted, so the geometric error is reported separately as "
               "max_radius_error.",
               "max_normal_velocity_ratio is left null: the P1 recovery velocity is in-plane by "
               "construction, so it cannot diagnose wall leakage. A separate exterior-probe "
               "(velocity_fd) audit is the honest measurement and is not yet wired in here."])


# --------------------------------------------------------------------------- C03
def case_c03():
    ref = _reference()["tables"]["C03_joukowski"]
    return _summary(
        "C03_joukowski_A0_4_8", status="blocked",
        geometry_source="Joukowski map z = zeta + 1/zeta, circle centre -0.1+0.1i, a=1.1045361",
        solver_mode="none: the repository has no 2D panel solver or Kutta closure in 2D",
        alpha_deg=None,
        differences={"reference_table": ref,
                     "gamma_formula": "Gamma = 4 pi U a sin(alpha + beta), beta = 5.1944289077 deg"},
        notes=["This is the cleanest Kutta/circulation standard model and it needs a sharp "
               "trailing edge in 2D. The 3D wing path closes the wake implicitly on a lifting "
               "surface and cannot represent this conformal airfoil, so the case is parked "
               "until a 2D panel path exists."])


# --------------------------------------------------------------------------- C04
def _thin_wing(aspect_ratio=16.0, chord_panels=8, thickness=0.02):
    from panel import p1_lifting_g1 as g1

    return g1.make_wing(chord_panels=chord_panels, span_panels=int(round(4 * aspect_ratio)),
                        thickness=thickness, camber=0.0, span=float(aspect_ratio),
                        chord=1.0, wake_panels=2, wake_length=5.)


def _solve(wing, alpha_deg, kutta="potential_jump"):
    from panel import p1_lifting_g1 as g1

    started = time.perf_counter()
    state = g1.solve(wing, alpha_deg, kutta=kutta, wall_formulation="collocation")
    return state, time.perf_counter() - started


def case_c04():
    """Thin symmetric wing: the 2*pi lift slope in the 3D lifting-line limit."""
    wing = _thin_wing()
    aspect_ratio = float(np.ptp(wing.xyz[:, 1]) ** 2 / wing.reference_area)
    rows, moments = [], []
    for alpha in (1.0, 2.0, 3.0):
        state, seconds = _solve(wing, alpha)
        span = 0.5 * (wing.xyz[wing.te_faces[:, 0], 1] + wing.xyz[wing.te_faces[:, 1], 1]) \
            if hasattr(wing, "te_faces") else wing.gamma_y
        gamma = np.asarray(state["gamma"], float)
        rows.append(dict(alpha_deg=alpha, CL=float(state["CL"]),
                         CL_gamma=float(state["CL_gamma"]),
                         cp_pressure_min=float(np.min(state["CD_pressure"])),
                         seconds=seconds))
        moments.append((alpha, float(state["CL"]), float(state["CM_y"])))
    slope = float(np.polyfit([r["alpha_deg"] for r in rows], [r["CL"] for r in rows], 1)[0])
    cl_alpha_rad = slope * 180.0 / np.pi
    lifting_line = TWO_PI * aspect_ratio / (aspect_ratio + 2.0)
    return _summary(
        "C04_thin_wing_AR16", status="ran",
        geometry_source="closed symmetric thin wing, thickness 0.02, AR 16, chord 1, 8x64 panels",
        solver_mode="potential_jump Kutta, collocation wall", alpha_deg=2.0,
        samples=dict(chordwise_panels=8, spanwise_panels=64, surface_faces=int(len(wing.xyz))),
        differences=dict(
            cl_alpha_per_rad=cl_alpha_rad,
            two_pi=TWO_PI,
            two_pi_relative=(cl_alpha_rad - TWO_PI) / TWO_PI,
            lifting_line_finite_AR=lifting_line,
            lifting_line_relative=(cl_alpha_rad - lifting_line) / lifting_line,
            note="2*pi is the two-dimensional limit; a finite wing must be lower. The finite-AR "
                 "lifting-line value is the correct comparison for this geometry."),
        levels=[dict(alpha_deg=a, CL=cl, CM_y=cm) for a, cl, cm in moments],
        notes=["Section lift is not extracted here; the wing CL is compared through the "
               "lifting-line relation, which is only valid for attached small-angle flow."])


# --------------------------------------------------------------------------- C05
def _elliptic_planform(aspect_ratio, sections=9, blend=0.85, tip_fraction=0.12):
    """Elliptic planform with a finite tip chord, so the tip strip stays well conditioned."""
    root_chord = 4.0 * SPAN / (np.pi * aspect_ratio)
    theta = np.linspace(0.0, 0.5 * np.pi * 0.97, sections)
    y = 0.5 * SPAN * np.sin(theta)
    chord = root_chord * np.cos(theta)
    y0 = 0.5 * SPAN * blend
    c0 = root_chord * np.sqrt(1.0 - blend ** 2)
    ramp = y > y0
    tip_chord = tip_fraction * root_chord
    chord[ramp] = c0 + (tip_chord - c0) * (y[ramp] - y0) / (y[-1] - y0)
    return y, chord, root_chord


def _grass_wing(y, chord, chord_panels=6, span_panels_per_segment=4):
    from panel.grass_mesh import build_grass_mesh, load_grass_wing

    sections = len(y)
    code = np.r_[np.full(8, 0.17), np.full(8, -0.17), 0.0, 0.0]
    geometry = dict(leading_edge_xyz=np.c_[np.zeros(sections), y, np.zeros(sections)],
                    chord=chord, twist_radians=np.zeros(sections),
                    cst=np.tile(code, (sections, 1)))
    return load_grass_wing(build_grass_mesh(geometry, 1.0, chord_panels=chord_panels,
                                            span_panels_per_segment=span_panels_per_segment))


def _trefftz(wing, state):
    from panel.trefftz_energy_exact import exact_trefftz_piecewise_linear

    gamma = np.asarray(state["gamma"], float)
    return exact_trefftz_piecewise_linear(np.asarray(wing.gamma_y, float),
                                          np.r_[0.0, gamma, 0.0], U=1.0, density=1.0,
                                          reference_area=float(wing.reference_area))


def case_c05(aspect_ratios=(4.0, 6.0, 8.0), alpha_deg=4.0):
    ref = np.array(_reference()["tables"]["C05_elliptic"]["rows"], float)
    rows = []
    for aspect_ratio in aspect_ratios:
        y, chord, root_chord = _elliptic_planform(aspect_ratio)
        wing = _grass_wing(y, chord)
        state, seconds = _solve(wing, alpha_deg)
        drag = _trefftz(wing, state)
        gamma = np.asarray(state["gamma"], float)
        shape = wing.gamma_y[1:-1]
        reference_shape = np.sqrt(np.clip(1.0 - (2.0 * shape / SPAN) ** 2, 0.0, None))
        measured = gamma / np.max(gamma)
        reference_shape = reference_shape / np.max(reference_shape)
        measured_cl_alpha = float(state["CL"]) / np.deg2rad(alpha_deg)
        exact_cl = float(np.interp(aspect_ratio, ref[:, 0], ref[:, 2]))
        exact_slope = TWO_PI * aspect_ratio / (aspect_ratio + 2.0)
        optimum = drag["CL_circulation"] ** 2 / (np.pi * aspect_ratio)
        rows.append(dict(
            aspect_ratio=aspect_ratio, root_chord=root_chord, faces=int(len(wing.xyz)),
            CL=float(state["CL"]), CL_gamma=float(state["CL_gamma"]),
            cl_alpha=measured_cl_alpha, cl_alpha_lifting_line=exact_slope,
            cl_alpha_relative=(measured_cl_alpha - exact_slope) / exact_slope,
            CL_reference_at_4deg=exact_cl,
            cdi_trefftz=drag["CDi_ideal"], cdi_lifting_line=optimum,
            e=float(optimum / drag["CDi_ideal"]),
            gamma_shape_rms=float(np.sqrt(np.mean((measured - reference_shape) ** 2))),
            seconds=seconds))
    refinement = []
    for sections, chordwise, spanwise in ((9, 6, 4), (13, 8, 6), (17, 10, 8)):
        y, chord, root_chord = _elliptic_planform(6.0, sections=sections)
        wing = _grass_wing(y, chord, chord_panels=chordwise,
                           span_panels_per_segment=spanwise)
        state, seconds = _solve(wing, alpha_deg)
        drag = _trefftz(wing, state)
        optimum = drag["CL_circulation"] ** 2 / (np.pi * 6.0)
        measured = np.asarray(state["gamma"], float)
        shape = wing.gamma_y[1:-1]
        reference_shape = np.sqrt(np.clip(1.0 - (2.0 * shape / SPAN) ** 2, 0.0, None))
        reference_shape = reference_shape / np.max(reference_shape)
        measured = measured / np.max(measured)
        refinement.append(dict(
            aspect_ratio=6.0, sections=sections, chordwise_panels=chordwise,
            spanwise_panels_per_segment=spanwise, faces=int(len(wing.xyz)),
            CL=float(state["CL"]),
            cl_alpha=float(state["CL"]) / np.deg2rad(alpha_deg),
            cl_alpha_lifting_line=TWO_PI * 6.0 / (6.0 + 2.0),
            cdi_trefftz=drag["CDi_ideal"], e=float(optimum / drag["CDi_ideal"]),
            gamma_shape_rms=float(np.sqrt(np.mean((measured - reference_shape) ** 2))),
            seconds=seconds))
        print(json.dumps({k: refinement[-1][k] for k in
                          ("sections", "chordwise_panels", "spanwise_panels_per_segment",
                           "faces", "CL", "e", "gamma_shape_rms", "seconds")}), flush=True)
    return _summary(
        "C05_elliptic_AR4_6_8_A4", status="ran",
        geometry_source="elliptic planform, span 8, 9 sections, finite tip chord",
        solver_mode="potential_jump Kutta, collocation wall, Trefftz drag",
        alpha_deg=alpha_deg,
        samples=dict(sections=9, chordwise_panels=6, spanwise_panels_per_segment=4),
        differences=dict(
            cl_reference="LLT CL = 2 pi AR/(AR+2) * alpha",
            cdi_reference="LLT CDi = CL^2/(pi AR), e = 1",
            gamma_reference="Gamma(y)/Gamma_0 = sqrt(1 - (2y/b)^2)"),
        levels=rows,
        refinement=dict(protocol="AR 6, sections/chordwise/spanwise refined together",
                        rows=refinement),
        notes=["The planform keeps a finite tip chord instead of a cusped tip: a cusped tip "
               "sweeps the trailing edge into the stream direction and the potential-jump "
               "closure becomes near singular (docs/KUTTA_CONDITION.md section 8).",
               "CDi is the exact Trefftz integral of the solved circulation, not the near-field "
               "pressure drag, which is unreliable on these coarse chordwise grids."])


# --------------------------------------------------------------------------- C06
def case_c06():
    ref = _reference()["tables"]["C06_ladson_80grit"]
    rows = np.array(ref["rows"], float)
    band = np.abs(rows[:, 0]) <= 8.3
    slope_deg = float(np.polyfit(rows[band, 0], rows[band, 1], 1)[0])
    return _summary(
        "C06_NACA0012_A0_4_10", status="blocked",
        geometry_source="TMR altered NACA 0012 (sharp trailing edge, 11.894% thickness)",
        solver_mode="none yet: needs the 2D path for a like-for-like Cp comparison",
        alpha_deg=None,
        differences=dict(
            ladson_80grit_rows=int(len(rows)),
            ladson_fit_slope_per_deg=slope_deg,
            ladson_fit_slope_per_rad=slope_deg * 180.0 / np.pi,
            fitting_band="|alpha| <= 8.3 deg",
            gregory_cp_alpha0_rows=len(_reference()["tables"]["C06_gregory_cp_alpha0"]["rows"]),
            gregory_cp_alpha10_rows=len(_reference()["tables"]["C06_gregory_cp_alpha10"]["rows"]),
            conditions="Ladson forces Re=6e6 M=0.15 80-grit; Gregory Cp Re=2.88e6, upper surface"),
        notes=["The force data and the pressure data come from different experiments at "
               "different Reynolds numbers; they must never be merged into one 'same test' row.",
               "Viscous total drag is not an inviscid solver target: only small-angle lift, the "
               "pressure distribution and the ideal induced drag are compared.",
               "A 3D high-aspect-ratio stand-in would need the section effective angle "
               "alpha_eff = alpha - alpha_i before comparing with 2D Cp, so it is not used as a "
               "substitute."])


CASES = dict(C01=case_c01, C02=case_c02, C03=case_c03, C04=case_c04, C05=case_c05,
             C06=case_c06)


def _print_case(record):
    print("## %s  [%s]" % (record["case_id"], record["status"]))
    if record["status"] == "blocked":
        print("   " + record["notes"][0])
    for level in record["levels"]:
        print("   " + json.dumps(level, default=float))
    for key, value in record["differences"].items():
        if isinstance(value, dict) and "rows" in value:
            print("   %s: %d reference rows" % (key, len(value["rows"])))
        elif isinstance(value, float):
            print("   %s: %.6g" % (key, value))
        else:
            print("   %s: %s" % (key, value))
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=sorted(CASES),
                        help="case to run; repeatable; default all")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "verification" / "calibration")
    args = parser.parse_args(argv)
    names = args.case or sorted(CASES)
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for name in names:
        started = time.perf_counter()
        record = CASES[name]()
        record["seconds"] = time.perf_counter() - started
        _print_case(record)
        records.append(record)
    (args.out / "summary.json").write_text(json.dumps(records, indent=1, default=float),
                                           encoding="utf-8")
    print("wrote", args.out / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
