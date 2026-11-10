"""Re-runnable acceptance comparisons: same calculation, two or more settings, report the difference.

There are no thresholds here on purpose.  A case runs one physical situation under
several settings (closures, mesh resolutions, planforms) and reports each quantity next
to the others plus the relative difference from the baseline setting, optionally next to
an analytic reference (2*pi lift slope, unit span efficiency).  The report is a table of
numbers: the reader, not the tool, decides whether a difference is acceptable.

Output: a report.json (machine readable, with environment and git commit) and a
report.md (the same numbers as tables) under --out, plus the markdown on stdout.

Usage:
    python -m tools.acceptance --list
    python -m tools.acceptance
    python -m tools.acceptance --case closures --case lift_slope
    python -m tools.acceptance --out runs/verification/acceptance/2026-11-10

Adding a case: decorate a function with @case(name, title, description); it returns a
dict with configs, baseline and rows, where every row is
    dict(quantity="CL", unit="-", values={"pressure": 0.34, "potential_jump": 0.35})
and may add reference=dict(label="2*pi", value=6.283).
"""
from __future__ import annotations

import argparse
import functools
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from panel import p1_lifting_g1 as g1  # noqa: E402
from panel.grass_mesh import build_grass_mesh, load_grass_wing  # noqa: E402
from panel.trefftz_energy_exact import exact_trefftz_piecewise_linear  # noqa: E402

ALPHA_DEG = 2.0
TWO_PI = 2.0 * np.pi
SPAN = 8.0
ROOT_CHORD = 2.0

CASES = {}


def case(name, title, description):
    """Register one comparison; see the module docstring for the return shape."""

    def register(run):
        CASES[name] = dict(name=name, title=title, description=description, run=run)
        return run

    return register


def _wing(aspect_ratio, chord_panels=8, thickness=0.12):
    return g1.make_wing(chord_panels=chord_panels, span_panels=int(round(4 * aspect_ratio)),
                        thickness=thickness, camber=0.0, span=float(aspect_ratio),
                        chord=1.0, wake_panels=2, wake_length=5.)


def _solve(wing, kutta, alpha_deg=ALPHA_DEG):
    started = time.perf_counter()
    state = g1.solve(wing, alpha_deg, kutta=kutta, wall_formulation="collocation")
    return state, time.perf_counter() - started


def _trefftz(wing, state):
    gamma = np.asarray(state["gamma"], float)
    return exact_trefftz_piecewise_linear(
        np.asarray(wing.gamma_y, float), np.r_[0.0, gamma, 0.0], U=1.0, density=1.0,
        reference_area=float(wing.reference_area))


def _span_efficiency(wing, alpha_deg=4.0, kutta="potential_jump"):
    state, _ = _solve(wing, kutta, alpha_deg)
    drag = _trefftz(wing, state)
    aspect_ratio = float(np.ptp(wing.xyz[:, 1])) ** 2 / float(wing.reference_area)
    optimum = drag["CL_circulation"] ** 2 / (np.pi * aspect_ratio)
    return dict(e=optimum / drag["CDi_ideal"], CL_alpha=float(state["CL"]) / np.deg2rad(alpha_deg),
                AR=aspect_ratio)


def _sections(sections=9, blend=0.85, tip_chord=0.24):
    theta = np.linspace(0.0, 0.5 * np.pi * 0.97, sections)
    y = 0.5 * SPAN * np.sin(theta)
    chord = ROOT_CHORD * np.cos(theta)
    y0 = 0.5 * SPAN * blend
    c0 = ROOT_CHORD * np.sqrt(1.0 - blend ** 2)
    ramp = y > y0
    chord[ramp] = c0 + (tip_chord - c0) * (y[ramp] - y0) / (y[-1] - y0)
    return y, chord


def _grass_wing(y, chord, chord_panels=6, span_panels_per_segment=4):
    sections = len(y)
    code = np.r_[np.full(8, 0.17), np.full(8, -0.17), 0.0, 0.0]
    geometry = dict(leading_edge_xyz=np.c_[np.zeros(sections), y, np.zeros(sections)],
                    chord=chord, twist_radians=np.zeros(sections),
                    cst=np.tile(code, (sections, 1)))
    mesh = build_grass_mesh(geometry, 1.0, chord_panels=chord_panels,
                            span_panels_per_segment=span_panels_per_segment)
    return load_grass_wing(mesh)


@case("closures", "Kutta closures on one wing",
      "Pressure Kutta and the Morino potential-jump closure on the same 8-by-64 panel grid. "
      "Both closures must agree in the fine-mesh limit; the gap on a coarse grid says how "
      "much of the answer is closure error.")
def closures():
    wing = _wing(16.0, 8)
    configs = ["pressure", "potential_jump"]
    numbers = {}
    for kutta in configs:
        state, seconds = _solve(wing, kutta)
        drag = _trefftz(wing, state)
        numbers[kutta] = dict(CL=float(state["CL"]), CD=float(state["CD_pressure"]),
                              CL_gamma=float(state["CL_gamma"]),
                              CDi=float(drag["CDi_ideal"]),
                              CL_circulation=float(drag["CL_circulation"]), seconds=seconds)
    rows = [dict(quantity=key, unit=unit, values={k: numbers[k][key] for k in configs})
            for key, unit in (("CL", "-"), ("CL_gamma", "-"), ("CD", "-"), ("CDi", "-"),
                              ("CL_circulation", "-"), ("seconds", "s"))]
    return dict(configs=configs, baseline="pressure", rows=rows,
                notes=["make_wing(AR 16, chord_panels 8, span_panels 64), alpha 2 deg, collocation wall."])


@case("refinement", "Chordwise refinement by closure",
      "Same wing at 4, 8 and 16 chordwise panels per section for both closures. "
      "A converged method changes little; this table is what makes a divergent one visible.")
def refinement():
    configs = ["%s c%d" % (kutta, panels) for kutta in ("pressure", "potential_jump") for panels in (4, 8, 16)]
    numbers = {}
    for kutta in ("pressure", "potential_jump"):
        for panels in (4, 8, 16):
            wing = _wing(16.0, panels)
            state, seconds = _solve(wing, kutta)
            numbers["%s c%d" % (kutta, panels)] = dict(
                CL=float(state["CL"]), CL_alpha=float(state["CL"]) / np.deg2rad(ALPHA_DEG),
                seconds=seconds)
    rows = [dict(quantity=key, unit=unit, values={k: numbers[k][key] for k in configs})
            for key, unit in (("CL", "-"), ("CL_alpha", "1/rad"), ("seconds", "s"))]
    return dict(configs=configs, baseline="potential_jump c8", rows=rows,
                notes=["The baseline is the 8-panel potential-jump solve; the change from 8 to 16 panels is the comparison that matters."])


@functools.lru_cache(maxsize=1)
def _lift_numbers():
    aspect_ratios = [4.0, 8.0, 16.0, 24.0]
    slopes = {}
    fits = {}
    for kutta in ("pressure", "potential_jump"):
        values = []
        for aspect_ratio in aspect_ratios:
            state, seconds = _solve(_wing(aspect_ratio), kutta)
            values.append(float(state["CL"]) / np.deg2rad(ALPHA_DEG))
        slopes[kutta] = values
        gain, intercept = np.polyfit(1.0 / np.asarray(aspect_ratios), 1.0 / np.asarray(values), 1)
        fits[kutta] = dict(a_inf=float(1.0 / intercept), slope=float(gain))
    return aspect_ratios, slopes, fits


@case("lift_slope", "Lift-curve slope at four aspect ratios",
      "CL_alpha at aspect ratios 4-24 for both closures, one column per aspect ratio. "
      "The differences show the finite-aspect-ratio effect and the closure gap at the same grid.")
def lift_slope():
    aspect_ratios, slopes, _ = _lift_numbers()
    configs = ["AR %.0f" % ar for ar in aspect_ratios]
    rows = [dict(quantity="CL_alpha " + kutta, unit="1/rad",
                 values={"AR %.0f" % ar: slopes[kutta][i] for i, ar in enumerate(aspect_ratios)})
            for kutta in ("pressure", "potential_jump")]
    return dict(configs=configs, baseline="AR 24", rows=rows,
                notes=["Baseline AR 24 is the closest to the two-dimensional limit; the delta columns are not a closure comparison."])


@case("lift_slope_fit", "Lifting-line fit against 2*pi",
      "Fits 1/CL_alpha = 1/a_inf + slope/AR over the four aspect ratios of the lift_slope case. "
      "The analytic anchors are a_inf = 2*pi and slope = 1/pi.")
def lift_slope_fit():
    _, slopes, fits = _lift_numbers()
    rows = [dict(quantity="a_inf fit", unit="1/rad", values={k: fits[k]["a_inf"] for k in fits},
                 reference=dict(label="2*pi", value=float(TWO_PI))),
            dict(quantity="d(1/a)/d(1/AR)", unit="-", values={k: fits[k]["slope"] for k in fits},
                 reference=dict(label="1/pi", value=float(1.0 / np.pi))),
            dict(quantity="CL_alpha at the highest AR", unit="1/rad",
                 values={k: slopes[k][-1] for k in fits})]
    return dict(configs=["pressure", "potential_jump"], baseline="pressure", rows=rows,
                notes=["Each fit is a straight-line fit through four points (aspect ratios 4, 8, 16, 24), not an asymptotic limit."])


@case("span_efficiency", "Elliptic planform against the Trefftz optimum",
      "Elliptic (bounded tip) and rectangular wings of the same span and area, potential-jump "
      "closure, CDi from the Trefftz plane. e = 1 is the elliptic optimum.")
def span_efficiency():
    elliptic = _grass_wing(*_sections())
    span = float(np.ptp(elliptic.xyz[:, 1]))
    rectangular = g1.make_wing(chord_panels=6, span_panels=32, thickness=0.12, camber=0.0,
                               span=span, chord=float(elliptic.reference_area) / span,
                               wake_panels=2, wake_length=5.)
    numbers = {"elliptic": _span_efficiency(elliptic), "rectangular": _span_efficiency(rectangular)}
    configs = ["elliptic", "rectangular"]
    rows = [dict(quantity="e", unit="-", values={k: numbers[k]["e"] for k in configs},
                 reference=dict(label="elliptic optimum", value=1.0)),
            dict(quantity="CL_alpha", unit="1/rad", values={k: numbers[k]["CL_alpha"] for k in configs}),
            dict(quantity="aspect ratio", unit="-", values={k: numbers[k]["AR"] for k in configs})]
    return dict(configs=configs, baseline="elliptic", rows=rows,
                notes=["Elliptic planform: 9 GRASS sections, chord_panels 6, 4 span panels per segment, 1560 faces.",
                       "Only the Trefftz-plane induced drag is used; the near-field CD is negative on this case at this mesh."])


@case("wake_material", "Wake material identity and rigid-flow response",
      "The wake material mapping must keep the declared trailing-edge attachment nodes and "
      "must not move any wake node under a uniform axial flow.  Deviations from those "
      "expectations are what fix/issue113-b3-material-wake-map changes.")
def wake_material():
    from dataclasses import replace

    from panel.p1_lifting_g2 import combine_wings
    from panel.p1_steady_free_wake import _one_wake_target, prepare_wake_material

    wing = g1.make_wing(chord_panels=3, span_panels=4, wake_panels=3)

    def material_of(wall, wake):
        group = combine_wings([replace(wing, xyz=wall, wake_xyz=wake)]).groups[0]
        return group, prepare_wake_material(group)

    def swept(sweep):
        wall, wake = wing.xyz.copy(), wing.wake_xyz.copy()
        wall[:, 0] += sweep * abs(wall[:, 1])
        wake[:, 0] += sweep * abs(wake[:, 1])
        return material_of(wall, wake)

    group, material = material_of(wing.xyz.copy(), wing.wake_xyz.copy())
    declared = np.sort(np.asarray(group.wake_attachment_nodes, int))
    used_attachments = material.roots if material.attachments is None else np.asarray(material.attachments, int)
    used_attachments = np.sort(used_attachments)
    rows = [
        dict(quantity="declared attachments", unit="nodes", values={"current": float(len(declared))}),
        dict(quantity="attachments used by the mapping", unit="nodes",
             values={"current": float(len(used_attachments))}),
        dict(quantity="used but not declared", unit="nodes",
             values={"current": float(len(np.setdiff1d(used_attachments, declared)))},
             reference=dict(label="expected", value=0.0)),
        dict(quantity="declared but unused", unit="nodes",
             values={"current": float(len(np.setdiff1d(declared, used_attachments)))},
             reference=dict(label="expected", value=0.0)),
        dict(quantity="attachments that are also graph roots", unit="nodes",
             values={"current": float(len(np.intersect1d(used_attachments, np.asarray(material.roots, int))))}),
    ]
    for label, sweep in (("unswept", 0.0), ("swept 0.2", 0.2)):
        group, material = swept(sweep)
        velocity = np.tile([1.0, 0.0, 0.0], (len(group.wake_xyz), 1))
        target, frozen = _one_wake_target(group, material, velocity, min_axial=0.08, freeze=True)
        nodes = np.unique(group.wake_faces)
        rows.append(dict(quantity="max axial-flow displacement, " + label, unit="chord",
                         values={"current": float(np.abs(target[nodes] - group.wake_xyz[nodes]).max())},
                         reference=dict(label="expected", value=0.0)))
        rows.append(dict(quantity="frozen nodes, " + label, unit="nodes",
                         values={"current": float(len(frozen))},
                         reference=dict(label="expected", value=0.0)))
    return dict(configs=["current"], baseline="current", rows=rows,
                notes=["Reference values are the analytic expectation, not another run.",
                       "make_wing(chord_panels 3, span_panels 4, wake_panels 3); uniform velocity [1,0,0]; swept geometry adds 0.2*|y| to x before the habitual wake is rebuilt."])

def _environment():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, timeout=20).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                                    text=True, timeout=20).stdout.strip())
    except Exception:
        commit, dirty = "", None
    import scipy
    import torch
    return dict(commit=commit, dirty=dirty, python=sys.version.split()[0],
                platform=platform.platform(), processor=platform.processor(),
                numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__,
                torch_threads=torch.get_num_threads(),
                generated=time.strftime("%Y-%m-%dT%H:%M:%S"))


def run_cases(names):
    report = dict(environment=_environment(), cases=[])
    for name in names:
        spec = CASES[name]
        started = time.perf_counter()
        print("running %s ..." % name, flush=True)
        result = spec["run"]()
        result.update(name=name, title=spec["title"], description=spec["description"])
        result["seconds"] = round(time.perf_counter() - started, 1)
        report["cases"].append(result)
        print("  done in %.1f s" % result["seconds"], flush=True)
    return report


def _format(value):
    if isinstance(value, float):
        if value != 0.0 and (abs(value) < 1e-3 or abs(value) >= 1e4):
            return "%.6g" % value
        return "%.6f" % value
    return str(value)


def _row_cells(row, configs, baseline):
    cells = []
    for config in configs:
        value = row["values"].get(config)
        cells.append("-" if value is None else _format(value))
    deltas = []
    base = row["values"].get(baseline)
    for config in configs:
        if config == baseline:
            continue
        value = row["values"].get(config)
        if value is None or not base:
            deltas.append("")
        else:
            deltas.append("%+.4g (%+.2f%%)" % (value - base, 100.0 * (value - base) / base))
    return cells, deltas


def render_markdown(report):
    env = report["environment"]
    lines = ["# Acceptance report", "",
             "Commit %s%s, %s, Python %s, NumPy %s, SciPy %s, torch %s (%d threads), %s." % (
                 (env["commit"] or "unknown")[:12], " (dirty tree)" if env["dirty"] else "",
                 env["platform"], env["python"], env["numpy"], env["scipy"], env["torch"],
                 env["torch_threads"], env["generated"]),
             "",
             "No thresholds: every table reports the numbers of each setting, the absolute and",
             "relative difference from the baseline, and an analytic reference where one exists.",
             ""]
    for entry in report["cases"]:
        configs = entry["configs"]
        baseline = entry["baseline"]
        lines += ["## %s" % entry["title"], "", entry["description"], "",
                  "configs: %s; baseline: %s; %.1f s." % (", ".join(configs), baseline, entry["seconds"]), ""]
        has_reference = any(row.get("reference") for row in entry["rows"])
        header = ["quantity", "unit"] + configs
        if len(configs) > 1:
            header += ["delta " + config for config in configs if config != baseline]
        if has_reference:
            header.append("reference")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in entry["rows"]:
            cells, deltas = _row_cells(row, configs, baseline)
            reference = row.get("reference")
            reference_cell = ""
            if reference:
                value = row["values"].get(baseline)
                relative = "" if not value else " (%+.2f%%)" % (100.0 * (value - reference["value"]) / reference["value"])
                reference_cell = "%s %s%s" % (reference["label"], _format(reference["value"]), relative)
            line = ["%s" % row["quantity"], "%s" % row["unit"]] + cells
            if len(configs) > 1:
                line += deltas
            if has_reference:
                line.append(reference_cell)
            lines.append("| " + " | ".join(line) + " |")
        lines.append("")
        if entry.get("notes"):
            lines += ["notes:"] + ["- " + note for note in entry["notes"]] + [""]
    return "\n".join(lines)


def _plate(n=13):
    """Jittered flat plate with a mid-chord potential cut, as an offline wing patch."""
    yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, n), np.linspace(-1.0, 1.0, n))
    points = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(n * n)))
    j0 = n // 2
    inner = ((np.abs(points[:, 0]) < 1.0) & (np.abs(points[:, 1]) < 1.0)
             & (np.abs(np.arange(n * n) % n - j0) > 0))
    rng = np.random.default_rng(19)
    points[inner, :2] += rng.uniform(-0.03, 0.03, (inner.sum(), 2))
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            faces.extend(([a, a + n, a + n + 1], [a, a + n + 1, a + 1]))
    cuts = np.array([[i * n + j0, (i + 1) * n + j0] for i in range(n - 1)], dtype=np.int64)
    return points, np.array(faces), cuts


@case("pressure_reconstruction", "Face-gradient reconstruction on a frozen field",
      "The issue #9 comparison, offline: the production unweighted normal equations, the same "
      "fit by untruncated QR, and an inverse-distance-weighted QR fit all see the identical "
      "field and stencil. Nothing is clipped, smoothed, truncated or regularized. The affine "
      "field has an exact answer; the curved field shows the template model error.")
def pressure_reconstruction():
    from panel.drag_consistency_audit import pressure_integral_audit
    from panel.pressure_reconstruction_probe import (
        face_frames, two_ring_neighbors, fit_gradients, cp)

    points, faces, cuts = _plate()
    centers, normal, tangent, second = face_frames(points, faces)
    ids = two_ring_neighbors(faces, cuts)
    flow = np.array([np.cos(np.deg2rad(4.0)), 0.0, np.sin(np.deg2rad(4.0))])
    fields = {
        "affine": (.05 * centers[:, 0] - .02 * centers[:, 1],
                   np.column_stack((np.full(len(faces), .05),
                                    np.full(len(faces), -.02), np.zeros(len(faces))))),
        "curved": (.05 * centers[:, 0] - .02 * centers[:, 1] + .15 * centers[:, 0] ** 2,
                   np.column_stack((.05 + .3 * centers[:, 0],
                                    np.full(len(faces), -.02), np.zeros(len(faces))))),
    }
    arms = ["normal", "QR", "distance-weighted QR"]
    modes = ["normal", "qr", "weighted"]

    def loads(cp_values):
        return pressure_integral_audit(
            points, faces, cp_values, flow, density=1.0, reference_area=2.0,
            reference_chord=1.0, reference_point=np.zeros(3))

    rows, ratios = [], {}
    for name, (mu, gradient) in fields.items():
        exact = cp(gradient, normal, flow)
        reference = loads(exact)
        cl = {}
        cm = {}
        rel = {}
        worst = {}
        for arm, mode in zip(arms, modes):
            gradient_fit, ratio, rejected = fit_gradients(points, faces, mu, ids, mode=mode)
            ratios[arm] = (ratio, rejected)
            values = cp(gradient_fit, normal, flow)
            audit = loads(values)
            cl[arm] = audit["CL_pressure"]
            cm[arm] = audit["CM_y"]
            rel[arm] = float(np.linalg.norm(values - exact) / np.linalg.norm(exact))
            worst[arm] = float(np.max(np.abs(values - exact)))
        rows.append(dict(quantity="CL (%s field)" % name, unit="-", values=cl,
                         reference=dict(label="analytic", value=reference["CL_pressure"])))
        rows.append(dict(quantity="CM_y (%s field)" % name, unit="-", values=cm,
                         reference=dict(label="analytic", value=reference["CM_y"])))
        rows.append(dict(quantity="Cp L2 error, relative to analytic (%s)" % name, unit="-",
                         values=rel))
        rows.append(dict(quantity="max |Cp - analytic| (%s)" % name, unit="-", values=worst))
    rows.append(dict(quantity="min stencil singular ratio", unit="-",
                     values={arm: float(np.nanmin(ratios[arm][0])) for arm in arms}))
    rows.append(dict(quantity="median stencil singular ratio", unit="-",
                     values={arm: float(np.nanmedian(ratios[arm][0])) for arm in arms}))
    rows.append(dict(quantity="faces skipped for rank", unit="-",
                     values={arm: float(ratios[arm][1]) for arm in arms}))
    return dict(configs=arms, baseline="normal", rows=rows,
                notes=["Affine field: every arm reproduces the analytic gradient, so a nonzero "
                       "error here means the stencil or the solve is wrong, not the template.",
                       "Curved field: the residual is the affine template model error, which is "
                       "the quantity issue #9 asks about. A weighted fit that does not reduce it "
                       "is reported as no improvement, not tuned."])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Re-runnable acceptance comparisons")
    parser.add_argument("--case", action="append", choices=sorted(CASES), default=None,
                        help="run only this case (repeatable); default runs every case")
    parser.add_argument("--out", default=None,
                        help="output directory; default runs/verification/acceptance/<stamp>")
    parser.add_argument("--list", action="store_true", help="list the cases and exit")
    parser.add_argument("--json-only", action="store_true", help="skip the markdown on stdout")
    args = parser.parse_args(argv)
    if args.list:
        for name in sorted(CASES):
            print("%-15s %s" % (name, CASES[name]["title"]))
        return 0
    names = args.case or sorted(CASES)
    report = run_cases(names)
    out = Path(args.out) if args.out else Path("runs/verification/acceptance") / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = render_markdown(report)
    (out / "report.md").write_text(markdown + "\n", encoding="utf-8")
    print()
    if not args.json_only:
        print(markdown)
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
