# Acceptance comparisons

`tools/acceptance.py` runs the same aerodynamic situation under two or more settings and
prints a table of numbers: every setting side by side, the absolute and relative
difference from a baseline setting, and an analytic reference where one exists.

There are **no thresholds** in it. That is deliberate. A threshold silently encodes one
mesh, one closure and one machine, and it turns every physics change into an argument
about the number instead of a reading of the trend. The report is the evidence; the
reader decides whether a difference is acceptable. A regression still shows up, because
the *difference* moves: the gap between closures that used to shrink with refinement can
start to grow, and the table says so.

## Running it

    python -m tools.acceptance                      # every case, about 2 minutes
    python -m tools.acceptance --list               # case names and titles
    python -m tools.acceptance --case closures --case lift_slope
    python -m tools.acceptance --out runs/verification/acceptance/<stamp>

The output directory receives `report.json` (machine readable: environment, git commit,
dirty flag, every row) and `report.md` (the same numbers as tables). With no `--out` the
directory defaults to `runs/verification/acceptance/<timestamp>`, which is not tracked.
`--json-only` suppresses the markdown on stdout.

Single environment, as everywhere else in this repository:
`C:/Users/zyx20/anaconda3/envs/myml/python.exe`, run from the repository root with
`PYTHONPATH=.` (or `python -m` from the root, which sets it up).

## Cases

| case | what varies | what the table reports |
| --- | --- | --- |
| `closures` | Kutta closure: pressure vs Morino potential jump | CL, circulation lift, near-field CD, Trefftz CDi, forward time, on one 8-by-64 panel grid |
| `lift_slope` | aspect ratio 4, 8, 16, 24 | CL_alpha per aspect ratio, both closures |
| `lift_slope_fit` | closure | a_inf and d(1/a)/d(1/AR) from a straight-line fit over those four aspect ratios |
| `refinement` | chordwise panels 4, 8, 16; both closures | CL, CL_alpha, forward time |
| `span_efficiency` | elliptic (bounded tip) vs rectangular planform, same span and area | e = CDi_opt/CDi from the Trefftz plane, CL_alpha, aspect ratio |
| `wake_material` | uniform axial flow, flat and swept geometry | attachment identity, wake-node displacement under rigid flow, frozen nodes |

Walls are collocation and alpha is 2 degrees except where a case says otherwise. The
potential-jump closure is a square Newton solve, the pressure closure is an overdetermined
least-squares problem; see `docs/KUTTA_CONDITION.md` for why the two disagree on coarse
meshes and why the default is still `pressure`.

## Reading the 2026-11-10 report

The report of 2026-11-10 (commit 04a9913 plus the uncommitted harness) reads:

- **Closures, AR 16, 8 chordwise panels** differ in lift by +20.64% (0.162879 pressure
  against 0.196502 potential jump) and in Trefftz CDi by +50.22%.
- **Chordwise refinement** makes them diverge differently: pressure CL goes 0.189956 ->
  0.162879 -> 0.074809 from c4 to c16 (-61.93% from c8), potential jump 0.209380 ->
  0.196502 -> 0.189504 (-3.56%). The potential-jump column is the one that stays put.
- **Lifting-line fit**: a_inf = 6.621889 for the potential jump (+5.39% against 2*pi) and
  6.955299 for pressure (+10.70%); the slope is 0.426935 against 0.318310 (1/pi) for the
  potential jump and 1.125701 for pressure.
- **Span efficiency, elliptic planform**: e = 0.978154 against 0.965248 for a rectangular
  wing of the same span and area; CL_alpha 4.434095 against the lifting line 4.518.
- **Wake material**: with this construction (`replace()` rebuilds the habitual wake, so the
  material labels are kept by hand) there is no displacement and no attachment mismatch.
  The mesh-edge marching path that loses transverse offsets is not exercised here; see
  the note below.

Near-field CD is negative in several cases because the chordwise grid is too coarse to
resolve leading-edge suction. That is why the drag comparisons use the Trefftz plane.

## Relationship to the GitHub acceptance work

`fix/issue113-b3-material-wake-map` adds three acceptance tests to
`test/test_wake_sensitivity_experiments.py`: declared attachment identity, a formerly
`strict xfail` axial-sweep case, and an oblique-flow affine drift check. The
`wake_material` case here mirrors the first and second as reported numbers. It does not
reproduce the branch defect yet, because the defect lives in the mesh-edge marching that
builds the material labels, and the case builds them by hand to keep the comparison about
the mapping instead of about the rebuild. Issue #113 also asks for wake displacement
sensitivity, wake length and resolution sweeps, root-row constraints and a conditioning
response; those are not cases yet.

## Adding a case

    @case("name", "Title", "What the difference means.")
    def name():
        return dict(configs=["a", "b"], baseline="a",
                    rows=[dict(quantity="CL", unit="-",
                               values={"a": 0.2, "b": 0.25},
                               reference=dict(label="2*pi", value=6.283))],
                    notes=["one line of context about the mesh and flow"])

Registered cases are listed by `--list` and run in name order. A row without a value for
a config renders as a dash; a reference whose value is zero omits the percentage, since a
relative error against zero is meaningless.

Tests: `pytest test/test_acceptance.py` covers the registry, the `--list` output and the
table formatting; it never solves anything.
