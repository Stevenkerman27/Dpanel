# Wake models: representation, evaluator, update

Status: architecture note for the B2 coupled material-wake experiment
(`panel/p1_coupled_material_wake.py`). Nothing here changes a production default:
`velocity_method='analytic_edges'` stays the default, and
`physical_accuracy_accepted` stays false.

## 1. Three independent axes

| axis | question | what selects it | what it does **not** change |
| --- | --- | --- | --- |
| representation | what object *is* the wake? | the wake-model choice (method 1/2/3) | the wall model, the Kutta condition, the solve |
| evaluator | how is the induced velocity of that object computed? | for method 2 only, `area_quadrature`; for any method, quadrature order / fan strips / near-far split | the geometry of the wake sheet, the unknowns, the loads |
| update | how does the wake geometry move between solves? | the B2 outer loop (`iterations`, `relaxation`, `max_backtracks`, geometry gates, per-node step caps) | the velocity model at a fixed geometry |

Confusing these three is what produced the earlier "five peer velocity methods"
surface: `projection_fan` is an area integrator inside method 2, not a wake
representation, and changing `fan_order` changes a number, not a model.

## 2. Method 1 -- P1 doublet sheet, unregularized (`analytic_edges`)

Default and reference. The wake is a continuous P1 doublet sheet: `mu` varies
linearly over every wake triangle, so the sheet carries two physical
contributions,

```
u(x) = 1/(4 pi) [ ∫_boundary  gamma_l(l) × (x - y) / |x-y|^3 dl
                 + ∫_area     gamma_f    × (x - y) / |x-y|^3 dA ]
gamma_l = linearly varying P1 trace on boundary / cut edges
gamma_f = grad_s(mu_f) × n_f     (in-panel distributed vorticity)
```

Status: production default, fully coupled to B2, evaluated exactly (edge
integrals in closed form for the linear trace, area term by quadrature). Its
value **on** the sheet is singular, so numbers taken at a wake node are not a
converged principal value and must not be reported as one. That limitation is
the reason method 2 exists, not a bug in method 1.

## 3. Method 2 -- regularized P1 sheet (`rm_regularized_p1`)

Same representation, same two contributions, one explicit Rosenhead-Moore core
`a`: every denominator becomes `(|x-y|^2 + a^2)^(3/2)`, on both the boundary
term and the area term. The P1 sheet is **not** replaced by constant-strength
edge filaments.

Status: opt-in research / production candidate; requires an explicit
`core_radius`. `area_quadrature` picks the integrator of the area term:

| `area_quadrature` | what it is | when to use |
| --- | --- | --- |
| `legacy_duffy` | fixed tensor-product Gauss-Legendre plus a single-vertex Duffy collapse | default, cheapest; aliases when the core is much smaller than the panel |
| `projection_fan` | projection fan around the closest point, per-panel strips | the mode that removes the alias; the one used for accuracy evidence |
| `projection_fan_adaptive` | projection fan near, composite Duffy far, a-posteriori budget | opt-in, self-gating; reports its own estimates |

`fan_order`, `fan_strips`, `fan_refinement_order`, `fan_refinement_strips`,
`near_ratio`, `adaptive_*` and `quadrature_order` are **numerical settings of
this evaluator**. They never select a model, and none of them may be presented
as a separate method.

## 4. Method 3 -- Lagrangian vortex filaments (documented, not wired)

A distinct representation: oriented filaments carrying circulation, induced
velocity by Biot-Savart / Rosenhead-Moore segment summation,

```
u(x) = Σ_filaments Gamma/(4 pi) ∫ dl × (x - y) / (|x-y|^2 + a^2)^(3/2)
```

It needs its own circulation transfer, filament connectivity, curved-filament
self-induction treatment and a P1-to-filament mapping before it can be coupled.
The segment kernel from PR #83 is a component and an oracle, not a solver. It is
deliberately **not** accepted as a `velocity_method`; `panel/wake_velocity_methods.py`
refuses it with an explanatory message.

## 5. Deprecated backend (`legacy_gauss`)

The pre-2026-10-30 edge Gauss evaluator. Kept only so frozen snapshots that
recorded `sampling_kernel='legacy_gauss'` still replay; it is not offered in the
CLI choices and is not documented as a model.

## 6. Numerical vs physical controls

| control | kind | effect |
| --- | --- | --- |
| `velocity_method` | physical (model) | chooses method 1 or 2 (3 reserved) |
| `core_radius` | physical (model parameter) | sets the RM core of method 2 |
| `area_quadrature` | numerical (method 2 evaluator) | which integrator computes the area term |
| `quadrature_order` | numerical | legacy edge Gauss order / Duffy order; ignored by method 1 |
| `fan_order`, `fan_strips` | numerical | resolution of the projection fan |
| `fan_refinement_*` | numerical (gate) | the refined configuration the sampling gate compares against |
| `near_ratio`, `adaptive_*` | numerical | near/far split and budget of `projection_fan_adaptive` |
| wake node count (mesh/geometry) | discrete geometry | where the sheet is sampled; **not** a velocity setting |
| `iterations`, `relaxation`, `max_backtracks` | update | how the mixed-inverse/kinematic map moves the sheet |

Wake node count and wake iteration count are properties of the geometry and of
the B2 loop. They change the answer by changing the discretisation and the
fixed point, so they must be reported next to any velocity-method comparison;
`fan_order`/`fan_strips` change only the quadrature error at fixed geometry.

## 7. What is coupled, and what is not

Coupled: wall and wake share the P1 potential representation; the wall solve is
`solve_multi` (Dirichlet potential trace), the pressure-Kutta condition ties the
sheet strength to the wall loading, and the B2 loop moves the wake with the
kinematic map under the geometry gates (root/attachment/area/fold/step/wall/
intersection) and per-node step caps. See
`docs/ISSUE84_T5_CONTACT_AND_STEP_CONTROL.md` for the gate semantics.

Not coupled: method 3; any filament-based geometry update; the Galerkin wall
weak form (implemented and measured in issue #71 but not yet the B2 wall
solver); a physical-accuracy claim of any kind.

## 8. API and compatibility

* `velocity_method` accepts `analytic_edges` (default) and `rm_regularized_p1`;
  `legacy_gauss` is accepted only to replay old snapshots.
* `area_quadrature` is only valid with method 2 and only with
  `legacy_duffy`, `projection_fan` or `projection_fan_adaptive`; asking for a fan
  with method 1 raises instead of being silently ignored.
* `python -m panel.run_p1_coupled_material_wake --describe-velocity-methods`
  prints the model table below.
* `panel/wake_velocity_methods.py` is the single source of truth for the model
  list, statuses, numerical-control descriptions and the error messages; the
  CLI choices and the solver guards both read it.

```
method                 index  status                            coupled
legacy_gauss               0  deprecated, replay compatibility only False
analytic_edges             1  reference / production default   True
rm_regularized_p1          2  opt-in research / production candidate True
lagrangian_filaments       3  future model, not coupled        False
```

Tests: `test/test_wake_velocity_methods.py` (registry, fail-closed guards, CLI
table, doc contract), plus the existing `test_p1_coupled_material_wake.py`,
`test_p1_regularized_sheet_velocity.py`, `test_p1_fan_quadrature_gate.py`,
`test_p1_adaptive_area.py`.
