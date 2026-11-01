"""Public surface of the wake induced-velocity models.

Three *models* exist; only the first two are wired to the coupled B2 solve, and
only the first is the production default.  Quadrature order, fan strips, near/far
splits and adaptive budgets are *numerical controls of a model*, never models of
their own -- this module is the single place where that distinction is recorded
so the CLI, the solver and the documentation cannot drift apart.  See
`docs/ARCHITECTURE_WAKE_MODELS.md` for the full page.

Method 1 -- `analytic_edges` (default, reference)
    Representation: continuous P1 doublet sheet, i.e. mu varies linearly over each
    wake triangle; the sheet carries two contributions,
        u(x) = 1/(4 pi) [ integral_boundary (gamma_l dl x (x - y)) / |x-y|^3
                          + integral_area gamma_f x (x - y) / |x-y|^3 dA ],
    with gamma_l the linearly varying boundary/cut trace and gamma_f =
    grad_s(mu_f) x n_f the in-panel distributed vorticity.
    Status: reference implementation, default for every production path.  Its
    on-sheet value is singular, so it must not be quoted as a converged
    principal value at its own nodes.

Method 2 -- `rm_regularized_p1` (research / production candidate)
    Same representation and the same two contributions, with one explicit
    Rosenhead-Moore denominator (|x-y|^2 + a^2)^(3/2) applied to both.
    `area_quadrature` selects how the area term is integrated
    (`legacy_duffy`, `projection_fan`, `projection_fan_adaptive`); `fan_order`,
    `fan_strips`, `fan_refinement_*`, `near_ratio`, `adaptive_*` are numerical
    settings of that backend.  Status: opt-in, requires an explicit
    `core_radius`.

Method 3 -- `lagrangian_filaments` (future model, NOT wired)
    Representation: oriented filaments carrying circulation, u(x) =
    sum Gamma/(4 pi) integral dl x (x - y) / (|x-y|^2 + a^2)^(3/2).  It needs its
    own circulation transfer, filament connectivity, self-induction treatment and
    a P1-to-filament mapping before it may be coupled.  The segment kernel of
    PR #83 is a component/oracle, not a solver; this module documents the model
    and deliberately refuses to accept it as a `velocity_method`.

Deprecated backend -- `legacy_gauss`
    The pre-2026-10-30 edge Gauss evaluator; kept only so frozen snapshots that
    recorded `sampling_kernel='legacy_gauss'` still replay.  It is not a peer
    method: it is not offered in the CLI choices and not documented as a model.
"""

from __future__ import annotations

PUBLIC_METHODS = ("analytic_edges", "rm_regularized_p1")
DOCUMENTED_ONLY_METHODS = ("lagrangian_filaments",)
DEPRECATED_METHODS = ("legacy_gauss",)
ACCEPTED_METHODS = PUBLIC_METHODS + DEPRECATED_METHODS
DEFAULT_METHOD = "analytic_edges"
AREA_QUADRATURES = ("legacy_duffy", "projection_fan", "projection_fan_adaptive")
DEFAULT_AREA_QUADRATURE = "legacy_duffy"
ARCHITECTURE_DOC = "docs/ARCHITECTURE_WAKE_MODELS.md"

METHOD_SPECS = {
    "analytic_edges": {
        "index": 1,
        "name": "P1 doublet sheet (unregularized)",
        "status": "reference / production default",
        "representation": "continuous P1 doublet sheet over wake triangles",
        "evaluator": "linear boundary trace + in-panel grad_s(mu) x n",
        "coupled": True,
        "selectable": True,
        "regularization": None,
        "notes": "singular on its own sheet; not a converged PV at the nodes",
    },
    "rm_regularized_p1": {
        "index": 2,
        "name": "Regularized P1 sheet (Rosenhead-Moore core)",
        "status": "opt-in research / production candidate",
        "representation": "continuous P1 doublet sheet over wake triangles",
        "evaluator": "same two contributions with (|r|^2 + a^2)^(3/2) on both",
        "coupled": True,
        "selectable": True,
        "regularization": "explicit core_radius a",
        "notes": "area_quadrature selects the area integrator; fan order/strips are "
                 "numerical controls",
    },
    "lagrangian_filaments": {
        "index": 3,
        "name": "Lagrangian vortex-filament wake",
        "status": "future model, not coupled",
        "representation": "oriented filaments carrying circulation",
        "evaluator": "Biot-Savart / Rosenhead-Moore segment summation",
        "coupled": False,
        "selectable": False,
        "regularization": "explicit segment core radius a",
        "notes": "needs circulation transfer, connectivity, self-induction and a "
                 "P1-to-filament mapping; PR #83 segment kernel is an oracle only",
    },
    "legacy_gauss": {
        "index": 0,
        "name": "Pre-2026-10-30 edge Gauss evaluator",
        "status": "deprecated, replay compatibility only",
        "representation": "P1 sheet, legacy edge quadrature",
        "evaluator": "edge Gauss sums",
        "coupled": False,
        "selectable": False,
        "regularization": None,
        "notes": "not a peer method; exists so frozen snapshots still replay",
    },
}

NUMERICAL_CONTROLS = {
    "quadrature-order": "legacy edge Gauss order; the Duffy area order under "
                        "area_quadrature='legacy_duffy'; ignored by analytic_edges",
    "area_quadrature": "which integrator evaluates the RM-P1 area term "
                       "(legacy_duffy | projection_fan | projection_fan_adaptive)",
    "fan-order": "projection-fan quadrature order (points per fan family), "
                 "a numerical accuracy/cost setting",
    "fan-strips": "number of fan strips the capped triangle is split into, "
                  "a numerical accuracy/cost setting",
    "fan-refinement-order/strips": "the refined configuration the a-priori gate "
                                   "compares against; must differ from the base",
    "near_ratio": "near/far split of projection_fan_adaptive, in circumradii",
    "adaptive-*": "budget, far order, levels and escalation cap of "
                  "projection_fan_adaptive",
}


def describe(method):
    """Return the spec of `method`; raise ValueError for unknown names."""
    try:
        spec = METHOD_SPECS[method]
    except KeyError:
        accepted = ", ".join(ACCEPTED_METHODS)
        raise ValueError(
            f"velocity_method must be one of {accepted} "
            f"(lagrangian_filaments is documented but not wired; quadrature "
            f"settings belong to area_quadrature); see {ARCHITECTURE_DOC}"
        ) from None
    return dict(spec, method=method)


def is_selectable(method):
    """True only for a method the solver accepts (public names)."""
    return method in ACCEPTED_METHODS


def is_public(method):
    return method in PUBLIC_METHODS


def method_label(method):
    spec = describe(method)
    return f"{spec['method']} (method {spec['index']}: {spec['name']})"


def wiring_error(method):
    """Return None if the solver may select `method`, else the reason string."""
    spec = describe(method)
    if spec["selectable"]:
        return None
    if spec["method"] in DOCUMENTED_ONLY_METHODS:
        return (f"{spec['method']} is documented as a distinct wake representation "
                f"but is not coupled to B2; see {ARCHITECTURE_DOC}")
    return (f"{spec['method']} is a deprecated comparison backend kept for "
            f"frozen-snapshot replay; it is not a public velocity method")


def table():
    """Rows for docs/CLI, in method-index order."""
    rows = []
    for method, spec in sorted(METHOD_SPECS.items(), key=lambda kv: kv[1]["index"]):
        rows.append(dict(spec, method=method))
    return rows


def format_table():
    lines = ["method                 index  status                            coupled"]
    for row in table():
        lines.append(f"{row['method']:<22} {row['index']:>5}  {row['status']:<32} "
                     f"{str(row['coupled']):<7}")
    return "\n".join(lines)
