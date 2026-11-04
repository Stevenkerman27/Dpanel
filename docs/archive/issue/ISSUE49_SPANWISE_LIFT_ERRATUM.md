# Issue #49 / PR #50 circulation-lift erratum

Read-only audit correction. The original `CL_gamma` used the **true TE
arc length ds** even though its docstring said **spanwise dy**. For a
swept/elliptic horizontal TE, lift from a Kutta-Joukowski line integral
uses the y-projection dy. The original TE arc length ds must remain in
the weak-Kutta T/M system and is not changed here.

The new audit reports `CL_gamma` from the station-space dy measure,
`CL_gamma_exposed_TE_span` from the exposed physical TE span, and
`CL_gamma_legacy_TE_arclength` for historical reconstruction. The
first two can differ at a wing-body contact gap. Do not force them equal.

Original PR50 frozen elliptic data, no new solve:

| panels | old gamma CL (ds) | corrected gamma CL (dy) | Cp CL |
|---:|---:|---:|---:|
|1028|0.394426687928|0.375065055347|0.37325|
|1610|0.394820366478|0.375919381433|0.38371|

The old numbers were inflated by about 5.16%/5.03% because the
elliptic TE length was 2.19279 instead of the 2.0 half-span. Previous
PR50 statements interpreting old pressure-minus-gamma gaps are
superseded. This is a diagnostic fix, NOT physical accuracy acceptance.

User reports VSPAERO comparison completed locally, but no VSPAERO run
is attached to PR50. Request original outputs plus version/mode,
CAD/mesh/TE/wake SHA, symmetry, reference quantities, alpha,
section Cp and CL/CDi/CM; do not select a method solely by matching
the VSPAERO integrated CL. Existing frozen files must stay unchanged.

Tests cover straight/swept edge, station/edge coverage, zero tip,
elliptic real-data regression, and invalid input rejection.
`physical_accuracy_accepted=false`; no merge or GRASS training.
