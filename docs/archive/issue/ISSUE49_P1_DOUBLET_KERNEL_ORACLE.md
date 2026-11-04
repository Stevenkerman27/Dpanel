# Issue 49: analytic triangular P1 doublet kernel oracle

This is the first isolated **operator-level** building block toward solving
a continuous P1 wall-doublet field on the *same planar triangular faces*.
It is not yet a P1 boundary-element solver or a Cp improvement claim.
The old constant-doublet kernel, wall mesh, wake, Gamma and default CLI
are all untouched. physical_accuracy_accepted=false.

For source triangle T and its barycentric basis N_a, compute
D_a(x) = integral_T N_a(q) * dG(x,q)/dn(q) dS(q).
Writing N_a(q)=1/3 + grad(N_a).(q-centroid), the exact analytic moment
uses the same edge-log and solid-angle expressions as the original P0
triangle kernel. Sum_a D_a(x) = D_P0(x) by basis partition of unity.
For a target **inside its own planar triangle**, the interior trace
uses exactly D_a(x) = -N_a(x)/2 (not a tiny off-surface offset).

The oracle intentionally uses NumPy scalar loops and limits target-source
pairs to 20,000. It is **not** AD-enabled and is **not** a production
assembly. No hypersingular manipulation, clipping, smoothing or invented
pressure value is involved.

11 pure NumPy tests actually passed locally: comparison to independent
44x44 Duffy-Gauss quadrature on rotated 3-D triangles, rigid transforms,
winding reversal, two on-panel trace identities, multiple interaction
blocks, malformed inputs and resource refusal. The source blob matches
the tested file. Full DPanel/JAX baseline/integration is still unrun.

Next steps must be separate PRs:
1. Compare the P1 sum against existing JAX P0 kernel on off-surface,
   mirrored, and trace-limited real triangles; reject conditioning failures.
2. Define legitimate shared-node **potential** space, separately duplicated
   traces only across verified *physical* potential cuts (not every
   pressure-stencil contact cut). Determine node equation vs unknown count
   and gauge before any LU; no fictitious square collocation.
3. Assemble a tiny non-lifting sphere/flat plate manufactured problem,
   then P1 G1 Kutta and numerical wall pressure against PR50 sphere/wing.
   Compare wall-normal total-flow and near-wall Cp across offsets
   0.01a-0.05a; true physical improvement requires independent anchors.
4. VSPAERO comparison is secondary to matching reference geometry and
   normalization. PR51 corrects an unrelated 5% dy/ds Gamma-CL audit bug.

Do not stack mesh refinement, G2 wakes or pressure-method changes onto
this kernel acceptance experiment. No merge or GRASS training.
