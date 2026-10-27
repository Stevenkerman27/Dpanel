# Issue #49 priority d - G0 sphere: continuous P1 doublet Dirichlet solve

Research-only result. The production solver (`panel/kernel.py`, `panel/half_native.py`,
`panel/solve.py`) is untouched, no default is changed and
`physical_accuracy_accepted=false` still holds. Every number below is recomputed by
`tools/vsp_evidence/p1/p1_g0_report.py`; nothing is hand-entered.

## 1. What is solved

Nodal P1 doublet layer on a closed body, interior Dirichlet condition:

```
D[mu] = -phi_inf,        mu = sum_a mu_a N_a(x),  N_a = P1 hat function
D[mu](x) = sum_a mu_a * (-2 arctan2(det, den) / 4pi + moment term)
```

The kernel is the same arctan2 solid-angle kernel and the same linear moment term as
`panel/kernel.py:19 triangle_potential` and `panel/p1_doublet_moment.py:8`, cross-checked
element-wise against the latter to 1e-12 (`test/test_p1_dirichlet.py`).

Collocation is at **panel centroids**, one equation per panel, one unknown per node,
i.e. an overdetermined system (M ~ 2N on a closed mesh) solved by least squares.

## 2. Why centroids and not vertices

A uniform doublet layer on a closed surface must reproduce the constant interior
potential -1, so the assembled operator must have row sums exactly -1. That is the
cheapest possible correctness test, and it separates the two collocation choices:

| level | faces | vertex-collocation row sum | centroid row sum | centroid spread |
| --- | --- | --- | --- | --- |
| 1 | 80 | -0.815524 | -1.000000000 | 5.6e-16 |
| 2 | 320 | -0.911275 | -1.000000000 | 1.2e-15 |
| 3 | 1280 | -0.956879 | -1.000000000 | 3.0e-15 |

Vertex collocation is short by O(1/N) because at a curved vertex the sum of the planar
face angles is smaller than 2*pi (the polyhedral curvature deficit), while the arctan2
kernel degenerates exactly on the surface: the free term that the kernel implies is the
smooth -1/2, and the missing part is not recovered by the arctan2 branch. At centroids
the smooth limit -B/2 is exact and the row sum is -1 to 2.8e-15 at every level.

This is the same physics that `panel/p1_gate_checks.py: self_trace_gate` measures on a
single panel: the free term is the solid-angle fraction c(x) = interior_angle/(4pi),
which is 1/2 at the centroid, 1/4 on an edge and angle/(4pi) at a vertex - so a constant
-1/2 free term is only valid where c(x) = 1/2.

## 3. Why no source term

For a closed body the doublet-only form is complete. Adding the single layer
`sigma = n . U_inf` and keeping the same interior condition rescales the boundary value:
on a sphere that layer already carries `phi_inf / 3`, so the solution moves to
`mu = 2 phi_inf` instead of the surface potential `1.5 phi_inf`, and the pressures are
O(1) wrong. Measured at level 2: mu_alpha = 2.0079, cp_rms = 1.2924 (compared with 1.5040 and
0.0391 for the validated doublet-only form). The variant is kept as a test, not as an option.

## 4. Results

Freestream `[U, 0, 0]`, unit sphere, exact surface potential `1.5 U cos(theta)`,
exact `Cp = 1 - 2.25 sin^2(theta)`.

| level | faces / nodes | mu amplitude | LS residual | Cp rms | Cp max | nose Cp | tail Cp | area |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 80 / 42 | 1.513567 | 1.32e-03 | 0.1143 | 0.2392 | 0.7006 | 0.7006 | 11.6659 |
| 2 | 320 / 162 | 1.503959 | 1.12e-03 | 0.0391 | 0.0871 | 0.9428 | 0.9428 | 12.3298 |
| 3 | 1280 / 642 | 1.501012 | 2.76e-04 | 0.0150 | 0.0385 | 0.9816 | 0.9816 | 12.5065 |

The amplitude of the solved doublet converges to the analytic 1.5 from above
(1.5136, 1.5040, 1.5010), and the Cp error decreases monotonically with an apparent order of 1.55 / 1.39
(a face-refinement factor of 4 per level, so a ratio of 2.9 is order ~1.5). That is what a
piecewise linear potential on a faceted sphere should give: the error is dominated by the
geometric faceting (total area 12.3298 instead of 4pi = 12.5664 at level 2), not by the solve.

Nose and tail Cp are identical by symmetry (0.981571 vs 0.981571 at level 3): both are stagnation
points on a closed body, and both approach 1 from below as the facet size vanishes.

## 5. Limits - what this does NOT show

- Closed, non-lifting body only: no wake sheet, no Kutta condition, no lifting surface, no
  mirror/half-model reduction. G1 (wing) still needs a doublet wake and the shed-group
  topology from `docs/ISSUE39_SHED_GROUPS_V2.md`.
- Cp is per-panel, recovered from the panel-wise constant *tangential* gradient of the P1
  potential; the normal component of the exact gradient is not representable and is not
  claimed (see `test_face_cp_uses_the_tangential_gradient`).
- The sphere mesh is a faceted icosphere, so part of the reported error is geometric. No
  Richardson/GCI extrapolation is claimed from three levels with a faceted geometry.
- The x64 guard is implicit: the module is float64-only and rejects other dtypes at the
  boundary; no 32-bit run is used for any number above.
- Nothing here is wired into the production solver, and no near-field/trajectory claim is
  made from these pressures.

## 6. Reproduction

```
cd runs/pushrepo
python -m pytest test/test_p1_dirichlet.py test/test_p1_gate_checks.py -q
python tools/vsp_evidence/p1/p1_g0_report.py    # rewrites this file and the JSON
```

Data: `docs/issue49_vspaero/p1/g0_sphere.json`. Gates: `docs/ISSUE49_P1_GATES.md`.
