# Issue #56, part D: a well-posed small G0 discretisation and its acceptance gates

Research only. `physical_accuracy_accepted=false`. Nothing here is imported by the production
solver, no default or rank gate changes, and `panel/kernel.py` is untouched. Evidence:
`docs/issue49_vspaero/p1/g0_acceptance.json` (regenerate with
`tools/vsp_evidence/p1/p1_g0_acceptance_report.py`), implementation
`panel/p1_g0_acceptance.py`, contract `test/test_p1_g0_acceptance.py` (6 cases).

## 1. The formulation, stated explicitly

| item | choice |
| --- | --- |
| discretisation | nodal collocation on the panel centroids, overdetermined least squares |
| unknown | one doublet density `mu` per surface node, interpreted as the potential trace |
| test rows | one row per panel centroid, `D mu = -phi_inf` |
| free term | the panel's own three basis rows are `-1/6` each, summing to `-1/2` |
| kernel | exact analytic P1 doublet coefficients on flat triangles, no quadrature |
| source term | none; the `sigma = -n . U_inf` single layer is absorbed by `-phi_inf` |
| constraints | none (part C showed there is no gauge to fix in this operator) |

The sign convention is not assumed: on a closed body the assembled operator satisfies
`A 1 = -1` row by row (part C, spread `3e-15`), which is the `-1/2` own jump plus the `-1/2`
from the rest of the surface. Solving `A mu = -phi_inf` then returns `mu = phi_s`, and the
surface velocity follows from the tangential gradient of `mu` with `Cp = 1 - |V_t|^2 / U^2`.

| level | nodes (unknowns) | panels (rows) | rank | overdetermined by |
| --- | --- | --- | --- | --- |
| 1 | 42 | 80 | 42 | 38 |
| 2 | 162 | 320 | 162 | 158 |
| 3 | 642 | 1280 | 642 | 638 |

## 2. Acceptance gates (all measured, none tuned)

| gate | L1 | L2 | L3 | observed order |
| --- | --- | --- | --- | --- |
| least-squares residual | 1.32e-3 | 1.12e-3 | 2.76e-4 | - |
| surface `Cp` rms | 1.143e-1 | 3.91e-2 | 1.50e-2 | 1.55, 1.39 |
| surface `Cp` max | 2.392e-1 | 8.71e-2 | 3.85e-2 | 1.46, 1.18 |
| off-surface potential, `r/a = 1.2` | 2.51e-2 | 7.05e-3 | 1.78e-3 | 1.83, 1.99 |
| off-surface potential, `r/a = 1.5` | 1.50e-2 | 4.03e-3 | 1.02e-3 | 1.90, 1.98 |
| off-surface potential, `r/a = 2` | 6.88e-3 | 1.84e-3 | 4.65e-4 | 1.90, 1.98 |
| off-surface potential, `r/a = 3` | 2.13e-3 | 5.68e-4 | 1.44e-4 | 1.91, 1.98 |
| radial velocity error, `r/a = 1.02` | 1.54e-1 | 5.79e-2 | 1.83e-2 | 1.41, 1.66 |
| radial velocity error, `r/a = 1.05` | 1.28e-1 | 3.90e-2 | 9.24e-3 | 1.72, 2.08 |
| radial velocity error, `r/a = 1.10` | 9.95e-2 | 2.64e-2 | 6.16e-3 | 1.91, 2.10 |
| `|CD|` (d'Alembert) | 1.6e-16 | 5.0e-16 | 5.1e-16 | - |
| total area / `4 pi` | 0.9284 | 0.9812 | 0.9952 | - |
| jump error at `offset = 1e-4` | 1.03e-4 | - | - | linear in offset |

Errors are relative where a scale exists (potential, `Cp`) and absolute in units of `U` for the
radial velocity. Orders are `log2` of consecutive ratios; they are reported as observed, not fitted
to an expectation.

## 3. What each gate is for

- **Off-surface potential** is the only gate that exercises the kernel away from the collocation
  points. It converges at order ~1.9 at every radius, and the near field is consistently harder
  than the far field (`2.51e-2` versus `2.13e-3` at L1).
- **Radial velocity** is the no-penetration statement. No penetration holds on the surface, so the
  fair test is the error against the analytic radial derivative at the same ring of offsets. It
  falls from `1.5e-1 U` to `1.8e-2 U` at `r/a = 1.02`; the residual is the price of a
  piecewise-constant tangential gradient, and it is the slowest gate.
- **The jump** is exact by construction: the measured error is `1.028e-3 / 1.028e-4 / 1.028e-5`
  for `offset = 1e-3 / 1e-4 / 1e-5`, i.e. purely `O(offset)` from the continuous neighbour
  contributions. The jump equals the doublet density, which pins the trace convention.
- **`|CD| ~ 1e-16`** is a structural sanity gate (a closed body under pressure integration), not
  a physics result: any closed mesh gives it, and it says nothing about `Cp` accuracy.
- **The least-squares residual alone would mislead.** It improves by only `4.8x` over two
  refinements while the `Cp` rms improves by `7.6x`, so the residual is a first gate, exactly as
  the issue requires, and never the acceptance argument.

## 4. Conclusions

1. The formulation is well posed: full column rank at every level, a unique least-squares
   solution, and no constraint or gauge is needed (part C).
2. It converges on three independent gates (surface `Cp`, off-surface potential, radial velocity)
   with observed orders between `1.4` and `2.0`; the coarsest mesh is the outlier, not the trend.
3. The trace conventions are pinned by measurement: `-1/2` own jump, `A 1 = -1` identity, jump
   equals density to `O(offset)`.
4. This is a statement about a *sphere* under a uniform stream. It does not transfer to a lifting
   wing: G1 still needs a Kutta condition that is a condition and not a definition (the linear
   `mu_up - mu_lo - Gamma = 0` row is satisfied by `Gamma = 0`), either pressure-type or through a
   real trailing-edge cut.

## 5. Not done

- Pressure-type Kutta and the real TE cut for G1 (the only remaining blocker on the path to a
  lifting P1 solve).
- A square operator with an explicit `+-1/2` diagonal and its own gauge analysis.
- Any production change: `panel/kernel.py`, the defaults and the rank gates are unchanged, and
  `physical_accuracy_accepted` stays `false`.