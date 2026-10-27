# Issue #49 priority d - G1 design: P1 wall + wake sheet + Kutta

Research-only. The production solver is untouched, no default changes and
`physical_accuracy_accepted=false` still holds. G0 (`docs/ISSUE49_P1_G0_SPHERE.md`,
commit `95b7ace`) established the closed-body P1 doublet operator; this document fixes
the three things G1 adds - the wake sheet, the Kutta row and the mirror rule - and
measures each of them in isolation before any wing coefficient is claimed.

## 1. The system G1 has to assemble

```
wall rows  : D_wall[mu] + W_wake[Gamma] = -phi_inf   (one per wall collocation point)
Kutta rows : mu_upper - mu_lower - Gamma_te = 0      (one per trailing edge station)
wake rows  : Gamma at station k+1 = Gamma at station k (one per wake node, stations fixed)
```

Unknowns are the wall doublet nodes `mu` plus the wake doublet nodes `Gamma`. The wake is a
*doublet* sheet, so its density is a potential jump and the Kutta condition says that the
jump vanishes at the trailing edge.

## 2. Wake sheet: the jump is the interpolated density

A doublet sheet with density `mu` produces the potential jump `mu` across itself. On a
2x2 triangle strip (unit chord, unit span) with uniform density 1, the measured jump
`phi(+h n) - phi(-h n)` converges to 1 as the probe height `h` decreases:

| probe height | max abs error of the jump | mean jump |
| --- | --- | --- |
| 1e-02 | 3.56e-02 | 0.971543411 |
| 1e-03 | 3.57e-03 | 0.997151870 |
| 1e-04 | 3.57e-04 | 0.999715185 |
| 1e-05 | 3.57e-05 | 0.999971518 |
| 1e-06 | 3.57e-06 | 0.999997152 |

With a *linear* density `mu = 2y + 0.5` the jump equals the interpolated density at the
panel centroids to 4.84e-06. The sheet carries `mu_h`, not the strip average, and that is what
makes the P1 wake consistent with a P1 wall. Close to the surface the arctan2 branch loses
absolute precision, so the measured error saturates at a few 1e-6: a kernel-evaluation
floor, not a discretisation effect.

Two consequences G1 must respect:

- a constant-density sheet seen from a point **in its own plane** has zero potential
  (measured 0.0e+00 at two in-plane points), because the potential is a solid angle;
  in-plane wake collocation would silently see nothing;
- far from a compact sheet the potential decays faster than a monopole (measured order
  3.10 / 3.05 at r = 10, 20, 40), so wake panels need no special far-field treatment here.

## 3. Mirror rule: the mirrored sheet carries -Gamma

Mirroring a sheet in the symmetry plane flips its normal, and a doublet is odd under a
normal flip. Measured on the strip with a linear density, comparing the mirrored geometry
against mirrored probe points:

| mirrored density | residual against the unmirrored field |
| --- | --- |
| `+density` | 0.00e+00 (the field equals `-phi`, it is not `phi`) |
| `-density` | 0.00e+00 |

So the half-model wake term must enter with the mirrored slope subtracted - exactly the
`W = (dw + wi) @ basis + (my - mi) @ slope` structure of `panel/half_native.py`, and the
discrete version of `_REFLECT = [1, -1, 1]`. Being an algebraic identity of the kernel it
holds for any density, so it is asserted as a test rather than measured per case.

## 4. Unknown/equation bookkeeping

```
unknowns  = n_wall_nodes + n_wake_nodes
equations = n_wall_rows + n_stations + (n_wake_nodes - n_stations)
```

The `n_stations` terms cancel, so the system is square only when `n_wall_rows = n_wall_nodes`.
On the real 396-panel wing/body half mesh (`surface.json`):

| coordinates | wall faces | referenced nodes | unreferenced vertices |
| --- | --- | --- | --- |
| 400 | 396 | 213 | 187 |

A closed triangle mesh has roughly two panels per node, so a square `n_wall_rows =
n_wall_nodes` system would need node collocation. G1 does **not** require it: the
overdetermined panel-centroid least squares of G0 already converged to the analytic sphere
pressures (level 3: `Cp` rms 0.0150), so G1 keeps one row per wall centroid plus one Kutta
row per trailing-edge station against a smaller unknown vector and reports the rank and the
residual. The assembled solid-angle free term is not needed either: with the mirrored half
the constant-layer row is exactly `-1` (section 7), which is the identity a node-collocation
`c(x)` would have been introduced to repair.

The unreferenced vertices are the wake strip nodes sharing the same coordinate array. They
are *not* wall unknowns - the same trap the gate audit found on the trapezoid meshes, where
16.4 percent of the vertices of the 12290-panel case are wake nodes.

| wake nodes | stations | unknowns | equations | square |
| --- | --- | --- | --- | --- |
| 22 | 21 | 235 | 418 | False |
| 44 | 21 | 257 | 440 | False |
| 385 | 21 | 598 | 781 | False |
| 770 | 21 | 983 | 1166 | False |

Those rows use `n_wall_rows = len(faces)`, i.e. the panel-row variant, and are listed to show
that the count cannot be closed that way. The node-row variant sets
`n_wall_rows = n_wall_nodes` by construction and is what G1 will assemble.

## 5. What G1 must prove before any coefficient is reported

1. The assembled wall block of the real half mesh satisfies the closed-surface identity
   (constant layer `-1` per row to machine zero, section 7); the coupled residual is
   reported, not concealed.
2. Wake transport keeps `Gamma` constant along the streamwise stations on a flat strip with
   a prescribed `Gamma(y)`, to machine zero.
3. A zero-lift case (`alpha = 0` on a symmetric wing) returns `Gamma = 0` and the G0
   pressures, so the wake coupling adds no spurious lift.
4. `CL_pressure` and `CL_circulation` agree to a stated tolerance on one mesh, and their
   difference is reported per mesh level instead of being called convergence.
5. Mirror residuals are measured on every case (the rule above is exact algebraically, so a
   non-zero residual is a bug).

## 6. Limits

- No wing coefficient is claimed yet: the wake sheet operator, the Kutta rows and the mirror
  rule are verified in isolation only.
- The production solver, its defaults and its thresholds are untouched; nothing here is
  imported outside `panel/p1_*.py` and its tests.
- x64 guard: the module is float64-only, mirroring the gate-2 requirement; no 32-bit number
  appears in this document.

## 7. First assembly on the real 396-panel wing/body mesh

Unknowns `213` (wall nodes) `+ 26` (wake knots) `= 239`; rows `396` wall `+ 44` Kutta `= 440`;
rank `239`, condition number `7.28e+02`.

The mirror rule is not an assumption any more. Adding the mirrored half block gives a
constant-layer row sum of `-0.713` (wrong), subtracting it gives `-1.000000000` (spread `2.9e-15`):


| variant | row sum | interpretation |
| --- | --- | --- |
| wall block alone | `-0.857` | open half sheet, the identity cannot hold |
| `wall + mirror` | `-0.713` | double counting the mirrored layer |
| `wall - mirror` | `-1.000000000` | closed doubled surface, exact |

The trailing edge of this CAD is **not** a coincident cut: 44 pairs, gap `0.0092` min / `0.0500`
median, `0` coincident pairs. The surface is therefore already closed by the trailing-edge
strip, and the potential is continuous along it.

| variant | `CL_pressure` | wall residual | `max abs Gamma` |
| --- | --- | --- | --- |
| Kutta + wake | 0.43203 | 2.461e-02 | 0.1630 |
| wake only | 0.41550 | 2.924e-02 | 0.5362 |
| body only | 0.44364 | 2.414e-02 | 0.0000 |

Production reference on the same 396-panel half mesh: `CL = 0.5312296425` (rank-6 pressure stencil).

Two facts kill the first formulation, and they are the finding of this section:

1. `mu_upper - mu_lower - Gamma = 0` is a **definition**, not a condition. With the surface
   closed, the Dirichlet solution is continuous in `mu`, the trailing-edge jump is only
   `0.00229` rms, and the least-squares system is happy with `Gamma = 0`: the measured
   `max abs Gamma` is `0.1630` and `sum(Gamma dy) = 0.00379`.
2. Removing the wake columns, or keeping them without the Kutta rows, moves `CL_pressure`
   by a few percent (`0.4436 / 0.4155 / 0.4320`), i.e. the three variants agree on a
   **non-circulatory** number. `CL_pressure = 0.43203` here is the cambered closed-body
   solution, 19 percent below the production lifting value, and it must not be reported
   as a P1 wing result.

Next step, one of two, and both need the Kutta condition to act as an equation:

- a pressure-type Kutta row (equal speed or equal `Cp` across the trailing edge), which is
  non-linear in `mu` and needs an outer iteration, or
- a cut seam: duplicate the trailing-edge nodes per side and let `Gamma` couple the two
  copies, as `panel/half_native.py` does with its `T`/`M` blocks and bank pairing.

Neither is a threshold relaxation: the rank gate, the defaults and `physical_accuracy_accepted
= false` are untouched, and no G1 coefficient is claimed.

Data: `docs/issue49_vspaero/p1/g1_design.json`. Generated by
`tools/vsp_evidence/p1/p1_g1_design.py`.
