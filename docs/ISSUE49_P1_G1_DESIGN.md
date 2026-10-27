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

A closed triangle mesh has roughly two panels per node, so `n_wall_rows = n_wall_nodes` fails
for the panel-row collocation of G0 and G1 cannot inherit it: the wall needs **node**
collocation (one row per wall node) and the free term has to come from the assembled solid
angle `c(x) = interior_angle / 4pi`, not from a constant `-1/2`.

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

1. Node-row wall collocation reproduces G0 on the sphere (row sums `-1` to machine zero with
   the assembled `c(x)`); the panel-row G0 result cannot be carried over.
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

Data: `docs/issue49_vspaero/p1/g1_design.json`. Generated by
`tools/vsp_evidence/p1/p1_g1_design.py`.
