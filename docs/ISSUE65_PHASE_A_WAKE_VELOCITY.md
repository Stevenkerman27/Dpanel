# issue #65 Phase A: an independent wake-sheet velocity kernel

Status: research only. `physical_accuracy_accepted = false`. Nothing in this
document changes a default, relaxes a gate, or claims convergence.

## 1. Why a new kernel

`docs/P1_FREE_WAKE_PROBE_FIX.md` recorded the failure of the experimental P1
steady free wake: the wake-node velocity was sampled at a hand-tuned probe
offset, and the iteration had **no fixed point** (the normal velocity in the
junction band was 1.77 U against sin(5 deg) = 0.087 U).  The probe offset was
never a definition: it had no limit statement, no jump condition, and it could
not represent recirculation (`Vx <= 0`).

Phase A replaces the heuristic with a definition that can be checked against
quantities derived independently: the velocity of a triangulated doublet sheet
with piecewise linear strength, expressed with exact Biot-Savart integrands only
(no core, no cutoff, no probe floor).

## 2. The math, and where each sign comes from

For a flat triangle with vertices `v0, v1, v2` (right-handed boundary
`v0 -> v1 -> v2`), constant strength `mu` and field point `p`,

```
phi(p) = mu * Omega(p) / (4*pi)
Omega(p) = 2 * atan2( r0 . (r1 x r2),  den )        r_i = v_i - p
den = la*lb*lc + (r0.r1)*lc + (r1.r2)*la + (r2.r0)*lb
u(p) = grad phi = closed_integral mu * t x r / |r|^3 dl
```

The second line is the Van Oosterom-Strackee signed solid angle; the third is
the exact vortex-ring velocity of the triangle boundary.  For variable `mu` the
surface vorticity is `gamma = grad_s(mu) x n` and

```
u(p) = 1/(4*pi) [ B(p) + gamma x ( S(p) - Omega(p) n ) ]
B(p) = closed_integral mu(x') * t x r / |r|^3 dl
S(p) = closed_integral (1/|r|) * (t x n) dl
```

The surface integral `int_S r/|r|^3 dS` was reduced to a boundary integral with
the surface-gradient theorem, but that reduction is **not complete**: it gives
only the tangential part.  The normal part is `-Omega * n`, which is exactly the
term that carries the vortex-sheet jump.  Measured identity residual (midpoint
rule, 12800 points, four field points): `2.4e-05, 2.2e-05, 5.7e-06, 1.1e-04`.
Dropping `-Omega n` is what makes a raw two-sided probe look like a velocity
definition while actually losing the jump.

## 3. Validation (all tolerances are measured, not chosen)

| oracle | quantity | measured |
| --- | --- | --- |
| closed-form segment Biot-Savart vs 400-point quadrature | `segment_velocity` | `< 1e-9` |
| exact solid-angle gradient, constant mu | `sheet_velocity` | `1.74e-11`, `4.34e-11` |
| exact vortex ring of the same triangle | `sheet_velocity` | `1.74e-11` |
| brute-force area quadrature of `phi`, linear mu | `sheet_velocity` | `4.2e-05`, `7.0e-05`, `2.1e-04`, `1.3e-05` |
| production kernel `panel.kernel.triangle_potential(linear_y=True)` | `-grad phi_kernel` | `1.8e-10`, `3.2e-10`, `1.6e-10` |
| analytic jump condition `jump = gamma x n` | `sheet_velocity_pv` | rel `5.4e-05`, normal jump `0.0` |
| principal-value convergence, `eps = 1e-2 ... 1e-5` | `sheet_velocity_pv` | `1.5e-07` |

Sign convention against the production kernel is a single global sign and is
asserted in `test/test_wake_velocity.py::test_matches_production_kernel_sign_and_value`:
the production kernel stores `D = -Omega/(4*pi)` (verified to all digits), so
`grad(mu0*D + dmu/dy*M) = -u` with `M` the kernel's `moment_y`.  `mu0` in the
velocity API is the strength **at the triangle centroid**, not at the origin.

Unit and scaling behaviour is asserted, not assumed: doubling `mu` doubles the
velocity (`< 1e-14`), and scaling the geometry by 3 while keeping `mu` divides it
by 3 (`< 1e-12`).

## 4. What this does and does not settle

Settled:

* the velocity of a piecewise-linear doublet sheet has a definition that agrees
  with five independent oracles at the 1e-10 to 1e-4 level;
* the on-sheet limit exists and its jump equals `gamma x n` — a physical jump,
  not a numerical artefact to be tuned away;
* the principal value is the two-sided average and is offset independent to
  `1.5e-07` on the test sheet.

Not settled (Phase B/C of issue #65):

* no material-node update, no pseudo-time marching, no topological gating;
* no statement about steady existence: a fixed point of the node update is still
  an open question, and Phase A makes no claim about it;
* the kernel is exact (zero core); any finite core radius would be a numerical
  parameter and must be labelled as such, not as physics;
* self-face (PV) handling for a material node is left to the caller: the node
  update must decide whether it needs the PV, one of the two one-sided limits,
  or the PV plus the local term, and that choice has to be argued in Phase B;
* `Vx > 0` is not required by the kernel, but recirculation has not been tested.

## 5. Reproduce

```powershell
$env:TMP='D:\3D\Projects\panel\runs\pipwork'; $env:TEMP=$env:TMP; $env:TMPDIR=$env:TMP
$env:PYTHONPATH='D:\3D\Projects\panel\runs\wake_kernel_git;D:\3D\Projects\panel\runs'
$env:DSH_TMP_ROOT='D:\3D\Projects\panel\runs\pipwork\dsh'; $env:JAX_ENABLE_X64='1'
& 'D:\3D\Projects\panel\.conda_env\python.exe' -m pytest test/test_wake_velocity.py -q -p dsh_tmp_plugin
```

Result: `11 passed`.  The oracle scripts travel with the code as
`tools/issue65/oracle_sheet.py` (solid-angle and area-quadrature oracles),
`tools/issue65/oracle_kernel.py` (production-kernel cross-check) and
`tools/issue65/oracle_identity.py` (surface-integral identity residual).
