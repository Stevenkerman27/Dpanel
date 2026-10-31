# Issue #84 - the subdivision rung through the production gate

Kernel-level subdivision is measured in `docs/ISSUE84_SLIVER_SUBDIVISION.md`. This note
answers the follow-up through the production path: does the rung actually clear the
2e-4 gate on the one target that failed task 2, and what does it cost?

Short answer: no. The rung leaves the velocity untouched to 9.6e-10 relative, and it
raises the reported gate from 1.902e-03 to 1.220e-01, because the gate sums per-face
estimates while the value is a vector sum in which those per-face errors cancel.

## 1. Setup

- Case: CAD G2 (`runs/half_cad_wingbody_v1/surface.json`), alpha = 5 deg, core radius
  0.01, pressure Kutta; solve 8.0 s.
- Field: `TotalVelocity(method='rm_regularized_p1', area_quadrature='projection_fan_adaptive')`,
  near fan 28:8 refined 48:8, far composite Duffy order 8 levels 1, budget 2e-4,
  `max_escalations=2`.
- Gate: the production `panel.p1_coupled_material_wake.limits(..., eps_fraction=0.05,
  ladder=[1, 0.5, 0.25, 0.125], order=48)`.
- Rung: `adaptive_subdivision_levels=0` (the task-2 ladder) versus 2 with
  `adaptive_subdivision_fan=(48, 8)`; the rung integrates each still-over-budget near
  face as `4**levels` pieces at 48:8 and reports the difference against 96:16 as its
  estimate.
- Targets: the ten `special_points` of the two wake groups (tip, root, trailing edge,
  sliver, high aspect), which include the point that failed task 2
  (xyz = [22.3333, 0.23767, 0.0], scale 0.018497).

## 2. Results

| target | gate, rung off | gate, rung on | rung faces/pieces | dv/|v| | time off (s) | time on (s) |
| --- | --- | --- | --- | --- | --- | --- |
| special_tip | 6.800e-11 | 6.800e-11 | 0 / 0 | 0.0 | 3.7 | 4.7 |
| special_root | 1.018e-07 | 1.018e-07 | 0 / 0 | 0.0 | 6.8 | 5.6 |
| special_trailing_edge | 1.902e-03 | **1.220e-01** | 2 / 32 | 9.55e-10 | 26.3 | 30.8 |
| special_sliver | 4.255e-09 | 4.255e-09 | 0 / 0 | 0.0 | 5.6 | 5.7 |
| special_high_aspect | 1.902e-03 | **1.220e-01** | 2 / 32 | 9.55e-10 | 26.8 | 30.8 |
| special_tip (group 2) | 3.837e-06 | 3.837e-06 | 0 / 0 | 0.0 | 6.4 | 6.3 |
| special_root (group 2) | 5.928e-06 | 5.928e-06 | 0 / 0 | 0.0 | 6.8 | 6.8 |
| special_trailing_edge (group 2) | 4.162e-06 | 4.162e-06 | 0 / 0 | 0.0 | 6.3 | 6.1 |
| special_sliver (group 2) | 4.598e-06 | 4.598e-06 | 0 / 0 | 0.0 | 6.3 | 6.3 |
| special_high_aspect (group 2) | 6.075e-06 | 6.075e-06 | 0 / 0 | 0.0 | 6.7 | 6.7 |

The two failure rows are the same physical target: `special_trailing_edge` and
`special_high_aspect` select the same worst point (as in the task-2 report).

Group 1 and group 2 of each set are the main wing and the raised tail; only the main
wing has the offending sliver.

## 3. What the numbers say

**The rung does not move the value.** On the offending target the velocity is
v = [0.996176, -3.013e-07, 0.011331] with |v| = 0.99624; the rung changes it by
9.51e-10 absolute, 9.55e-10 relative. That is bit-level agreement between two very
different quadratures: the order ladder's final configuration (fan order and strips
escalated twice, 26.3 s) and 32 pieces at 48:8 (30.8 s). Two independent schemes
agreeing to 1e-9 is the strongest evidence in this study that the velocity at this
point is resolved - consistent with the kernel-level result that depth-2 pieces at
48:8 sit 3.1e-06 from the depth-4 truth.

**The gate moves the wrong way.** Rung off reports 1.902e-03; rung on reports
1.220e-01, 64x larger. The mechanism is in the estimator, not the value:

- the gate reports `estimate.sum() / |sum(base)|`, a sum of *per-face norms*;
- the rung replaces the stuck faces' estimate with |fine - coarse| on each of the 32
  pieces, so the reported number keeps every piece's disagreement with its own sign
  removed;
- the underlying exact quantity is the difference of the *vector sums*, and that is
  9.5e-10.

So on this target the per-face sum overstates the aggregate change by about six orders
of magnitude. The old 1.902e-03 was not honest either - it under-reported, because base
and refined fans failed in the same direction - and the new 1.220e-01 is not a bound on
the value.

**Nothing else changes.** The eight targets that the order ladder already resolves have
identical gate values and bit-identical velocities with the rung enabled, and the rung
costs nothing there (`subdivision_faces = 0`). Concurrency caveat: these timings were
taken while the task-3 battery was running, so the absolute seconds are inflated; the
only timing claim made here is the +4.5 s / +17% on the offending target.

## 4. Recommendation

Per-face norms are the right quantity for allocating a budget across faces; they are
the wrong quantity to report as the gate, because the functional that matters (the
velocity at a node) is a vector sum in which per-face errors cancel, sometimes almost
exactly. Two options, both cheap:

1. report the gate as the norm of the summed difference (vector aggregate), and keep the
   per-face sum visible as a separate diagnostic for allocation;
2. keep the per-face gate but pair it with a single paired-rerun of the aggregate at a
   second configuration, which is what this note effectively did by hand.

Either changes the meaning of a reported number, so it is a contract decision for the
cloud, not something to switch under the task-2 battery. The rung itself is implemented
and opt-in (`adaptive_subdivision_levels`, default 0) - it is a usable extra rung, but
it is not a gate fix and must not be presented as one.

## 5. What is not claimed

- Not claimed: the 2e-4 gate is met on the offending target. It is not, under either
  rung setting.
- Not claimed: 1.220e-01 is the true error, or that 1.902e-03 is. Neither is a bound on
  the value; the best value-level evidence is the 9.55e-10 agreement between two
  schemes plus the kernel-level 3.1e-06 against the depth-4 truth.
- Not claimed: timings reflect an idle machine.
- Coverage: ten special targets, one case, one core radius.

## 6. Reproduction

```
python -m tools.issue84.subdivision_rung_check \
    --main-half-surface D:/3D/Projects/panel/runs/half_cad_wingbody_v1/surface.json \
    --sets trailing_edge high_aspect sliver tip root \
    --levels 0 2 --subdivision-fan 48:8 \
    --out docs/issue84/subdivision_rung_gate.json
```

Artifact: `docs/issue84/subdivision_rung_gate.json` (10 targets, 219.5 s total).