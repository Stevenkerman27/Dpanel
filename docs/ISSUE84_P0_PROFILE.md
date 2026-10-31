# Issue #84 P0: where the coupled material-wake solve actually spends time

Status: measurement only. No solver, default or gate changed. physical_accuracy_accepted=false still holds for every number below.

## 1. Retracted estimate

An earlier estimate of about 7.7 h per G2 iteration multiplied the pressure-Kutta least_squares evaluation count (85) by the free-node count (378) by one regularized velocity evaluation. That model is **withdrawn**:

* solve_coupled_material_wake runs the pressure-Kutta least_squares **before** it constructs TotalVelocity and before it calls limits() (panel/p1_coupled_material_wake.py:485-560), so the regularized kernel is not inside the residual evaluations at all;
* the driver is the number of target points that limits() pushes through evaluate_many, which this report measures directly.

No cache keyed on a least_squares iteration is therefore valid, and none is proposed here.

## 2. Opt-in counters

panel/velocity_profile.py adds enable()/disable()/reset()/snapshot()/write() and a _profile helper used by four call sites:

| counter group | where |
| --- | --- |
| limits.* (calls, points, ladder_steps, targets, refined_targets, tag.<t>.*) | panel/p1_coupled_material_wake.py:195-226 |
| total.* (call.count, evaluate_many calls/targets/method) | panel/p1_coupled_material_wake.py:120,166-171 |
| area_terms.* (calls, triangles, node_sets, subtriangles, seconds) | panel/p1_projection_duffy.py |
| fan.* (calls, strips, subtriangles, region counts) | panel/p1_projection_duffy.py::projection_fan |

test/test_velocity_profile.py (5 tests) proves that the counters are inert when disabled, that limits(tag=...) returns bitwise identical values with and without a label, and that peak_rss_mb() is finite and positive.

peak_rss_mb() reads the process peak working set through kernel32.K32GetProcessMemoryInfo; the psapi.GetProcessMemoryInfo alias returns 0 on this Windows build and is only a fallback.

## 3. Target-set composition of one B2 iteration

Measured on the frozen G2 CAD case (runs/half_cad_wingbody_v1/surface.json, alpha 5 deg, --kutta pressure, analytic edges, two iterations requested):

| quantity | value |
| --- | --- |
| limits.calls in the recorded round | 4 (two groups x two target sets; the run blocked at iteration 0) |
| wake-face sample points | 794 |
| free-node sample points | 393 |
| ladder steps | 16 (4 factors x 2 signs) |
| velocity evaluations per iteration | 9,496 |
| evaluate_many calls | 16 |
| wall clock, whole process | 14.123 s |
| peak RSS | 157.203 MB |

The floor is 2 x 1187 x 4 = 9,496 sign-resolved targets: 794 wake-face centroids plus 393 free nodes, each sampled at four ladder radii on both sides of the sheet. A non-analytic method adds one more refinement pass at twice the order, i.e. another 2 x 1187 targets.

## 4. Cost of a single target, per fan configuration

109 fixed points (7 legacy-quadrature rejects + 102 epsilon rejects, rebuilt from the G2 rejection tables), one evaluation each, on the same 1666-triangle sheet (872 wall + 794 wake). Every point projects onto a triangle vertex.

| configuration | ms/point | kernel s | projection s | fan calls/point | sub-triangles/point | x legacy |
| --- | --- | --- | --- | --- | --- | --- |
| 16/8 | 239.4 | 18.976 | 5.333 | 1,666 | 14,382 | 1.10 |
| 24/8 | 468.8 | 42.231 | 5.422 | 1,666 | 14,382 | 2.16 |
| 28/8 | 618.3 | 57.469 | 5.430 | 1,666 | 14,382 | 2.84 |
| 48/8 | 1,740.8 | 171.876 | 5.489 | 1,666 | 14,382 | 8.01 |
| legacy_duffy/48 | 217.4 | n/a | n/a | 0 | 0 | 1.00 |

The fan partition (closest-point projection plus sector split) is at most 9 % of the point cost; 91-97 % is the Duffy kernel, and the kernel scales with sub_triangles * order**2. Sub-triangles per point follow 8*vertex_regions + 16*edge_regions + 24*interior_regions.

## 5. Extrapolated iteration cost

Model: iteration_seconds = (ladder targets + refinement targets) x measured per-target cost. It is a linear scaling of a measurement, not a measurement of a whole iteration; section 6 measures the same quantity a second, independent way.

| method | per-target s | evaluations | iteration |
| --- | --- | --- | --- |
| analytic_edges (measured) | 0.0015 | 9,496 | 14 s |
| 24/8 (ladder + refinement) | 0.469 | 11,870 | 1.5 h |
| 28/8 (ladder only) | 0.618 | 9,496 | 1.6 h |
| 28/8 (ladder + refinement) | 0.618 | 11,870 | 2.0 h |
| 48/8 (ladder + refinement) | 1.741 | 11,870 | 5.7 h |

The fan-order flag, not the limits order, sets this cost; see section 7. Every extra ladder factor and the refinement pass cost a full re-evaluation.


## 6. B2 target set, measured

**production --quadrature-order 48** (--max-points 6, method rm_regularized_p1, fan 28/8)

| group | set | sampled | total | measured s | s/point | extrapolated s |
| --- | --- | --- | --- | --- | --- | --- |
| main_wing | free_node | 6 | 378 | 36.858 | 6.143 | 2,322.1 |
| main_wing | wake_face | 6 | 770 | 37.754 | 6.292 | 4,845.1 |
| raised_tail | free_node | 6 | 15 | 37.412 | 6.235 | 93.5 |
| raised_tail | wake_face | 6 | 24 | 37.548 | 6.258 | 150.2 |

Extrapolated whole iteration: **7,411 s** (2.06 h).

The counters verify the linear model on the sampled points: 6 points cost 48 ladder targets plus 12 refinement targets, i.e. 10 evaluations per point, and targets_full_set is exactly 10 x the set size.

**aligned --quadrature-order 28** (--max-points 6, method rm_regularized_p1, fan 28/8)

| group | set | sampled | total | measured s | s/point | extrapolated s |
| --- | --- | --- | --- | --- | --- | --- |
| main_wing | free_node | 6 | 378 | 37.090 | 6.182 | 2,336.7 |
| main_wing | wake_face | 6 | 770 | 38.049 | 6.342 | 4,883.0 |
| raised_tail | free_node | 6 | 15 | 37.040 | 6.173 | 92.6 |
| raised_tail | wake_face | 6 | 24 | 37.539 | 6.257 | 150.2 |

Extrapolated whole iteration: **7,462 s** (2.07 h).

The counters verify the linear model on the sampled points: 6 points cost 48 ladder targets plus 12 refinement targets, i.e. 10 evaluations per point, and targets_full_set is exactly 10 x the set size.

The two runs differ by 51.5 s on the same points, i.e. 0.69 %. Changing --quadrature-order from 48 to 28 changes nothing in the fan path, which section 7 shows directly. The cost is set by fan_order and fan_strips.

## 7. Which flag actually controls the fan area integral

Four fixed wake-face centroids, G2, core 0.01, fan_strips 8, ladder 1 and 0.5 (so each point is sampled four times plus two refinement samples):

| field | limits order | max quadrature error | max epsilon error | seconds |
| --- | --- | --- | --- | --- |
| fan 28/8 | 48 | 0.000e+00 | 1.674e-05 | 15.039 |
| fan 28/8 | 28 | 0.000e+00 | 1.674e-05 | 15.116 |
| fan 16/8 | 48 | 0.000e+00 | 1.674e-05 | 5.780 |

| comparison | max abs difference | max relative difference |
| --- | --- | --- |
| fan28/limits48 vs fan28/limits28 | 0.000e+00 | 0.000e+00 |
| fan28/limits48 vs fan16/limits48 | 2.562e-06 | 2.578e-06 |

Conclusions, all measured rather than argued:

* limits(order=...) is inert in the regularized path: fan 28 with limits order 48 and fan 28 with limits order 28 return bitwise identical means. panel/p1_regularized_sheet_velocity.py:153-160 forwards self.fan_order to area_velocity instead of the order argument it receives;
* fan_order is the binding flag: fan 28 and fan 16 differ, here by 2.6e-06 on four smooth centroids and by up to 7.8e-4 on the 109-point set (16/8 vs 48/8);
* the quadrature channel of the sampling gate is dead in the fan path: the refinement pass re-evaluates area_terms with the same fan order, so quadrature_error is exactly 0.0 on every one of these runs. Only epsilon_error still measures anything;
* this is reported, not repaired. Making limits' order reach the fan, or gating on a fan-order refinement instead, changes what the gate certifies and belongs in the P1/P2 change set.

## 8. Geometry update

| quantity | value |
| --- | --- |
| status | iteration_limit |
| limits.calls | 12 |
| velocity evaluations | 28,488 |
| limits.refined_targets | 0 |
| wall clock | 31.930 s |
| peak RSS | 124.809 MB |

Three sampling rounds ran for --iterations 2 (12 limits calls, 28,488 = 3 x 9,496), so an accepted geometry update adds one round of the same size; it does not change the per-point cost. The analytic run above blocked after the first round, which is why it records 4 calls.

## 9. What this implies for P1/P2

* A whole-iteration cache is only legitimate if the same target coordinate recurs in the same iteration with the same field state; the ladder samples 2*N*4 distinct coordinates per set, so reuse is limited to repeated sets.
* Reuse across the ladder exists at the level of quadrature nodes and triangle geometry, which is what P1 caches.
* The 91-97 % kernel share is work, not overhead: batching it (P2) removes Python loops and temporaries but not the sub_triangles * order**2 volume. Cutting the iteration from hours to minutes needs a cheaper per-face integration for the far field, decided only after the P2 batching measurement.

## 10. P1 read-only caches, measured

| configuration | ms/point before | ms/point after | change | node-set s before | node-set s after |
| --- | --- | --- | --- | --- | --- |
| legacy_duffy/48 | 217.4 | 216.2 | -0.58 % | 0.00000 | 0.00000 |
| 16/8 | 239.4 | 235.7 | -1.54 % | 0.02346 | 0.00017 |
| 24/8 | 468.8 | 465.5 | -0.70 % | 0.02896 | 0.00018 |
| 28/8 | 618.3 | 619.1 | +0.13 % | 0.03159 | 0.00020 |
| 48/8 | 1,740.8 | 1,738.5 | -0.13 % | 0.04958 | 0.00019 |

* p1_projection_duffy._duffy_nodes is now an lru_cache keyed on the order only and hands out read-only arrays; the node-set stage drops from 0.0235 s to 0.0002 s over 102 targets at order 16 (cache.duffy_hits = 102), but that stage was 0.1 % of a target;
* RegularizedP1SheetField now builds read-only per-triangle edge lengths, centroid, longest/shortest edge, aspect and circumradius once per field. Nothing in the production path consumes them yet - they exist for a near/far split - so they cannot show a speedup, and the measured changes above are noise plus the node cache;
* verdict: P1 is in place and worth at most about 1.5 %, so node sets, triangle geometry and fan partitions are not where an iteration goes. Any hour-to-minute change has to come from the per-target kernel volume itself (14,382 sub-triangles x order**2 nodes at fan 28/8), i.e. from P2/P3, not from caching.

test/test_p1_readonly_caches.py (5 tests) covers cache identity, read-only flags, clear_caches, bad orders, bitwise equality of cold and warm area_terms and of the field velocity.

## 11. Reproduce

```
python -m pytest test/test_velocity_profile.py -q
python -m tools.issue84.velocity_profile --main-half-surface runs/half_cad_wingbody_v1/surface.json --csv quad:runs/issue84/projection_gate_quad.csv --csv eps:runs/issue84/projection_gate_eps.csv --case g2 --alpha 5 --core-radius 0.01 --configs legacy_duffy 16:8 24:8 28:8 48:8 --out docs/issue84/velocity_profile_109.json
python -m panel.run_p1_coupled_material_wake --case g2 --main-half-surface runs/half_cad_wingbody_v1/surface.json --alpha 5 --kutta pressure --iterations 2 --profile-json docs/issue84/profile_g2_analytic.json --out runs/issue84/g2_analytic_prof
python -m panel.run_p1_coupled_material_wake --case g2 --main-half-surface runs/half_cad_wingbody_v1/surface.json --alpha 5 --kutta signed_speed --iterations 2 --profile-json docs/issue84/profile_g2_signed_speed.json --out runs/issue84/g2_signed_speed_prof
python -m tools.issue84.b2_target_profile --main-half-surface runs/half_cad_wingbody_v1/surface.json --method rm_regularized_p1 --core-radius 0.01 --area-quadrature projection_fan --fan-order 28 --fan-strips 8 --quadrature-order 48 --max-points 6 --out docs/issue84/b2_target_profile_prod.json
python -m tools.issue84.b2_target_profile --main-half-surface runs/half_cad_wingbody_v1/surface.json --method rm_regularized_p1 --core-radius 0.01 --area-quadrature projection_fan --fan-order 28 --fan-strips 8 --quadrature-order 28 --max-points 6 --out docs/issue84/b2_target_profile_aligned.json
python -m tools.issue84.fan_order_semantics --main-half-surface runs/half_cad_wingbody_v1/surface.json --case g2 --core-radius 0.01 --fan-strips 8 --points 4 --out docs/issue84/fan_order_semantics.json
python -m tools.issue84.velocity_profile --main-half-surface runs/half_cad_wingbody_v1/surface.json --csv quad:runs/issue84/projection_gate_quad.csv --csv eps:runs/issue84/projection_gate_eps.csv --case g2 --alpha 5 --core-radius 0.01 --configs legacy_duffy 16:8 24:8 28:8 48:8 --out docs/issue84/velocity_profile_109_p1.json
python -m pytest test/test_p1_readonly_caches.py -q
python -m tools.issue84.render_p0_profile_doc
```
