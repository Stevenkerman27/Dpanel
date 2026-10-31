# Issue 84 / PR #91 — does parameter-domain subdivision fix the trailing-edge sliver?

Receipt [6539951007](https://github.com/Stevenkerman27/Dpanel/pull/91#issuecomment-6539951007) reported one target above the declared gate:
the T2 battery's single failing physical point, `xyz = (22.333, 0.238, 0.0)`, `gate = 1.902e-03` against a declared
`THRESHOLDS["gate_max"] = 2e-4`. I asked the cloud to choose between

* **(a)** subdivision of the near-singular triangle in the parameter domain, and
* **(b)** allowing four escalation rounds instead of two,

and wrote that I would answer (a) with a prototype rather than wait. This document is that answer.

Short version: **(a) works, (b) does not.** Two levels of midpoint subdivision with the fan at `48:8` per piece
reaches `6.2e-06` on that point — 32x inside the gate — for `3.0 s` on a 100-face near set, while order-only
escalation plateaus at `1.2e-02`, 59x *above* the gate, no matter how many rounds are spent.

---

## 1. Setup

| item | value |
| --- | --- |
| case | `g2` (CAD half wing+body, `half_cad_wingbody_v1`), `alpha = 5 deg`, pressure Kutta |
| core radius | `0.01` |
| point | `(22.333, 0.238, 0.0)` — on the sheet plane, at the tie of a `1405.65` aspect sliver |
| face scale | `0.0185` (longest centroid radius) |
| near set | `100` of `1666` sheet triangles (`dist <= 2*scale or dist <= core`) |
| kernel | `r / (|r|^2 + core^2)^{3/2}` per unit area, linear in the face strength |

The integral is linear in the sheet strength, so subdividing a triangle and integrating each piece with the
same kernel is an exact re-partition of the same mathematics — it changes the quadrature, not the physics.
`split_faces` splits by edge midpoints (four sub-triangles per level), keeping face-major ordering so the
per-face contributions can be summed back. `area_velocity_split` then applies the production fan to each
piece and contracts with that piece's (constant) face strength.

Reference values come from a much heavier run of the *same* family: subdivision depth 4 (25,600 pieces)
with the fan at `24:8`, plus a `32:8` self-check.

## 2. Fan-family ladder

`err` is `|v - v_ref| / |v_ref|` against the depth-4 `24:8` reference (`|v_ref| = 9.306912e-01`);
`|v|` is quoted for the `z` component, which carries essentially all of the magnitude here.

| configuration | pieces | `\|v\|` | `err` | seconds |
| --- | --- | --- | --- | --- |
| `0:28:8` (production) | 100 | 8.970782e-01 | 3.612e-02 | 0.08 |
| `0:48:8` | 100 | 9.205912e-01 | 1.085e-02 | 0.30 |
| `0:96:8` (two escalation rounds) | 100 | 9.417034e-01 | 1.183e-02 | 1.18 |
| `0:192:8` | 100 | 9.301959e-01 | 5.322e-04 | 4.63 |
| `1:48:8` | 400 | 9.293441e-01 | 1.447e-03 | 0.90 |
| `1:96:8` | 400 | 9.313607e-01 | 7.193e-04 | 3.50 |
| `2:16:8` | 1600 | 9.223445e-01 | 8.968e-03 | 0.42 |
| `2:24:8` | 1600 | 9.358476e-01 | 5.540e-03 | 0.83 |
| `2:32:8` | 1600 | 9.363618e-01 | 6.093e-03 | 1.39 |
| **`2:48:8`** | 1600 | 9.306855e-01 | **6.160e-06** | 3.01 |
| `2:64:8` | 1600 | 9.302863e-01 | 4.351e-04 | 5.24 |
| `2:96:8` | 1600 | 9.306983e-01 | 7.626e-06 | 11.66 |
| reference `4:24:8` | 25600 | 9.306912e-01 | 0 | 11.22 |
| reference check `4:32:8` | 25600 | 9.306731e-01 | 1.946e-05 | 18.93 |

Three readings matter.

1. **Order-only escalation stalls.** Production `28:8` sits at `3.6e-02`; two rounds (`96:8`) only reach
   `1.2e-02`, and the extra round to `192:8` still leaves `5.3e-04` at 4.6 s. The production gate on this
   point (`1.9e-03`) is consistent with that stall: it is the base-versus-refined difference, which shrinks
   because both members of the pair are wrong in the same direction, not because the quadrature is converging.
   Option (b) therefore cannot close the target however many rounds it is given.
2. **Subdivision is what unlocks the order.** At depth 2 the same `48:8` fan per piece lands at `6.2e-06`,
   and depth 1 alone is not enough (`1.4e-03`). The effective piece size at depth 2 is `0.0046 < core = 0.01`,
   which is exactly the condition under which the fan's collapsed (polar) coordinate can resolve the peak.
3. **Fan order is not a monotone knob on a sliver.** `2:32:8 → 6.1e-03`, `2:48:8 → 6.2e-06`,
   `2:64:8 → 4.4e-04`, `2:96:8 → 7.6e-06`: an aliasing window, not a convergent sequence. The same
   non-monotonicity was seen on whole faces (`0:48:8` beats `0:96:8`). The gate is still safe here because
   it compares `N` against `2N` and `32` against `64` differ by `3e-03`, but **no single order may be quoted
   as "the accurate one"**, and the T1 statement that `gate ≈ true` is family-internal (section 3).

## 3. Cross-check with a second quadrature family

A gate that compares the fan against itself cannot see a family-wide error. I therefore built an
independent rule: uniform midpoint subdivision to depth `d` plus the Dunavant 7-point degree-5 rule on each
sub-triangle — no collapsing map, no strips, no shared nodes or weights with the fan.

| uniform depth | sub-triangles | `\|v\|` | change vs previous |
| --- | --- | --- | --- |
| 4 | 25,600 | 9.894053e-01 | — |
| 5 | 102,400 | 9.683675e-01 | 2.1% |
| 6 | 409,600 | 9.288096e-01 | 4.1% |
| 7 | 1,638,400 | 9.169956e-01 | 1.3% |

It does not converge: four levels, 1.6M nodes, and it is still moving by 1.3% per level, drifting away from
the fan's stable `9.3067e-01` (`2.0e-03` apart). Raising the depth further is not affordable, so the question
became *which family is failing*, and the answer is structural.

**Mechanism.** When the target lies on the sheet plane, the area kernel behaves like `1/rho^2`, whose
surface integral is only *logarithmically* regularised by the core: the value scales like `ln(R/core)`. A rule
with element size `h` effectively replaces `core` by `max(core, h)`, so its value drifts like `1/ln(R/h)`
instead of converging — a relative change of a few percent per halving, exactly what the table shows. The
fan does not suffer from this: its collapsing coordinate integrates the radial direction analytically, which is
precisely why it was adopted.

The discriminating experiment lifts the same target off the sheet, where the logarithm disappears:

| `z` offset | uniform `d6 -> d7` | fan `48:8` vs `96:8` | fan `48:8` vs uniform `d7` |
| --- | --- | --- | --- |
| 0 (on sheet) | 1.29e-02 | 1.38e-05 | 1.49e-02 |
| 0.01 | 3.54e-03 | 6.05e-07 | 9.94e-03 |
| 0.05 | 5.13e-03 | 2.33e-09 | 7.84e-04 |
| 0.20 | 8.38e-05 | 2.46e-15 | 3.72e-06 |

At `z = 0.2` the two families agree to `3.7e-06` — better than the uniform family's own self-convergence
(`8.4e-05`) — so both implementations are correct where the uniform rule is trustworthy. The disagreement
appears only as the target approaches the sheet, and there the fan is the family whose answer is stable under
refinement of its own parameters (`1e-05`). Conclusion: **the uniform rule is not an oracle for on-sheet
targets, and the fan is not obviously wrong there** — but neither does this experiment *prove* the fan's
absolute error on the sheet. That proof needs a genuinely analytic reference (edge/solid-angle form with
core regularisation), which does not exist yet.

Consequence for T1: the `gate ≈ true` comparison in `docs/ISSUE84_FAN_QUADRATURE_GATE.md` section 4 used a
heavier fan configuration as truth. That is a legitimate self-consistency check, not an error bound, and its
`1.5x`-to-`3x` agreement should be read with that limit. I am not retracting the T1 numbers; I am bounding
what they certify.

## 4. Answer to (a)

* Subdivision is the right mechanism: **depth 2, fan `48:8` per piece**, `6.2e-06` on the failing point,
  `3.0 s` for a 100-face near set (`2.5x` the stalled `96:8` escalation, `38x` production `28:8`).
* Depth 1 is not sufficient (`1.4e-03`) and depth 2 with a low order is not either (`2:32:8 -> 6.1e-03`), so
  the rung must fix both the depth and the order; `48:8` at depth 2 is the cheapest configuration tested
  that clears the gate.
* The natural place to put it is the existing escalation ladder as an extra rung — subdivide the faces that
  exhaust their per-face budget, then integrate at `48:8` — which fits the a-posteriori estimator already in
  `panel/p1_adaptive_area.py` without changing the default (`analytic_edges`) or relaxing any tolerance.

## 5. What this document does not claim

* **One point.** It is the T2 failure, so it is the relevant one, but nothing here says how many of the 1187
  B2 targets would need the subdivision rung.
* **No absolute error bound on the sheet** (section 3): the fan's on-sheet value is self-consistent to
  `~1e-05`, and that is all that is established.
* **No implementation.** The rung is not in `panel/p1_adaptive_area.py`; this is a standalone prototype in
  `tools/issue84/sliver_subdivision.py`.
* **Timings are wall clock on a loaded machine** (the T3 battery was running concurrently); they are good to
  maybe 20%, and only their ratios should be quoted.
* The `0:48:8`-beats-`0:96:8` aliasing is reported, not explained.

## 6. Reproduce

```
set PYTHONPATH=D:/3D/Projects/panel/runs/fix_master;D:/3D/Projects/panel/runs
set TMP=D:/3D/Projects/panel/runs/pipwork
D:/3D/Projects/panel/.conda_env/python.exe -u -m tools.issue84.sliver_subdivision \
    --main-half-surface D:/3D/Projects/panel/runs/half_cad_wingbody_v1/surface.json \
    --configs "0:28:8,0:48:8,0:96:8,0:192:8,1:48:8,1:96:8,2:16:8,2:24:8,2:32:8,2:48:8,2:64:8,2:96:8" \
    --out docs/issue84/sliver_subdivision.json

python runs/issue84/oracle_offset.py        # two-family comparison, on and off the sheet
python runs/issue84/oracle_convergence.py   # uniform-family depth ladder
```

Artifacts: `docs/issue84/sliver_subdivision.json`, `docs/issue84/oracle_offset.json`,
`docs/issue84/oracle_convergence.json`.