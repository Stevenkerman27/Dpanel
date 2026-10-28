# PR #61 follow-up: anisotropic two-sided PV probe and the real-CAD steady free wake

**Status: research, \`physical_accuracy_accepted=false\`. Nothing here is a
converged free-wake result.** Production \`panel/kernel.py\` and the P0 half-model
path are untouched.

## 1. What failed on the real CAD (head \`313987a\`, unmodified)

Command (worktree \`runs/pr61_git\`, HEAD \`313987a\`):

    python -m panel.run_p1_steady_wake \
      --main-half-surface runs/half_cad_wingbody_v1/surface.json \
      --alpha 5 --iterations 12 --reference-area 4.0 --out runs/fw_head

| it | CL | CL_gamma | main-wing normal slip RMS / max |
| --- | --- | --- | --- |
| 0 | +0.339459 | +0.329910 | 0.1774 / 0.800 |
| 1 | +0.322217 | +0.285863 | 0.4576 / 1.857 |
| 2 | — | — | \`ValueError: main_wing: wake streamwise marching requires Vx>0\` |

The slip *grew*, so this was not slow convergence: the relaxation pushed the
sheet away from the stream surface until a node stalled in x.

## 2. Root cause: the probe height is set by the wrong edge

The DPanel half-model wake is one flat strip at z = 0 running from x = 1.09c to
x = 22.3c. Measured on the 770-triangle main-wake sheet:

- median shortest edge 0.100c, p10 0.050c, minimum 9.2e-3c;
- edge aspect (longest / shortest) median 8.7, p90 **260**, maximum **1405**;
- the original probe height 0.055*sqrt(area) sits at up to **1.46x the shortest
  edge**, and on **13 %** of faces above half of it.

\`sqrt(area)\` is set by the long edge, so on a sliver the two probe points fall
inside the face's own near field: the centred difference no longer estimates a
principal value, it reads the sheet's own singular layer.

## 3. Fix

\`panel/p1_steady_free_wake.py\`:

1. \`_shortest_edges(mesh)\`; \`mean_wake_velocity(..., probe_floor_fraction=0.0)\`
   takes \`h = max(probe_fraction*sqrt(area), probe_floor_fraction*shortest_edge)\`.
   The isotropic default is unchanged (floor 0 is bit-identical to \`313987a\`).
2. \`_one_wake_target(..., min_axial=0.08, freeze=True)\` no longer raises on axial
   stagnation: it holds the node and reports \`frozen_nodes\`. \`--strict-marching\`
   restores the old hard failure.
3. Rate-limited, maskable relaxation: \`max_step_chord\`, \`relax_min_abs_y\`
   (freeze the wing-body junction strip), \`slip_gate\` (relax only where the
   sheet is already within the gate of a stream surface). Every iteration record
   now carries \`frozen_nodes\`, \`relaxed_nodes\` and the sheet quality metrics
   (\`shortest_edge_median\`, \`aspect_p90\`, \`probe_over_shortest_max\`).

## 4. Initial-field acceptance (issue #62's criterion)

\`tools/vsp_evidence/p1/free_wake_probe.py\` solves the undeformed G2 model once
and measures the sheet velocity at four probe settings
(\`docs/issue49_vspaero/p1/free_wake_probe.json\`):

| floor | main slip RMS | main slip max | tail slip RMS | probe/short (worst) |
| --- | --- | --- | --- | --- |
| 0.0 | 0.2994 | 1.7677 | 0.0476 | 1.46 |
| 0.3 | 0.2395 | 1.1650 | 0.0545 | 1.46 |
| 0.6 | 0.1774 | 0.8003 | 0.0985 | 1.46 |
| 1.2 | 0.1106 | 0.4346 | 0.0743 | 1.46 |

Even at floor 1.2 the worst face still probes at 1.46x its short edge: that face
is the sliver at the wing-body junction, i.e. exactly where the spanwise Gamma
basis changes sign. The estimator is therefore honest only away from that strip.

## 5. Iteration acceptance (same mesh, small steps)

All runs use the probe floor 0.6 and the frozen-node guard.

| run | settings | CL(0) -> CL(last) | main slip | tail slip |
| --- | --- | --- | --- | --- |
| guard only | relax .25, no cap | 0.339459 -> 0.242948 | 0.1774 -> 1.3227 | 0.0985 -> 0.5810 |
| guard + mask + cap | relax .15, cap .05, \`--relax-min-y .35\` | 0.339459 -> 0.355323 | 0.1774 -> 0.2750 | 0.0985 -> 0.0738 |
| small step | relax .05, cap .01 | 0.339459 -> 0.336367 | 0.1774 -> 0.1954 | 0.0985 -> 0.0875 |

Per-iteration main-wing slip in the small-step run:
0.1774, 0.1752, 0.1714, 0.1786, 0.1954 (local maximum 2.63 at the last step),
while the tail decreases monotonically (0.0985, 0.0954, 0.0919, 0.0897, 0.0875).
In the masked run the displacement saturates the cap (0.05c) at *every*
iteration, i.e. the marching asks for a larger move than the cap allows forever.

**Conclusion: no fixed point is reached.** The crash is fixed and the iteration
is bounded, but the main-wing sheet keeps deforming because the frozen-x
streamline march cannot represent a sheet that wants to fold (near the junction
the local normal velocity reaches 1.77 U against sin(5 deg) = 0.087 U). Claiming
convergence would be false; the honest statement is "bounded, not converged,
geometry not stable".

## 6. Recommendation

Free wake stays an **explicit opt-in experimental mode** (issue #62, item 5).
The next step is not more damping but a Lagrangian wake: free vortex filaments
with x-deformable nodes and a Biot-Savart PV kernel with an analytic self term,
so that the junction vortex and the roll-up are represented instead of sampled.
Until that exists, PR #61 must not be described as converged, and the fixed-wake
G2 (PR #60) remains the reference configuration.

## 7. Reproduction

    cd runs/pr61_git
    python -m pytest test/test_p1_steady_free_wake_guards.py -q      # 4 passed
    python -m panel.run_p1_steady_wake --main-half-surface ...\
      --probe-floor-fraction .6 --relaxation .05 --max-step-chord .01 --iterations 4
    python tools/vsp_evidence/p1/free_wake_probe.py

Test totals in this worktree: \`test_p1_steady_free_wake_guards.py\` +
\`test_p1_steady_free_wake.py\` + \`test_p1_lifting_g2.py\` + \`test_p1_lifting_g1.py\`
= **16 passed / 3.8 s**.
