# issue #65 Phase B1: material nodes on a P1 doublet sheet

Status: research only. physical_accuracy_accepted = false. No default is changed,
no coefficient convergence is claimed, and no solver in the production path calls
this module yet.

## 1. What is being replaced

The P1 steady free wake in panel/p1_steady_free_wake.py estimates the wake velocity
with two hand-tuned parameters:

- mean_wake_velocity samples the potential gradient on both sides of each face
  centroid at a height h = probe_fraction * sqrt(area), then averages the two sides;
- _one_wake_target marches the spanwise coordinates with dy/dx, which requires
  Vx > 0 and freezes the node otherwise.

docs/P1_FREE_WAKE_PROBE_FIX.md measured what that costs: across probe floor
fractions 0.0 / 0.3 / 0.6 / 1.2 the main-wing slip RMS falls 0.2994 / 0.2395 /
0.1774 / 0.1106 while the worst node stays at 1.46 times the shortest edge, and the
iteration has no fixed point (normal velocity in the junction band 1.77 U against
sin(5 deg) = 0.087 U). Both parameters move the answer, so neither is a property of
the flow.

Phase A (panel/wake_velocity.py, PR #66) removed one source of arbitrariness: the
sheet velocity is now defined exactly, as a solid-angle term plus its boundary
integrals, with the jump condition u+ - u- = gamma x n verified to 5.4e-5 and the
normal jump exactly zero.

## 2. The node velocity used here

Phase B evaluates that same kernel at a node and takes the mean of the two one-sided
limits,

    u_node = 1/2 [ u(x + eps n) + u(x - eps n) ],  eps -> 0

which is the principal value plus the local term, i.e. the material velocity of the
sheet. eps is 0.05 times the shortest incident edge at that node, refined along the
ladder (1, 1/2, 1/4, 1/8); the reported convergence is the relative change between
the last two levels of that ladder, so the limit is measured, not assumed.

The jump is kept as an output, not discarded: DeltaCp = Cp_plus - Cp_minus =
-2 Vmean . (Vplus - Vminus) / U^2 in the convention of the half-wake audit, and the
algebraic identity between that and (|Vminus|^2 - |Vplus|^2)/U^2 is reported as a
residual. Measured residual on the reference case: 7.1e-14.

Nothing else is added: no vortex core, no cutoff, no probe floor.

## 3. The two estimators are not the same operator

Iteration 0 means both estimators see the very same strength field: the sheet has an
elliptic spanwise loading mu = 0.3 sqrt(1 - (y/span)^2) on a 6x6 strip (49 nodes, 72
faces, freestream (1,0,0)). A gradient of the potential at a point is the exact
velocity at that point, so the old two-sided probe can be reproduced without a
finite-difference error term; that is what probe_velocity_face_centers does.

    probe vs nodal PV, core faces (32)   max 0.260 U   (relative to |U|)
    probe vs nodal PV, all faces (72)    max 0.675 U

So the probe offset is not a small correction: at the parameter the P1 wake actually
used (probe_fraction = 0.055) the two estimators disagree by a quarter of the
freestream on faces away from every edge. Any convergence claim built on the probe
value inherits that gap.

## 4. Where the inviscid node velocity is defined at all

Measured eps convergence, reference case:

    region                       nodes   median      max
    core (1<i<nx, |y|<0.8 span)     25   1.396e-2   5.442e-2
    near tip (|y| >= 0.8 span)      14   7.400e-2   7.400e-2
    TE attachment row (i = 0)        7   2.720e-2   3.700e-2
    boundary nodes                  24

The ordering is the physics, not noise. A doublet sheet whose strength jumps at its
lateral edge carries a concentrated vortex line there, and the elliptic loading has
an infinite dGamma/dy at the tip: the inviscid velocity diverges as eps shrinks and
no ladder can converge it. With a linear spanwise loading the effect is stronger
(core convergence 2.2e-2, dCp RMS 51.7 dominated by the edge), which is why the
reference case uses elliptic loading with zero strength at the free edge.

Consequence, and the reason the gate exists: those nodes are refused, not averaged.
A finite core would make them finite, but a core is a numerical parameter and would
have to be labelled as one, so it is not introduced here.

Slip on the same case, area weighted, three streamwise bands: 0.3466 / 0.3420 /
0.3466. The Cp jump separates cleanly: 53.03 in the band nearest the TE cut (the cut
itself carries the bound vortex), 5.02e-4 in the middle band, normal jump exactly 0.

## 5. Marching the nodes

Six arc-length steps, ds = 0.08, gates at max_eps_convergence = 2e-2:

    step  accepted  core  slip RMS  dCp RMS   eps conv  min area  aspect p90
    1       21      15     0.3485    37.887    0.617     0.02694    2.47
    2       16      15     0.3805    33.101    2.060     0.01907    2.84
    3       15      15     0.4451    87.034    4.910     0.01521    3.15
    4       14      13     0.4185   192.151   10.300     0.01450    3.75
    5       13      13     0.5169   193.281    9.930     0.00637    3.94
    6       10      10     0.5222   192.842    9.930     0.00656    4.32

Refusals are counted per gate: convergence 21..32 nodes, roots 7, and no node was
ever refused for area, fold, wall or self-intersection in this case. The sheet rolls
up, the minimum area falls by a factor 4, the node velocity loses its eps limit and
the gate progressively stops the sheet. The wake does not reach a fixed point within
six steps, which is the same conclusion as the probe study, now with a stated reason
instead of a tuned floor.

The two modes differ only in speed: arc_length moves each node ds along its own
streamline, pseudo_time moves it ds * u / |U|. Directions agree to 8.5e-7 deg on the
first step.

## 6. What this does not claim

- mu and Gamma are not re-solved while the nodes move; this is iteration 0 of the
  comparison issue #65 asks for, and the coupled re-solve is Phase B2.
- No lift, drag or moment is reported, no Kutta residual, no CL_pressure vs
  CL_gamma comparison: the sheet is synthetic and has no wing attached.
- No steady free wake is claimed, and the TE attachment row is held fixed by
  construction, so trailing-edge movement is not tested.
- The march is a synthetic geometry, not a real CAD case; G1 and G2 come later.

## 7. Reproduce

    # from the repository root, with the project interpreter
    python -m panel.run_wake_nodes --steps 6 --json runs/b1_report.json
    pytest test/test_wake_nodes.py -q

Reference numbers above are from the 6x6 elliptic case with the default flags.
