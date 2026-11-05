# B3 wake-map PR #114 review

Reviewed revision: `febedcca2d86e1ef2669c8776fcdc93f66d1760f`.
Baseline: `01214d4`. Date: 2026-11-05.
Environment: Windows x64, myml Python 3.13, NumPy 2.4.6,
Torch 2.13.0+cu132, CPU, BLAS/OMP/MKL one thread.

The PR was tested in a detached checkout without changing its solver sources.
It is not merged. `physical_accuracy_accepted=false`.

## Reproduction

In a checkout of the reviewed revision, with the single myml Python environment:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m pytest -q test/test_wake_sensitivity_experiments.py test/test_p1_steady_free_wake.py -p no:cacheprovider
python -m pytest -q test -p no:cacheprovider
```

To check the original frozen B3 inputs with the reviewed revision imported:
set `PYTHONPATH` to that checkout, run from that checkout, and invoke the
review script from the baseline repository:

```powershell
python <baseline>/tools/wake_map_review.py --inputs <baseline> --out <fresh-output.json>
```

This checks regenerated geometry against the original `initial_state.npz`,
then attempts prescribed axial/oblique velocity maps and a pressure-Kutta
first update at alpha=4 degrees, relaxation=0.3. The original local CAD inputs
and frozen snapshot files are required. It does not replace or overwrite the
original experiment evidence.

## Findings

1. Blocking runtime error at `panel/p1_steady_free_wake.py:348`.
   `drift=np.zeros_like(original)` has shape `(N,3)` while the integrated
   transverse velocity has shape `(2,)`. Adding them raises
   `ValueError: operands could not be broadcast together with shapes (3,) (2,)`.
   The eventual assignment to `target[node,1:]` also requires a two-component
   drift. Both manufactured axial cases, the oblique case and two existing
   solver tests fail. The original code runs those existing solver tests.
2. Incorrect fixture/expectation at
   `test/test_wake_sensitivity_experiments.py:33`. The new metadata test creates
   an unswept rectangular wake, where declared attachments and graph roots
   are both `[0,1,2,3,4]`, then asserts that the attachment set has nodes outside
   the roots. A swept fixture is needed for that distinguishing assertion;
   the equality of metadata and declared attachments is a separate property.
3. Constructor compatibility regression at `panel/p1_steady_free_wake.py:40`.
   `WakeMaterial.attachments` is a new required argument, but existing direct
   construction at `test/test_wake_step_scale.py:23` is unchanged. Seven tests
   now raise `TypeError: WakeMaterial.__init__() missing 1 required positional
   argument: 'attachments'`. These cover coupled-material step scales,
   clipping and truncation policy. Although this PR does not change the G2 T5
   algorithm, its shared material container still affects these consumers.

Focused PR result: **6 failed, 7 passed** (1.78 s).
Focused baseline result: **10 passed, 1 xfailed** (2.15 s).
The PR removes the original strict xfail but its replacement does not pass.

Full PR result: **25 failed, 803 passed, 22 skipped** (210.77 s).
Ten failure IDs match the prior full-suite log
`runs/verification/native_blas/fixed_pytest.log`. The other fifteen consist of
seven map-dimension failures, seven constructor failures and one incorrect
metadata fixture assertion. These include the former xfail and two newly
added tests; they are not fifteen newly regressed previously passing tests.
Baseline guards/step-scale tests were rerun separately:
**12 passed** (1.97 s).

The historical eleventh failure, the CAD pressure-floor test, skips in the
detached checkout because its ignored CAD input is absent there. It was
rerun with `PANEL_ROOT` pointing at the original repository and still fails;
see `cad_floor_pytest.log`. This skip is not a fixed baseline defect.

The frozen three-case check took 94.96 s. All five geometry/material arrays
match the original evidence. Iteration-0 CL values are
`0.256144450850489 / 0.403535637981288 / 0.278756973434849`, with exactly zero
Gamma/Cp/wake-coordinate difference from the saved states. All three axial
and oblique target evaluations and all three actual first updates fail with
the dimension error. Declared attachment/root counts are `7/7`, `63/1`,
and `83/2`; explicit metadata construction works on these inputs, but its
use in a completed wake update remains unvalidated.

Raw logs and the three-case receipt are stored alongside this document.
The first-step prerequisite blocks the unchanged PR's 8-step/60-step
four-way ablation. Execution failure is not evidence that the intended map
repair succeeds or fails physically. No production algorithm, relaxation,
contact tolerance, or velocity model was changed for this review.
