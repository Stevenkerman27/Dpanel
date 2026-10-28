# Issue #71: chordwise divergence of the P1 lifting chain (c=4..32)

Date: 2026-10-29. Status: **negative result**, `physical_accuracy_accepted=false`,
production defaults unchanged. This note records what the local runs show and which
explanations were tested and rejected; it does not propose a fix.

## 1. Setup

G1 synthetic wing, `make_wing(chord_panels=c, span_panels=8, camber=.02)`,
`combine_wings` -> `assemble_multi` -> `solve_multi(..., kutta='pressure', max_nfev=200)`,
alpha = 5 deg. Environment: `.conda_env` Python 3.11.17, NumPy 2.4.6, JAX 0.10.2 (CPU).

## 2. Result: the sequence is not merely non-convergent, it changes sign

| c | faces | sum(gamma) | CL | CL_gamma | max|V| | nfev |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 144 | +1.7369 | +0.457263 | +0.434215 | 2.43 | 5 |
| 6 | 216 | +1.5190 | +0.388562 | +0.379741 | 2.11 | 5 |
| 8 | 288 | +0.9847 | +0.244532 | +0.246185 | 2.40 | 5 |
| 12 | 432 | -0.5453 | -0.158040 | -0.136330 | 7.88 | 25 |
| 16 | 576 | -1.8036 | -0.450010 | -0.450908 | 11.52 | 7 |
| 24 | 864 | -2.9811 | -0.731416 | -0.745273 | 11.41 | 200 |
| 32 | 1152 | -3.3409 | -0.818735 | -0.835234 | 13.55 | 200 |

CL and CL_gamma stay together (they do not disagree), the circulation crosses zero
near c=10, and the peak surface speed grows from 2.4 to 13.55 (Cp down to -181).
At c=8 the maximum sits at x/c=0.802, y=-2.0 (wing tip), z=0.005 - the tip trailing
corner, i.e. a sharp-edge singularity amplified by the P1 face gradient, not a
physical suction peak.

## 3. Why the circulation flips sign: the Kutta equation barely depends on Gamma

- On the same solutions, ||W Gamma|| / ||A mu|| = 0.005 / 0.005 / 0.003 (c=4/6/8):
  the wall potential system is almost insensitive to circulation.
- The pressure Kutta residual is already small: mean |V| on the trailing-edge band
  is 1.155 (top) / 1.231 (bottom) at c=4 and 1.148 / 1.115 at c=12, yet sum(gamma)
  moves from +1.74 to -0.55 over the same range.
- So the Kutta equation is nearly flat in Gamma; the value it selects is dominated by
  discretisation noise, and a change of mesh flips its sign. Being 'compatible' with
  the wall fit is not the same as being identifiable.

## 4. What is already excluded (local oracle runs)

- **Wall operator, solve and gradient are consistent.** On a sphere with the exact
  solution mu = 1.5 U x, the scattered P1 collocation residual is 6.01e-03 / 1.65e-03 /
  4.30e-04 / 1.10e-04 at 80/320/1280/5120 faces (order ~1.97), cond 35-72, and the P1
  surface speed matches 1.5 U sin(theta) to 0.7% / 0.2% with v.n = 1e-16.
- **Pressure integration does not cancel badly.** sum(area * n) = -2.4e-16 / -5.2e-18 /
  -1.0e-15 relative 2.6e-16, so the '1' in Cp = 1 - |V|^2 integrates to exactly zero and
  CL(full) equals CL(keep only |V|^2) to the digit (ratio 1.000).
- **Trailing-edge topology is not the cause.** `panel/p1_te_trace_audit.py` reports
  ok=true on exactly these failing cases (7 TE pairs, 8 attached seam edges at c=4/6/8)
  and ok=true on the local real CAD fixture (88 pairs, 90 attached seam edges).
- **Not the Kutta branch, not the wake face count, not the nonlinear stopping rule:**
  signed-speed Kutta collapses identically, wake panels 3/6/10 give the same CL to six
  digits, and nfev 80/200/600 leaves the wall residual at ~1.5e-3.

## 5. What this means for the next step

1. Establish a velocity-field convergence test first: track max|V| and the gradient
   error on sharp leading/trailing/tip corners as c increases; the divergence above is
   itself that measurement.
2. Stop selecting Gamma from the |V| of the last chordwise cell. Use a trailing-edge
   integral condition on the potential jump (Morino-style), or remove the edge
   singularity before forming the pressure.
3. Only then compare a Galerkin/higher-order formulation (see
   `docs/ISSUE71_P1_GALERKIN_TE_TRACE_AUDIT.md`, section 6).

## 6. Reproduce

```powershell
Set-Location runs/pr70_git
$env:PYTHONPATH='<worktree>;D:/3D/Projects/panel/runs'
$env:TMP='D:/3D/Projects/panel/runs/pipwork'; $env:TEMP=$env:TMP; $env:TMPDIR=$env:TMP
& 'D:/3D/Projects/panel/.conda_env/python.exe' D:/3D/Projects/panel/runs/pr71_te_scale2.py
& 'D:/3D/Projects/panel/.conda_env/python.exe' D:/3D/Projects/panel/runs/pr71_closure.py
```

The scripts live under `runs/` on the working machine and are not part of the repo; the
tables above are the durable record.