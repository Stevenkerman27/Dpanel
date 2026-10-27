# Analytic ideal-Trefftz integral for piecewise-linear Gamma (research)

Issue #49 / PR #54 follow-up. This is a **read-only numerical
quadrature correction**, not new circulation, Kutta, wake dynamics,
or production drag. The legacy `ideal_trefftz_piecewise_linear`
samples induced downwash at segment midpoints. Its resulting number
**changes when a piecewise-linear Gamma is subdivided without changing
the function**. That is quadrature error, not a new physical model.

For both physical tips Gamma=0, integration by parts gives the energy

```text
w(y) = -PV integral Gamma'(s)/(y-s) ds / (4*pi)
D_i = -rho integral Gamma(y)*w(y) dy
    = -rho/(4*pi) sum_ij m_i*m_j*J_ij
m_i = (Gamma(b_i)-Gamma(a_i))/(b_i-a_i)
J_ij = integral_[a_i,b_i] integral_[a_j,b_j] log|y-s| ds dy
     = F(b_i-a_j)-F(a_i-a_j)-F(b_i-b_j)+F(a_i-b_j)
F(t) = 0.5*t*t*log|t| - 0.75*t*t; F(0)=0
```

Therefore for a *fixed piecewise-linear Gamma* the energy can be
evaluated analytically without station-midpoint downwash quadrature.
The new `panel/trefftz_energy_exact.py` implements this on an explicitly
**full-span** Gamma, zero at both tips, with a bounded segment count
and chunked O(N²) cross-segment matrix. Calling code must reconstruct
the physically consistent mirrored station values itself. Validation
of source handedness, full span, reference area and the far-wake
assumptions remains the caller's responsibility.

## Independent tests actually executed

`python -m pytest -q -p no:cacheprovider test/test_trefftz_energy_exact.py`
=> **16 passed / 0.15 s**, with source/test Git blobs equal to the
executed files. Tests include:
- Gamma(y)=1-|y| on [-1,1] with Sref=2,U=rho=1:
  exact CDi=log(2)/pi and CL=1.
- 2/4/8/16x subdivision of the **same** piecewise-linear Gamma must
  preserve both CL and CDi to floating point tolerance.
- Density/speed/Gamma/coordinate/reference scaling and translation,
  no circulation, invalid tip values, repeated y and input budget.
- Independent frozen 12290-wall trapezoid at 5deg, Gamma16,
  taken from `docs/issue49_vspaero/gamma/dp_s005_a5/gamma_profile.json`
  on `research/issue49-vspaero-evidence@d24e4db`.

## Exact same circulation, seven quadrature results

| Original interval subdivisions (old midpoint method) | CDi_ideal |
|---:|---:|
| 1 | 0.0102758545335 |
| 2 | 0.0108529738957 |
| 4 | 0.0111186303578 |
| 8 | 0.0112416745197 |
| 16 | 0.0112996369585 |
| 32 | 0.0113274327229 |
| 64 | 0.0113409580368 |
| **Analytic logarithmic energy** | **0.0113541664454** |

The coarse value used in the original DPanel `dpanel.json` is
**10.49% below this exact integration of the SAME Gamma**. PR #34/#54
used 32 subdivisions per interval and reported CDi=0.01132743; it is
within ~0.24% of the analytic value, while 16-to-64 Gamma DOF changes
the common-grid result by only ~0.04%.

The prior span-efficiency estimate based on Gamma CL and coarse
CDi=0.01027585 was about e=1.059 (>1 for a planar, elliptical ideal
benchmark); with fixed Gamma and the analytic CDi it is around
e=0.959. Hence the apparent e>1 **cannot be used on its own to
diagnose unphysical circulation/wake**. Using *pressure* CL instead of
Gamma CL still yields a different e and the two force estimators must
not be conflated.

## Hard limitations

This analytic expression integrates the chosen **semi-infinite flat
wake ideal lifting-line model** for a piecewise-linear full-span Gamma.
It is not a finite DPanel wake-momentum calculation, a viscous drag
prediction, a free-wake solution, or the same discrete force channel
as VSPAERO `CDiw`. It does not establish DPanel's physical drag
accuracy, and does not imply an arbitrary non-planar wing or
multi-wake state is accurately modelled by this ideal estimator.

Do not make the new routine the production default until it is
checked on NACA/elliptic analytic loading, VSPAERO pinned wake
configurations and several true Gamma resolutions. Preserve every
historical numeric record with an explicit erratum instead of
overwriting it. No mesh change, no GRASS training.
`physical_accuracy_accepted=false`.
