# tools/issue71 — evidence scripts for issue #71

These are the four measurements behind `docs/ISSUE71_VELOCITY_AND_KUTTA_TESTS.md`.
They import the production modules (`panel.p1_lifting_g1`, `panel.p1_lifting_g2`) and only
read them; no production default is changed.

Run from a checkout whose `panel/` is on `PYTHONPATH`, e.g.

```powershell
$env:PYTHONPATH='<checkout>;$env:PYTHONPATH'
$env:TMP='<scratch>'; $env:TEMP=$env:TMP; $env:TMPDIR=$env:TMP
& python tools/issue71/pointwise_velocity.py
```

| script | what it measures |
| --- | --- |
| `pointwise_velocity.py` | |V| at fixed mid-span surface points as the chordwise mesh is refined |
| `bisect_cut.py` | the same, with no TE cut, with the cut and Gamma=0, and with Gamma!=0 |
| `fixed_gamma.py` | prescribed elliptical circulation, plus the kut_weight / max_nfev dependence |
| `kutta_variants.py` | Kutta estimators that avoid the trailing-edge-adjacent cell |

All four are single-file scripts with a `main()`; they print tables and write nothing.