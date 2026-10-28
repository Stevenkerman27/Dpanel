# Explicit fluid-cut differentiable panels

Status on 2026-10-25: local work paused at the user's request. Geometry and
discrete derivative matrices pass in the declared scope; general robustness and
physical accuracy remain unaccepted. See [handoff](../PAUSE_HANDOFF.md) and
[method and evidence](../docs/PANEL_ROBUSTNESS.md).

From the repository root, using a Python environment with JAX, NumPy, SciPy and
threadpoolctl:

```powershell
python aero.py --case cases/panel_two_bodies.json --out runs/NEW_CASE --export-surface
python aero.py --case cases/panel_cad_wingbody.json --out runs/NEW_CAD --export-surface
python panel/solve.py --surface cases/panel_cut/aft_body.json --out runs/NEW_AFT --derivatives
```

The first command uses a complete public surface fixture and requires no CAD
installation. The CAD command additionally requires Gmsh; `cad.py` first tries
the existing local runtime under `audit/runtime/python`, then normal Python
imports. `panel/requirements.txt` records the tested solver versions. The
independent audit uses a separate environment with NumPy, Numba and threadpoolctl.
The report script additionally uses Matplotlib.

```powershell
python panel/verify.py --out runs/NEW_MATRIX --audit-python C:/Users/zyx20/anaconda3/python.exe
python panel/verify_topology.py --out runs/NEW_TOPOLOGY
python panel/verify_kernel.py --out runs/NEW_KERNEL
python panel/audit.py runs/NEW_AFT --off-surface
python panel/report.py --matrix runs/NEW_MATRIX --topology runs/NEW_TOPOLOGY --out docs/NEW_EVIDENCE
```

All experiment directories must be new. Direct solver output contains
`surface_input.json`, `validation.json`, source snapshots, `fields.npz`, and
`result.json`; failures retain `failure.json`. The case driver additionally
preserves commands, logs and `manifest.json`. Exported `surface.vtu` contains
unmodified cell Cp, component labels, pressure forces and moments. `wake.vtu`
contains the actual fluid-side sheet and its centroid jump.

The model solves incompressible Laplace flow on closed, consistently oriented
triangle walls. The wake lies in normalized z=0; TE is explicitly declared,
piecewise linear x(y), and monotone in y. Wake triangles conform to circulation
strips. Jump is constant downstream and piecewise linear across span; contact
projections impose equal unknown coefficients. The only fixed zero circulation
values are the wing tips in the supplied examples. No central TE is invented.
The finite wake ends 30 reference chords beyond the last TE station.

AD covers angle, an explicit thickness bulge and span scaling at fixed topology.
It does not differentiate CAD Boolean operations, remeshing or contact topology
changes. Surface Cp comes from tangent least squares on a fixed, geometry-ranked
stencil. This reconstruction cannot verify physical no-penetration: use the
independent field audit. There is no Cp clipping, matrix regularization, singular
value truncation or collocation displacement.
