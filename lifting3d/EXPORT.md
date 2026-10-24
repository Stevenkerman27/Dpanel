# Surface VTU export

`export_surface.py` exports saved surface data from a converged lifting3d run.
It reads `field.npz`, `result.json`, and the matching mesh metadata, then writes
ASCII VTU files that ParaView can inspect. It uses only Python's standard
library and NumPy; it does not recalculate or alter Cp.

Run from the repository root:

```powershell
python lifting3d\export_surface.py --run lifting3d\results\wingbody_initial\a5 --mesh lifting3d\meshes\wingbody02_curv32\mesh.npz --out lifting3d\results\export_v2
```

The output directory must not already exist. This is intentional: exports do
not overwrite prior results. By default only wall tags 10, 20, and 30 are
included. Add `--include-wake` to include wake tags 201 and 202; farfield tag
100 is never exported.

`wall_surface.vtu` contains the original Tri6 surface geometry and connectivity.
Its cell arrays are:

- `cellcp_area_mean`: area-weighted mean of the saved Gauss-point Cp values;
  for visualization only.
- `tag`: source boundary tag.
- `raw_vn_rms`: area-weighted RMS of the saved normal velocity.
- `integral_Cp_n_dS`: the three-component per-face sum
  `sum_q(Cp_q * normal_q * areaweight_q)`. This is integrated directly from
  the saved quadrature values, not reconstructed from a cell-average Cp and
  average normal.

`raw_gauss_samples.vtu` stores each original surface quadrature sample as a
VTK vertex cell. Point arrays are `rawCp`, `velocity`, `normal`, `areaweight`,
`tag`, `source_face_id`, and `gauss_index`. Cp, velocity, normals, and
quadrature weights are copied unchanged from `field.npz`.

Normals point outward from the fluid domain; on solid walls they point into the
solid. Coordinates retain the saved mesh units (this reference run uses chord
length 1, so coordinates are numerically in chord units). `areaweight` is
quadrature `dS` in the corresponding squared coordinate units. `rawCp` is the
dimensionless pressure coefficient. Exported velocity is the saved
nondimensional velocity. Force coefficients in the metadata are formed by
dividing the direct pressure integral by the saved reference area.

`export_metadata.json` records source and exporter SHA-256 hashes, output
hashes, selected/excluded tags, conventions, and component force comparisons
against `result.json`. `test_export_surface.py` reads both VTU files back as
XML and checks exact sample-value and Tri6 connectivity round trips, per-face
quadrature integrals, and component-force agreement to an absolute tolerance
of `1e-12`.

Reference run:

```powershell
python lifting3d\test_export_surface.py
```

The scaled check reuses `results/invariance_v1/large_shifted`: chord 1000,
origin `[100,-200,50]`, Sref 4000000. Exported coordinates remain in the
saved physical mesh units; area weights scale with chord squared. Coordinates,
Cp and connectivity round-trip exactly, and the force coefficient error is
`1.63e-15`. Transform-source and current-mesh hashes are distinguished in the
metadata. The original unscaled check remains under `export_v1`; the additional
scaled report is `results/export_scaled_v1/scaled_validation.json`.
