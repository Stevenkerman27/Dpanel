# PR #22 round 2: exact mirror oracle on the real half mesh, and the real wing-body CAD

Head under test: `25e91f797221e1320f4ceb345b49e44e0b1aea2b` (`feature/occ-half-domain-cad-mesh`).
Environment: Python 3.11.17, JAX 0.10.2 (CPU, x64), NumPy 2.4.6, SciPy 1.17.1, Gmsh 4.15.2
from `audit/runtime/python`, `TMP/TEMP/TMPDIR=runs/pipwork`.

## 1. Tests

- `pytest -q test/test_half_surface_mesh.py test/test_half_native_panel.py test/test_half_mirror_oracle.py`
  -> **15 passed, 23.55 s, exit 0** with `test_contact_root_station_is_one_shared_mirror_orbit` deselected.
- That last test needs pytest's `tmp_path`. Under this DSH sandbox `tempfile.mkdtemp` directories are not
  listable/writable (the same defect already documented for pip and numba), so pytest fails inside
  `cleanup_dead_symlinks` with `PermissionError: [WinError 5]` *before and after* the test body.
  I therefore executed exactly the same assertions from `runs/p22b_manual/run_contact_test.py`, writing the
  fixture into `runs/p22b_manual/` (a directory created by the harness): `half_circulation_unknowns 1`,
  `full_oracle_circulation_unknowns 1`, `expanded_half_state_vs_full` max abs **9.71e-17** / relative L2
  **5.70e-16**, assertions passed. No tolerance, fixture or solver code was changed.

## 2. Exact mirror oracle on the real OCC half mesh

```
python -m panel.verify_half_mirror \
  --surface runs/half_cad_wing_only_v2/surface.json \
  --out runs/half_cad_wing_mirror_v1.json --derivatives --max-half-panels 1000
```

Exit 0, wall clock about 9 s (log created 02:48:53, report written 02:49:02), peak working set **651.1 MB** (sampled every 20 ms).
`status = exact_mirror_discretization_verified`; source SHA-256 `be02ac088bd4551521328fee6c1da70818916aef1be254b1ef2aaa9fe2ec66a6`;
half 276 wall / 24 gamma / 300 unknowns; oracle full 552 wall / 47 gamma / 599 unknowns;
`physical_accuracy_accepted = false`.

| quantity | max abs | relative L2 |
|---|---|---|
| `A_half` vs `A_full[R,:]E` | 1.038e-16 | 7.357e-17 |
| RHS | 4.163e-17 | 4.645e-16 |
| half mu vs full positive | 2.165e-15 | 4.556e-15 |
| half gamma vs full positive | 3.719e-15 | 5.762e-15 |
| expanded half state vs full | 3.719e-15 | 4.186e-15 |
| raw Cp, right side | 1.457e-13 | 4.495e-14 |
| pressure forces, right side | 9.714e-17 | 4.264e-15 |
| pressure moments, right side | 9.216e-17 | 4.339e-15 |
| full force | 3.997e-15 | 3.196e-15 |
| full moment | 1.915e-15 | 1.502e-14 |
| CL | 1.554e-15 | 3.035e-15 |
| CM | 7.772e-16 | 1.497e-14 |

CL: half 0.5120946880723137, oracle full 0.5120946880723122.

Station-geometry VJP (`dCL/dstation_y` style derivative of the same objective): half AD and full AD agree to
2e-16 absolute; against three step sizes of central differences:

| h | half AD | full AD | half abs err | full abs err |
|---|---|---|---|---|
| 1e-3 | -0.003870430016543074 | -0.003870430016543273 | 8.02e-13 | 5.24e-13 |
| 1e-4 | -0.003870430016543074 | -0.003870430016543273 | 9.12e-13 | 8.27e-12 |
| 1e-5 | -0.003870430016543074 | -0.003870430016543273 | 3.25e-12 | 2.18e-13 |

This is the machine-precision identity the previous round could not claim: on the real OCC half mesh the
half-native operator is exactly the positive-row restriction of the mirrored full operator, for matrix, RHS,
state, Cp, panel forces/moments, integrated loads and the station derivative. The unrelated 488-panel Gmsh
full mesh is deliberately not used here.

## 3. Real wing-body half CAD: first attempt succeeds

```
python -m panel.cad_half --wing-cad lifting3d/results/cad_inputs_v1/naca4412_wing_body/wing.step \
  --body-cad lifting3d/results/cad_inputs_v1/naca4412_wing_body/body.step \
  --trailing-edge lifting3d/results/cad_inputs_v1/naca4412_wing_body/trailing_edge.json \
  --size 0.3 --far-size 1.5 --radius 5 --wake-strips 8 --out runs/half_cad_wingbody_v1
```

Exit 0 on the first try, 2.388 s. `manifest.json`: `half_surface_created_not_physical_accuracy`,
396 right wall / 385 right wake / 22 gamma, `geometry_is_from_OCC_half_domain=true`,
`full_aircraft_mesh_generated=false`.

| quantity | value |
|---|---|
| nodes / wall faces / wake faces | 400 / 396 / 385 |
| components | **[20, 30]** (wing and body both present) |
| min y over all nodes | 0.0 |
| nodes exactly on y=0 | 35 (`max_plane_snap` 7.58e-17) |
| boundary edges / all on y=0 | 28 / 28 |
| non-manifold edges | 0 |
| TE edges / upper / lower banks | 44 / 44 / 44 |
| cut edges: TE / smooth wall contact | 44 / **6** |
| stations / gamma DOF | 26 / 22, dof `[0,0,0,0,1..21,-1]` (root contact folds 4 stations) |
| wall area / wake area / reference area | 6.851866 / 59.921578 / 4.0 |

So the geometry contract holds: real wing-body wetted surface with two components, six non-TE
`smooth_wall_contact` cut edges, 44 genuine exposed TE edges with both banks, a conforming closed seam
on the symmetry plane, no artificial cap inside the wall set, and no cross-plane panels.

## 4. Wing-body native solve: blocked on a degenerate pressure stencil

`python -m panel.run_half_native --surface runs/half_cad_wingbody_v1/surface.json --mode lu` fails:

```
ValueError: Half pressure reconstruction stencil rank deficient
  panel/half_native.py:232  (validate_half_configuration)
```
`runs/half_cad_wingbody_lu_v1/report.json`: `status=failed`, peak working set 164.1 MB, exit 1.

Diagnosis (`runs/p22b_manual/diag_stencil.py`, `runs/p22b_manual/diag_face.py`): exactly 2 of 396 faces
have a rank-1 stencil, both on component 30 (body) at the wing-body junction TE:

- face 238: center (0.98726, 0.29828, -1.5e-04), area 4.78e-04, singular values [7.14e-02, 0.0];
- face 359: center (0.98708, 0.29814, +3.55e-03), area 5.02e-04, singular values [7.14e-02, 0.0].

Both are TE slivers whose 9-slot stencil contains only 4 distinct neighbours (all on one spanwise line) plus
5 self-pads. The contact cut edges are excluded from the pressure neighbours by design
(`_pressure_neighbors_from_half(..., cut_edges)`), so these two slivers keep a collinear one-sided stencil.
Minimum singular ratio over the other 394 faces is fine (median 0.663).

I did not relax the 1e-6 gate, did not re-include contact edges, did not delete or fuse anything, and did not
treat the y=0 cap as a wall. The wing-only half mesh (276 faces) still solves: LU true residual 1.799e-15.

## 5. Conclusions

1. The exact mirror identity now holds on the real OCC half mesh at machine precision, including loads and the
   station-geometry VJP, so the half reduction is verified for this discretization rather than merely
   approximated. Physical accuracy remains unaccepted.
2. The real wing-body half surface is generated correctly on the first attempt and passes every geometry
   contract item; its native solve is blocked by two junction TE slivers with a collinear stencil.
3. The blocker is in the neighbour/stencil policy for contact-adjacent TE slivers, not in the CAD cut, the
   winding, the seam or the loads; fixing it needs a deliberate decision (for example a wider stencil ring or
   a documented one-sided fallback with a rank guard), not a tolerance bump.

## 6. Artifacts

- `runs/half_cad_wing_mirror_v1.json` (+ `runs/half_cad_wing_mirror_v1.log`), peak `runs/pr22b_peak/mirror/`.
- `runs/half_cad_wingbody_v1/{surface,manifest,gmsh.log,te_check_0,te_check_1,source_copy_removal}.json`.
- `runs/half_cad_wingbody_lu_v1/report.json` (failed) and `.log`.
- `runs/p22b_manual/{run_contact_test.py,diag_stencil.py,diag_face.py,show_mirror.py}`.
