# PR #22 local verification: OCC positive-half CAD mesher on the real NACA4412 wing STEP

Head under test: `be4336023ae059f30e11cf33c7afe70ae3101985` (base `feature/half-native-panel-operator`, PR #21).
Environment: Windows 10, Python 3.11.17, NumPy 2.4.6, SciPy 1.17.1, JAX 0.10.2 (CPU, x64),
Gmsh 4.15.2 from `audit/runtime/python`, `.conda_env` interpreter, `TMP/TEMP/TMPDIR=runs/pipwork`.

## 1. Test suite

`pytest -q test/test_half_surface_mesh.py test/test_half_native_panel.py` -> **14 passed, 18.27 s, exit 0**
before the fix and **14 passed, 18.22 s, exit 0** after it. No skips, no tolerance changes.

## 2. Geometry attempt 1: the real STEP fails (kept as evidence)

Command:

```
python -m panel.cad_half --wing-cad lifting3d/results/cad_inputs_v1/naca4412_wing_body/wing.step \
  --trailing-edge lifting3d/results/cad_inputs_v1/naca4412_wing_body/trailing_edge.json \
  --size 0.3 --far-size 1.5 --radius 5 --wake-strips 8 --out runs/half_cad_wing_only_v1
```

Result: exit 1. `runs/half_cad_wing_only_v1/failure.json` records stage
`half_OCC_boolean_and_contact_pass_0` and
`ValueError: Half fluid has no explicit y=0 symmetry boundary` at `panel/cad_half.py:217`.

Root cause (measured, `runs/pr22_plane_probe4.py`): OCC inflates the bounding box of a
trimmed planar face by the shape tolerance. The genuine y=0 cap of the positive-half fluid
has `getBoundingBox` y in `[-5.0e-06, +5.0e-06]` (area 99.917, `getType` = `Plane`), while
the gate accepted only `max(|ymin|, |ymax|) < 1e-7`. Meshing that face produces 514 triangles
whose nodes are all exactly `y = 0.0`, so the cap is planar and correctly placed - only the
bounding-box predicate was too tight. `Geometry.Tolerance` is `1e-08`.

Other fluid boundary faces sit at `|y| >= 2.0` (BSpline wing surfaces) or `|y| = 5.0`
(far-field box), so surface kind plus a `1e-5` bbox window separates the cap uniquely.

## 3. Fix (commit c46328f, branch local/pr22-half-cad-plane-gate)

`panel/cad_half.py`, two changes, both stricter rather than looser:

1. Cap identification: `max(|ymin|,|ymax|) < 1e-5 and gmsh.model.getType(2, tag) == "Plane"`.
2. New post-mesh assertions on generated nodes: every wall/wake node must satisfy `y >= -1e-12`,
   and every artificial cap node must satisfy `|y| <= 1e-12`. The acceptance criterion therefore
   moves from an inflated OCC bounding box to the meshed geometry that the solver actually uses.

## 4. Geometry attempt 2: accepted (`runs/half_cad_wing_only_v2`, exit 0, 1.289 s)

| quantity | value |
|---|---|
| status | `half_surface_created_not_physical_accuracy` |
| full aircraft mesh generated | false |
| right wall panels | 276 |
| right wake panels | 360 |
| independent mu / gamma | 276 / 24 |
| nodes / triangles / wake triangles | 318 / 276 / 360 |
| minimum y over all nodes | 0.0 (wall and wake both 0.0) |
| nodes exactly on y=0 | 15 |
| boundary edges | 8, all on y=0 (single closed root seam) |
| non-manifold edges | 0 |
| TE edges / upper / lower | 48 / 48 / 48 |
| cut edges | 48 |
| stations / DOF | 25 in [0, 2] / first 0, last -1 |
| wall area / wake area / reference area | 4.132074 / 60.0 / 4.0 |
| source-copy removal | 15 of 15 removed, fluid mass 499.8362242413775 unchanged |
| TE trim check | passed, 20 segments, 60 samples, max boundary distance 2.22e-16 |

The contract items hold: `y >= 0` everywhere, a positive-length seam (8 edges), no artificial
cap inside the wall set (the cap is excluded as a symmetry plane), TE upper/lower banks present
for all 48 edges, `gamma_station_dof[0] = 0` and `gamma_station_dof[-1] = -1`.

## 5. Solve on the real half mesh

Both runs are separate processes with peak working set sampled every 20 ms;
`physical_accuracy_accepted` stays false.

| quantity | LU (`runs/half_cad_wing_lu_v1`) | GMRES (`runs/half_cad_wing_gmres_v1`, restart 40, block 32, near 32, tol 1e-10) |
|---|---|---|
| independent state unknowns | 300 | 300 |
| true relative residual | 1.799e-15 | 2.793e-11 |
| iterations / matvecs | - | 30 / 32 |
| first / hot seconds | 0.6160 / 0.04545 | 6.060 / 1.267 |
| peak working set | 398.0 MB | 817.2 MB |
| CL | 0.5120946880723137 | 0.5120946881028932 |
| CD_pressure | 2.3115615079420286e-04 | 2.311561507784721e-04 |
| CM_y | -0.07101168133661905 | -0.0710116813361876 |
| max abs Cp_raw range | [-3.4932807388650033, 0.8416499832197042] | differs by 3.1e-11 |

LU vs GMRES: dCL 3.06e-11, dCD 1.6e-14, dCM_y 4.3e-13, dforce 7.5e-11.
At this size the streamed GMRES path is slower and uses more memory than LU, exactly as in the
22-unknown fixture; no speed or memory advantage is claimed from it.

## 6. Independent full-model reference on the same STEP (context, not acceptance)

Production full-domain mesher and solver on the same STEP (`runs/pr22_full_only_v1`,
`runs/pr22_full_solve_v2`): 488 wall panels, 600 wake panels, 80 TE edges, 41 stations,
CL 0.5224613411, CM_y -0.0760176423, cp_min -3.1377241, true residual 1.68e-15, cond 294.25.

Half vs full: CL -2.0%, CM_y -6.6%, dimensional lift 2.4998 (twice half) vs 2.0825 (-16.7%).
CD_pressure must not be compared: it is `F . [cos a, 0, sin a] / (q S)`, a cancellation of two
O(0.1) numbers (0.0066 vs 0.00057 before normalization), so 0.00166 vs 0.00023 is not a 7x error.

These two meshes are not mirror images of each other (488 vs 2 x 276 panels, 80 vs 48 TE edges,
41 vs 25 stations: Gmsh meshes each side independently), so this is a discretization-level
comparison and cannot certify the reduction. A machine-precision identity requires a full surface
built by mirroring the half JSON, which PR #22 does not provide.

## 7. Scope and limits

- PR #22 delivers generation plus a native solve. It does not deliver the mirrored full surface,
  so the exact identity `A_half == A_full[R, :] E` is still only proven at the 22-unknown fixture
  scale (PR #21), not on this real OCC mesh.
- The half wall panels are not exact mirrors of one another on this mesh, so the reduction is an
  approximation here; `physical_accuracy_accepted` remains false.
- No body CAD, no sideslip, no coarser/finer sweep, no 23536-panel run, no GPU.

## 8. Artifacts

- `runs/half_cad_wing_only_v1/failure.json` - retained first failure.
- `runs/half_cad_wing_only_v2/{manifest,surface,te_check_0,te_check_1,source_copy_removal,fixture_check}.json`.
- `runs/half_cad_wing_lu_v1/report.json`, `runs/half_cad_wing_gmres_v1/report.json`,
  `runs/half_cad_wing_gmres_v2.log` (iteration counts).
- Peak working sets: `runs/pr22_peak/lu/process_peak_rss_mb.json`, `runs/pr22_peak/gmres/...`.
- Full reference: `runs/pr22_full_only_v1/surface.json`, `runs/pr22_full_solve_v2/result.json`.
- Helper scripts: `runs/pr22_check.py`, `runs/pr22_cmp.py`, `runs/pr22_plane_probe3.py`,
  `runs/pr22_plane_probe4.py`, `runs/peak_wrap.py`.
