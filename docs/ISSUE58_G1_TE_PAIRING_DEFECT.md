# Issue #58 / PR #59: the half-CAD trailing edge is sharp, and the adapter pairs it wrongly

State: research note. `physical_accuracy_accepted=false`. No production module, default or gate changed.

## 1. What the schema actually stores

`dpanel.half_planar_cut.v1` (here `runs/half_cad_wingbody_v1/surface.json`, 396 wall faces,
400 coordinates, 385 wake faces, reference area 4.0, reference chord 1.0):

| field | shape | meaning |
| --- | --- | --- |
| `trailing_edges` | (44, 2) | **TE edge chain**: consecutive vertex-index pairs, `[[6,123],[123,35],[35,124],...]`, max index 144, every value referenced by a wall face |
| `upper_te_faces` | (44,) | upper wall triangle adjacent to each TE edge |
| `lower_te_faces` | (44,) | lower wall triangle adjacent to the same TE edge |
| `gamma_stations` | (26,) | `0.0 ... 2.0`; `gamma_station_dof` = `[0,0,0,0,1..21,-1]`, i.e. 26 stations but 22 basis cells |

Each `(upper_te_faces[k], lower_te_faces[k])` pair shares **exactly two nodes**, 44 times out of 44.
Those two nodes are the TE edge. The wing therefore has a **sharp, zero-thickness trailing edge**.

The number previously reported as a TE gap (min 0.0092, median 0.0500, "~5% chord") is the distance
between the **centroids** of the two TE-adjacent triangles. They lie on opposite sides of a common
edge, so their centroid separation is roughly two thirds of the local panel height
(median wall edge 0.2162). It is not a base thickness, and it is not evidence of a blunt base.

## 2. What PR #59 does with it

`panel/p1_lifting_g1.from_dpanel_half_surface` feeds `trailing_edges` into the `(upper_dof, lower_dof)`
slot:

- the potential-trace duplication is guarded by `if u == l`, which holds **0 times out of 44**, so no
  lower-bank TE unknown is ever created, there is no circulation degree of freedom, and `Gamma` is
  forced to the minimum-norm (zero) value by the least-squares solve;
- the 44 jump rows difference `mu` between **adjacent TE nodes along the span**, not across the wake,
  so they do not express a potential jump at all.

Measured on that mesh (mirrored to 792 wall faces, 398 nodes, 770 wake triangles, 49 gamma,
88 Kutta rows), pressure-Kutta mode:

| alpha | CL | CL_gamma | CD_pressure | pressure kutta rms | signed kutta rms |
| --- | --- | --- | --- | --- | --- |
| -5.0 | -0.027412 | +0.006203 | +0.027122 | 2.5e-5 | 2.9e-2 |
| 0.0 | -0.030749 | +0.003848 | -0.001631 | 2.5e-5 | 2.9e-2 |
| +5.0 | -0.023190 | +0.001732 | -0.037439 | 2.5e-5 | 2.9e-2 |
| +10.0 | -0.006674 | -0.000381 | -0.073568 | 2.5e-5 | 2.9e-2 |

`max |gamma| = 0.0139`. The pressure condition is satisfied at `Gamma = 0`; the signed condition is
infeasible (3% of `U`). The synthetic sharp-TE wing from the same commit gives
`CL = 0.321120` (signed) and `0.319473` (pressure) at 5 degrees, so the closure works when the
assembly has a circulation DOF.

## 3. Probe with the pairing repaired

`tools/vsp_evidence/p1/g1_te_pairing_probe.py` rebuilds only the adapter: TE pairs from
`upper_te_faces`/`lower_te_faces`, the lower-bank DOF duplicated at each shared TE node, rows sorted
by `y`. Result: 444 DOFs (398 + 46 duplicates), 89 jump rows, 49 gamma.

| mode | alpha | CL | CL_gamma | max abs gamma | kutta rms |
| --- | --- | --- | --- | --- | --- |
| signed_speed | 0 | +0.046600 | +0.040903 | 0.1447 | 4.0e-14 |
| signed_speed | 5 | +0.090023 | +0.022262 | 0.2488 | 4.7e-14 |
| pressure | 0 | +0.058644 | +0.051976 | 0.1490 | 2.5e-6 |
| pressure | 5 | +0.136754 | +0.067771 | 0.2605 | 6.3e-6 |

Both conditions become exactly satisfiable once a circulation DOF exists, and `Gamma` is no longer
zero. But `CL(5)` is still 0.090-0.137 against the production P0 value `0.5312296425`, and the
recovered `Gamma(y)` is spiky (one lobe at 0.26, the rest near 1e-2). In the probe the TE rows span
`y = -1.5 ... 1.95` while wall and wake span `0 ... 2` and mirror to `-2 ... 2`, so the
`gamma` station and tip anchoring is also wrong: the prescribed `Gamma = 0` tips sit outside the real
TE, and the interpolation is not anchored at the true TE extent.

## 4. Order of work this implies

1. Build TE pairs from `upper_te_faces`/`lower_te_faces`; duplicate the lower-bank potential trace at
   the shared TE nodes. Keep `trailing_edges` as the TE polyline only.
2. Anchor `gamma_y` at the actual TE y range with tip `Gamma = 0` at the real tip node.
3. Re-run on the identical mesh and require `CL(5)` to approach P0 `0.5312296425` before reading any
   alpha or TE-thickness trend; report `CL_pressure - CL_gamma` and `Gamma(y)` alongside.
4. Reconcile with the earlier G1 assembly (`6b5f780`): same mesh, `Gamma ~ 0`, it reported
   `CL_pressure = 0.43203` where PR #59 reports `-0.0232`. Both cannot be right.

A genuinely blunt base remains a legitimate subject, but it is Mode B: it needs its own geometry and
its own closure, and it is not what this CAD contains.

## 5. Reproduction

```
git worktree add --detach runs/pr59_git 670bd0d7
set PYTHONPATH=D:\3D\Projects\panel\runs\pr59_git
python -m panel.run_p1_lifting_g1 --half-surface runs/half_cad_wingbody_v1/surface.json \
       --alpha -5 -2.5 0 2.5 5 10 --kutta pressure --out runs/pr59_alpha
python tools/vsp_evidence/p1/g1_te_pairing_probe.py
```

Receipts: PR #59 comment 6459522775, issue #58 comment 6459538116.