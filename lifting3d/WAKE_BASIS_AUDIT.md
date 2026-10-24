# Wake Basis Audit

## 24-Strip Audit

The new sample is `meshes/strips_v1/wing015_s24/mesh.npz` with
`results/strips_v1/wing015_s24_a5`. The run and mesh metadata hashes agree.
The mesh contains 24 strips, 25 stations (including zero-jump tips), 23
unknown gamma values, and 696 paired P2 wake faces. Upper/lower face
coordinates and node pairing agree exactly.

The audit independently reconstructs the nodal B row from the two adjacent
station weights and the solved gamma values. Across all 1,445 cut-node pairs,
`B gamma - (phi_upper-phi_lower)` has RMS `1.04e-16` and absolute maximum
`4.44e-16`. The reported station jump values also match `[0, gamma, 0]`
exactly.

At all 18 order-3 triangle quadrature points on each wake face, results are:

| Quantity | Area-weighted RMS | Absolute maximum |
|---|---:|---:|
| `Delta_phi_P2 - Gamma(y)` | 1.39570995e-16 | 5.96744876e-16 |
| `d_x Delta_phi_P2` | 6.14542082e-16 | 1.28785871e-14 |
| `d_y Delta_phi_P2 - Gamma'(y)` | 3.34226083e-15 | 1.46549439e-14 |

All 696 faces lie within one gamma interval; zero faces cross an interior
station. The kink classification uses the full six-node `y` range with a
`2e-12` coordinate tolerance. This verifies the intended basis/mesh
compatibility for this planar, zero-tilt sample to floating-point precision.
It is a representation audit, not a flow, lift, or crossflow validation.

The 24-strip run reports `CL_pressure=0.33302093` and `CL_jump=0.33680919`.
Those values are recorded only as context. This isolated audit cannot establish
that the representation change caused an aerodynamic change or resolves the
prior pressure diagnostics.

## Legacy 244-Face Sample

The selected paired run and mesh are compatible topologically but not
representation-exact for the stated wake jump. The mesh provides paired P2
tri6 faces with coincident physical coordinates. The field is constructed from
the same mesh hash and reports the prescribed 53 interior `gamma` values at 55
spanwise stations (including zero-valued tip endpoints).

The solver assigns the piecewise-linear `Gamma(y)` values at wake nodes. The
resulting P2 trace is not generally the same function inside an element that
crosses a station kink. The audit evaluated the actual upper-minus-lower P2
trace and its physical `x,y` derivatives at all 18 order-3 triangle quadrature
points on every wake face.

| Quantity | Area-weighted RMS | Absolute maximum |
|---|---:|---:|
| `Delta_phi_P2 - Gamma(y)` | 1.02034898e-2 | 4.05405172e-2 |
| `d_x Delta_phi_P2` | 1.81917052e-2 | 1.95257999e-1 |
| `d_y Delta_phi_P2 - Gamma'(y)` | 9.81953418e-2 | 5.85069125e-1 |

All 244 of 244 wake faces are classified as crossing at least one interior
station. Classification uses the minimum and maximum `y` among the six P2 face
nodes and a station-boundary tolerance of `2e-12` in the mesh coordinate units;
a station coincident with that tolerance band is treated as a boundary, not an
interior crossing. The group therefore has no single-interval control faces
in this mesh.

## Independent Patch Control

For the legacy mesh, the control assigns the exact quadratic

```text
Gamma_quad(y) = A * (1 - (2*(y-y_min)/span - 1)^2)
A = max(abs([0, gamma, 0])) = 0.20716078151179254
```

at each P2 face node, then evaluates the P2 interpolant and mapped derivatives
at the same quadrature points. All three controls pass at floating-point
roundoff:

| Quantity | Area-weighted RMS | Absolute maximum |
|---|---:|---:|
| `Delta_phi_P2 - Gamma_quad(y)` | 2.37813062e-17 | 1.17961196e-16 |
| `d_x Delta_phi_P2` | 1.51416092e-16 | 2.22044605e-15 |
| `d_y Delta_phi_P2 - Gamma_quad'(y)` | 1.55220862e-16 | 2.55351296e-15 |

This confirms the trace evaluation and physical derivative mapping for a
polynomial represented exactly by the P2 basis. It does not validate the
solver's flow model.

## Interpretation and Mesh Constraint

The legacy observed mismatch verifies the representation concern: nodal
piecewise-linear values interpolated with P2 shape functions do not preserve a
piecewise-linear function across its kinks. In particular, the P2 interpolant
can vary with `x` even though the intended `Gamma(y)` is constant in `x`.
Whether this contributes to the reported lift difference or any wake pressure
diagnostic is not established by this audit; no causal inference is made.

Making each kink line a conforming boundary of the wake triangulation is a
valid sufficient construction for this planar wake and Cartesian span
coordinate, and the 24-strip mesh passes that geometric/basis check. Every
wake triangle must lie wholly within one interval
`[y_k,y_(k+1)]`; simply inserting station vertices without splitting all
crossing triangles does not suffice. On such a triangle, the prescribed
function is affine in physical `y`, and P2 reproduces it exactly when the six
nodal jump values are sampled from that same affine function. Consequences are
`Delta_phi=Gamma(y)`, `d_x Delta_phi=0`, and the correct constant
`d_y Delta_phi` within each interval.

Boundary cases to enforce when constructing strips:

- The first and last stations must coincide with the wake span limits and have
  the prescribed zero tip jump. Do not extrapolate the first or last interior
  slope beyond those limits.
- Every strip boundary must be shared conformingly by neighboring triangles;
  no triangle may straddle a kink, including triangles adjacent to the tips.
- Mid-edge P2 nodes on a strip boundary must remain on that boundary. Paired
  upper/lower coordinates and node ordering must agree exactly.
- The statement is affine in physical `y`. If the wake parameter coordinate is
  curved or nonlinearly mapped, interpolate against the intended physical
  `y`, not an assumed reference coordinate.
- A kink landing on an existing vertex or edge is a boundary case, within a
  documented geometric tolerance; it should not be classified as an interior
  crossing.

## Inputs and Reproducibility

Input mesh:
`meshes/wing015_curv32/mesh.npz`

Input field and run:
`results/paired_v1/wing015_curv32_a5/field.npz`,
`results/paired_v1/wing015_curv32_a5/result.json`

The selected result is a converged `mach=0`, `alpha=5 deg` run; it is used
only as the existing field to audit. Its reported `CL_pressure` and `CL_jump`
are context, not evidence of causality. This audit does not claim a verified
crossflow, shock, or corrected aerodynamic coefficient.

Re-run the current 24-strip audit from the project root with:

```powershell
python lifting3d\audit_wake_basis.py --ncpu 8
```

The script writes `wake_nodes_B.csv`, `wake_gauss_points.csv`, `wake_faces.csv`,
and `summary.json` under `lifting3d/audit/wake_basis_s24_v1/`. The summary
records SHA-256 hashes for the mesh, field, result JSON, and audit script, and
includes a read-only numerical comparison to the legacy result summary.

24-strip run hashes:

| Input or script | SHA-256 |
|---|---|
| `mesh.npz` | `622b9704c4d3759d0859d20e92a3af4bec29d8229557449614101a08169e0998` |
| `field.npz` | `cdbc4f3bf1d8cfd5d666fe0aad4f780052bff88c91b059a7ea45e7547dd23452` |
| `result.json` | `d2148adeaaa2f9825266cfed13466e95f1c074a89536deee36d43596d5920521` |
| `audit_wake_basis.py` | `4352796bc4c0b0e4ef0d86593c13c1212384182c9e8e6b5b39f17f04e3057df3` |

Legacy 244-face input and historical script hashes:

Recorded hashes for this run:

| Input or script | SHA-256 |
|---|---|
| `mesh.npz` | `2bd2e2dc3ad7892dbaa382be97ff6021b9a8cac354b93205adefde9703c659f9` |
| `field.npz` | `e464857dac0ced2ace5059d3deb8216de0e27212d7f5459cdb490f1e2f35fc75` |
| `result.json` | `12ff9e81b0cf5371f8ae6f740c59c1ef65c89565eb86d7b3ecfc8bb38dee7bd5` |
| `audit_wake_basis.py` | `6263166c1b479872e93306bab0cf1dfe7a44a988f76864f4903e918d05e61cf5` |
