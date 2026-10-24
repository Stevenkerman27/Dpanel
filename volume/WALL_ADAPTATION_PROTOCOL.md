# One-Pass Wall-Aware Sphere Adaptation Protocol

## Frozen run

This is one pre-registered comparison, not a parameter sweep. Use the existing
source CAD sphere mesh `volume/meshes/p2/sphere_coarse/mesh.npz`, M=0,
`sphere-exact`, P2 tetra10, quadrature order 3, and NCPU=8 for solve and
indicator. Recompute the source solution and source indicator in the new
output directory. Refuse to continue unless the indicator result's
`field_sha256` equals the exact source `field.npz` hash. Freeze `preflight.json`
before invoking the CAD remesher.

Use a union of:

1. All source corner vertices of Dörfler 50% marked tetrahedra.
2. All source corner vertices of tetrahedra adjacent to physical wall faces
   (`boundary_tags != 100`).

Set target size to `0.5 * base_size` on that union, then hold all farfield
corner vertices (`boundary_tags == 100`) at `base_size`. Every other source
corner vertex remains at `base_size`. The wall residual and curved-wall
geometry are separate objectives from the volume-energy Dörfler indicator;
this choice is fixed in advance and is not a fitted coefficient.

Run exactly once with `run_adaptation.py --wall-refine --ncpu 8` and a new,
empty output directory such as `audit/adaptation_sphere_wallrefine_v1/`.
The driver default without `--wall-refine` remains the historical
`0.65 * base_size` Dörfler-only method.

## Unchanged physical gates

Keep the v7 gates unchanged:

- Analytic sphere Cp area RMS and maximum absolute errors must each be
  strictly lower than the freshly recomputed same-mesh baseline.
- Absolute lift and drag errors against zero must each be no larger than
  baseline.
- Raw wall area-weighted RMS `v.n`, raw wall area-weighted RMS `rho*v.n`, and
  absolute integrated total-boundary mass closure must each be no larger than
  baseline.
- Weak free-node residual must be at most `1e-10`.
- Sampled curved Jacobian determinants must stay positive and body/farfield
  tags must be present.

The indicator total is diagnostic only. The existing global fine sphere is
reported as a separate, non-nested reference in the efficiency table; it is
not an acceptance gate or a strict nested-convergence comparison.
