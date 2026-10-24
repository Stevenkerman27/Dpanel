# Volume mesh builder

`build_mesh.py` uses the Gmsh Python API and OpenCASCADE to create a spherical
farfield and subtract a closed body before generating a first-order tetrahedral
mesh, then applies two Netgen optimization passes. The wingbody is a geometry
prototype: an eight-point symmetric airfoil-like polygon extruded as a straight
wing and fused with an ellipsoidal fuselage. This is only a coarse polygonal
approximation, not an exact NACA 0012 profile. It does not import an OpenVSP
model or an OpenVSP mesh.

The Gmsh Python package and its Windows DLL are expected in
`audit/runtime/python`. Example invocations from the project root:

```powershell
C:/Users/zyx20/anaconda3/python.exe volume/build_mesh.py --shape sphere --size 1.0 --radius 1.0 --out volume/meshes/sphere_coarse
C:/Users/zyx20/anaconda3/python.exe volume/build_mesh.py --shape sphere --size 0.55 --radius 1.0 --out volume/meshes/sphere_fine
C:/Users/zyx20/anaconda3/python.exe volume/build_mesh.py --shape ellipsoid --size 0.5 --radius 1.0 --out volume/meshes/ellipsoid_coarse
C:/Users/zyx20/anaconda3/python.exe volume/build_mesh.py --shape wingbody --size 0.8 --radius 1.0 --out volume/meshes/wingbody_coarse
```

`--size` is Gmsh's target characteristic length in model units; `--radius`
sets the sphere reference radius and all other dimensions scale from it. The
farfield is a sphere of radius `5 * radius`. Ellipsoid semiaxes are
`(3 * radius, radius, radius)`. The dilated ellipsoid is partitioned at its
center by an internal OCC plane before the farfield subtraction; this preserves
the external ellipsoid exactly while avoiding a Gmsh closed B-spline meshing
failure seen without the partition.

Each output directory contains `mesh.npz` and `mesh.json`. The NPZ arrays are:

- `points`: `(n, 3)` float coordinates.
- `tets`: `(m, 4)` zero-based point indices, with positive signed volume.
- `boundary_triangles`: `(b, 3)` zero-based point indices, oriented out of the
  fluid volume. Body-wall normals consequently point into the body.
- `boundary_tags`: `(b,)` integer tags: 100 farfield, 10 sphere or ellipsoid,
  20 fuselage, 30 wing.
- `boundary_tet`: `(b,)` index of the unique adjacent tetrahedron. Face groups
  are concatenated by local face, so their parent tet indices are tiled in the
  same group order.

Boundary faces are extracted from tetrahedron face incidence, not from a
separate surface mesh, so each exported boundary triangle has exactly one
adjacent tet. Their normals are oriented using the opposite tet vertex.
Legacy P1 wingbody tags were assigned with an ellipsoid level-set tolerance of
0.15 and are historical prototype data, not OCC provenance. That prototype
uses chord/fuselage x, thickness y, span z and is not aligned with the
FEM alpha x-z convention.

Run the independent boundary checks with:

```powershell
C:/Users/zyx20/anaconda3/python.exe volume/test_mesh.py
```

`--rebuild-boundary --out volume/meshes/<name>` repairs boundary arrays and
metadata from an existing mesh without rerunning Gmsh. Before revision 2, the
face-to-tet mapping used `repeat` although faces were concatenated by face
group. The original NPZ and JSON are preserved in each `original_v1/`
directory. The ellipsoid's first failed attempt and successful retry logs are
also retained in its output directory.

## Curved P2 Meshes

`build_curved_mesh.py` generates second-order tetrahedral fluid meshes directly
from OCC geometry. The accepted ellipsoid candidate is generated with:

```powershell
C:/Users/zyx20/anaconda3/python.exe volume/build_curved_mesh.py --shape ellipsoid --size 0.5 --radius 1.0 --out volume/meshes/curved/ellipsoid_s05_v4
```

The script also generates an OCC wingbody candidate:

```powershell
C:/Users/zyx20/anaconda3/python.exe volume/build_curved_mesh.py --shape wingbody --size 0.8 --radius 1.0 --out volume/meshes/curved/wingbody_s08_v2
```

This wingbody is still a geometry prototype: an eight-point airfoil-like
polygon, not exact NACA 0012 and not an OpenVSP import. Its chord/fuselage axis
is x, span y and thickness z. OCC Boolean-result BSpline fuselage faces receive
tag 20 and planar wing faces receive tag 30 by CAD boundary surface entity
membership, without a level-set threshold. The patches are adjoining and are
not expected to close individually. No trailing-edge wake curve is exported
yet.

The solver-ready curved ellipsoid mesh is
`volume/meshes/curved/ellipsoid_s05_v4/mesh.npz`; the OCC-tagged curved wingbody
candidate is `volume/meshes/curved/wingbody_s08_v2/mesh.npz`. Failed and
superseded candidates remain in their own directories without being
overwritten.

Run `volume/test_mesh.py` to check the legacy P1 outputs and all successful
curved meshes, including P2 edge-node adjacency, outward orientation,
closed-wall area vectors, array/source hashes and recorded Jacobian checks.

## Curved P2 Acceptance Protocol

The P2 acceptance candidate must be generated from the analytic OpenCASCADE
fluid-domain geometry, with the ellipsoid and farfield retained as CAD surfaces.
Do not obtain an accepted mesh by shrinking a failed projection or clipping
curved nodes. Gmsh's native second-order construction and high-order mesh
optimizers are permitted; the final exported coordinates must remain on their
intended analytic CAD surfaces.

Accept a candidate only when all of these checks pass:

- Volume elements are quadratic tetra10 with vertex order `0,1,2,3`, followed
  by edge nodes `01,12,20,03,13,23`. Boundary elements are triangle6 with
  vertices followed by edge nodes `01,12,20`.
- Every triangle6 vertex face is a unique one-sided face of its recorded
  adjacent tetra10; `boundary_tet` identifies that tetrahedron, not a positional
  coincidence in the element arrays.
- All tetra10 Jacobian determinants are strictly positive on a dense
  barycentric lattice of at least order 12. Record the minimum, sample count,
  and method. A failed candidate retains metadata and no accepted `mesh.npz`.
- Projected body nodes satisfy the analytic sphere/ellipsoid surface equation;
  farfield nodes satisfy the farfield sphere equation. Record maximum CAD
  residuals, quality statistics, surface tags, and source geometry parameters.
- Boundary orientation points out of the fluid. The closed body and farfield
  area-vector sums are near zero.

Wingbody labels must come from OCC Boolean-result boundary face provenance,
not level-set tolerances. The intended coupled-solver axes are fuselage/chord
x, span y and thickness z. An explicit trailing-edge curve/wake input remains
future work; the current curved wingbody candidate does not claim to contain it.
