# Planform Mesh / Wake / Kutta Independent Audit

This is a read-only audit of the three saved `results/planforms_v1` meshes and their existing weak-Kutta fields. It does not invoke Gmsh or rerun a solver. Input files are hashed below; source files are inspected but not modified.

## Findings

**No node-Kutta cancellation was demonstrated.** Across the three saved meshes, upper/lower TE pairs have distinct global node IDs and distinct selected adjacent-wall-tet sets, with at least two selected wall tetrahedra per trace. The suspected same-row cancellation mechanism is therefore not supported for these meshes. This is a selection-coverage audit, not a full matrix-rank or alternate-mesh proof.

**No planform mapping or strip-crossing defect was found in these three saved meshes.** The independently evaluated local quarter-chord sweep, taper chord, TE x, section thickness mapping, wake strip interval membership and weak-Kutta station support are reported per mesh. A clean geometry/topology result is not aerodynamic accuracy certification.

## Independent Measurements

### taper

- Geometry: `b=4.0`, root chord=`1.0`, taper=`0.5`, quarter-chord sweep=`0.0 deg`, reference area=`3.0`; 24593 nodes, 14080 tetrahedra.
- Surface map: 36148 interior spanwise P2 surface samples; max TE x error=`0.000e+00`, max TE z=`0.000e+00`, max TE/Gamma-station y error=`2.220e-16`; absolute-thickness versus analytic NACA section RMS/max/p99=`2.677e-05/3.319e-04/1.041e-04`.
- Planform endpoint check on interior-span z=0 wing boundary corner nodes (tip caps excluded): 335 nodes; not near expected LE/TE=`0`, maximum x/c endpoint error=`2.220446049250313e-16`.
- Wake: 486 upper triangles, upper/lower coordinate max error=`0.000e+00`; Gamma-station crossings=`0`; empty strips=`[]`; faces by strip=`[19, 19, 19, 19, 19, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 19, 19, 19, 19]`.
- Solved potential jump measured from `phi[upper]-phi[lower]`: max/RMS `|dGamma/dx|`=`6.217e-15/7.627e-16`; max nodal mismatch from the saved piecewise-linear Gamma=`4.441e-16`.
- Node Kutta coverage: 23 pairs; upper/lower same node=`0`; identical adjacent-wall-tet sets=`0`; predicted zero rows=`0`; minimum selected wall tets per side=`2/2`.
- Weak Kutta coverage: selected TE traces=`96` (solver recorded `96`); Gamma hats covered=`23/23`, missing indices=`[]`.
- Existing weak solve: `converged_subsonic_unvalidated_lifting`, converged=`True`, CL pressure/Gamma=`0.380885994/0.384626809`. Mesh/wake geometric checks passed=`True`.
- SHA256: mesh `16237a970ac650442a1729a584f89a495cb251928c53d0ca05368e3d79d70fe0`, field `d99ddde0fcc6d41835b03bf0d0cc86a48404751ef9ad2406e80a4211aa9f93ca`, metadata `032b1989e6f1fcc5e73fb27dd3dbaf7a093696ed8857021ba58b9b22883c7531`, builder snapshot `a1186ea493608b736e59b0591459ffbece23ba1e5938e11f7b74e458ae9f595a`, solver snapshot `9d6a4d77592fdef0e0ac7c43f10b147019b3628a7ebffefcaeca9ed49eb2e272`.

### sweep

- Geometry: `b=4.0`, root chord=`1.0`, taper=`1.0`, quarter-chord sweep=`20.0 deg`, reference area=`4.0`; 27993 nodes, 16066 tetrahedra.
- Surface map: 36632 interior spanwise P2 surface samples; max TE x error=`4.441e-16`, max TE z=`0.000e+00`, max TE/Gamma-station y error=`2.220e-16`; absolute-thickness versus analytic NACA section RMS/max/p99=`2.725e-05/2.596e-04/1.111e-04`.
- Planform endpoint check on interior-span z=0 wing boundary corner nodes (tip caps excluded): 349 nodes; not near expected LE/TE=`0`, maximum x/c endpoint error=`4.440892098500626e-16`.
- Wake: 456 upper triangles, upper/lower coordinate max error=`0.000e+00`; Gamma-station crossings=`0`; empty strips=`[]`; faces by strip=`[19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19]`.
- Solved potential jump measured from `phi[upper]-phi[lower]`: max/RMS `|dGamma/dx|`=`5.940e-15/9.251e-16`; max nodal mismatch from the saved piecewise-linear Gamma=`4.233e-16`.
- Node Kutta coverage: 23 pairs; upper/lower same node=`0`; identical adjacent-wall-tet sets=`0`; predicted zero rows=`0`; minimum selected wall tets per side=`2/2`.
- Weak Kutta coverage: selected TE traces=`96` (solver recorded `96`); Gamma hats covered=`23/23`, missing indices=`[]`.
- Existing weak solve: `converged_subsonic_unvalidated_lifting`, converged=`True`, CL pressure/Gamma=`0.331743995/0.334445467`. Mesh/wake geometric checks passed=`True`.
- SHA256: mesh `c9487e6428a6c2f5a5c5de6a869933e011fb0c68d370b0b56fc577b51b9193bc`, field `395d59cb7890d19c01e6699c7ea6aac7d0003b6ff6304052d202ed6ceab92ab0`, metadata `b86760fc467f3fa58cb67f8fbb363d7d3bd4fcf8b9ab1523b8adc8d8d05de211`, builder snapshot `a1186ea493608b736e59b0591459ffbece23ba1e5938e11f7b74e458ae9f595a`, solver snapshot `9d6a4d77592fdef0e0ac7c43f10b147019b3628a7ebffefcaeca9ed49eb2e272`.

### combined

- Geometry: `b=4.0`, root chord=`1.0`, taper=`0.5`, quarter-chord sweep=`20.0 deg`, reference area=`3.0`; 26303 nodes, 15036 tetrahedra.
- Surface map: 41816 interior spanwise P2 surface samples; max TE x error=`2.220e-16`, max TE z=`0.000e+00`, max TE/Gamma-station y error=`2.220e-16`; absolute-thickness versus analytic NACA section RMS/max/p99=`2.958e-05/3.381e-04/1.192e-04`.
- Planform endpoint check on interior-span z=0 wing boundary corner nodes (tip caps excluded): 355 nodes; not near expected LE/TE=`0`, maximum x/c endpoint error=`4.440892098500626e-16`.
- Wake: 472 upper triangles, upper/lower coordinate max error=`0.000e+00`; Gamma-station crossings=`0`; empty strips=`[]`; faces by strip=`[19, 19, 19, 19, 19, 19, 19, 19, 21, 21, 21, 21, 21, 21, 21, 21, 19, 19, 19, 19, 19, 19, 19, 19]`.
- Solved potential jump measured from `phi[upper]-phi[lower]`: max/RMS `|dGamma/dx|`=`5.662e-15/8.138e-16`; max nodal mismatch from the saved piecewise-linear Gamma=`4.441e-16`.
- Node Kutta coverage: 23 pairs; upper/lower same node=`0`; identical adjacent-wall-tet sets=`0`; predicted zero rows=`0`; minimum selected wall tets per side=`2/2`.
- Weak Kutta coverage: selected TE traces=`96` (solver recorded `96`); Gamma hats covered=`23/23`, missing indices=`[]`.
- Existing weak solve: `converged_subsonic_unvalidated_lifting`, converged=`True`, CL pressure/Gamma=`0.380668108/0.383219519`. Mesh/wake geometric checks passed=`True`.
- SHA256: mesh `7e316c4c98a0726f69860d13867d3d6f1d66ec6d470b73d67f4abc3c38023b9a`, field `6e0f41a92a15c4f48a8d09ecb21490115ed07953dbff2df0aaf3a834b47c3ee0`, metadata `06549a9ce4eea0c5bb152d0d8a9cbc22409177b8bac3a284632308d2e5820094`, builder snapshot `a1186ea493608b736e59b0591459ffbece23ba1e5938e11f7b74e458ae9f595a`, solver snapshot `9d6a4d77592fdef0e0ac7c43f10b147019b3628a7ebffefcaeca9ed49eb2e272`.

## Source-Level Interpretation

- Taper and sweep are encoded as `c(y)=1+(taper−1)|2y/span|`, quarter-chord x=`|y| tan(sweep)`, and LE x=`x_qc+0.25(1−c)`. The TE is consequently `x_qc+0.25+0.75c`; the audit recomputes this against stored TE nodes and section surface samples.
- Wake strip boundaries are created at the same uniform `gamma_stations`; each saved wake triangle is checked by its corner-y interval. The saved field’s jump is independently extracted from duplicated-coordinate `phi` traces, then fitted against local x and y on each quadratic wake triangle.
- Weak Kutta uses TE edges detected from the local TE mapping and integrates against the spanwise Gamma hats. The audit separately checks every internal hat has positive TE-edge support and compares the selected edge count against the saved solver diagnostic.
- The analytic NACA comparison uses the absolute surface z because the stored boundary contains both reflected upper and lower skins. The OCC boundary is a spline through 81 cosine-spaced NACA samples, so small differences from direct analytic evaluation are a CAD interpolation diagnostic, not by themselves a defect.
- The above checks cover only these three meshes and the source snapshots associated with them. They do not prove arbitrary taper/sweep, fuselage interference, tilted-wake geometry, or aerodynamic convergence.

Read-only current source SHA256:

- `build_mesh.py`: `28139c4a87f6d22f4feef6ce9511e03d96d7f4b9441607536f3414b74e7b27dc`.
- `solve.py`: `9d6a4d77592fdef0e0ac7c43f10b147019b3628a7ebffefcaeca9ed49eb2e272`.
- `analyze.py`: `c9fd4705771ca91e9d475b2f8c6fec0654c1663338432b28d8c6a8970eb5c85e`.
- `tilt_mesh.py`: `ebdec8c4311c7f37f1137eb421c79c9a501944dcad337962a9b0f5adeb6a1e15`.

Audit script SHA256: `26fdbe228f5d283e6160c5cf1fb101ea16fc7da706c3d3e03427a99b449e1e9c`.
