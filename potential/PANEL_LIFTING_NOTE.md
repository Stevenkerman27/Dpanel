# Hess-Smith Joukowski lifting-panel reference

Run from the repository root:

```powershell
python potential/panel_airfoil.py --panels 128 --alpha 5 --epsilon 0.08 --eta 0 --out audit/panel_airfoil_hesssmith/n128_ap5
```

The script discretizes the mapped Joukowski contour counterclockwise, solves one
constant source strength per straight panel and one constant vortex-sheet
strength over the complete surface, and applies impermeability at every panel
center plus an equal-sign sum of the two trailing-edge tangential velocities.
It does not impose a separate source-mass constraint or alter the solved normal
velocity.

The source line-integral convention follows `panel_cylinder.py`: the exterior
normal self trace is `-1/2`. The counterclockwise vortex kernel is the source
velocity rotated as `(source_y, -source_x)`, with self tangential trace `+1/2`.
The circulation reported for the sheet is `vortex_strength * perimeter`;
positive circulation is counterclockwise, so the lift reference is
`CL = -2 Gamma / chord`.

`cp_exact` evaluates the analytic Joukowski-mapped cylinder flow at mapped
parameter midpoints. The numerical pressure is collocated at straight-panel
centers. These points do not coincide on a polygon, so their Cp difference
includes geometry and sampling error as well as the panel discretization error.
Force and lift coefficients use the chord measured from at least 65,537 mapped
surface samples; `polygon_chord` is also recorded to expose panel geometry error.
