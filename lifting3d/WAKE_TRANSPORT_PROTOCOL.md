# Experimental Wake Transport Protocol

## Scope

This module is an experimental graph-based wake-line transport predictor. It
transports material span coordinate `eta` downstream in a prescribed velocity
field and extends the resulting displacement to reference volume nodes. It is
not a coupled wake/flow iteration, a free-vortex-sheet method, or a validated
Euler solution.

The wake is represented as a single-valued graph

```text
y = y(x, eta)
z = z(x, eta)
Gamma = Gamma(eta)
```

This representation requires `qx > 0` everywhere on the supplied transport
grid. It cannot represent a wake that folds back in `x`, has a vertical tangent
to the x direction, self-intersects, or requires multiple `y,z` values for one
`(x,eta)`.

## Coupling API

`lifting3d/wake_transport.py` provides exactly these interfaces:

```python
displacement = integrate_displacement(x, eta, velocity)
deformed_points = extend_displacement(
    reference_points, x, eta, displacement, radius, span)
```

`x` and `eta` are strictly increasing one-dimensional arrays. For the current
normalized wing coordinates, `x[0]=1` is the trailing-edge plane and the
downstream grid reaches `x=0.5+R`, where `R=radius`. `eta` covers exactly
`[-span/2, span/2]`. The velocity array has shape `(nx, neta, 3)` and components
`(qx,qy,qz)`. The first return has shape `(nx,neta,2)` with components `(dy,dz)`.
The second return has shape `(npoints,3)` and contains absolute deformed
normalized coordinates, always rebuilt from `reference_points`.

The slope field and integration are

```text
dy/dx = qy/qx
dz/dx = qz/qx
dy(x[0],eta) = dz(x[0],eta) = 0
```

Each x interval uses a cumulative trapezoid of the two endpoint slopes. The
velocity is never clipped or sign-modified. Any nonfinite velocity or any
`qx <= 0` sample raises `ValueError`; the full input grid is checked, not only
the subset later queried by volume nodes.

## Volume Extension

At each reference node with `x_ref > 1`, the kernel queries a
`RegularGridInterpolator` built from `(x,eta,displacement)`. Its material
coordinate is `eta_ref=y_ref`. For `|eta_ref|>span/2`, the query coordinate is
clamped to the nearest span endpoint; the displacement is then tapered outside
the span. Thus the tip displacement, rather than a zero extrapolation, is the
exterior taper's amplitude.

The node weights and update are

```text
S_eta = max(1 - max(|eta_ref|-span/2, 0)/(R-span/2), 0)^2
S_z   = max(1 - |z_ref|/R, 0)^2
y_new = y_ref + S_eta*S_z*interp(dy)(x_ref, clamp(eta_ref))
z_new = z_ref + S_eta*S_z*interp(dz)(x_ref, clamp(eta_ref))
x_new = x_ref
```

Consequently, nodes at `x_ref<=1` remain unchanged, as do nodes at `y_ref=+/-R`
and `z_ref=+/-R`. The upstream farfield plane is fixed by `x_ref<=1`; the
downstream plane `x=0.5+R` is intentionally allowed to move. The span taper
fades to zero at `|eta_ref|=R`. Interpolation does not extrapolate in x:
downstream reference nodes must lie inside the supplied x grid or the call is
rejected.

For the current wing topology, the solid body lies at or upstream of the
trailing edge `x=1`, so the explicit upstream rule preserves it. A future
geometry with solid nodes downstream of `x=1` must not use this inference;
the API would need an explicit fixed-node mask before such a case can be
supported.

## Verification

Run from the project root:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python lifting3d\test_wake_transport.py
```

The test writes `lifting3d/results/wake_transport_v1/summary.json`. Analytic
checks cover uniform oblique flow, zero side flow, an x/eta-coupled velocity
field with x-linear slopes, TE displacement, upstream/body preservation,
spanwise clamping and taper, z/y farfield boundaries, permitted downstream
outflow-plane displacement, and rejection of zero and negative `qx`. The
coupled velocity test is integrated exactly by the trapezoid rule because its
slope is linear in x; it does not establish accuracy for arbitrary CFD fields.
The summary records test outcomes and source hashes.

## Limits Before Coupling

The kernel only transports graph coordinates from a supplied velocity sample.
It does not:

- solve for velocity on the updated geometry or iterate wake shape to
  convergence;
- enforce wake pressure continuity, a Kutta condition, Kelvin circulation
  conservation, vortex-sheet strength evolution, or a viscous model;
- test the deformed volume mesh for positive P2 Jacobians or element quality;
- prove independence of the interpolated result from wake-grid spacing,
  quadrature, or flow-sample placement.

Before a driver uses its output, it must rebuild from the immutable reference
mesh on every iteration, verify mesh Jacobians and boundary/topology
invariants, and reject invalid geometry without clipping the transport
velocity. Flow, wake-shape, load, and conservation convergence criteria must
remain separate. Passing the unit tests certifies only the algebra and boundary
rules of this isolated kernel, not a 3D free-wake solution.
