# M=0 Linear Angle Batch Protocol

## Scope

This is an offline linear-superposition path for converged, fixed-mesh, zero-Mach solves. It does not modify or call a nonlinear update at each requested angle, and it does not infer a lift curve or interpolate Cp. Each requested raw state is made from the two saved potential/Gamma/velocity basis states, then Cp and pressure loads are recomputed from that state.

The two inputs must be direct converged solutions at alpha 0 and a nonzero beta on byte-identical meshes and with identical solve, P2, protocol, quadrature, Kutta, and wake settings. Each basis source snapshot is checked against the hashes stored in its result, and both basis provenance records must match each other. The basis need not match whatever solver source is current later. Basis outputs must contain finite complete raw fields. The implementation rejects nonzero Mach, mismatched basis provenance/settings, failed/nonconverged bases, or a degenerate beta.

For beta basis angle beta and target alpha, the coefficients are

`w0 = cos(alpha) - sin(alpha) * cot(beta)`

`wb = sin(alpha) / sin(beta)`.

For any linear state `f`, `f(alpha) = w0*f(0) + wb*f(beta)`. This reconstructs the response to unit x and z incident velocity components. Both total potentials include their respective freestream component; the formula applies to them as solved with the same gauge.

At Mach zero, velocity is linear in the incident flow and `Cp = 1 - dot(u,u)`. Cp is recomputed at every saved surface quadrature sample, including wake/farfield faces. Component force is reintegrated from the raw wall samples as `sum(Cp*n*dS)/Sref`, with the same pressure convention as the direct solver. Geometry, normals, area weights, connectivity, face tags, and adjacent-element IDs are copied unchanged.

If a `gamma_station_dof` map is present, it must agree between both fields/results and the mesh when the mesh carries that map. Reduced Gamma stays reduced in `field.npz`; the full station jump is expanded by the map, with `-1` stations set to zero, for `potential_jump`, `cl_jump`, and station output. The map and `root_wake_support` are retained in the synthesized field/result. Normalization length and origin are inherited and checked against the mesh metadata.

The batch result contains no fabricated Newton history. It reports the two basis final weak-residual norms and the conservative propagated bound `abs(w0)*R0 + abs(wb)*Rb`. This is a bound on the weighted basis residual magnitudes, not a separately assembled target-angle weak residual. No target-angle residual is reassembled in the online path.

For source history, every available basis snapshot among `solve.py`, `p2.py`, `PROTOCOL.md`, `schur.py`, `SCHUR_PROTOCOL.md`, `root_wake.py`, and `ROOT_WAKE_PROTOCOL.md` is SHA256-recorded. Required result hash fields are checked against their matching snapshots; optional module/protocol snapshots must be present and byte-identical across the two bases. The basis provenance is not compared to current source files, so archived basis remain usable after the solver evolves.

The independent regression case is alpha 2.5 degrees, directly solved using the saved mesh, `--mach 0`, and the basis Kutta, quadrature, linear solver, and `--ncpu 8` settings. Its full available source snapshot set is recorded. If source hashes and settings match the basis, raw potential, Gamma, surface velocity, Cp, and integrated force are compared with a maximum absolute tolerance of `1e-9` and marked a same-source regression. If source hashes or settings differ, all differences are still recorded, but the result is explicitly marked diagnostic-only and is not judged against the same-source tolerance. The direct solve output is retained separately. The basis solutions and this direct solve are offline costs; only the combination/reintegration time is the online cost.

This method inherits the existing linear M=0 discretization and its physical limitations. It is not a per-angle nonlinear convergence certification or an aerodynamic accuracy claim.
