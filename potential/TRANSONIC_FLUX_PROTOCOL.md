# Independent Transonic Flux Kernel Protocol

## Scope

This is a preregistered, independent numerical experiment. It does not edit or
certify `solve_airfoil.py`, does not solve a two-dimensional transonic flow,
and does not turn the one-dimensional E-O auxiliary evolution into a 2D
validation.

The source is Osher, Whitlow and Hafez, NASA TM-85751, Section VI, printed
pages 38-40 (PDF pages 39-41). The page images were checked locally. The paper
defines

```text
J_inv = det(d(xi)/d(x))
U_i   = sum_j a_ij phi_xj
a_ij  = sum_k (xi_i,xk)(xi_j,xk)
```

so `U_i` is the general contravariant velocity without a determinant. Its
coordinate conservation flux is `rho U_i/J_inv`. With the forward map
`J_phys=det(d(x)/d(xi))=1/J_inv`, the same flux is
`J_phys rho U_i`. Thus the metric factor belongs on the flux, not on
`rho U_i` after taking a coordinate divergence. The paper's equations (6.1)
and (6.2) split this metric-weighted directional flux using the E-O
construction. They are a difference scheme with one-sided metric-gradient
factors; they are not the candidate centered-density FV interpolation tested
below.

The independently implemented Cartesian face kernel freezes the reconstructed
tangential gradient at a face and applies the scalar E-O split to the normal
gradient on that face. A single integrated face flux is returned and reused
with opposite signs by neighboring cells. This is an FV construction
motivated by Section VI's transformed directional flux, not a transcription
of (6.1)/(6.2), and no multidimensional entropy theorem is claimed.

## Candidate Under Test

The rejected-or-retained density-biased FV candidate is explicitly

```text
qbar = (qL+qR)/2
Mtilde^2 = M_inf^2 qbar^2 / beta(qbar)
nu = max(0, 1-1/Mtilde^2)
rho_f = (rhoL+rhoR)/2 - (nu/2) sign(qbar) (rhoR-rhoL)
h_cand(qR,qL) = -qbar rho_f
```

where `beta(q)=1-(gamma-1) M_inf^2 (q^2-1)/2`. This is the prior proposed
center-plus-bias FV formula, not the paper's unnumbered density-difference
stencil on printed page 28. The local Mach sensor and face placement are
candidate choices, not uniquely prescribed by TM-85751.

## Frozen Acceptance Thresholds

Use `gamma=1.4`, `M_inf=0.5,0.8`, double precision and at least 8 CPU threads
available to numerical libraries. No clipping, added viscosity, post-step
filter, or sign alteration is allowed.

1. The candidate must be consistent to `2e-12`, have first-argument
   non-increasing and second-argument non-decreasing violations no larger than
   `2e-10`, and satisfy the full interval E-flux inequality with maximum
   positive violation no larger than `2e-10`. Test dense positive and negative
   endpoint pairs and dense interior states. Any violation rejects this
   candidate for transonic use; retain the violating states and magnitude.
2. The candidate's smooth manufactured face-flux evaluation uses a periodic
   positive subsonic analytic state, exact pointwise physical mass flux, and
   center-sampled endpoint states. The error must decrease at least by a factor
   of `3.2` on each mesh doubling for `N=64,128,256,512` (observed order at
   least 1.68). Also evaluate the conservative flux-divergence residual at cell
   centers against the exact derivative for the same manufactured profile;
   its L-infinity error must decrease by at least a factor of `3.0` at each
   doubling (observed order at least 1.58). Report every error even if another
   gate rejects the candidate.
3. The separate E-O face kernel must satisfy its scalar E-flux dense-state
   check to `2e-10`; free-stream closed-polygon balance and rigid-rotation
   covariance must each be within `2e-12` relative scale; 1D degeneration must
   match the established E-O split to `2e-12`.
4. Passing the kernel checks means only that these algebraic/geometric unit
   tests pass. It does not establish a 2D fully-discrete entropy inequality,
   convergence to a multidimensional weak solution, or a transonic airfoil
   result. Those are later, independent gates.

The script writes all candidate violations and smooth errors under
`potential/results/transonic_flux_v1/`, together with source/protocol hashes,
CPU-thread environment and explicit per-gate status. An expected candidate
rejection is a successful experiment run, not an accepted method.

## Discrete Quantities and Orientation

For a physical face with oriented area vector `S`, let `n=S/|S|` and take
left/right gradients relative to that orientation. The tangential component
used by this kernel is the average tangential gradient. In the positive-flow
orientation, call the scalar split with first argument equal to the
downstream/right state and second argument equal to the upstream/left state:

```text
h(q1,q0) = -m(q1) + s(q1)-s(q0)
m(qn,qt) = rho(qn,qt) qn
s = 0                             for qn <= qn_star
s = m(qn,qt)-m(qn_star,qt)       for qn > qn_star
```

Here `beta=1-(gamma-1)M_inf^2(qn^2+qt^2-1)/2`,
`rho=beta^(1/(gamma-1))`, and `qn_star` is the nonnegative root of
`partial(m)/partial(qn)=0` with the frozen `qt`. For reversed flow, reverse
the face orientation and swap/negate the two normal states, then reverse the
integrated flux sign. This preserves a single conservative face value; it
does not authorize treating either contravariant coordinate component as
physical speed without its metric.

The physical Cartesian face flux returned by the kernel is `-h*|S|` for
positive oriented flow. An arbitrary mapped face uses the same physical
area vector; equivalently its coordinate flux density is `J_phys rho U_i`
times the computational face interval. Tests use Cartesian polygon geometry
so the area vectors are explicit and no Jacobian can be silently duplicated.

## Interpretation

The one-dimensional E-O Riemann study in `results/entropy_v1/` previously
passed all 16 runs (`M_inf=0.5,0.8`, compression/expansion, four meshes),
including its chosen fully-discrete quadratic-entropy flux check. That is
only the scalar auxiliary law. This protocol separately assesses a proposed
FV density bias and a local normal E-O face kernel. Neither outcome is a
two-dimensional transonic-flow certification.
