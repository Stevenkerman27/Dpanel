# Issue #49 Phase 1: geometry-consistent affine face-gradient probe

This is an **experimental read-only gradient operator**, not a change to
the DPanel flow solve or to production Cp. Triangular wall faces, JAX
kernel, wake, Kutta matrix and CAD are unchanged.

Input: the **original frozen cut-preserving face stencil** from PR #36,
full mirrored face centers/normals and one potential per face. It never
reselects neighbors or joins the two TE banks. The 3D ambient affine fit
uses an untruncated QR solve, with SVD for **rank diagnosis only**.
The output is then projected tangentially at the original target face.
Exactly planar support falls back to an identifiable 2D tangent fit.

Critical limitation: a geometrically curved but rank-poor patch is
reported **unsupported** with NaN gradient. A patch with neighbor normal
angle >45 deg is similarly masked; this illustrative cutoff is part
of the experiment, not a proven optimal physical constant. No
pseudoinverse, regularizer, Cp clipping or arbitrary patch expansion
may conceal an unsupported face. Only complete coverage permits global
Cp/CL/CD/CM A/B; partial coverage must report missing faces and area,
not silently integrate the supported subset as a whole aircraft.

Five standalone pure-NumPy tests passed: ambient 3D affine
reproduction on curved support, exactly planar 2D projection,
weak 3D curvature leverage rejection, gauge/scale/mirror covariance,
and malformed input rejection. SHA blobs equal tested source files.

**Local acceptance to run next:** reuse the exact original PR29/PR36
396/1205/1966 frozen state + neighbor IDs, source/mesh SHA validated.
Repeat all six PR36 manufactured potentials, compare original QR,
cut-aware vertex candidate, exact-node P1 *reference*, and this new
3D-face estimator. Report all-valid coverage by face and area,
min 3D rank, normal-angle rejections, absolute and normalized
gradient/Cp errors, signed/absolute CL/CD/CM errors, and worst
locations; do not change pressure or Gamma data. Only if full coverage
and manufactured accuracy improve consistently may a separate
real-frozen-field A/B be attempted.

Even a successful manufactured test does not prove wall-field or
near-surface physical accuracy. PR #50 sphere offset errors and
independent VSPAERO/CFD/experiment comparisons remain mandatory.
Phase 2 true P1 *solve* is still absent; PR #52 contains only its
analytical doublet-kernel oracle. physical_accuracy_accepted=false;
no merge/default/training change.
