"""Matrix-free wall/wake panel operator: A v, A^T v, GMRES, half-model gate.

The assembled system is the one used by differentiable.solve_fields:

    A = [[D, W], [-T, M]]      x = [mu, gamma]

D and W are never materialised. Every product evaluates frozen target blocks
of the analytic triangle kernel and contracts them with the vector at once, so
peak memory is O(block * panels) instead of O(panels^2). This is NOT a fast
multipole method: one application is still O(panels^2) work, only the storage
changes. No approximation, cutoff or regularisation is introduced.
"""
import numpy as np
import scipy.linalg as sla
import jax
import jax.numpy as jnp

from .kernel import triangle_potential


def _np(x):
    return np.asarray(x, dtype=np.float64)


class PanelOperator:
    """Matrix-free A, A^T and RHS of one prepared run."""

    def __init__(self, inputs, block=64):
        self.block = int(block)
        t = inputs["topology"]
        vertices = _np(inputs["vertices"])
        station_y = _np(inputs["station_y"])
        tri = vertices[_np(t.faces).astype(np.int64)]
        self.tri = tri
        self.nf = tri.shape[0]
        area_vector = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) / 2
        area = np.linalg.norm(area_vector, axis=1)
        self.area = area
        self.normal = area_vector / area[:, None]
        self.center = tri.mean(axis=1)
        self.sigma = -(self.normal @ _np(inputs["freestream"]))
        self.wake = vertices[_np(t.wake_faces).astype(np.int64)]
        self.station_y = station_y
        interval = _np(t.wake_interval).astype(np.int64)
        hat = _np(t.hat_map)
        self.ng = hat.shape[1]
        dy = station_y[interval + 1] - station_y[interval]
        fraction = (self.wake.mean(axis=1)[:, 1] - station_y[interval]) / dy
        self.basis = (1 - fraction[:, None]) * hat[interval] + fraction[:, None] * hat[interval + 1]
        self.slope = (hat[interval + 1] - hat[interval]) / dy[:, None]
        gauss = np.asarray([.5 - np.sqrt(3.) / 6, .5 + np.sqrt(3.) / 6])
        ends = vertices[_np(t.te_edges).astype(np.int64)]
        te_y = ends[:, 0, 1, None] * (1 - gauss) + ends[:, 1, 1, None] * gauss
        ti = _np(t.te_interval).astype(np.int64)
        tf = (te_y - station_y[ti]) / (station_y[ti + 1] - station_y[ti])
        te_basis = (1 - tf[..., None]) * hat[ti] + tf[..., None] * hat[ti + 1]
        length = np.linalg.norm(ends[:, 1] - ends[:, 0], axis=1)
        weights = np.broadcast_to(length[:, None] / 2, te_y.shape)
        self.mass = np.einsum("eq,eqi,eqj->ij", weights, te_basis, te_basis)
        self.integrals = np.einsum("eq,eqi->ei", weights, te_basis)
        self.measure = self.integrals.sum(axis=0)
        self.mass = self.mass / self.measure[:, None]
        self.upper = _np(t.upper_te).astype(np.int64)
        self.lower = _np(t.lower_te).astype(np.int64)
        self.n_te = len(self.upper)
        self._tri_j = jnp.asarray(self.tri)
        self._wake_j = jnp.asarray(self.wake)
        basis_j, slope_j = jnp.asarray(self.basis), jnp.asarray(self.slope)

        @jax.jit
        def _doublet(targets, rows):
            _, doublet = triangle_potential(targets, self._tri_j, interior_diagonal=True,
                                            block_size=0, row_index=rows)
            return doublet

        @jax.jit
        def _wake(targets):
            dw, my = triangle_potential(targets, self._wake_j, source=False,
                                        linear_y=True, block_size=0)
            return dw @ basis_j + my @ slope_j

        @jax.jit
        def _source(targets):
            single, _ = triangle_potential(targets, self._tri_j, block_size=0)
            return single

        self._doublet_jit = _doublet
        self._wake_jit = _wake
        self._source_jit = _source

    # ---- dense blocks, evaluated per frozen target block --------------
    def doublet_rows(self, i0, i1):
        rows = jnp.arange(i0, i1)
        return _np(self._doublet_jit(jnp.asarray(self.center[i0:i1]), rows))

    def doublet_rows_indexed(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        return _np(self._doublet_jit(jnp.asarray(self.center[rows]), jnp.asarray(rows)))

    def wake_rows(self, i0, i1):
        return _np(self._wake_jit(jnp.asarray(self.center[i0:i1])))

    def kutta(self, mu):
        diff = mu[self.upper] - mu[self.lower]
        return (self.integrals * diff[:, None]).sum(axis=0) / self.measure

    def kutta_transpose(self, q):
        c = (self.integrals * (q / self.measure)[None, :]).sum(axis=1)
        out = np.zeros(self.nf)
        np.add.at(out, self.upper, c)
        np.add.at(out, self.lower, -c)
        return out

    def warm(self):
        """Compile the block kernels once (first call is otherwise cold)."""
        self.doublet_rows(0, min(self.block, self.nf))
        self.wake_rows(0, min(self.block, self.nf))
        self._source_jit(jnp.asarray(self.center[0:min(self.block, self.nf)]))

    # ---- operator products --------------------------------------------
    def matvec(self, v):
        mu, gamma = v[:self.nf], v[self.nf:]
        out = np.empty(self.nf + self.ng)
        for i0 in range(0, self.nf, self.block):
            i1 = min(i0 + self.block, self.nf)
            out[i0:i1] = self.doublet_rows(i0, i1) @ mu + self.wake_rows(i0, i1) @ gamma
        out[self.nf:] = -self.kutta(mu) + self.mass @ gamma
        return out

    def rmatvec(self, w):
        p, q = w[:self.nf], w[self.nf:]
        out = np.zeros(self.nf + self.ng)
        for i0 in range(0, self.nf, self.block):
            i1 = min(i0 + self.block, self.nf)
            out[:self.nf] += self.doublet_rows(i0, i1).T @ p[i0:i1]
            out[self.nf:] += self.wake_rows(i0, i1).T @ p[i0:i1]
        out[:self.nf] -= self.kutta_transpose(q)
        out[self.nf:] += self.mass.T @ q
        return out

    def source_apply(self, values):
        """Single-layer matrix times a per-panel vector, blockwise (never stored)."""
        out = np.empty(self.nf + self.ng)
        for i0 in range(0, self.nf, self.block):
            i1 = min(i0 + self.block, self.nf)
            single = self._source_jit(jnp.asarray(self.center[i0:i1]))
            out[i0:i1] = _np(single) @ values
        out[self.nf:] = 0.0
        return out

    def rhs(self):
        return self.source_apply(self.sigma)

    def dense(self):
        """Materialise A once; only used to audit the matrix-free products."""
        mat = np.zeros((self.nf + self.ng, self.nf + self.ng))
        for i0 in range(0, self.nf, self.block):
            i1 = min(i0 + self.block, self.nf)
            mat[i0:i1, :self.nf] = self.doublet_rows(i0, i1)
            mat[i0:i1, self.nf:] = self.wake_rows(i0, i1)
        tested_t = np.zeros((self.nf, self.ng))
        np.add.at(tested_t, self.upper, self.integrals)
        np.add.at(tested_t, self.lower, -self.integrals)
        mat[self.nf:, :self.nf] = -tested_t.T / self.measure[:, None]
        mat[self.nf:, self.nf:] = self.mass
        return mat

    def wake_loads(self, mu, gamma):
        """Lift from the circulation channel: 2 * sum(gamma_i * dy_i) / S_ref."""
        return 2.0 * float(np.sum(gamma * np.diff(self.station_y)))

def gmres(matvec, b, x0=None, restart=80, rtol=1e-10, maxiter=2000, precond=None,
          transpose=False):
    """Restarted GMRES with left preconditioning; residual is the true one."""
    b = np.asarray(b, dtype=np.float64)
    n = b.size
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=np.float64)
    bnorm = np.linalg.norm(b)
    if bnorm == 0.0:
        return x, {"iterations": 0, "true_residual": 0.0, "residuals": [0.0]}

    def pre(r):
        return r if precond is None else precond.solve(r, transpose=transpose)

    residuals = []
    it = 0
    while it < maxiter:
        r = b - matvec(x)
        scaled = float(np.linalg.norm(r) / bnorm)
        residuals.append(scaled)
        if scaled <= rtol:
            return x, {"iterations": it, "true_residual": scaled, "residuals": residuals}
        v0 = pre(r)
        beta = float(np.linalg.norm(v0))
        if beta == 0.0:
            return x, {"iterations": it, "true_residual": scaled, "residuals": residuals}
        basis = [v0 / beta]
        span = min(restart, maxiter - it)
        hess = np.zeros((span + 1, span))
        cycle = None
        for j in range(span):
            w = pre(matvec(basis[j]))
            for i in range(j + 1):
                hess[i, j] = float(basis[i] @ w)
                w = w - hess[i, j] * basis[i]
            hess[j + 1, j] = float(np.linalg.norm(w))
            rhs = np.zeros(j + 2)
            rhs[0] = beta
            y = np.linalg.lstsq(hess[:j + 2, :j + 1], rhs, rcond=None)[0]
            cycle = x + sum(y[i] * basis[i] for i in range(j + 1))
            estimate = float(np.linalg.norm(rhs - hess[:j + 2, :j + 1] @ y) / bnorm)
            it += 1
            if estimate <= rtol or hess[j + 1, j] <= 1e-15:
                break
            basis.append(w / hess[j + 1, j])
        x = cycle
        true = float(np.linalg.norm(b - matvec(x)) / bnorm)
        residuals.append(true)
        if true <= rtol:
            return x, {"iterations": it, "true_residual": true, "residuals": residuals}
    true = float(np.linalg.norm(b - matvec(x)) / bnorm)
    return x, {"iterations": it, "true_residual": true, "residuals": residuals}


def _bisect(centers, indices, depth):
    """Split a panel set along its longest axis; frozen spatial blocks."""
    if depth == 0 or len(indices) < 8:
        return [indices]
    pts = centers[indices]
    axis = int(np.argmax(np.ptp(pts, axis=0)))
    order = indices[np.argsort(pts[:, axis], kind="stable")]
    half = len(order) // 2
    return _bisect(centers, order[:half], depth - 1) + _bisect(centers, order[half:], depth - 1)


class NearFieldPreconditioner:
    """P = [[blockdiag(D), 0], [-T, M]]; exact local LU plus the small mass/Kutta block."""

    def __init__(self, op, depth=3):
        self.op = op
        self.blocks = [np.asarray(b, dtype=np.int64)
                       for b in _bisect(op.center, np.arange(op.nf), depth) if len(b)]
        self.factor = [sla.lu_factor(op.doublet_rows_indexed(b)[:, b]) for b in self.blocks]
        self.mass_lu = sla.lu_factor(op.mass)
        self.mass_lu_t = sla.lu_factor(op.mass.T)
        self.entries = int(sum(len(b) ** 2 for b in self.blocks));
        self.stride = int(np.max([len(b) for b in self.blocks]))

    def solve(self, r, transpose=False):
        op = self.op
        nf, ng = op.nf, op.ng
        out = np.zeros(nf + ng)
        if not transpose:
            for block, fac in zip(self.blocks, self.factor):
                out[block] = sla.lu_solve(fac, r[block])
            out[nf:] = sla.lu_solve(self.mass_lu, r[nf:] + op.kutta(out[:nf]))
        else:
            gamma = sla.lu_solve(self.mass_lu_t, r[nf:])
            out[nf:] = gamma
            rhs = r[:nf] + op.kutta_transpose(gamma)
            for block, fac in zip(self.blocks, self.factor):
                out[block] = sla.lu_solve(fac, rhs[block], trans=1)
        return out

def _pair_map(points, tol):
    """Strict mirror map: j pairs with i when points[j] == mirror_y(points[i])."""
    n = points.shape[0]
    target = points * np.array([1.0, -1.0, 1.0])
    d = np.linalg.norm(points[None, :, :] - target[:, None, :], axis=2)
    best = np.argmin(d, axis=1)
    residual = d[np.arange(n), best]
    return best, residual


def mirror_gate(op, tol=1e-9, station_tol=1e-6, station_scale=None, freestream=None):
    """Decide whether the mesh admits the R A E half-model reduction.

    Returns status "ok" only for an exactly mirror-paired bijection including
    every panel, every circulation station and every trailing-edge edge. Any
    unpaired panel, non-involutive pairing, centreline-straddling triangle or
    mismatched TE pair yields status "blocked" with the evidence attached.
    """
    scale = float(np.linalg.norm(op.center.max(axis=0) - op.center.min(axis=0)))
    atol = tol * max(scale, 1e-12)
    pair, residual = _pair_map(op.center, atol)
    unpaired = int(np.sum(residual > atol))
    bijective = len(set(pair.tolist())) == op.nf
    involution = bool(np.all(pair[pair] == np.arange(op.nf)))
    self_panels = int(np.sum(pair == np.arange(op.nf)))
    y_centers = np.abs(op.center[:, 1])
    straddling = int(np.sum((pair == np.arange(op.nf)) & (y_centers > atol)))
    pair_scale = float(np.max(op.station_y) - np.min(op.station_y))
    stol = station_tol * max(pair_scale, 1e-12) if station_scale is None else station_scale
    sdiff = np.abs(op.station_y[:, None] + op.station_y[None, :])
    spair = np.argmin(sdiff, axis=1)
    sunpaired = int(np.sum(sdiff[np.arange(len(op.station_y)), spair] > stol))
    sinvolution = bool(np.all(spair[spair] == np.arange(len(op.station_y))))
    # XZ reflection flips y only. The z-based upper/lower TE bank labels
    # remain unchanged: mirrored upper -> upper, mirrored lower -> lower.
    # Test the paired bank *together*, not two independent set memberships.
    upper_lookup = {}
    for edge_id, face in enumerate(op.upper):
        upper_lookup.setdefault(int(face), []).append(edge_id)
    te_paired = int(sum(
        any(int(pair[lower_face]) == int(op.lower[j])
            for j in upper_lookup.get(int(pair[upper_face]), ()))
        for upper_face, lower_face in zip(op.upper, op.lower)
    ))
    reasons = []
    if unpaired:
        reasons.append("%d/%d panels have no exact mirror partner (tol %.1e)" % (unpaired, op.nf, atol))
    if not bijective:
        reasons.append("mirror map is not injective (%d distinct partners)" % len(set(pair.tolist())))
    if not involution:
        reasons.append("mirror map is not an involution")
    if self_panels:
        reasons.append("%d self-mirror wall panel(s); current half reducer requires fixed-point-free wall pairs" % self_panels)
    if sunpaired:
        reasons.append("%d/%d circulation stations have no mirror partner (tol %.1e)" % (sunpaired, len(op.station_y), stol))
    if not sinvolution:
        reasons.append("station mirror map is not an involution")
    if te_paired != op.n_te:
        reasons.append("%d/%d TE edges do not mirror onto an existing lower-TE edge" % (op.n_te - te_paired, op.n_te))
    sideslip = None
    if freestream is not None:
        flow = np.asarray(freestream, dtype=np.float64)
        sideslip = float(abs(flow[1]) / max(np.linalg.norm(flow), 1e-30))
        if sideslip > 1e-12:
            reasons.append("freestream carries a sideslip component (|v_y|/|v| = %.3e); "
                           "the mirror symmetry is broken" % sideslip)
    return {
        "status": "ok" if not reasons else "blocked",
        "reasons": reasons,
        "panels": int(op.nf),
        "paired_panels": int(op.nf - unpaired),
        "unpaired_panels": unpaired,
        "bijective": bijective,
        "involution": involution,
        "self_panels": self_panels,
        "centreline_straddling": straddling,
        "nearest_mirror_median": float(np.median(residual)),
        "nearest_mirror_max": float(residual.max()),
        "tolerance": atol,
        "stations": int(len(op.station_y)),
        "paired_stations": int(len(op.station_y) - sunpaired),
        "station_involution": sinvolution,
        "te_edges": int(op.n_te),
        "te_edges_paired": te_paired,
        "sideslip_ratio": sideslip,
        "mirror": pair,
        "station_mirror": spair,
    }


class HalfModelReduction:
    """R A E reduction of an exactly mirror-invariant operator, wall and stations."""

    def __init__(self, nf, ng, mirror, station_mirror):
        self.nf, self.ng = nf, ng
        self.mirror = np.asarray(mirror, dtype=np.int64)
        self.station_mirror = np.asarray(station_mirror, dtype=np.int64)
        index = np.arange(nf)
        self.half = np.flatnonzero(self.mirror > index)
        if len(self.half) * 2 != nf:
            raise ValueError("half-model reduction needs a fixed-point-free pairing")
        s_index = np.arange(ng)
        self.station_half = np.flatnonzero(self.station_mirror > s_index)
        self.station_self = np.flatnonzero(self.station_mirror == s_index)
        self.size = len(self.half) + len(self.station_half) + len(self.station_self)

    def expand(self, x):
        """E: reduced state -> full state."""
        mu = np.zeros(self.nf)
        mu[self.half] = x[:len(self.half)]
        mu[self.mirror[self.half]] = x[:len(self.half)]
        gamma = np.zeros(self.ng)
        a = len(self.half)
        b = a + len(self.station_half)
        gamma[self.station_half] = x[a:b]
        gamma[self.station_mirror[self.station_half]] = x[a:b]
        gamma[self.station_self] = x[b:]
        return np.concatenate([mu, gamma])

    def restrict(self, y):
        """R: full residual -> reduced residual."""
        mu, gamma = y[:self.nf], y[self.nf:]
        return np.concatenate([mu[self.half], gamma[self.station_half], gamma[self.station_self]])

    def apply(self, op_matvec, x):
        return self.restrict(op_matvec(self.expand(x)))

    def apply_transpose(self, op_rmatvec, w):
        full = np.zeros(self.nf + self.ng)
        a = len(self.half)
        full[self.half] = w[:a]
        b = a + len(self.station_half)
        full[self.nf + self.station_half] = w[a:b]
        full[self.nf + self.station_self] = w[b:]
        v = op_rmatvec(full)
        mu, gamma = v[:self.nf], v[self.nf:]
        out_mu = mu[self.half] + mu[self.mirror[self.half]]
        out_g = gamma[self.station_half] + gamma[self.station_mirror[self.station_half]]
        return np.concatenate([out_mu, out_g, gamma[self.station_self]])

    def rhs(self, op_rhs):
        return self.restrict(op_rhs())