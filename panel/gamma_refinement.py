"""Nested circulation spaces on the SAME planar wake and fixed wall mesh.

Research only. Clipping partitions integration support; it does not move the
wake or remesh the wall. There is no pressure correction, regularization or
new geometric VJP. Coarse contact plateaus and zero tip circulation survive.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GammaSpace:
    stations: np.ndarray
    dofs: np.ndarray
    hat: np.ndarray
    prolongation: np.ndarray


def _station_data(stations, dofs):
    y, raw = np.asarray(stations, float), np.asarray(dofs)
    if (y.ndim != 1 or len(y) < 2 or raw.shape != y.shape
            or not np.issubdtype(raw.dtype, np.integer)
            or not np.isfinite(y).all() or y[0] != 0
            or np.any(np.diff(y) <= 0)):
        raise ValueError('Need increasing nonnegative stations and integer DOFs')
    d = raw.astype(np.int64, copy=True)
    if d[0] < 0 or d[-1] != -1 or np.any(d[:-1] < 0):
        raise ValueError('Only the outboard tip may have prescribed zero Gamma')
    n = int(d.max())+1
    if set(d[d >= 0]) != set(range(n)):
        raise ValueError('Gamma DOFs must be contiguous and present')
    for i in range(n):
        where = np.flatnonzero(d == i)
        if where[-1]-where[0]+1 != len(where):
            raise ValueError('A contact plateau must be a contiguous station interval')
    hat = np.zeros((len(y), n))
    hat[np.arange(len(y)-1), d[:-1]] = 1.
    return y.copy(), d, hat


def refine_space(stations, dofs, factor=1):
    """P maps OLD free Gamma values to the nested NEW free Gamma values."""
    y, d, old_hat = _station_data(stations, dofs)
    if (isinstance(factor, bool) or not isinstance(factor, (int, np.integer))
            or factor < 1):
        raise ValueError('factor must be a positive integer')
    ys, ds = [y[0]], [d[0]]
    rows = list(np.eye(old_hat.shape[1]))
    for k in range(len(y)-1):
        for j in range(1, factor):
            t = j/factor
            ys.append(y[k]+t*(y[k+1]-y[k]))
            if d[k] == d[k+1]:
                ds.append(d[k])
            else:
                ds.append(len(rows))
                rows.append((1-t)*old_hat[k]+t*old_hat[k+1])
        ys.append(y[k+1])
        ds.append(d[k+1])
    ys, ds = np.asarray(ys), np.asarray(ds, dtype=np.int64)
    if np.any(np.diff(ys) <= 0):
        raise ValueError('Refined station spacing is below floating-point resolution')
    h = np.zeros((len(ys), len(rows)))
    live = ds >= 0
    h[np.flatnonzero(live), ds[live]] = 1.
    return GammaSpace(ys, ds, h, np.asarray(rows))


def basis_at(points_y, space):
    y = np.asarray(points_y, dtype=float)
    if (not np.isfinite(y).all() or np.any(y < space.stations[0])
            or np.any(y > space.stations[-1])):
        raise ValueError('Evaluation point outside circulation support')
    i = np.searchsorted(space.stations, y, side='right')-1
    i = np.minimum(i, len(space.stations)-2)
    t = (y-space.stations[i])/(space.stations[i+1]-space.stations[i])
    return (1-t[..., None])*space.hat[i]+t[..., None]*space.hat[i+1]


def _clip(poly, bound, keep_above):
    out = []
    for a, b in zip(poly, poly[1:]+poly[:1]):
        sa, sb = a[1]-bound, b[1]-bound
        ina = sa >= 0 if keep_above else sa <= 0
        inb = sb >= 0 if keep_above else sb <= 0
        if ina != inb:
            c = a+(bound-a[1])/(b[1]-a[1])*(b-a)
            c[1] = bound
            out.append(c)
        if inb:
            out.append(b.copy())
    clean = []
    for v in out:
        if not clean or not np.array_equal(v, clean[-1]):
            clean.append(v)
    if len(clean) > 1 and np.array_equal(clean[0], clean[-1]):
        clean.pop()
    return clean


def split_wake(triangles, stations, *, max_pieces=20000):
    """Exact planar triangle/slab clipping with parent-wise area/moment checks.

    Only exactly zero-area fan pieces are discarded. Positive slivers are not
    clipped away. Original wall vertices/faces and TE bank indices are unused.
    """
    tri, y = np.asarray(triangles, float), np.asarray(stations, float)
    if (tri.ndim != 3 or tri.shape[1:] != (3, 3) or len(tri) == 0
            or not np.isfinite(tri).all() or y.ndim != 1 or len(y) < 2
            or not np.isfinite(y).all() or np.any(np.diff(y) <= 0)
            or max_pieces < 1):
        raise ValueError('Finite wake triangles, increasing stations and budget required')
    av = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])/2
    scale = max(float(np.ptp(tri.reshape(-1, 3), axis=0).max()), 1.)
    if np.ptp(tri[:, :, 2]) > 1e-12*scale or np.any(av[:, 2] <= 0):
        raise ValueError('Only nondegenerate constant-z, positive-z wake triangles')
    if tri[:, :, 1].min() < y[0] or tri[:, :, 1].max() > y[-1]:
        raise ValueError('Wake extends beyond station support; do not silently crop')
    children, parents, intervals = [], [], []
    for p, face in enumerate(tri):
        a = max(0, int(np.searchsorted(y, face[:, 1].min(), side='right')-1))
        b = min(len(y)-2, int(np.searchsorted(y, face[:, 1].max(), side='left')))
        for k in range(a, b+1):
            poly = _clip(list(face.copy()), y[k], True)
            poly = _clip(poly, y[k+1], False) if poly else []
            for j in range(1, len(poly)-1):
                piece = np.asarray([poly[0], poly[j], poly[j+1]])
                signed = np.cross(piece[1]-piece[0], piece[2]-piece[0])[2]
                if signed == 0:
                    continue
                if signed < 0:
                    raise ValueError('Clipping reversed a positive wake triangle')
                if len(children) >= max_pieces:
                    raise MemoryError('Wake integration-piece budget exceeded')
                children.append(piece)
                parents.append(p)
                intervals.append(k)
    if not children:
        raise ValueError('No wake integration support')
    pieces = np.asarray(children)
    parent, interval = np.asarray(parents), np.asarray(intervals)
    area = np.cross(pieces[:, 1]-pieces[:, 0], pieces[:, 2]-pieces[:, 0])[:, 2]/2
    sum_area = np.bincount(parent, weights=area, minlength=len(tri))
    moment = np.zeros((len(tri), 3))
    np.add.at(moment, parent, area[:, None]*pieces.mean(axis=1))
    expected = av[:, 2, None]*tri.mean(axis=1)
    area_err = float(np.max(np.abs(sum_area-av[:, 2])/av[:, 2]))
    moment_err = float(np.max(np.abs(moment-expected)/(av[:, 2, None]*scale)))
    if max(area_err, moment_err) > 5e-12:
        raise ArithmeticError('Clipped wake did not conserve each parent area/first moment')
    return pieces, parent, interval, dict(
        parent_count=len(tri), piece_count=len(pieces),
        area_relative_max=area_err, first_moment_scaled_max=moment_err)


def wake_basis(pieces, interval, space):
    dy = space.stations[interval+1]-space.stations[interval]
    f = (pieces.mean(axis=1)[:, 1]-space.stations[interval])/dy
    basis = (1-f[:, None])*space.hat[interval]+f[:, None]*space.hat[interval+1]
    slope = (space.hat[interval+1]-space.hat[interval])/dy[:, None]
    return basis, slope


def weak_te(endpoints, upper, lower, wall_count, space):
    """Unscaled K=integral(hat*wall-jump), M=integral(hat*hat), ds physical.

    Split at every station crossing BEFORE two-point Gauss. Thus products of
    piecewise P1 functions are integrated exactly, not underintegrated when
    refined stations lie inside the original TE edges.
    """
    e = np.asarray(endpoints, float)
    up, lo = np.asarray(upper), np.asarray(lower)
    if (e.ndim != 3 or e.shape[1:] != (2, 3) or len(e) == 0
            or not np.isfinite(e).all() or up.shape != (len(e),)
            or lo.shape != up.shape or wall_count < 1
            or not np.issubdtype(up.dtype, np.integer)
            or not np.issubdtype(lo.dtype, np.integer)
            or min(up.min(), lo.min()) < 0
            or max(up.max(), lo.max()) >= wall_count or np.any(up == lo)):
        raise ValueError('Real TE endpoints and distinct valid upper/lower wall banks required')
    if e[:, :, 1].min() < space.stations[0] or e[:, :, 1].max() > space.stations[-1]:
        raise ValueError('TE outside station support')
    lengths = np.linalg.norm(e[:, 1]-e[:, 0], axis=1)
    if np.any(lengths <= 0):
        raise ValueError('Zero-length TE edge')
    samples, weights, owners = [], [], []
    nodes = np.array([.5-np.sqrt(3)/6, .5+np.sqrt(3)/6])
    for i, edge in enumerate(e):
        dy = edge[1, 1]-edge[0, 1]
        breaks = [0., 1.]
        if dy != 0:
            cuts = (space.stations-edge[0, 1])/dy
            breaks += cuts[(cuts > 0) & (cuts < 1)].tolist()
        breaks = np.unique(breaks)
        for a, b in zip(breaks[:-1], breaks[1:]):
            ts = a+(b-a)*nodes
            samples.extend(edge[0, 1]+ts*dy)
            weights.extend([lengths[i]*(b-a)/2]*2)
            owners.extend([i]*2)
    weights, owners = np.asarray(weights), np.asarray(owners)
    h = basis_at(np.asarray(samples), space)
    mass = h.T@(weights[:, None]*h)
    integrals = np.zeros((len(e), space.hat.shape[1]))
    np.add.at(integrals, owners, weights[:, None]*h)
    measure = integrals.sum(axis=0)
    if np.any(measure <= 0):
        raise ValueError('Refined Gamma DOF has no exposed-TE test support')
    np.linalg.cholesky(mass)  # Reject rank loss; no cutoff/regularization.
    K = np.zeros((len(measure), wall_count))
    for i in range(len(e)):
        K[:, up[i]] += integrals[i]
        K[:, lo[i]] -= integrals[i]
    length_error = abs(weights.sum()-lengths.sum())/lengths.sum()
    if length_error > 5e-13:
        raise ArithmeticError('TE integration length changed')
    return dict(K=K, M=mass, measure=measure, sample_hat=h,
                sample_weight=weights, sample_edge=owners,
                length_relative_error=float(length_error))


def wake_operator(targets, pieces, interval, space, *, block=32,
                  max_pairs=1000000):
    """Existing JAX analytic kernel, streamed targets; no new kernel formula."""
    import jax
    import jax.numpy as jnp
    from .kernel import triangle_potential
    if not jax.config.x64_enabled:
        raise ValueError('Explicit JAX float64 required')
    points = np.asarray(targets, float)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) == 0
            or not np.isfinite(points).all() or block < 1 or max_pairs < 1):
        raise ValueError('Finite targets and positive block/pair budget required')
    if min(block, len(points))*len(pieces) > max_pairs:
        raise MemoryError('Kernel-pair budget exceeded; reduce target block')
    basis, slope = wake_basis(pieces, interval, space)
    wake = jnp.asarray(pieces)
    image = wake[:, (0, 2, 1), :]*jnp.array([1., -1., 1.])
    b, s = jnp.asarray(basis), jnp.asarray(slope)

    @jax.jit
    def action(p):
        d, m = triangle_potential(p, wake, source=False, linear_y=True, block_size=0)
        di, mi = triangle_potential(p, image, source=False, linear_y=True, block_size=0)
        return (d+di)@b+(m-mi)@s

    result = np.empty((len(points), space.hat.shape[1]))
    for start in range(0, len(points), block):
        result[start:start+block] = np.asarray(action(jnp.asarray(points[start:start+block])))
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite analytic wake influence; no core/fallback applied')
    return result


def relative_error(a, b):
    a, b = np.asarray(a), np.asarray(b)
    scale = float(np.max(np.abs(a)))
    error = float(np.max(np.abs(a-b)))
    return dict(max_abs=error, relative_to_peak=error/scale if scale else None)


def nested_checks(old_W, old_te, new_W, new_te, P):
    pairs = dict(wake_action=(old_W, new_W@P),
                 weak_mass=(old_te['M'], P.T@new_te['M']@P),
                 weak_wall_jump=(old_te['K'], P.T@new_te['K']))
    report = {name: relative_error(a, b) for name, (a, b) in pairs.items()}
    for name, (a, b) in pairs.items():
        if not np.allclose(a, b, rtol=5e-10, atol=5e-12):
            raise ArithmeticError(f'Nested-space identity failed: {name}: {report[name]}')
    return report


def solve_augmented(body, rhs_wall, W, te):
    """Unregularized dense reference. Re-solving is mandatory when Gamma changes."""
    import warnings
    from scipy.linalg import LinAlgWarning, lu_factor, lu_solve
    D, rhs = np.asarray(body), np.asarray(rhs_wall)
    n = len(rhs)
    if D.shape != (n, n) or W.shape != (n, len(te['measure'])):
        raise ValueError('Body/wake block shapes mismatch')
    matrix = np.block([[D, W], [-te['K']/te['measure'][:, None],
                                te['M']/te['measure'][:, None]]])
    vector = np.r_[rhs, np.zeros(len(te['measure']))]
    with warnings.catch_warnings():
        warnings.simplefilter('error', LinAlgWarning)
        state = lu_solve(lu_factor(matrix, check_finite=True), vector)
    residual = matrix@state-vector
    norm_rhs = float(np.linalg.norm(vector))
    error = float(np.linalg.norm(residual))
    relative = error/norm_rhs if norm_rhs else error
    if not np.isfinite(state).all() or relative > 1e-10:
        raise ArithmeticError(f'Refined solve failed true residual gate: {relative}')
    return state[:n], state[n:], dict(true_relative_residual=relative,
        residual_normalization='norm(rhs)' if norm_rhs else 'absolute zero-RHS',
        wall_residual_max=float(np.max(np.abs(residual[:n]))),
        weak_kutta_scaled_residual_max=float(np.max(np.abs(residual[n:]))))
