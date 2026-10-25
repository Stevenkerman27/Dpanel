"""True right-half-only lifting/wake regression versus an independent full image.

The full aircraft mesh is materialized *only inside the test* as an oracle.
Production half input never stores left wall/wake vertices or unknowns.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from panel.kernel import triangle_potential
from panel.half_native import (
    prepare_half_topology, validate_half_configuration,
    solve_half_fields, HalfPanelOperator,
)
from panel.half_native_gmres import (
    HalfNearFieldPreconditioner, solve_half_gmres,
)

jax.config.update("jax_enable_x64", True)


def half_wing():
    ys = np.array([0., .5, 1.])
    ring = np.array([[0., 0., 0.], [.5, 0., .1],
                     [1., 0., 0.], [.5, 0., -.1]])
    vertices = np.concatenate([ring+[0., y, 0.] for y in ys]).tolist()
    faces, te = [], []
    for i in range(2):
        for j in range(4):
            a, b = 4*i+j, 4*i+(j+1)%4
            c, d = 4*(i+1)+j, 4*(i+1)+(j+1)%4
            faces.extend(((a, b, d), (a, d, c)))
        te.append([4*i+2, 4*(i+1)+2])
    tip = len(vertices)
    vertices.append([.5, 1., 0.])
    for j in range(4):
        faces.append([tip, 8+j, 8+(j+1)%4])
    far = []
    for y in ys:
        far.append(len(vertices))
        vertices.append([4., float(y), 0.])
    wake = []
    for i in range(2):
        wake.extend([
            [4*i+2, far[i], far[i+1]],
            [4*i+2, far[i+1], 4*(i+1)+2],
        ])
    x = np.asarray(vertices)
    faces = np.asarray(faces, dtype=np.int32)
    te = np.asarray(te, dtype=np.int32)
    wake = np.asarray(wake, dtype=np.int32)
    normals = np.cross(x[faces[:, 1]]-x[faces[:, 0]],
                       x[faces[:, 2]]-x[faces[:, 0]])
    upper, lower = [], []
    for edge in te:
        adj = [i for i, f in enumerate(faces)
               if set(map(int, edge)).issubset(set(map(int, f)))]
        assert len(adj) == 2
        upper.append(next(i for i in adj if normals[i, 2] > 0))
        lower.append(next(i for i in adj if normals[i, 2] < 0))
    t = prepare_half_topology(
        x, faces, wake, te, upper, lower, ys,
        [0, 1, -1], [10]*len(faces), stencil_rings=2)
    return dict(
        vertices=jnp.asarray(x), station_y=jnp.asarray(ys), topology=t,
        freestream=jnp.array([1., 0., .08]), density=jnp.array(1.225),
        reference_area=jnp.array(2.), reference_chord=jnp.array(1.),
        reference_point=jnp.array([.25, 0., 0.]),
    )


def full_oracle(inputs):
    """Independently assemble the full 2N + 3 unknown matrix for parity."""
    x = np.asarray(inputs["vertices"])
    t = inputs["topology"]
    n = len(t.faces)
    positive = np.flatnonzero(x[:, 1] > 0)
    mirror = np.arange(len(x))
    mirror[positive] = len(x)+np.arange(len(positive))
    xx = np.vstack((x, x[positive]*[1., -1., 1.]))
    faces = np.vstack((np.asarray(t.faces),
                       mirror[np.asarray(t.faces)[:, ::-1]]))
    wake = np.vstack((np.asarray(t.wake_faces),
                      mirror[np.asarray(t.wake_faces)[:, ::-1]]))
    te = np.vstack((np.asarray(t.te_edges),
                    mirror[np.asarray(t.te_edges)[:, ::-1]]))
    upper = np.r_[np.asarray(t.upper_te), np.asarray(t.upper_te)+n]
    lower = np.r_[np.asarray(t.lower_te), np.asarray(t.lower_te)+n]
    stations = np.array([-1., -.5, 0., .5, 1.])
    hat = np.zeros((5, 3))
    hat[1:4] = np.eye(3)
    tri = jnp.asarray(xx[faces])
    center = tri.mean(axis=1)
    av = jnp.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])/2
    sigma = -(av/jnp.linalg.norm(av, axis=1)[:, None])@inputs["freestream"]
    S, D = triangle_potential(center, tri, interior_diagonal=True)
    wtri = jnp.asarray(xx[wake])
    Dw, My = triangle_potential(center, wtri, source=False, linear_y=True)
    wi = np.searchsorted(stations, np.asarray(wtri.mean(axis=1)[:, 1]),
                         side="right")-1
    dy = stations[wi+1]-stations[wi]
    frac = (wtri.mean(axis=1)[:, 1]-stations[wi])/dy
    basis = (1-frac[:, None])*hat[wi]+frac[:, None]*hat[wi+1]
    slope = (hat[wi+1]-hat[wi])/dy[:, None]
    W = Dw@basis+My@slope
    gauss = np.array([.5-np.sqrt(3.)/6, .5+np.sqrt(3.)/6])
    ends = xx[te]
    te_y = ends[:, 0, 1, None]*(1-gauss)+ends[:, 1, 1, None]*gauss
    ti = np.searchsorted(stations, te_y, side="right")-1
    tf = (te_y-stations[ti])/(stations[ti+1]-stations[ti])
    tb = (1-tf[..., None])*hat[ti]+tf[..., None]*hat[ti+1]
    weights = np.broadcast_to(
        np.linalg.norm(ends[:, 1]-ends[:, 0], axis=1)[:, None]/2, te_y.shape)
    mass = jnp.einsum("eq,eqi,eqj->ij", weights, tb, tb)
    integrals = jnp.einsum("eq,eqi->ei", weights, tb)
    measure = integrals.sum(axis=0)
    incidence = jnp.zeros((len(te), len(faces)))
    rows = jnp.arange(len(te))
    incidence = incidence.at[rows, upper].add(1.)
    incidence = incidence.at[rows, lower].add(-1.)
    A = jnp.block([[D, W], [-(integrals.T@incidence)/measure[:, None],
                               mass/measure[:, None]]])
    rhs = jnp.concatenate((S@sigma, jnp.zeros(3)))
    E = np.zeros((2*n+3, n+2))
    E[:n, :n] = np.eye(n)
    E[n:2*n, :n] = np.eye(n)
    E[2*n+0, n+1] = 1.
    E[2*n+1, n+0] = 1.
    E[2*n+2, n+1] = 1.
    R = np.r_[np.arange(n), 2*n+1, 2*n+2]
    return A, rhs, E, R, jnp.asarray(xx), jnp.asarray(faces)


def full_pressure_oracle(inputs):
    A, rhs, _, _, xyz, faces = full_oracle(inputs)
    mu = jnp.linalg.solve(A, rhs)[:len(faces)]
    tri = xyz[faces]
    av = jnp.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])/2
    normal = av/jnp.linalg.norm(av, axis=1)[:, None]
    centers = tri.mean(axis=1)
    tangent = tri[:, 1]-tri[:, 0]
    tangent = tangent/jnp.linalg.norm(tangent, axis=1)[:, None]
    second = jnp.cross(normal, tangent)
    t = inputs["topology"]
    n = len(t.faces)
    neighbor = np.asarray(t.neighbor_ghost_ids)
    left = np.where(neighbor < n, neighbor+n, neighbor-n)
    neighbors = jnp.asarray(np.vstack((neighbor, left)))
    delta = centers[neighbors]-centers[:, None]
    dl = jnp.sum(delta*tangent[:, None], axis=2)
    dm = jnp.sum(delta*second[:, None], axis=2)
    dmu = mu[neighbors]-mu[:, None]
    ll, mm, lm = (jnp.sum(a, axis=1) for a in (dl*dl, dm*dm, dl*dm))
    bl, bm = jnp.sum(dl*dmu, axis=1), jnp.sum(dm*dmu, axis=1)
    det = ll*mm-lm*lm
    gl, gm = (bl*mm-bm*lm)/det, (bm*ll-bl*lm)/det
    flow = inputs["freestream"]
    velocity = ((tangent@flow+gl)[:, None]*tangent
                +(second@flow+gm)[:, None]*second)
    cp = 1-jnp.sum(velocity**2, axis=1)/jnp.dot(flow, flow)
    q = .5*inputs["density"]*jnp.dot(flow, flow)
    force_panel = -q*cp[:, None]*av
    moment_panel = jnp.cross(
        centers-inputs["reference_point"], force_panel)
    return cp, force_panel.sum(axis=0), moment_panel.sum(axis=0)


def test_half_only_matrix_rhs_and_state_equal_true_full_reference():
    inputs = half_wing()
    validate_half_configuration(
        inputs["vertices"], inputs["station_y"], inputs["topology"],
        inputs["freestream"], inputs["reference_point"])
    out = solve_half_fields(**inputs)
    A, rhs, E, R, _, _ = full_oracle(inputs)
    np.testing.assert_allclose(
        out["matrix_half"], np.asarray(A)[R]@E, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(
        out["rhs_half"], np.asarray(rhs)[R], atol=2e-12, rtol=2e-12)
    reference = np.asarray(jnp.linalg.solve(A, rhs))
    n = len(inputs["topology"].faces)
    np.testing.assert_allclose(
        out["mu_half"], reference[:n], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        out["gamma_half"], reference[[2*n+1, 2*n+2]],
        rtol=1e-9, atol=1e-9)


def test_half_raw_pressure_force_and_moment_close_against_full_oracle():
    inputs = half_wing()
    out = solve_half_fields(**inputs)
    cp, force, moment = full_pressure_oracle(inputs)
    n = len(inputs["topology"].faces)
    np.testing.assert_allclose(out["Cp_raw_half"], cp[:n],
                               atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(out["force"], force, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(out["moment"], moment, atol=1e-9, rtol=1e-9)


def test_half_geometry_and_station_vjp_against_three_central_differences():
    data = half_wing()
    x, y = data["vertices"], data["station_y"]
    dx = jnp.stack((.01*x[:, 0], .02*x[:, 1], .01*x[:, 2]), axis=1)
    dy = .02*y
    def loss(theta):
        updated = {**data, "vertices": x+theta*dx, "station_y": y+theta*dy}
        out = solve_half_fields(**updated)
        return out["CL"]+.37*out["CM"][1]+.01*jnp.sum(out["panel_forces_half"][:, 0])
    derivative = jax.grad(loss)(jnp.array(0.))
    assert abs(float(derivative)) > 1e-7
    for h in (1e-3, 1e-4, 1e-5):
        fd = (loss(h)-loss(-h))/(2*h)
        np.testing.assert_allclose(derivative, fd, rtol=1e-5, atol=1e-8)


def test_streamed_half_action_transpose_and_dot_identity():
    data = half_wing()
    op = HalfPanelOperator(data["vertices"], data["station_y"],
                           data["topology"], data["freestream"], block_size=7)
    out = solve_half_fields(**data)
    n = op.state_size
    x = jnp.asarray(np.random.default_rng(3).normal(size=n))
    w = jnp.asarray(np.random.default_rng(4).normal(size=n))
    np.testing.assert_allclose(jax.jit(op.rhs)(), out["rhs_half"],
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jax.jit(op.matvec)(x),
                               out["matrix_half"]@x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jax.jit(op.rmatvec)(w),
                               out["matrix_half"].T@w, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jnp.dot(w, op.matvec(x)),
                               jnp.dot(x, op.rmatvec(w)), atol=1e-11)


def test_half_cpu_preconditioned_gmres_primal_and_transpose():
    data = half_wing()
    op = HalfPanelOperator(data["vertices"], data["station_y"],
                           data["topology"], data["freestream"], block_size=7)
    pre = HalfNearFieldPreconditioner(op, max_block_size=6)
    dense = solve_half_fields(**data)
    A, b = np.asarray(dense["matrix_half"]), np.asarray(dense["rhs_half"])
    result, meta = solve_half_gmres(
        op, preconditioner=pre, tolerance=1e-10)
    np.testing.assert_allclose(result, np.linalg.solve(A, b),
                               rtol=2e-9, atol=1e-10)
    assert meta["true_relative_residual"] < 1e-10
    seed = np.linspace(-.2, .7, op.state_size)
    adj, adj_meta = solve_half_gmres(
        op, rhs=seed, transpose=True, preconditioner=pre, tolerance=1e-10)
    np.testing.assert_allclose(adj, np.linalg.solve(A.T, seed),
                               rtol=1e-8, atol=1e-9)
    assert adj_meta["true_relative_residual"] < 1e-10
    np.testing.assert_allclose(seed@result, adj@b, rtol=1e-8, atol=1e-9)


def test_half_rejects_sideslip_reference_out_of_plane_and_unclosed_mesh():
    data = half_wing()
    t = data["topology"]
    with pytest.raises(ValueError, match="zero sideslip"):
        validate_half_configuration(
            data["vertices"], data["station_y"], t,
            jnp.array([1., .03, .08]), data["reference_point"])
    with pytest.raises(ValueError, match="reference on symmetry plane"):
        validate_half_configuration(
            data["vertices"], data["station_y"], t,
            data["freestream"], jnp.array([.25, .01, 0.]))
    f = np.asarray(t.faces)[:-1]
    with pytest.raises(ValueError, match="closed 2-manifold"):
        prepare_half_topology(
            data["vertices"], f, t.wake_faces, t.te_edges, t.upper_te,
            t.lower_te, data["station_y"], [0, 1, -1],
            [10]*len(f), stencil_rings=2)


def test_right_half_json_contract_matches_inline_fixture():
    from pathlib import Path
    from panel.run_half_native import load_half_surface
    source = Path(__file__).resolve().parents[1] / "cases" / "panel_half" / "thick_wing_two_strip.json"
    loaded = load_half_surface(source)
    ref = half_wing()
    for name in ("vertices", "station_y", "freestream", "density",
                 "reference_area", "reference_chord", "reference_point"):
        np.testing.assert_allclose(loaded[name], ref[name], rtol=0., atol=0.)
    for name in ("faces", "wake_faces", "te_edges", "upper_te", "lower_te",
                 "hat_map", "wake_interval", "te_interval", "component_ids"):
        np.testing.assert_allclose(getattr(loaded["topology"], name),
                                   getattr(ref["topology"], name),
                                   rtol=0., atol=0.)
