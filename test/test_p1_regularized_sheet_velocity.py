import numpy as np
import pytest

from panel.p1_analytic_sheet_velocity import AnalyticSheetField
from panel.p1_regularized_sheet_velocity import RegularizedP1SheetField


def direct_p1_regularized_velocity(point, xyz, faces, values, core, order=80):
    """Independent per-face edge + Duffy area quadrature; no trace aggregation."""
    p = np.asarray(point, float)
    xyz, faces, values = np.asarray(xyz, float), np.asarray(faces, int), np.asarray(values, float)
    q, w = np.polynomial.legendre.leggauss(order)
    z, wz = (q+1.0)/2.0, w/2.0
    boundary = np.zeros(3)
    area_term = np.zeros(3)
    for face, nodal in zip(faces, values):
        tri = xyz[face]
        av = np.cross(tri[1]-tri[0], tri[2]-tri[0])
        a2 = np.linalg.norm(av)
        normal = av/a2
        g1 = np.cross(tri[2]-tri[0], normal)/a2
        g2 = np.cross(normal, tri[1]-tri[0])/a2
        grad = nodal[0]*(-g1-g2)+nodal[1]*g1+nodal[2]*g2
        grad -= normal*np.dot(grad, normal)
        gamma = np.cross(grad, normal)

        # Directly integrate each oriented face edge, retaining all internal
        # traces; this is independent of the implementation's edge combining.
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = tri[i], tri[j]
            d = b-a
            length = np.linalg.norm(d)
            tangent = d/length
            s = z*length
            source = a[None, :]+s[:, None]*tangent
            r = p[None, :]-source
            mu = nodal[i]+(s/length)*(nodal[j]-nodal[i])
            den = (np.einsum("ij,ij->i", r, r)+core**2)**1.5
            boundary += np.sum((wz*length*mu/den)[:, None]
                               *np.cross(tangent[None, :], r), axis=0)

        # Independent tensor-product Duffy quadrature for the triangle area term.
        u, v = np.meshgrid(z, z, indexing="ij")
        ww = wz[:, None]*wz[None, :]*(1-u)
        bary = np.stack(((1-u)*(1-v), u, (1-u)*v), axis=-1)
        source = bary@tri
        r = p-source
        den = (np.einsum("ij,ij->i", r.reshape(-1, 3), r.reshape(-1, 3))
               +core**2)**1.5
        area_integral = np.sum(
            (ww.reshape(-1)/den)[:, None]*r.reshape(-1, 3), axis=0)*a2
        area_term += np.cross(gamma, area_integral)
    return np.array([1., 0., 0.])-(boundary+area_term)/(4*np.pi)


def fixture():
    xyz = np.array([[0., 0., 0.], [1.2, 0.1, 0.2], [0.2, 1.1, -0.1]])
    faces = np.array([[0, 1, 2]])
    values = np.array([[0.3, -0.2, 0.8]])
    return xyz, faces, values


def test_regularized_p1_field_matches_independent_edge_and_area_quadrature():
    xyz, faces, values = fixture()
    point = np.array([0.37, -0.42, 1.3])
    core = 0.13
    field = RegularizedP1SheetField([(xyz, faces, values)], core_radius=core,
                                   quadrature_order=48)
    got = field(point)
    expected = direct_p1_regularized_velocity(point, xyz, faces, values, core, order=100)
    np.testing.assert_allclose(got, expected, rtol=3e-11, atol=3e-12)


def test_linear_strength_edge_first_moment_matches_direct_quadrature():
    xyz, faces, values = fixture()
    point = np.array([0.2, -0.3, 0.7])
    core = 0.04
    field = RegularizedP1SheetField([(xyz, faces, values)], core_radius=core)
    # A single triangle has three uncancelled boundary edges; the independent
    # per-face edge integral in the full-field oracle includes the P1 slope.
    got = field(point)
    expected = direct_p1_regularized_velocity(point, xyz, faces, values, core, order=160)
    np.testing.assert_allclose(got, expected, rtol=3e-11, atol=3e-12)


def test_shared_continuous_internal_edge_cancels_but_boundary_edges_remain():
    xyz = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    # A globally affine P1 trace: the shared diagonal has identical values.
    nodal_global = np.array([0., 1., 2., 1.])
    values = nodal_global[faces]
    field = RegularizedP1SheetField([(xyz, faces, values)], core_radius=0.1)
    assert field.n_boundary_edges == 4


def test_in_plane_p1_gradient_is_retained_not_reduced_to_edge_filaments():
    xyz, faces, values = fixture()
    point = np.array([0.4, 0.3, 0.9])
    core = 0.08
    varying = RegularizedP1SheetField([(xyz, faces, values)], core_radius=core)
    constant_values = np.full_like(values, np.mean(values))
    constant = RegularizedP1SheetField([(xyz, faces, constant_values)], core_radius=core)
    assert np.linalg.norm(varying.gamma[0]) > 0
    assert np.linalg.norm(constant.gamma[0]) <= 1e-15 * max(
        1.0, np.linalg.norm(constant.grad_face[0]))
    assert np.linalg.norm(varying(point)-constant(point)) > 1e-5


def test_zero_core_limit_recovers_existing_analytic_p1_field_off_sheet():
    xyz, faces, values = fixture()
    point = np.array([0.3, -0.4, 1.6])
    exact = AnalyticSheetField([(xyz, faces, values)])(point)
    cores = np.array([0.08, 0.04, 0.02, 0.01, 0.005])
    errors = []
    for core in cores:
        field = RegularizedP1SheetField([(xyz, faces, values)], core_radius=core,
                                       quadrature_order=48)
        errors.append(np.linalg.norm(field(point)-exact))
    slope = np.polyfit(np.log(cores), np.log(errors), 1)[0]
    assert abs(slope-2.0) < 0.05
    ratios = np.asarray(errors)/cores**2
    assert np.max(np.abs(ratios[-3:]/ratios[-1]-1.0)) < 0.03


def test_positive_core_gives_finite_velocity_on_triangle_and_its_edges():
    xyz, faces, values = fixture()
    field = RegularizedP1SheetField([(xyz, faces, values)], core_radius=0.025,
                                   quadrature_order=40)
    points = [xyz.mean(axis=0), 0.5*(xyz[0]+xyz[1]), xyz[0]]
    for point in points:
        assert np.isfinite(field(point)).all()


def test_core_and_geometry_scale_covariance():
    xyz, faces, values = fixture()
    point = np.array([0.2, -0.5, 1.2])
    core = 0.09
    base = RegularizedP1SheetField([(xyz, faces, values)], core_radius=core)(point)
    scale = 3.7
    scaled = RegularizedP1SheetField([(scale*xyz, faces, values)],
                                     core_radius=scale*core)(scale*point)
    # Freestream is unchanged; induced velocity scales as inverse length.
    np.testing.assert_allclose(scaled-[1., 0., 0.],
                               (base-[1., 0., 0.])/scale,
                               rtol=3e-12, atol=3e-13)


def test_multiple_parts_are_added_without_cross_part_topology_cancellation():
    xyz, faces, values = fixture()
    point = np.array([0.3, -0.4, 1.6])
    core = 0.1
    one = RegularizedP1SheetField([(xyz, faces, values)], core_radius=core)
    two = RegularizedP1SheetField([(xyz, faces, values), (xyz, faces, values)],
                                  core_radius=core)
    np.testing.assert_allclose(two(point)-[1., 0., 0.],
                               2*(one(point)-[1., 0., 0.]), rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("core", [0., -0.1, np.nan, np.inf])
def test_invalid_core_radius_is_rejected(core):
    xyz, faces, values = fixture()
    with pytest.raises(ValueError, match="core_radius"):
        RegularizedP1SheetField([(xyz, faces, values)], core_radius=core)


def test_zero_area_and_invalid_strength_data_are_rejected():
    xyz, faces, values = fixture()
    bad = xyz.copy()
    bad[2] = bad[1]
    with pytest.raises(ValueError, match="degenerate"):
        RegularizedP1SheetField([(bad, faces, values)], core_radius=0.1)
    with pytest.raises(ValueError, match="nodal strengths"):
        RegularizedP1SheetField([(xyz, faces, values[:, :2])], core_radius=0.1)
