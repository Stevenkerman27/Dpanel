import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from tools.web import geometry, params


@pytest.mark.parametrize("raw", [
    {},
    {"twist_root_deg": 6, "twist_tip_deg": -1, "dihedral_deg": 8},
    {"thickness_root": .03, "thickness_tip": .03, "camber_root": .1, "camber_tip": .1},
])
def test_step_has_smooth_leading_edge_and_preserves_trailing_edge(tmp_path, monkeypatch, raw):
    values, _ = params.normalise(raw)
    stations = params.full_stations(values)
    curves = []
    add = gmsh.model.occ.addBSpline
    finalize = gmsh.finalize

    def capture(points, *args, **kwargs):
        tag = add(points, *args, **kwargs)
        curves.append(tag)
        return tag

    monkeypatch.setattr(gmsh.model.occ, "addBSpline", capture)
    monkeypatch.setattr(gmsh, "finalize", lambda: None)
    try:
        info = geometry.write_step(tmp_path / "wing.step", values)
        assert info["volume"] > 0
        assert len(curves) == 2 * len(stations)
        for i, station in enumerate(stations):
            upper, lower = curves[2*i:2*i+2]
            up_bounds = gmsh.model.getParametrizationBounds(1, upper)
            lo_bounds = gmsh.model.getParametrizationBounds(1, lower)
            u = np.asarray(gmsh.model.getDerivative(1, upper, [float(up_bounds[0][0])]))
            v = np.asarray(gmsh.model.getDerivative(1, lower, [float(lo_bounds[1][0])]))
            np.testing.assert_allclose(u / np.linalg.norm(u), v / np.linalg.norm(v), atol=1e-10)
            te = np.asarray([station["te_x"], station["y"], station["te_z"]])
            np.testing.assert_allclose(gmsh.model.getValue(1, upper, [float(up_bounds[1][0])]), te, atol=1e-10)
            np.testing.assert_allclose(gmsh.model.getValue(1, lower, [float(lo_bounds[0][0])]), te, atol=1e-10)
            if station["y"] == 0:
                place = geometry._placement(station)
                origin = np.asarray(place(0, 0))
                chord_axis = (np.asarray(place(1, 0)) - origin) / station["chord"]
                thickness_axis = (np.asarray(place(0, 1)) - origin) / station["chord"]
                for curve, bounds, sign in ((upper, up_bounds, 1), (lower, lo_bounds, -1)):
                    q = np.linspace(bounds[0][0], bounds[1][0], 201)
                    points = np.asarray(gmsh.model.getValue(1, curve, q.tolist())).reshape(-1, 3) - origin
                    x = points @ chord_axis / station["chord"]
                    z = points @ thickness_axis / station["chord"]
                    expected = np.array([geometry._camber_line(t, station["camber"], station["position"])
                                         + sign * geometry._half_thickness(t, station["tc"]) for t in x])
                    assert np.max(np.abs(z - expected)) < .001
    finally:
        if gmsh.isInitialized():
            finalize()


def test_step_rejects_insufficient_section_points(tmp_path):
    values, _ = params.normalise({})
    with pytest.raises(ValueError, match="four points"):
        geometry.write_step(tmp_path / "wing.step", values, count=3)
