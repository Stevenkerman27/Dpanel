import numpy as np
import pytest

from panel.mesh_feature_size import FeatureSizePolicy, _distance, _polyline


def model(*, ratio=.25, radius=.1):
    return {"schema": "dpanel.mesh_features.v1", "units": "m", "features": [
        {"name": "leading_edge", "target_ratio": ratio,
         "radius_chords": radius,
         "points": [[0, 0, 0], [0, 1, 0]]},
    ]}


def policy(**kwargs):
    return FeatureSizePolicy(model(**kwargs), origin=[0, 0, 0], chord=1,
                             wall_size=.2)


def test_polyline_finite_clamped_segment_distance():
    line = _polyline([[0, 0, 0], [0, 1, 0]], "leading_edge")
    assert _distance(np.array([.03, .4, 0]), line) == pytest.approx(.03)
    assert _distance(np.array([.03, 2, 0]), line) == pytest.approx(np.hypot(.03, 1))


def test_le_min_and_outside_callback_noop():
    p = policy()
    assert p(1, 8, 0, .5, 0, .2) == pytest.approx(.05)
    assert p(2, 8, .2, .5, 0, .2) == .2
    assert p(2, 8, .2, .5, 0, .03) == .03


def test_band_is_bounded_monotone_and_c1():
    p = policy()
    samples = [p.size((d, .4, 0), .2) for d in np.linspace(0, .1, 31)]
    assert min(samples) >= .05 - 1e-13
    assert max(samples) <= .2 + 1e-13
    assert np.all(np.diff(samples) >= -1e-15)
    eps = 1e-5
    assert abs((p.size((eps,.4,0),.2)-p.size((0,.4,0),.2))/eps) < .001
    assert abs((p.size((.1,.4,0),.2)-p.size((.1-eps,.4,0),.2))/eps) < .001


def test_same_shape_normalized_from_original_cad_frame():
    d = model()
    d["features"][0]["points"] = [[10, 0, -2], [10, 3, -2]]
    p = FeatureSizePolicy(d, origin=[10, 0, -2], chord=3, wall_size=.2)
    assert p.size((0,.5,0),.2) == pytest.approx(.05)


def test_multi_features_smallest_wins_without_enlarging_background():
    d = model()
    d["features"].append({"name": "trailing_edge", "target_ratio": .5,
                           "radius_chords": .3,
                           "points": [[.05,0,0],[.05,1,0]]})
    p = FeatureSizePolicy(d, origin=[0,0,0],chord=1,wall_size=.2)
    assert p.size((0,.5,0),.2) == pytest.approx(.05)
    assert p.size((.05,.5,0),.2) <= .1
    assert p.size((0,.5,0),.04) == .04


def test_declared_right_and_full_span_same_size():
    d = model()
    d["features"][0]["points"] = [[0,-1,0],[0,0,0],[0,1,0]]
    p = FeatureSizePolicy(d, origin=[0,0,0],chord=1,wall_size=.2)
    assert p.size((.02,-.5,0),.2) == pytest.approx(p.size((.02,.5,0),.2))


@pytest.mark.parametrize("change", [
    lambda d: d.update(schema="other"),
    lambda d: d.update(units="normalized"),
    lambda d: d.update(features=[]),
    lambda d: d["features"].append(dict(d["features"][0])),
    lambda d: d["features"][0].update(target_ratio=.01),
    lambda d: d["features"][0].update(radius_chords=0),
    lambda d: d["features"][0].update(points=[[0,0,0],[0,0,0]]),
    lambda d: d["features"][0].update(points=[[0,0,0],[float("nan"),0,0]]),
])
def test_invalid_policy_rejected(change):
    d = model()
    change(d)
    with pytest.raises(ValueError):
        FeatureSizePolicy(d, origin=[0,0,0],chord=1,wall_size=.2)


def test_bad_coordinates_rejected():
    p = policy()
    for xyz, lc in [([np.nan,0,0],.2), ([0,0,0],0), ([0,0,0],-1)]:
        with pytest.raises(ValueError):
            p.size(xyz,lc)
