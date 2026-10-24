"""Circulation station equivalence classes on discovered CAD wall contacts."""
import numpy as np


def contact_station_map(stations, intervals):
    stations = np.asarray(stations, dtype=float)
    intervals = np.asarray(intervals, dtype=float).reshape(-1, 2)
    assert stations.ndim == 1 and len(stations) >= 3 and np.isfinite(stations).all()
    assert np.all(np.diff(stations) > 0.), "Circulation stations must increase strictly"
    assert np.isfinite(intervals).all()
    if len(intervals):
        assert np.all(intervals[:, 1] > intervals[:, 0]), "Contact intervals need positive width"
        assert np.all(intervals[:, 0] > stations[0]) and np.all(intervals[:, 1] < stations[-1]), "Wall contact reaching a wing tip requires a different circulation boundary model"
        assert np.all(intervals[1:, 0] > intervals[:-1, 1]), "Merge overlapping contact projections first"
        for endpoint in intervals.ravel():
            assert np.min(abs(stations-endpoint)) < 1e-10, "Contact endpoints must be explicit wake stations"
    groups = np.arange(len(stations))
    for left, right in intervals:
        selected = (stations >= left-1e-12) & (stations <= right+1e-12)
        groups[selected] = int(np.flatnonzero(selected)[0])
    mapping = np.full(len(stations), -1, dtype=int)
    _, mapping[1:-1] = np.unique(groups[1:-1], return_inverse=True)
    return mapping
