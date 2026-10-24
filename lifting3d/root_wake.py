"""Independent geometry and reduced-basis candidate for an aft-body wake root."""
import argparse
import json

import numpy as np


def ellipsoid_root_interval(te_x, body_center_x, semi_axis_x, body_radius, half_span):
    parameters = np.array([te_x, body_center_x, semi_axis_x, body_radius, half_span])
    if not np.isfinite(parameters).all():
        raise ValueError("TE and ellipsoid dimensions must be finite")
    if semi_axis_x <= 0.0 or body_radius <= 0.0 or half_span <= 0.0:
        raise ValueError("Ellipsoid axes and wing half-span must be positive")
    section = (te_x - body_center_x) / semi_axis_x
    if abs(section) >= 1.0:
        raise ValueError("Wing TE plane does not cut the ellipsoid in a finite root interval")
    root_half_width = body_radius * np.sqrt(1.0 - section**2)
    if root_half_width <= 0.0:
        raise ValueError("Ellipsoid/TE contact is tangent; a finite root interval is required")
    if root_half_width >= half_span:
        raise ValueError("Body occupies the full wing span at the TE; no exposed TE remains")
    return np.array([-root_half_width, root_half_width])


def merge_root_basis(stations, root_interval, half_span):
    stations = np.asarray(stations, dtype=float)
    root_interval = np.asarray(root_interval, dtype=float)
    if stations.ndim != 1 or len(stations) < 2 or not np.isfinite(stations).all():
        raise ValueError("Stations must be a finite one-dimensional array")
    if np.any(np.diff(stations) <= 0.0):
        raise ValueError("Stations must be strictly increasing")
    if root_interval.shape != (2,) or not np.isfinite(root_interval).all():
        raise ValueError("Root interval must contain two finite endpoints")
    if root_interval[0] >= root_interval[1]:
        raise ValueError("Root interval must have positive width")
    if abs(stations[0] + half_span) > 1.0e-12 or abs(stations[-1] - half_span) > 1.0e-12:
        raise ValueError("Stations must include both wing tips")
    if root_interval[0] <= stations[0] or root_interval[1] >= stations[-1]:
        raise ValueError("Body occupies the full wing span at the TE; no exposed TE remains")
    if not np.any(np.isclose(stations, root_interval[0], rtol=0.0, atol=1.0e-12)):
        raise ValueError("Insert the negative root endpoint as a wake station first")
    if not np.any(np.isclose(stations, root_interval[1], rtol=0.0, atol=1.0e-12)):
        raise ValueError("Insert the positive root endpoint as a wake station first")
    inside_root = (stations >= root_interval[0] - 1.0e-12) & (stations <= root_interval[1] + 1.0e-12)
    if not inside_root.any() or inside_root[0] or inside_root[-1]:
        raise ValueError("Root interval must leave exposed trailing-edge stations on both sides")

    station_dof = np.full(len(stations), -1, dtype=int)
    station_dof[inside_root] = 0
    next_dof = 1
    for station in range(1, len(stations) - 1):
        if not inside_root[station]:
            station_dof[station] = next_dof
            next_dof += 1
    return station_dof


def expand_station_values(station_dof, dof_values):
    station_dof = np.asarray(station_dof, dtype=int)
    dof_values = np.asarray(dof_values, dtype=float)
    if station_dof.ndim != 1 or dof_values.ndim != 1:
        raise ValueError("Station map and circulation values must be one-dimensional")
    if np.any(station_dof >= len(dof_values)) or len(dof_values) == 0:
        raise ValueError("Station map refers to a missing circulation degree of freedom")
    station_values = np.zeros(len(station_dof))
    free = station_dof >= 0
    station_values[free] = dof_values[station_dof[free]]
    return station_values


def mapped_weak_test_support(stations, station_dof, exposed_intervals):
    stations = np.asarray(stations, dtype=float)
    station_dof = np.asarray(station_dof, dtype=int)
    exposed_intervals = np.asarray(exposed_intervals, dtype=float)
    dof_count = int(station_dof.max()) + 1
    support = np.zeros((len(exposed_intervals), dof_count))
    for edge, interval in enumerate(exposed_intervals):
        edge_left, edge_right = interval
        if not np.isfinite([edge_left, edge_right]).all() or edge_left >= edge_right:
            raise ValueError("Each exposed TE interval must have finite positive width")
        if edge_left < stations[0] or edge_right > stations[-1]:
            raise ValueError("Exposed TE interval lies outside the wing span")
        for segment in range(len(stations) - 1):
            left = max(edge_left, stations[segment])
            right = min(edge_right, stations[segment + 1])
            if right <= left:
                continue
            width = right - left
            fraction_left = (left - stations[segment]) / (stations[segment + 1] - stations[segment])
            fraction_right = (right - stations[segment]) / (stations[segment + 1] - stations[segment])
            for station, weight in [
                (segment, 0.5 * width * ((1.0 - fraction_left) + (1.0 - fraction_right))),
                (segment + 1, 0.5 * width * (fraction_left + fraction_right)),
            ]:
                dof = station_dof[station]
                if dof >= 0:
                    support[edge, dof] += weight
    return support


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncpu", type=int, default=8)
    args = parser.parse_args()
    assert args.ncpu == 8, "This small independent research script records NCPU=8"

    wing_half_span = 2.0
    te_x = 1.0
    body_aft = 1.5
    body_length = 4.0
    body_radius = 0.35
    body_center_x = body_aft - body_length / 2.0
    root_interval = ellipsoid_root_interval(
        te_x, body_center_x, body_length / 2.0, body_radius, wing_half_span)
    stations = np.unique(np.r_[
        np.linspace(-wing_half_span, wing_half_span, 9),
        root_interval,
    ])
    station_dof = merge_root_basis(stations, root_interval, wing_half_span)
    root_dof = int(station_dof[np.flatnonzero(np.isclose(stations, 0.0))[0]])
    exposed_intervals = np.array([
        [-wing_half_span, root_interval[0]],
        [root_interval[1], wing_half_span],
    ])
    weak_support = mapped_weak_test_support(stations, station_dof, exposed_intervals)

    summary = {
        "status": "candidate_geometry_and_basis_only",
        "scope": "Rectangular wing and x-aligned ellipsoid; planar symmetry-plane wake cut only; no solver or mesh integration",
        "ncpu_requested": args.ncpu,
        "geometry": {
            "wing_te_x": te_x,
            "wing_half_span": wing_half_span,
            "body_aft": body_aft,
            "body_length": body_length,
            "body_radius": body_radius,
            "body_center_x": body_center_x,
            "root_interval_y": root_interval.tolist(),
        },
        "gamma_basis": {
            "stations": stations.tolist(),
            "station_to_dof": station_dof.tolist(),
            "root_dof": root_dof,
            "exposed_te_intervals": exposed_intervals.tolist(),
            "weak_test_support_integrals": weak_support.tolist(),
            "weak_test_support_count_per_dof": (weak_support > 0.0).sum(axis=0).tolist(),
        },
        "interpretation": "The merged root basis is constant on the body-covered TE interval and retains support on each adjacent exposed TE interval.",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
