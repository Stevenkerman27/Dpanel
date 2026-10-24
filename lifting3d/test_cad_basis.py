"""Independent noncentral and multiple contact-interval checks."""
import numpy as np
from cad_basis import contact_station_map

stations = np.array([-1.3, -.8, -.5, -.2, .1, .4, .7, 1., 1.6, 2.4])
intervals = np.array([[-.5, .1], [.7, 1.6]])
mapping = contact_station_map(stations, intervals)
assert np.array_equal(mapping, [-1, 0, 1, 1, 1, 2, 3, 3, 3, -1])
gamma = np.array([.3, .7, -.2, .4])
expanded = np.zeros(len(stations))
expanded[1:-1] = gamma[mapping[1:-1]]
assert np.array_equal(expanded, [0., .3, .7, .7, .7, -.2, .4, .4, .4, 0.])
assert np.array_equal(contact_station_map(stations, []), np.r_[-1, np.arange(8), -1])
for invalid in [[[-1.3, -.5]], [[-.5, .1], [0., 1.]], [[-.55, .1]]]:
    rejected = False
    try:
        contact_station_map(stations, invalid)
    except AssertionError:
        rejected = True
    assert rejected
print("CAD contact basis: noncentral/multiple intervals, zero tips and explicit invalid-contact rejection passed")
