"""Uniform physical velocity must survive paired gradient recovery on flat and warped meshes."""
import json
from pathlib import Path

import numpy as np

from wake_velocity import sample_velocity

root = Path(__file__).resolve().parent
source = root/"results"/"strips_sequence_v1"/"s24"
original = dict(np.load(source/"mesh"/"mesh.npz"))
field = dict(np.load(source/"solve"/"field.npz"))
x, eta = np.linspace(1., 5.5, 31), np.linspace(-2., 2., 25)
flow = np.array([.9, .07, .13])
records = []
for name in ["flat", "warped"]:
    mesh = dict(original)
    points = original["points"].copy()
    if name == "warped":
        downstream = np.maximum(points[:, 0]-1., 0.)
        taper = (1.-abs(points[:, 2])/5.)**2
        points[:, 1] += .01*downstream*taper*np.maximum(1.-(points[:, 1]/5.)**2, 0.)
        points[:, 2] += .02*downstream*taper
    mesh["points"] = points
    trial_field = dict(field)
    trial_field["phi"] = points@flow
    velocity, diagnostics = sample_velocity(mesh, trial_field, x, eta)
    error = float(np.max(abs(velocity-flow)))
    records.append(dict(test=name+"_uniform_physical_velocity", error=error, passed=error < 1e-11,
                        diagnostics=diagnostics))
print(json.dumps(records, indent=2))
assert all(row["passed"] for row in records)
