"""Harmonic mesh extension must reproduce affine displacement on the same curved reference map."""
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from mesh_motion import prepare_harmonic, extend_harmonic

threadpool_limits(8)
root = Path(__file__).resolve().parent
mesh = dict(np.load(root/"meshes"/"wing03"/"mesh.npz"))
points = mesh["points"]
fixed, free, factor, coupling = prepare_harmonic(points, mesh)
records = []
for name, displacement in [
    ("rigid_translation", np.broadcast_to([.1, -.2, .05], points.shape)),
    ("affine_shear", np.column_stack([np.zeros(len(points)), .01*points[:, 2], .02*points[:, 0]])),
]:
    expected = points+displacement
    prescribed = points.copy()
    prescribed[fixed] = expected[fixed]
    actual = extend_harmonic(points, prescribed, fixed, free, factor, coupling)
    error = float(np.max(abs(actual-expected)))
    boundary_error = float(np.max(abs(actual[fixed]-prescribed[fixed])))
    records.append(dict(test=name, max_coordinate_error=error, prescribed_boundary_error=boundary_error,
                        passed=error < 1e-11 and boundary_error < 1e-14))
print(json.dumps(records, indent=2))
assert all(row["passed"] for row in records)
