"""Harmonic interior mesh motion; prescribed wake, solid and farfield coordinates remain explicit."""
import numpy as np
from scipy.sparse.linalg import splu

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/"volume"))
import p2


def prepare_harmonic(reference_points, mesh):
    _, stiffness, _, _, _ = p2.evaluate(reference_points, mesh["tets"],
                                       reference_points[:, 0], 0., 3, True)
    fixed = np.unique(np.r_[mesh["boundary_triangles"].ravel(),
                             mesh["wake_upper"].ravel(), mesh["wake_lower"].ravel()])
    free = np.setdiff1d(np.arange(len(reference_points)), fixed)
    factor = splu(stiffness[free][:, free].tocsc())
    coupling = stiffness[free][:, fixed].tocsr()
    return fixed, free, factor, coupling


def extend_harmonic(reference_points, prescribed_points, fixed, free, factor, coupling):
    displacement = prescribed_points-reference_points
    displacement[free] = factor.solve(-coupling@displacement[fixed])
    return reference_points+displacement
