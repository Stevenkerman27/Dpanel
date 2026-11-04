"""Post-processing map from the P1 state to loads, shared by S2 capture and torch stages.

The production solves compute their loads inside solve()/solve_multi(); the S2 parity
needs the same map applied to a NumPy state and to a torch state, so it is factored here
and checked against the production numbers in capture (see the assertion in
tools.issue92.s2_rank_parity.capture) before any parity number is quoted.
"""
from __future__ import annotations

import numpy as np


def _freestream(alpha_deg):
    alpha = np.deg2rad(alpha_deg)
    return np.array([np.cos(alpha), 0.0, np.sin(alpha)]), np.array([-np.sin(alpha), 0.0, np.cos(alpha)])


def loads_single(ops, wing, alpha_deg, state):
    """G1 loads: CL, CL_gamma, CD_pressure, CM_y for one wing."""
    grad, mesh = ops["grad"], ops["mesh"]
    nm = ops["A"].shape[1]
    mu, gamma = state[:nm], state[nm:]
    velocity = np.einsum("fij,j->fi", grad, mu)
    cp = 1.0 - np.sum(velocity ** 2, axis=1)
    force = -np.einsum("f,f,fi->i", cp, mesh.areas, mesh.normals)
    U, lift = _freestream(alpha_deg)
    full = np.r_[0.0, gamma, 0.0]
    moment = np.sum(np.cross(mesh.centers - wing.reference_point,
                             -(cp * mesh.areas)[:, None] * mesh.normals), axis=0)
    return dict(CL=float(force @ lift / wing.reference_area),
                CL_gamma=float(2 * np.trapezoid(full, wing.gamma_y) / wing.reference_area),
                CD_pressure=float(force @ U / wing.reference_area),
                CM_y=float(moment[1] / (wing.reference_area * wing.chord)))


def loads_multi(ops, mesh, alpha_deg, state):
    """G2 loads: CL, CL_gamma, CD_pressure, CM_y for the combined body."""
    grad, ms = ops["grad"], ops["mesh"]
    nm = ops["A"].shape[1]
    mu, gamma = state[:nm], state[nm:]
    velocity = np.einsum("fij,j->fi", grad, mu)
    cp = 1.0 - np.sum(velocity ** 2, axis=1)
    face_forces = -(cp * ms.areas)[:, None] * ms.normals
    net = face_forces.sum(axis=0)
    moment = np.cross(ms.centers - mesh.reference_point, face_forces).sum(axis=0)
    U, lift = _freestream(alpha_deg)
    total = 0.0
    for group in mesh.groups:
        values = gamma[ops["slices"][group.name]]
        full = np.r_[0.0, values, 0.0]
        total += 2 * np.trapezoid(full, group.gamma_y)
    return dict(CL=float(net @ lift / mesh.reference_area),
                CL_gamma=float(total / mesh.reference_area),
                CD_pressure=float(net @ U / mesh.reference_area),
                CM_y=float(moment[1] / (mesh.reference_area * mesh.reference_chord)))
