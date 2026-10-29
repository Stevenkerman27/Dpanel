"""Local Neumann-wedge singularity oracle for DPanel research.

For a locally straight edge with fluid wedge angle Omega and homogeneous
no-penetration on both faces, the leading nonconstant separated harmonic mode
is phi = r**lambda * cos(lambda*theta), lambda = pi/Omega.
This is an analytical reference only; it is NOT yet inserted into the solver.
"""
from __future__ import annotations
import numpy as np


def fluid_wedge_angle_from_solid(beta):
    """Return exterior-fluid cross-sectional angle 2*pi-beta for a convex solid."""
    beta = np.asarray(beta, dtype=float)
    if not np.isfinite(beta).all() or np.any(beta <= 0) or np.any(beta >= np.pi):
        raise ValueError("solid interior wedge angle beta must be in (0, pi)")
    return 2*np.pi-beta


def neumann_wedge_exponent(fluid_angle):
    """Leading nonconstant exponent lambda=pi/Omega for 2-D Neumann wedge."""
    omega = np.asarray(fluid_angle, dtype=float)
    if not np.isfinite(omega).all() or np.any(omega <= np.pi) or np.any(omega >= 2*np.pi):
        raise ValueError("exterior convex-edge fluid angle Omega must be in (pi, 2*pi)")
    return np.pi/omega


def neumann_edge_mode(r, theta, fluid_angle):
    """Return phi, radial velocity and angular velocity for r^lambda cos(lambda theta).

    theta is measured into the fluid from one face, with 0 <= theta <= Omega.
    angular_velocity means (1/r) dphi/dtheta, i.e. the physical e_theta
    velocity component. The mode is harmonic away from r=0 and has zero
    normal derivative on both wedge faces.
    """
    r = np.asarray(r, dtype=float)
    theta = np.asarray(theta, dtype=float)
    omega = np.asarray(fluid_angle, dtype=float)
    if not np.isfinite(r).all() or not np.isfinite(theta).all() or not np.isfinite(omega).all():
        raise ValueError("inputs must be finite")
    if np.any(r <= 0) or np.any(theta < 0) or np.any(theta > omega):
        raise ValueError("require r>0 and theta within the fluid wedge")
    lam = neumann_wedge_exponent(omega)
    phi = r**lam*np.cos(lam*theta)
    vr = lam*r**(lam-1)*np.cos(lam*theta)
    vtheta = -lam*r**(lam-1)*np.sin(lam*theta)
    return {"lambda": lam, "potential": phi, "velocity_r": vr,
            "velocity_theta": vtheta,
            "speed": np.sqrt(vr*vr+vtheta*vtheta)}


def edge_enrichment_trace(r, face, fluid_angle):
    """Trace of the leading Neumann singular mode on either wedge face.

    face=0 is theta=0; face=1 is theta=Omega. For lambda=pi/Omega the
    traces are +r**lambda and -r**lambda. This is a candidate local enrichment
    shape, not a complete Galerkin basis or a production-solver modification.
    """
    r = np.asarray(r, dtype=float)
    if not np.isfinite(r).all() or np.any(r < 0):
        raise ValueError("r must be finite and nonnegative")
    lam = neumann_wedge_exponent(fluid_angle)
    if face == 0:
        sign = 1.0
    elif face == 1:
        sign = -1.0
    else:
        raise ValueError("face must be 0 or 1")
    return sign*r**lam
