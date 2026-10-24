"""Analytic constant triangular single/double-layer potentials and gradients.

G = 1/(4*pi*|x-y|), D = integral dG/dn_y. Positive source coefficient
has inward velocity with this potential convention. No vortex core or clipping.
"""
import math

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def influence(targets, panels, normals, trace_panel, kind):
    """kind=0: integral G and grad_x G; kind=1: integral dG/dn_y and its gradient.

trace_panel[i] gives the coincident panel whose EXTERIOR trace is requested,
or -1 for an ordinary field point. No distance threshold chooses the side.
"""
    potential = np.empty((len(targets), len(panels)))
    velocity = np.empty((len(targets), len(panels), 3))
    for i in prange(len(targets)):
        for j in range(len(panels)):
            r = panels[j] - targets[i]
            radius = np.sqrt(np.sum(r*r, axis=1))
            n = normals[j]
            cross12 = np.cross(r[1], r[2])
            determinant = np.dot(r[0], cross12)
            denominator = (radius[0]*radius[1]*radius[2]
                           + np.dot(r[0],r[1])*radius[2]
                           + np.dot(r[1],r[2])*radius[0]
                           + np.dot(r[2],r[0])*radius[1])
            omega = -2.*math.atan2(determinant, denominator)
            height = -np.dot(r[0], n)
            if trace_panel[i] == j:
                omega = 2.*math.pi
                height = 0.
            value = 0.
            grad = np.zeros(3)
            valid = True
            for k in range(3):
                q = (k+1) % 3
                edge = panels[j,q] - panels[j,k]
                length = np.sqrt(np.dot(edge,edge))
                cross = np.cross(r[k],r[q])
                dot = np.dot(r[k],r[q])
                product = radius[k]*radius[q]
                # Rationalization avoids subtracting nearly equal positive lengths.
                plus = product+dot
                if dot < 0.:
                    plus = np.dot(cross,cross)/(product-dot)
                if plus <= 0. or product == 0. or length == 0.:
                    valid = False
                    break
                if kind == 0:
                    gap = 2.*plus/(radius[k]+radius[q]+length)
                    line_integral = math.log1p(2.*length/gap)
                    outward = np.cross(edge/length,n)
                    value += np.dot(r[k],outward)*line_integral
                    grad -= outward*line_integral
                else:
                    grad -= cross*(radius[k]+radius[q])/(product*plus)
            if not valid:
                potential[i,j] = np.nan
                velocity[i,j,:] = np.nan
            elif kind == 0:
                potential[i,j] = (value-height*omega)/(4.*math.pi)
                velocity[i,j,:] = (grad-n*omega)/(4.*math.pi)
            else:
                potential[i,j] = omega/(4.*math.pi)
                velocity[i,j,:] = grad/(4.*math.pi)
    return potential, velocity
