"""The six explicit quadratic triangle shapes, shared with independent patch tests."""
import numpy as np


def shape_functions(reference):
    barycentric = np.column_stack([1.-reference.sum(axis=1), reference])
    barycentric_gradient = np.array([[-1., -1.], [1., 0.], [0., 1.]])
    values = np.column_stack([barycentric*(2.*barycentric-1.),
                              4.*barycentric[:, 0]*barycentric[:, 1],
                              4.*barycentric[:, 1]*barycentric[:, 2],
                              4.*barycentric[:, 2]*barycentric[:, 0]])
    gradients = np.empty((len(reference), 6, 2))
    gradients[:, :3] = (4.*barycentric-1.)[:, :, None]*barycentric_gradient
    for edge, (left, right) in enumerate([(0, 1), (1, 2), (2, 0)]):
        gradients[:, 3+edge] = 4.*(barycentric[:, left, None]*barycentric_gradient[right]
                                  +barycentric[:, right, None]*barycentric_gradient[left])
    return values, gradients
