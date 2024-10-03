import numpy as np

from . import utils as u
import common

def cross_ratio_det(p: np.ndarray, q: np.ndarray):
    return p[0] * q[1] - q[0] * p[1]

def cross_ratio(points: np.ndarray):
    u.validate_arrays_of_homogeneous_vectors(4, points)

    [x1, x2, x3, x4] = points
    ratio = cross_ratio_det(x1, x2) * cross_ratio_det(x3, x4) / (cross_ratio_det(x1, x3) * cross_ratio_det(x2, x4))

    return ratio

