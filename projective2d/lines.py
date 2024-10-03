import numpy as np
import common
import common.utils
from utils import validate_homogeneous_vectors

IDEAL_LINE = np.array([0, 0, 1.0])

def line_intersection(l1: np.ndarray, l2: np.ndarray) -> np.ndarray: 
    """
    Parameters:
        l1 (np.ndarray): A 3-element projective line.
        l2 (np.ndarray): A 3-element projective line.

    Returns:
        np.ndarray: The intersection point of the two lines.

    Raises:
        ValueError: If an input array is not of shape (3,).
    """

    validate_homogeneous_vectors(l1, l2)
    return np.cross(l1, l2)

def line_through_points(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Parameters:
        p (np.ndarray): A 3-element homogeneous vector.
        q (np.ndarray): A 3-element homogeneous vector.

    Returns:
        np.ndarray: The line between the two points.

    Raises:
        ValueError: If the input arrays are not of shape (3,).
    """

    validate_homogeneous_vectors(p, q)

    l = np.cross(p, q)

    return l

def lines_check_parallel(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    """
    Parameters:
        l1 (np.ndarray): A 3-element homogeneous vector.
        l2 (np.ndarray): A 3-element homogeneous vector.

    Raises:
        ValueError: If the input arrays are not of shape (3,).
    """

    inters = line_intersection(l1, l2)
    is_on_ideal_line = np.isclose(inters[2], 0)
    return is_on_ideal_line