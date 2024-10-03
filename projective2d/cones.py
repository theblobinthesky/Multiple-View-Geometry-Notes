import numpy as np

DUAL_CIRCULAR_POINT_CONIC = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0.0]])

def validate_cones(*cones: np.ndarray):
    for v in cones:
        if v.shape != (6,):
            raise ValueError(f"A cone must have shape (6,), not {v.shape}.")


def validate_cone_points(cone_points: np.ndarray):
    if cone_points.shape != (6, 3):
        raise ValueError(f"The cone points have shape (6, 3), not {cone_points.shape}.")

def get_cone_matrix(cone: np.ndarray):
    """
    Parameters:
        cone (np.ndarray): A 6-element homogeneous cone.

    Returns:
        np.ndarray: The symmetric cone matrix.

    Raises:
        ValueError: If the input cone is not of shape (6,).
    """

    validate_cones(cone)

    [a, b, c, d, e, f] = cone
    return np.ndarray([
        [a, b/2, d/2], 
        [b/2, c, e/2], 
        [d/2, e/2, f]
        ])

def cone_is_singular(cone: np.ndarray):
    M = get_cone_matrix(cone)
    return np.isclose(np.linalg.det(M), 0)

def cone_through_points(points: np.ndarray) -> np.ndarray:
    validate_cone_points(points)

    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    A = np.stack([xs ** 2, xs * ys, ys ** 2, xs * zs, ys * zs, zs ** 2], axis=1)
    cone = np.linalg.solve(A, np.zeros((6,)))

    return cone

def cone_get_dual_matrix(cone_matrix: np.ndarray):
    return np.conj(cone_matrix).T

def transform_cone_matrix(matrix: np.ndarray, transform: np.ndarray):
    inv = np.linalg.inv(transform)
    return inv.T @ matrix @ inv

def transform_dual_cone_matrix(matrix: np.ndarray, transform: np.ndarray):
    return transform @ matrix @ transform.T

def angle_between_vectors(v1: np.ndarray, v2: np.ndarray, dual_circular_point_conic: np.ndarray = DUAL_CIRCULAR_POINT_CONIC):
    dot = v1.T @ dual_circular_point_conic @ v2
    v1_len = v1.T @ dual_circular_point_conic @ v1
    v2_len = v2.T @ dual_circular_point_conic @ v2
    return dot / np.sqrt(v1_len * v2_len)

def vectors_orthogonal(v1: np.ndarray, v2: np.ndarray, dual_circular_point_conic: np.ndarray = DUAL_CIRCULAR_POINT_CONIC):
    return np.isclose(angle_between_vectors(v1, v2, dual_circular_point_conic), 0)

def extract_transform_from_dual_circual_point_conic_matrix(conic_matrix: np.ndarray):
    U, S, _ = np.linalg.svd(conic_matrix)

    if not np.allclose(S, DUAL_CIRCULAR_POINT_CONIC):
        raise ValueError("The given conic matrix is not a transformed dual circular point conic.")

    return U 