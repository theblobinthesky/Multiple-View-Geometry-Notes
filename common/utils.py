import numpy as np

def validate_homogeneous_vectors(n: int, *vectors: np.ndarray):
    """
    Parameters:
        n: Projective space dimension.
        *lines: Variable number of n-element arrays.

    Raises:
        ValueError: If the vector is not of shape (n + 1,).
    """

    for v in vectors:
        if v.shape != (n + 1,):
            raise ValueError(f"A homogeneous {n}-vector must have shape ({n + 1},), not {v.shape}.")

def validate_transforms(n: int, *transforms: np.ndarray):
    for v in transforms:
        if v.shape != (n + 1, n + 1):
            raise ValueError(f"A homogeneous {n}-transform must have shape ({n + 1}, {n + 1}), not {v.shape}.")

def validate_arrays_of_homogeneous_vectors(n: int, point_count: int, *vectors: np.ndarray):
    for v in vectors:
        if v.shape != (point_count, n + 1):
            raise ValueError(f"An array of {point_count} homogeneous {n}-vectors must have shape ({point_count}, {n + 1}), not {v.shape}.")

def normalize_homogeneous_vectors(*vectors: np.ndarray) -> list[np.ndarray]:
    vectors = [v / np.linalg.norm(v) for v in vectors]
    return vectors

def get_null_space_vector(M: np.ndarray, rtol = 1e-4):
    _, S, Vh = np.linalg.svd(M)
    # if S[-1] >= S[0] * rtol:
    #    raise ValueError("The matrix has no null space.")

    return Vh[-1]

def symmetric_matrix_to_vector(M: np.ndarray) -> np.ndarray:
    if len(M.shape) != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("Matrix is not symmetric.")

    n = M.shape[0]
    return M[np.triu_indices(n)]

def vector_to_symmetric_matrix(v: np.ndarray) -> np.ndarray:
    if len(v.shape) != 1:
        raise ValueError("Vector is not of shape (n,).")

    n = int(np.sqrt(2 * v.shape[0]))
    M = np.zeros((n, n))
    ind = np.triu_indices(n)
    
    M[ind] = v
    M = M + M.T - np.diag(np.diagonal(M))
    return M