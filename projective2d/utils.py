import common.utils as cu
from functools import partial
import numpy as np

validate_homogeneous_vectors = partial(cu.validate_homogeneous_vectors, 2)
validate_transforms = partial(cu.validate_transforms, 2)
validate_arrays_of_homogeneous_vectors = partial(cu.validate_arrays_of_homogeneous_vectors, 2)

def canonical_unembed(vector: np.ndarray) -> np.ndarray:
    validate_homogeneous_vectors(vector)

    if np.isclose(vector[2], 0):
        raise ValueError("Ideal points have no euclidian counterpart.")

    return vector[:2] / vector[2]

def canonical_embed(vector: np.ndarray) -> np.ndarray:
    return np.array([vector[0], vector[1], 1.0])

def canonical_embed_array(vectors: np.ndarray) -> np.ndarray:
    if len(vectors.shape) != 2 or vectors.shape[1] != 2:
        raise ValueError("Canonical embedding only works on euclidean 2-vectors.")
    
    return np.concat([vectors, np.ones((vectors.shape[0], 1))], axis=1)