import common.utils as cu
from functools import partial

validate_homogeneous_vectors = partial(cu.validate_homogeneous_vectors, 1)
validate_transforms = partial(cu.validate_transforms, 1)
validate_arrays_of_homogeneous_vectors = partial(cu.validate_arrays_of_homogeneous_vectors, 1)