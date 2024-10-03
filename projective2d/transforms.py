import numpy as np
import projective1d as p1
import utils as u
import lines
import common
from typing import Tuple

def transform_is_affine(transform: np.ndarray) -> bool:
    u.validate_transforms(transform)

    is_affine = np.allclose(transform[1, :], np.array([0, 0, 1.0]))
    return is_affine

def validate_affine_transform(transform: np.ndarray):
    if not transform_is_affine(transform):
        raise ValueError("The transform has to be affine.")

def decompose_affine_transform(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decomposes an affine transformation into a secondary orthogonal transform
    and a orthogonal change of basis to a scale matrix.

    Parameters:
        transform (np.ndarray): A 2d projective transform.

    Returns:
        tuple:
            - np.ndarray: The secondary orthogonal transform P.
            - np.ndarray: The change of basis orthogonal transform Q.
            - np.ndarray: The scale factors S.

    Raises:
        ValueError: If the input array is not of shape (3, 3).
    
    Notes:
        The relationship `transform = P @ (Q @ S @ Q^H)` holds.
    """

    validate_affine_transform(transform)

    U, S, Vh = np.linalg.svd(transform)
    R = U @ Vh
    return R, Vh, S
    
def decompose_projective_transform(transform: np.ndarray):
    """
    Decomposes a transformation into a pure similarity, a pure affinity and a pure projectivity.

    Parameters:
        transform (np.ndarray): A 2d projective transform.

    Returns:
        tuple:
            - np.ndarray: The pure similarity transform S.
            - np.ndarray: The pure affine transform A.
            - np.ndarray: The pure projective transform P.

    Raises:
        ValueError: If the input array is not of shape (3, 3).

    Note:
        The relationship `transform = S @ A @ P` holds.
    """
    
    v = transform[2, 0:2]
    mu = transform[2, 2]
    t = transform[0:2, 2] / mu

    R, Khat = np.linalg.qr(transform - t @ v.T)
    s = np.linalg.det(s)
    K = Khat / s

    S = np.eye(3)
    S[:2, :2] = s * R
    S[:2, 2] = t

    A = np.eye(3)
    A[:2, :2] = K

    P = np.eye(3)
    P[2, 2] = mu
    P[2, :2] = v

    return S, A, P

def get_lai_tr(inters1: np.ndarray, inters2: np.ndarray) -> np.ndarray:
    line_at_inf = lines.line_through_points(inters1, inters2)
    line_at_inf /= np.linalg.norm(line_at_inf)

    M = np.eye(3)
    M[2, :] = line_at_inf # Todo: might be the wrong shape
    # M = np.linalg.inv(M)
    # Important: This is already the inverse!

    return M

def affine_extraction_with_orthogonal_lines(rect_points: np.ndarray) -> np.ndarray:
    """
    Performs affine rectification by calculating the line at infinity from two orthogonal pairs of parallel lines.

    Parameters:
        rect_points (np.ndarray): An array of rectangle points of shape (4, 3).

    Returns:
        np.ndarray: The transformation with the pure projective component removed (e.g. resets the line at infinity).

    Raises:
        ValueError: If the input array is not of shape (4, 3).
    """

    u.validate_arrays_of_homogeneous_vectors(4, rect_points)
    
    x1, x2, x3, x4 = rect_points
    pair_1_1 = lines.line_through_points(x1, x2) 
    pair_1_2 = lines.line_through_points(x3, x4)

    pair_2_1 = lines.line_through_points(x2, x3)
    pair_2_2 = lines.line_through_points(x4, x1)

    inters1 = lines.line_intersection(pair_1_1, pair_1_2)
    inters2 = lines.line_intersection(pair_2_1, pair_2_2)
    M = get_lai_tr(inters1, inters2)

    return M

def affine_extraction_with_length_ratios(l1_points: np.ndarray, l1_tr_points: np.ndarray, l2_points: np.ndarray, l2_tr_points: np.ndarray) -> np.ndarray:
    u.validate_arrays_of_homogeneous_vectors(3, l1_points, l1_tr_points, l2_points, l2_tr_points)

    def points_to_ratios(points: np.ndarray) -> np.ndarray:
        d01 = np.linalg.norm(points[1] - points[0])
        d12 = np.linalg.norm(points[2] - points[1])
        return np.array([d01, d12])

    def ratios_to_ideal_point_rel_dist(ratios: np.ndarray, tr_ratios: np.ndarray) -> np.ndarray:
        x = p1.transforms.cross_ratio(np.array([[0, 1], [ratios[0], 1], [ratios[0] + ratios[1], 1], [1, 0.0]]))
        q = tr_ratios[0] - (tr_ratios[0] * tr_ratios[1]) / ((x - tr_ratios[0] / (tr_ratios[0] + tr_ratios[1])) * (tr_ratios[0] + tr_ratios[1]))
        return q / tr_ratios[0] # should this be tr_ratios[0] or ratios[0]?

    def points_to_ideal_point_rel_dist(points: np.ndarray, tr_points: np.ndarray) -> np.ndarray:
        ratios = points_to_ratios(points)
        tr_ratios = points_to_ratios(tr_points)
        return ratios_to_ideal_point_rel_dist(ratios, tr_ratios)

    q1 = points_to_ideal_point_rel_dist(l1_points, l1_tr_points)
    q2 = points_to_ideal_point_rel_dist(l2_points, l2_tr_points)

    tr_ideal1 = u.canonical_unembed(l1_tr_points[0]) + q1 * (u.canonical_unembed(l1_tr_points[1]) - u.canonical_unembed(l1_tr_points[0]))
    tr_ideal2 = u.canonical_unembed(l2_tr_points[0]) + q2 * (u.canonical_unembed(l2_tr_points[1]) - u.canonical_unembed(l2_tr_points[0]))
    M = get_lai_tr(u.canonical_embed(tr_ideal1), u.canonical_embed(tr_ideal2))

    return M, tr_ideal1, tr_ideal2

def rectification_stratified_with_orthogonal_lines(rect_points: np.ndarray) -> np.ndarray:
    """
    Performs stratified (e.g. two step) rectification with affine extraction followed by rectification.
    
    Parameters:
        rect_points (np.ndarray): An array of rectangle points of shape (4, 3).

    Returns:
        np.ndarray: The transformation with the pure projective- and affine component removed.

    Raises:
        ValueError: If the input array is not of shape (4, 3).
    """
    H = affine_extraction_with_orthogonal_lines(rect_points)
    rect_points = np.array([H @ p for p in rect_points])

    rect_sides = [lines.line_through_points(rect_points[i], rect_points[(i + 1) % 4]) for i in range(4)]
    [l, m, n, o] = rect_sides
    A = np.array([
        [l[0] * m[0], l[0] * m[1] + l[1] * m[0], l[1] * m[1]],
        [n[0] * o[0], n[0] * o[1] + n[1] * o[0], n[1] * o[1]]
    ])

    s = common.utils.get_null_space_vector(A)
    # s is only defined up to scale, so the potential sign flip ensures positive definiteness.
    s *= np.sign(s[0])

    S = common.utils.vector_to_symmetric_matrix(s)
    K = np.linalg.cholesky(S)

    M = np.eye(3) 
    M[:2, :2] = K

    return np.linalg.inv(M) @ H

def rectification_with_orthogonal_lines(rect_points: np.ndarray, l_shape_points: np.ndarray) -> np.ndarray:
    """
    Performs rectification with the image of the dual circular points conic.
    
    Parameters:
        rect_points (np.ndarray): An array of rectangle points of shape (4, 3).
        l_shape_points (np.ndarray): An array of orthogonal lines of shape (3, 3).

    Returns:
        np.ndarray: The transformation with the pure projective- and affine component removed.

    Raises:
        ValueError: If the input array is not of shape (4, 3).
    """

    rect_sides = [lines.line_through_points(rect_points[i], rect_points[(i + 1) % 4]) for i in range(4)]
    l_shape_sides = [lines.line_through_points(l_shape_points[i], l_shape_points[i + 1]) for i in range(2)]
    
    def gen_eq(a: np.ndarray, b: np.ndarray):
        return [a[0] * b[0], 
                (a[0] * b[1] + a[1] * b[0]) / 2.0, 
                a[1] * b[1], 
                (a[0] * b[2] + a[2] * b[0]) / 2.0, 
                (a[1] * b[2] + a[2] * b[1]) / 2.0, 
                a[2] * b[2]]

    A = np.array([
        gen_eq(rect_sides[0], rect_sides[1]), 
        gen_eq(rect_sides[0], rect_sides[3]),
        gen_eq(rect_sides[2], rect_sides[1]),
        gen_eq(rect_sides[2], rect_sides[3]),

        gen_eq(l_shape_sides[0], l_shape_sides[1]),
        ])
    
    n = common.utils.get_null_space_vector(A)
    n *= np.sign(n[0])
    [a, b, c, d, e, f] = n

    # The conic matrix has 1/2 factors inside of it for convenience reasons.
    # Don't forget to adjust!
    C = np.array([
        [a, b/2, d/2],
        [b/2, c, e/2],
        [d/2, e/2, f]
    ])

    S = C[:2, :2]
    A = np.linalg.cholesky(S)
    v = np.linalg.solve(S, C[:2, 2])

    H = np.eye(3)
    H[:2, :2] = A
    H[2, :2] = v.T @ A

    return np.linalg.inv(H)

def rotation_matrix_from_angle(angle: float):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])

def scale_matrix_from_factor(factor: float) -> np.ndarray:
    S = np.eye(3)
    S[:2, :2] *= factor
    return S

def translation_matrix_from_vector(vector: np.ndarray):
    T = np.eye(3)
    T[:2, 2] = u.canonical_unembed(vector)
    return T

def transform_remove_euclidean(transform: np.ndarray, from_domain_pt: np.ndarray, to_image_pt: np.ndarray, scale: float):
    """
    Parameters:
        transform (np.ndarray): A projective 2-transform.
        from_domain_pt (np.ndarray): The source point (not transformed).
        to_image_pt (np.ndarray): The destination point (transformed).

    Returns:

    Raises:
    """

    u.validate_transforms(transform)
    u.validate_homogeneous_vectors(from_domain_pt, to_image_pt)

    from_image_pt = transform @ from_domain_pt
    T1 = translation_matrix_from_vector(u.canonical_embed(-u.canonical_unembed(from_image_pt)))
    S = scale_matrix_from_factor(scale)
    T2 = translation_matrix_from_vector(to_image_pt)

    return T2 @ S @ T1 @ transform


def transform_remove_translation(transform: np.ndarray, from_pt: np.ndarray, to_pt: np.ndarray, to_scale: float):
    """
    Parameters:
        transform (np.ndarray): A projective 2-transform.
        from_pt (np.ndarray): The source point (not transformed).
        to_pt (np.ndarray): The target point (already transformed).
        to_scale (float): The scaleup factor.

    Returns:

    Raises:
    """

    u.validate_transforms(transform)
    u.validate_homogeneous_vectors(from_pt, to_pt)

    from_pt = transform @ from_pt
    M = np.eye(3)
    M[:2, :2] *= to_scale
    M[:2, 2] = -u.canonical_unembed(from_pt)

    N = np.eye(3)
    N[:2, 2] = u.canonical_unembed(to_pt)

    return N @ M @ transform

def transform_scale(transform: np.ndarray, scale: float) -> np.ndarray:
    u.validate_transforms(transform)

    M = np.eye(3)
    M[:2, :2] *= scale

    return M @ transform