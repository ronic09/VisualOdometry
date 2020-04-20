import logging
import numpy as np
from scipy.linalg import svd


def estimate_pose_dlt(pts2d, p_w_corners, k):
    # Estimates the pose of a camera using set of 2D - 3D correspondences and a given camera matrix
    # p: [nx2] vector containing the undistorted coordinates of the 2D points
    # P: [nx3] vector containing the 3D point positions
    # K: [3x3] camera matrix
    # M: [3x4] projection matrix under the form M = [R | t] where R is a rotation matrix. M encodes the transformation
    # that maps points from the world frame to the camera frame

    # Convert 2D points to normalized coordinates
    p_normalized = np.linalg.inv(k).dot((np.insert(pts2d, 2, 1, axis=1)).T)
    logging.debug('This is normalized points: \n %s' % p_normalized)

    # Build the Q-Matrix
    num_corners = np.size(p_normalized, 1)
    mat_q = np.zeros((num_corners * 2, 12))

    for i in range(num_corners):
        u = p_normalized[0, i]
        v = p_normalized[1, i]

        mat_q[2 * i, 0:3] = p_w_corners[i]
        mat_q[2 * i, 3] = 1
        mat_q[2 * i, 8:11] = -u * p_w_corners[i]
        mat_q[2 * i, 11] = -u

        mat_q[2 * i + 1, 4:7] = p_w_corners[i]
        mat_q[2 * i + 1, 7] = 1
        mat_q[2 * i + 1, 8:11] = -v * p_w_corners[i]
        mat_q[2 * i + 1, 11] = -v

    # SVD - Singular value decomposition
    u, s, vt = svd(mat_q)
    mat_m = vt[-1, :]
    mat_m = mat_m.reshape(3, 4)

    # Extract [R | t] with the correct scale from M ~[R | t]
    if np.linalg.det(mat_m[: , 0: 3]) < 0:
        mat_m = -mat_m;
    r = mat_m[:, 0:3]

    # Find the closest orthogonal matrix to R
    # https: // en.wikipedia.org / wiki / Orthogonal_Procrustes_problem
    u, s, vt = svd(r);
    r_tilde = u.dot(vt)

    # Normalization scheme using the Frobenius norm:
    # recover the unknown scale using the fact that r_corr is a true rotation matrix
    alpha = np.linalg.norm(r_tilde, ord='fro') / np.linalg.norm(r, ord='fro');

    # Build M with the corrected rotation and scale
    mat_m = np.column_stack((r_tilde, alpha * mat_m[:, 3]))

    return mat_m
