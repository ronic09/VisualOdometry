import numpy as np
import logging
import math


def pose_vector_to_transformation_matrix(pos_vec):

    # Converts a 6x1 pose vector into a 4x4 transformation matrix
    omega = np.array(pos_vec[0:3:1])
    logging.debug('This is the omega of the pose vector: \n %s \n' % omega)

    t = np.array(pos_vec[3:6:1])
    logging.debug('This is the translation of the pose vector: \n %s \n' % t)

    theta = np.linalg.norm(omega)
    k = omega/theta
    kx = k[0]
    ky = k[1]
    kz = k[2]
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    R = np.identity(3) + np.array(math.sin(theta)).dot(K) + np.array(1 - math.cos(theta)).dot(K.dot(K))
    T = np.identity(4)
    T[0:R.shape[0], 0:R.shape[1]] = R
    T[0:t.shape[0], 3] = t
    return T


