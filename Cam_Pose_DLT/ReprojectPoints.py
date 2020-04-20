import numpy as np

def reproject_points(p, m, k):

    # Re-projection of 3D points given a projection matrix
    # P: [nx3] coordinates of the 3d points in the world frame
    # M: [3x4] projection matrix
    # K: [3x3] camera matrix
    # return: [nx2] coordinates of the reprojected 2d points

    p_homo = k.dot(m).dot(np.insert(p, 3, 1, axis=1).T)
    p_homo[0, :] = p_homo[0, :] / p_homo[2, :]
    p_homo[1, :] = p_homo[1, :] / p_homo[2, :]
    p_reproj = p_homo[0:2, :]

    return p_reproj
