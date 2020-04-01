import numpy as np
from Pose_Cube.DistortPoints import distort_points


def project_points(points_3d, cam_intrinsics_k, dist_vector_d=np.zeros([4, 1])):
    # Projects 3d points to the image plane (3xN), given the camera matrix k(3x3) and distortion coefficients d(4x1)
    num_points = points_3d[1, :].size

    # get normalized coordinates
    x_norm = points_3d[0, :] / points_3d[2, :]
    y_norm = points_3d[1, :] / points_3d[2, :]

    # apply distortion
    x_d = distort_points([x_norm, y_norm], dist_vector_d)
    x_norm_dist = x_d[0]
    y_norm_dist = x_d[1]

    # convert to pixel coordinates
    projected_points = cam_intrinsics_k.dot([x_norm_dist, y_norm_dist, np.ones([num_points])])
    projected_points = projected_points[0:2, :]
    return projected_points
