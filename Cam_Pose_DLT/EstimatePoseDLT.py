import logging
import numpy as np


def estimate_pose_dlt(p_w_corners, k):
    # Estimates the pose of a camera using set of 2D - 3D correspondences and a given camera matrix
    # p: [nx2] vector containing the undistorted coordinates of the 2D points
    # P: [nx3] vector containing the 3D point positions
    # K: [3x3] camera matrix
    # M: [3x4] projection matrix under the form M = [R | t] where R is a rotation matrix. M encodes the transformation
    # that maps points from the world frame to the camera frame

    # Convert 2D points to normalized coordinates
    p_normalized = np.linalg.inv(k).dot(p_w_corners.T)
    print('This is k: \n %s' % k)
    print('This is p_w_corners: \n %s' % p_w_corners)
    print('This is p_normalized: \n %s' % p_normalized)
