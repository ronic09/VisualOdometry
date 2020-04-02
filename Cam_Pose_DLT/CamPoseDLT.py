import logging
import cv2
import argparse
import os
from PIL import Image
import numpy as np
from Cam_Pose_DLT.EstimatePoseDLT import estimate_pose_dlt


logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    filename='CamPoseDLT.log',
                    filemode='w',
                    level=logging.DEBUG)


def main():
    # Load an undistorted image and the detected corners
    filepath  = 'images/'
    img_index = 1;
    try:
        img = Image.open(filename)

    except IOError:
        pass

    k = np.loadtxt("data/K.txt")
    p_w_corners = np.loadtxt("data/p_w_corners.txt")





    K = load('./data/K.txt');

    p_W_corners = 0.01 * load('./data/p_W_corners.txt');
    num_corners = length(p_W_corners);

    % Load
    the
    2
    D
    projected
    points(detected
    on
    the
    undistorted
    image)
    all_pts2d = load('./data/detected_corners.txt');
    pts2d = all_pts2d(img_index,:);
    pts2d = reshape(pts2d, 2, 12)
    ';
    pose_dlt = estimate_pose_dlt(p_w_corners, k)


if __name__ == "__main__":
    main()
