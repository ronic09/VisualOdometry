import logging
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from PIL import Image, ImageDraw
from Cam_Pose_DLT.EstimatePoseDLT import estimate_pose_dlt
from Cam_Pose_DLT.ReprojectPoints import reproject_points


def cam_pose_dlt(image, image_index, p_w_corners, reprojection='false'):

    # Load an undistorted image and the detected corners
    img_index = image_index;
    file_path = 'data/images_undistorted/%s' % image

    try:
        img = Image.open(file_path)

    except IOError:
        pass

    k = np.loadtxt("data/K.txt")

    # Load the 2D projected points (detected on the undistorted image)
    all_pts2d = np.loadtxt('./data/detected_corners.txt');
    pts2d = all_pts2d[img_index - 1, :].reshape((12, 2))

    # Estimate camera pose with DLT
    m_dlt = estimate_pose_dlt(pts2d, p_w_corners, k)
    logging.debug('The estimated camera pose (rotation and translation): \n %s \n' % m_dlt)

    if not reprojection:
        # Produce a 3d plot containing the corner positions and a visualization of the camera axis
        reprojected_pts = reproject_points(p_w_corners, m_dlt, k)
        logging.debug('The reprojected points: \n %s \n' % reprojected_pts)

        draw = ImageDraw.Draw(img)

        for i in range(reprojected_pts.shape[1]):
            draw.point((reprojected_pts[0, i], reprojected_pts[1, i]), fill=128)
            draw.point((pts2d[i, 0], pts2d[i, 1]), fill=1000)

        img.show()

    return m_dlt
