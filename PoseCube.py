import logging
import numpy as np
from PoseVectorToTransformationMatrix import pose_vector_to_transformation_matrix
from ProjectPoints import project_points
from PIL import Image, ImageDraw

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    filename='Log_PoseCube.log',
                    filemode='w',
                    level=logging.DEBUG)

def main():

    img_index = 1
    filename = "images_undistorted/img_%04d" % img_index + ".jpg"

    try:
        img = Image.open(filename)
        #img.show()

    except IOError:
        pass

    # load camera poses
    pose_vector = np.loadtxt("poses.txt")
    # load camera intrinsics
    K = np.loadtxt("K.txt")
    D = np.loadtxt("D.txt")

    # 3-D corner positions of checkerboard
    square_size = 0.04
    num_corners_x = 9
    num_corners_y = 6
    num_corners = num_corners_x * num_corners_y

    xv, yv = np.meshgrid(np.linspace(0, num_corners_x - 1, num_corners_x),
                         np.linspace(0, num_corners_y - 1, num_corners_y), sparse=False, indexing='ij')
    t_x = np.ravel(xv) * square_size
    t_y = np.ravel(yv) * square_size
    t_z = np.zeros(num_corners)

    p_w_corners = [t_x, t_y, t_z]
    logging.debug('The world coordinates: \n %s' % p_w_corners)

    # Project the corners on the image.
    # Compute the 4x4 homogeneous transformation matrix that maps points from the world to the camera coordinate frame
    t_c_w = pose_vector_to_transformation_matrix(pose_vector[img_index])
    logging.debug('The transformation matrix: \n %s' % t_c_w)

    # Transform 3d points from world to current camera pose
    p_c_corners = t_c_w.dot(np.row_stack((p_w_corners, np.ones([1, num_corners]))))
    p_c_corners = p_c_corners[0:3, :]
    logging.debug('The 3d points in the current camera pose: \n %s' % p_c_corners)

    # Project 3d points to image plane
    projected_points = project_points(p_c_corners, K, D)
    logging.debug('The image coordinates of the projection points: \n %s' % projected_points)

    # Draw corners on image
    draw = ImageDraw.Draw(img)
    draw.point((projected_points[0, :], projected_points[1, :]), fill=128)
    img.show()


if __name__ == "__main__":
    main()