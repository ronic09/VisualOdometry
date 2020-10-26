import logging
import numpy as np
from Pose_Cube.PoseVectorToTransformationMatrix import pose_vector_to_transformation_matrix
from Pose_Cube.ProjectPoints import project_points
from PIL import Image, ImageDraw


def draw_cube(image_path, image_index, corners=False):

    filename = image_path
    img_index = image_index

    try:
        img = Image.open(filename)

    except IOError:
        pass

    # load camera poses
    pose_vector = np.loadtxt("poses.txt")
    # load camera intrinsics
    K = np.loadtxt("K.txt")
    D = np.loadtxt("D.txt")

    square_size = 0.04

    # Project the corners on the image.
    # Compute the 4x4 homogeneous transformation matrix that maps points from the world to the camera coordinate frame
    t_c_w = pose_vector_to_transformation_matrix(pose_vector[img_index - 1])
    logging.debug('The transformation matrix: \n %s \n' % t_c_w)

    draw = ImageDraw.Draw(img)

    # draw corners if required
    if corners:

        # 3-D corner positions of checkerboard
        num_corners_x = 9
        num_corners_y = 6
        num_corners = num_corners_x * num_corners_y

        xv, yv = np.meshgrid(np.linspace(0, num_corners_x - 1, num_corners_x),
                             np.linspace(0, num_corners_y - 1, num_corners_y), sparse=False, indexing='ij')
        t_x = np.ravel(xv) * square_size
        t_y = np.ravel(yv) * square_size
        t_z = np.zeros(num_corners)

        p_w_corners = [t_x, t_y, t_z]
        logging.debug('The world coordinates: \n %s \n' % p_w_corners)

        # Transform 3d points from world to current camera pose
        p_c_corners = t_c_w.dot(np.row_stack((p_w_corners, np.ones([1, num_corners]))))
        p_c_corners = p_c_corners[0:3, :]
        logging.debug('The 3d points in the current camera pose: \n %s \n' % p_c_corners)

        # Project 3d points to image plane
        projected_points = project_points(p_c_corners, K, D)
        logging.debug('The image coordinates of the projection points: \n %s \n' % projected_points)

        # Draw corners on image
        for i in range(num_corners):
            j = i
            draw.point((projected_points[0, i], projected_points[1, j]), fill=128)
        img.show()

    # Draw a cube
    offset_x = square_size * 2
    offset_y = square_size
    s = square_size * 2
    num_corners_cube = 8
    [x, z, y] = np.meshgrid(np.linspace(0, 1, 2), np.linspace(-1, 0, 2), np.linspace(0, 1, 2))
    cube = [offset_x + x * s, offset_y + y * s, z * s]
    p_w_cube_x = np.concatenate([cube[0][0][0], cube[0][0][1], cube[0][1][0], cube[0][1][1]])
    p_w_cube_y = np.concatenate([cube[1][0][0], cube[1][0][1], cube[1][1][0], cube[1][1][1]])
    p_w_cube_z = np.concatenate([cube[2][0][0], cube[2][0][1], cube[2][1][0], cube[2][1][1]])

    p_c_cube = t_c_w.dot([p_w_cube_x, p_w_cube_y, p_w_cube_z, np.ones([num_corners_cube])])
    p_c_cube = p_c_cube[0:3, :]
    cube_pts = project_points(p_c_cube, K, D)

    #draw top of cube
    draw.line((cube_pts[0, 0], cube_pts[1, 0], cube_pts[0, 1], cube_pts[1, 1]), fill=200, width=2)
    draw.line((cube_pts[0, 0], cube_pts[1, 0], cube_pts[0, 2], cube_pts[1, 2]), fill=200, width=2)
    draw.line((cube_pts[0, 2], cube_pts[1, 2], cube_pts[0, 3], cube_pts[1, 3]), fill=200, width=2)
    draw.line((cube_pts[0, 1], cube_pts[1, 1], cube_pts[0, 3], cube_pts[1, 3]), fill=200, width=2)

    # draw bottom of cube
    draw.line((cube_pts[0, 4], cube_pts[1, 4], cube_pts[0, 5], cube_pts[1, 5]), fill=200, width=2)
    draw.line((cube_pts[0, 4], cube_pts[1, 4], cube_pts[0, 6], cube_pts[1, 6]), fill=200, width=2)
    draw.line((cube_pts[0, 6], cube_pts[1, 6], cube_pts[0, 7], cube_pts[1, 7]), fill=200, width=2)
    draw.line((cube_pts[0, 5], cube_pts[1, 5], cube_pts[0, 7], cube_pts[1, 7]), fill=200, width=2)

    # draw mid layer of cube
    draw.line((cube_pts[0, 0], cube_pts[1, 0], cube_pts[0, 4], cube_pts[1, 4]), fill=200, width=2)
    draw.line((cube_pts[0, 1], cube_pts[1, 1], cube_pts[0, 5], cube_pts[1, 5]), fill=200, width=2)
    draw.line((cube_pts[0, 2], cube_pts[1, 2], cube_pts[0, 6], cube_pts[1, 6]), fill=200, width=2)
    draw.line((cube_pts[0, 3], cube_pts[1, 3], cube_pts[0, 7], cube_pts[1, 7]), fill=200, width=2)
    #img.show()

    return img
