import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from Cam_Pose_DLT.CamPoseDLT import cam_pose_dlt
from matplotlib.animation import FuncAnimation

# Compute segments (required for 3d quiver update)


def compute_segs(position, rotation, axis):
    x = position[0]
    y = position[1]
    z = position[2]
    u = rotation[0, axis]
    v = rotation[1, axis]
    w = rotation[2, axis]

    return x, y, z, u, v, w


# Update quivers for animation


def animate(i, pos_all, rot_mat_all, quivers_x, quivers_y, quivers_z):
    segs_x = np.array(compute_segs(pos_all[i], rot_mat_all[i], 0)).reshape(6, -1)
    segs_y = np.array(compute_segs(pos_all[i], rot_mat_all[i], 1)).reshape(6, -1)
    segs_z = np.array(compute_segs(pos_all[i], rot_mat_all[i], 2)).reshape(6, -1)
    new_segs_x = [[[x, y, z], [u, v, w]] for x, y, z, u, v, w in zip(*segs_x.tolist())]
    new_segs_y = [[[x, y, z], [u, v, w]] for x, y, z, u, v, w in zip(*segs_y.tolist())]
    new_segs_z = [[[x, y, z], [u, v, w]] for x, y, z, u, v, w in zip(*segs_z.tolist())]
    quivers_x.set_segments(new_segs_x)
    quivers_y.set_segments(new_segs_y)
    quivers_z.set_segments(new_segs_z)

    return quivers_x, quivers_y, quivers_z


def main():
    # Load corners world coordinates
    p_w_corners = 0.01 * np.loadtxt("data/p_w_corners.txt")

    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-ext", "--extension", required=False, default='jpg', help="extension name. default is 'jpg'.")
    ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
    args = vars(ap.parse_args())

    # Arguments
    dir_path = 'data/images_undistorted/'
    ext = args['extension']
    output = args['output']

    # Detect all available images
    images = []
    for f in os.listdir(dir_path):
        if f.endswith(ext):
            images.append(f)

    images.sort()
    num_frames = len(images)

    # get M-Matrix for each image
    i = 0
    pos_all = []
    rot_mat_all = []
    for img in images:
        mat_m = cam_pose_dlt(img, i, p_w_corners)
        r_c_w = mat_m[0:3, 0:3]
        t_c_w = mat_m[0:3, 3]
        rot_mat = r_c_w.T
        pos = -rot_mat.dot(t_c_w)
        pos_all.append(pos)
        rot_mat_all.append((rot_mat))
        i = i + 1

    # Create figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')

    # Generate initial quivers
    segment_x = compute_segs(pos_all[0], rot_mat_all[0], 0)
    segment_y = compute_segs(pos_all[0], rot_mat_all[0], 1)
    segment_z = compute_segs(pos_all[0], rot_mat_all[0], 2)

    quivers_x = ax.quiver(*segment_x, colors='r', length=0.1, normalize=True)
    quivers_y = ax.quiver(*segment_y, colors='g', length=0.1, normalize=True)
    quivers_z = ax.quiver(*segment_z, colors='b', length=0.1, normalize=True)

    # Generate reference point (World coordinates)
    for i in range(len(p_w_corners)-1):
        ax.scatter(p_w_corners[i, 0], p_w_corners[i, 1], p_w_corners[i, 2])

    ani = FuncAnimation(fig, animate, fargs=(pos_all, rot_mat_all, quivers_x, quivers_y, quivers_z), frames=num_frames,
                        interval=10, blit=False)

    plt.show()


if __name__ == "__main__":
    main()