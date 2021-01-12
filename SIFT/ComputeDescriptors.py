import numpy as np
from SIFT.GaussianFilterMatlab import gauss2D_matlab
from SIFT.DerotatePatch import derotate_patch
from scipy import ndimage


def compute_descriptors(blurred_images, keypoint_locations, rotation_invariant = 1):

    for octave in keypoint_locations.keys():
        # define gaussian filter with magic multiplication of 1.5 taken from Lowe's paper
        gaussian_filter = gauss2D_matlab((16, 16), 16 * 1.5)
        # check what images actually contain keypoints s.t. only relevant image indexes have to be searched
        relevant_img_idx = list(set(list[2] for list in keypoint_locations[octave]))
        # only gp through relevant images
        for img_idx in relevant_img_idx:
            # calculate image gradient magnitudes and directions
            grad_x, grad_y = np.gradient(blurred_images[octave][img_idx])
            grad_mag = np.sqrt(grad_x**2.0 + grad_y**2.0)
            grad_dir = np.arctan2(grad_y, grad_x) * (180 / np.pi)

            # get coordinates of keypoints for relevant image index
            img_kp_coord = [t[0:2] for t in keypoint_locations[octave] if t[2] == img_idx]
            # generate empty descriptor array
            img_descriptor = np.zeros([len(img_kp_coord), 128])
            # generate empty locator validation array - only keypoints far enough from image border are valid
            # reason: descriptor requires gradient information of surrounding pixels
            is_valid_locator = np.zeros(len(img_kp_coord))
            # get coordinates for each keypoint
            for kp_idx in range(len(img_kp_coord)):
                kp_row = img_kp_coord[kp_idx][0]
                kp_col = img_kp_coord[kp_idx][1]
                img_row_size, img_col_size = blurred_images[octave][img_idx].shape

                # TODO make sure that keypoints are not too close (not same spot); example - first keypoint
                if kp_row > 7 and kp_col > 7 and kp_row < img_row_size - 7 and kp_col < img_col_size - 7:
                    is_valid_locator[kp_idx] = 1
                    grad_mag_loc = grad_mag[kp_row - 8: kp_row + 8, kp_col - 8: kp_col + 8]
                    grad_mag_loc_weighted = grad_mag_loc * gaussian_filter
                    grad_dir_loc = grad_dir[kp_row - 8: kp_row + 8, kp_col - 8: kp_col + 8]

                    grad_mag_loc_derotated_weighted = grad_mag_loc_weighted
                    grad_dir_loc_derotated = grad_dir_loc
                    if rotation_invariant == 1:
                        # compute dominatnt direction through looking at the most common orientation in the histogram,
                        # spaced at 10 degrees --> from -180° to 180 results in 37 bins
                        bins_angles = np.arange(-180, 180, 10)
                        orientation_hist = np.histogram(grad_dir_loc, bins=bins_angles, weights=grad_mag_loc_weighted)
                        bin_dominant_direction = orientation_hist[1][orientation_hist[0].argmax()]
                        # take average of the two bin edges to calculate principal direction
                        grad_dir_loc_principal = (bin_dominant_direction + 10) / 2

                        # derotate patch
                        patch_derotated = derotate_patch(blurred_images[octave][img_idx], [kp_row, kp_col], 16, grad_dir_loc_principal)






    return
