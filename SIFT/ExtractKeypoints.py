import numpy as np
from scipy import signal


def extract_keypoints(diff_gaussian_images, num_scales, contrast_threshold):
    kpt_locations = {}

    for octave in diff_gaussian_images.keys():
        keypoints = []
        for i in range(1, num_scales):
            target_image = diff_gaussian_images.get(octave)[i]
            lower_image = diff_gaussian_images.get(octave)[i - 1]
            upper_image = diff_gaussian_images.get(octave)[i + 1]

            # Remember: Python: Dim 0 - Rows; Dim 1 - Columns
            # Visual odometry: u (x) - Columns; v (y) - Rows
            for u in range(1, target_image.shape[1] - 1):
                for v in range(1, target_image.shape[0] - 1):
                    if (target_image[v, u] >= contrast_threshold
                            and target_image[v, u] > target_image[v - 1, u] and target_image[v, u] > target_image[
                                v + 1, u]
                            and target_image[v, u] > target_image[v - 1, u - 1] and target_image[v, u] > target_image[
                                v, u - 1] and target_image[v, u] > target_image[v + 1, u - 1]
                            and target_image[v, u] > target_image[v - 1, u + 1] and target_image[v, u] > target_image[
                                v, u + 1] and target_image[v, u] > target_image[v + 1, u + 1]
                            and target_image[v, u] > lower_image[v - 1, u - 1] and target_image[v, u] > lower_image[
                                v, u - 1] and target_image[v, u] > lower_image[v + 1, u - 1]
                            and target_image[v, u] > lower_image[v - 1, u] and target_image[v, u] > lower_image[
                                v, u] and target_image[v, u] > lower_image[v + 1, u]
                            and target_image[v, u] > lower_image[v - 1, u + 1] and target_image[v, u] > lower_image[
                                v, u + 1] and target_image[v, u] > lower_image[v + 1, u + 1]
                            and target_image[v, u] > upper_image[v - 1, u - 1] and target_image[v, u] > upper_image[
                                v, u - 1] and target_image[v, u] > upper_image[v + 1, u - 1]
                            and target_image[v, u] > upper_image[v - 1, u] and target_image[v, u] > upper_image[
                                v, u] and target_image[v, u] > upper_image[v + 1, u]
                            and target_image[v, u] > upper_image[v - 1, u + 1] and target_image[v, u] > upper_image[
                                v, u + 1] and target_image[v, u] > upper_image[v + 1, u + 1]):
                        # in keypoints list: first entry is u (columns), second entry is v (rows)
                        keypoints.append([u, v, i, target_image[v, u]])
        kpt_locations[octave] = keypoints

    return kpt_locations
