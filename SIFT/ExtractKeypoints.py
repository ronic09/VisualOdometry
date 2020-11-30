import numpy as np
from scipy import signal


def extract_keypoints(diff_gaussian_images, num_scales, contrast_threshold):
    print(type(diff_gaussian_images))
    num_octaves = len(diff_gaussian_images.keys())
    kpt_locations = {}

    counter = 0

    for octave in diff_gaussian_images.keys():
        keypoints = []
        for i in range(1, num_scales):
            target_image = diff_gaussian_images.get(octave)[i]
            lower_image = diff_gaussian_images.get(octave)[i - 1]
            upper_image = diff_gaussian_images.get(octave)[i + 1]
            for u in range(1, target_image.shape[0] - 1):
                for v in range(1, target_image.shape[1] - 1):
                    if (target_image[u, v] >= contrast_threshold
                            and target_image[u, v] > target_image[u - 1, v] and target_image[u, v] > target_image[
                                u + 1, v]
                            and target_image[u, v] > target_image[u - 1, v - 1] and target_image[u, v] > target_image[
                                u, v - 1] and target_image[u, v] > target_image[u + 1, v - 1]
                            and target_image[u, v] > target_image[u - 1, v + 1] and target_image[u, v] > target_image[
                                u, v + 1] and target_image[u, v] > target_image[u + 1, v + 1]
                            and target_image[u, v] > lower_image[u - 1, v - 1] and target_image[u, v] > lower_image[
                                u, v - 1] and target_image[u, v] > lower_image[u + 1, v - 1]
                            and target_image[u, v] > lower_image[u - 1, v] and target_image[u, v] > lower_image[
                                u, v] and target_image[u, v] > lower_image[u + 1, v]
                            and target_image[u, v] > lower_image[u - 1, v + 1] and target_image[u, v] > lower_image[
                                u, v + 1] and target_image[u, v] > lower_image[u + 1, v + 1]
                            and target_image[u, v] > upper_image[u - 1, v - 1] and target_image[u, v] > upper_image[
                                u, v - 1] and target_image[u, v] > upper_image[u + 1, v - 1]
                            and target_image[u, v] > upper_image[u - 1, v] and target_image[u, v] > upper_image[
                                u, v] and target_image[u, v] > upper_image[u + 1, v]
                            and target_image[u, v] > upper_image[u - 1, v + 1] and target_image[u, v] > upper_image[
                                u, v + 1] and target_image[u, v] > upper_image[u + 1, v + 1]):
                        keypoints.append([u, v, i, target_image[u, v]])
        kpt_locations[octave] = keypoints

    return kpt_locations
