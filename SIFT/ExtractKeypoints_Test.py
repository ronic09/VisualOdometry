import pandas as pd
import numpy as np
from scipy import signal

def extract_keypoints_test(diff_gaussian_images, num_scales, contrast_threshold):

    print(type(diff_gaussian_images))
    num_octaves = len(diff_gaussian_images.keys())
    kpt_locations = {}
    keypoints = []
    counter = 0

    for octave in diff_gaussian_images.keys():
        for i in range(1, num_scales):
            target_image = diff_gaussian_images.get(octave)[i]
            lower_image = diff_gaussian_images.get(octave)[i - 1]
            upper_image = diff_gaussian_images.get(octave)[i + 1]
            for u in range(1, target_image.shape[0] - 1):
                for v in range(1, target_image.shape[1] - 1):
                    sub_target = target_image[u - 1: u + 2, v - 1: v + 2]
                    sub_lower = lower_image[u - 1: u + 2, v - 1: v + 2]
                    sub_upper = upper_image[u - 1: u + 2, v - 1: v + 2]
                    max_target_coord = np.argmax(sub_target)
                    max_target = np.amax(sub_target)
                    max_lower = np.amax(sub_lower)
                    max_upper = np.amax(sub_upper)
                    # Test if max index refers to pixel in the center of the sub-array.
                    # Remember: 4 is index of central pixel in a flattened  3 x 3 array.
                    if max_target_coord == 4 and max_target >= contrast_threshold:
                        if max_target > max_lower and max_target > max_upper:
                            keypoints.append([u, v, max_target])




    return