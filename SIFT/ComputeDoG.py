import numpy as np

def compute_DoG (blurred_images, num_scales):

    images_per_octave = num_scales + 3
    diff_gaussian_images = {}

    for octave in blurred_images:
        diff_gaussian = []

        for j in range(images_per_octave - 1):
            diff_gaussian.append(blurred_images.get(octave)[j + 1] - blurred_images.get(octave)[j])

        diff_gaussian_images[octave] = diff_gaussian

    return diff_gaussian_images