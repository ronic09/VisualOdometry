import numpy as np
from scipy.ndimage import gaussian_filter

def compute_blurred_images(image_pyramid, num_scales, sigma_orig):

    images_per_octave = num_scales + 3
    blurred_images = {}
    for octave in range(len(image_pyramid)):
        gaussian_img = []
        for sigma_intervals in range(-1, images_per_octave-1):
            sigma = 2 ** (sigma_intervals / num_scales) * sigma_orig
            gaussian_img.append(gaussian_filter(image_pyramid[octave], sigma))

        blurred_images[octave] = gaussian_img

    return blurred_images