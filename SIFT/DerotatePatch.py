import math
import numpy as np

#TODO: CHECK IF (AND HOW) IT WORKS
def derotate_patch(image, location, patch_size, origin):
    # The patch can overlap at most sqrt(2)/2 * patch_size over the image edge.
    # To prevent this, pad the image with zeros
    padding = math.ceil(np.sqrt(2) * patch_size / 2);
    derotated_patch = np.zeros(shape=(patch_size, patch_size))
    padded_img = np.pad(image, padding, 'constant')

    # compute derotated patch
    for px in range(patch_size):
        for py in range(patch_size):
            x_origin = px - patch_size / 2
            y_origin = py - patch_size / 2

            # rotate patch by angle ori
            x_rotated = math.cos(math.pi * origin / 180) * x_origin - math.sin(math.pi * origin / 180) * y_origin
            y_rotated = math.sin(math.pi * origin / 180) * x_origin + math.cos(math.pi * origin / 180) * y_origin

            # move coordinates to patch
            x_patch_rotated = location[1] + x_rotated
            y_patch_rotated = location[0] - y_rotated

            # sample image (using nearest neighbor sampling as opposes to more accurate bilinear sampling)
            y_img_padded = math.ceil(y_patch_rotated + padding)
            x_img_padded = math.ceil(x_patch_rotated + padding)
            derotated_patch[py, px] = padded_img[y_img_padded, x_img_padded]

    return derotated_patch
