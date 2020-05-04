import logging
import cv2
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from Corner_Detection.ShiThomasi import shi_thomasi

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    filename='CornerDetection.log',
                    filemode='w',
                    level=logging.DEBUG)


def main():
    # Randomly chosen parameters that seem to work well - can you find better ones?
    corner_patch_size = 9;
    harris_kappa = 0.08;
    num_keypoints = 200;
    nonmaximum_supression_radius = 8;
    descriptor_radius = 9;
    match_lambda = 4;

    try:
        img = Image.open('./data/000000.png')

    except IOError:
        pass

    # Part 1 - Calculate Corner Response Functions
    # Shi - Tomasi

    # grad[0] = Ix, grad[1] = Iy
    grad = shi_thomasi(img)

    fig, (ax_orig, ax_Ix, ax_Iy) = plt.subplots(3, 1, figsize=(6, 10))
    ax_orig.imshow(img, cmap='gray')
    ax_orig.set_title('Original')
    ax_orig.set_axis_off()
    ax_Ix.imshow(np.absolute(grad[0]), cmap='gray')
    ax_Ix.set_title('Derivative x-axis')
    ax_Ix.set_axis_off()
    ax_Iy.imshow(np.angle(grad[1]), cmap='gray')  # hsv is cyclic, like angles
    ax_Iy.set_title('Derivative y-axis')
    ax_Iy.set_axis_off()
    plt.show()


if __name__ == "__main__":
    main()
