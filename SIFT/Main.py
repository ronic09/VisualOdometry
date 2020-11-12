import logging
import cv2
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import cProfile
from PIL import Image, ImageDraw
from SIFT.ComputeImagePyramid import compute_image_pyramid
from SIFT.ComputeBlurredImages import compute_blurred_images
from SIFT.ComputeDoG import compute_DoG
from SIFT.ExtractKeypoints import extract_keypoints

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    filename='SIFT.log',
                    filemode='w',
                    level=logging.DEBUG)


def main():

    rotation_inv = 1
    rotation_img2_deg = 60
    # Scales per octave
    num_scales = 3
    # Number of octaves
    num_octaves = 5
    sigma = 1.0
    contrast_threshold = 0.04
    img = Image.open('images/img_1.jpg').convert('L')
    img_2 = Image.open('images/img_2.jpg').convert('L')
    # Rescaling of the original image for speed
    rescale_factor = 0.3


    rescaled_img = img.resize((int(img.size[0] * rescale_factor), int(img.size[1] * rescale_factor)))
    image_pyramid = compute_image_pyramid(rescaled_img, num_octaves)
    blurred_images = compute_blurred_images(image_pyramid, num_scales, sigma)
    diff_gaussian_images = compute_DoG(blurred_images, num_scales)
    tmp_kpt_locations = extract_keypoints(diff_gaussian_images, num_scales)


if __name__ == "__main__":
    main()