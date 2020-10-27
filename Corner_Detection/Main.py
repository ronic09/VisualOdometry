import logging
import cv2
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from Corner_Detection.ShiTomasi import shi_tomasi
from Corner_Detection.Harris import harris
from Corner_Detection.SelectKeyPoints import select_key_points
from Corner_Detection.DescribeKeypoints import describe_keypoints
from Corner_Detection.MatchDescriptors import match_descriptors

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
    # Shi - Tomasi score and Harris score
    shi_tomasi_score = shi_tomasi(img, corner_patch_size)
    harris_score = harris(img, corner_patch_size, harris_kappa)

    fig, ((ax_orig, ax_rShi), (ax_orig2, ax_rHarris)) = plt.subplots(2, 2, figsize=(20, 10))
    ax_orig.imshow(img, cmap='gray')
    ax_orig.set_title('Original')
    ax_orig.set_axis_off()
    ax_rShi.imshow(np.absolute(shi_tomasi_score), cmap='rainbow')
    ax_rShi.set_title('Shi-Thomasi Score')
    ax_rShi.set_axis_off()
    ax_orig2.imshow(img, cmap='gray')
    ax_orig2.set_title('Original')
    ax_orig2.set_axis_off()
    ax_rHarris.imshow(np.absolute(harris_score), cmap='rainbow')
    ax_rHarris.set_title('Harris Score')
    ax_rHarris.set_axis_off()
    plt.show()

    # Part 2 - Select key points and perform non-maximum surpression
    keypoints = select_key_points(harris_score, num_keypoints, nonmaximum_supression_radius)

    # Ugly work around to convert gray scale image to RGB
    # 1. Open with CV2 with RGB codec
    # 2. Convert to PIL image
    img_cv = cv2.imread('./data/000000.png', 1)
    img_col = Image.fromarray(img_cv)

    draw = ImageDraw.Draw(img_col)
    for i in range(len(keypoints[0])):
        top = (keypoints[1, i], keypoints[0, i] - 3)
        bottom = (keypoints[1, i], keypoints[0, i] + 3)
        left = (keypoints[1, i] - 3, keypoints[0, i])
        right = (keypoints[1, i] + 3, keypoints[0, i])
        draw.line([top, bottom], fill=200, width=2)
        draw.line([left, right], fill=200, width=2)

    frame = cv2.cvtColor(np.array(img_col), cv2.COLOR_RGB2BGR)
    cv2.imshow('image', frame)
    cv2.waitKey()

    # Part 3: Build (2 * radius + 1) ^ 2 patch descriptor
    # Descriptor simply analyses intensity values of pixels
    descriptors = describe_keypoints(img, keypoints, descriptor_radius)

    # Depict intensity levels of the 16 highest Harris scores descriptors
    fig, axes = plt.subplots(4, 4)
    axes = axes.ravel()
    for i in range(16):
        axes[i].imshow(np.absolute(descriptors[:, :, i]), cmap='rainbow')
    plt.show()

    # Part 4: Generate descriptors for 2nd image and match descriptors
    try:
        img_2 = Image.open('./data/000001.png')

    except IOError:
        pass

    harris_score_2 = harris(img_2, corner_patch_size, harris_kappa)
    keypoints_2 = select_key_points(harris_score_2, num_keypoints, nonmaximum_supression_radius)
    descriptors_2 = describe_keypoints(img_2, keypoints_2, descriptor_radius)

    matches = match_descriptors(descriptors, descriptors_2, match_lambda)

    # Ugly work around to convert gray scale image to RGB
    # 1. Open with CV2 with RGB codec
    # 2. Convert to PIL image
    img_cv_2 = cv2.imread('./data/000001.png', 1)
    img_col_2 = Image.fromarray(img_cv_2)

    draw = ImageDraw.Draw(img_col_2)
    for i in range(len(keypoints[0])):
        if matches[i] != -1:
            pix_img_1 = (keypoints[1, matches[i]], keypoints[0, matches[i]])
            pix_img_2 = (keypoints_2[1, i], keypoints_2[0, i])
            draw.line([pix_img_1, pix_img_2], fill=(124, 252, 0), width=2)

    frame = cv2.cvtColor(np.array(img_col_2), cv2.COLOR_RGB2BGR)
    cv2.imshow('image', frame)
    cv2.waitKey()

    # Part 5: Run corner detection algorithm for all images

    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
    ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
    args = vars(ap.parse_args())

    # Arguments
    dir_path = 'data/'
    ext = args['extension']
    output = args['output']

    images = []
    for f in os.listdir(dir_path):
        if f.endswith(ext):
            images.append(f)

    images.sort()

    # Determine the width and height from the first image
    image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(image_path)
    cv2.imshow('video', frame)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

    image_index = 1

    #for image in images:



if __name__ == "__main__":
    main()
