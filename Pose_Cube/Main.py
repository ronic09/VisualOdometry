import logging
import cv2
import argparse
import os
import numpy as np
from Pose_Cube.DrawCube import draw_cube

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    filename='PoseCube.log',
                    filemode='w',
                    level=logging.DEBUG)


def main():

    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-ext", "--extension", required=False, default='jpg', help="extension name. default is 'jpg'.")
    ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
    args = vars(ap.parse_args())

    # Arguments
    dir_path = 'images/'
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

    for image in images:

        image_path = os.path.join(dir_path, image)

        pil_image = draw_cube(image_path, image_index)
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        image_index = image_index + 1
        out.write(frame)  # Write out frame to video

        cv2.imshow('video', frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
            break

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()

    print("The output video is {}".format(output))


if __name__ == "__main__":
    main()
