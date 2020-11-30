from PIL import Image, ImageDraw
import numpy as np

def compute_image_pyramid(image, number_octaves):

    normalized_img = (image - np.amin(image)) / (np.amax(image) - np.amin(image))
    image_pyramid = [normalized_img]
    for i in range (1, number_octaves):
        image_resized = image.resize((int(image.size[0] / 2), int(image.size[1] / 2)))
        normalized_img = (image_resized - np.amin(image_resized)) / (np.amax(image_resized) - np.amin(image_resized))
        image_pyramid.append(normalized_img)
        image = image_resized

        # For testing purposes
        # image.show()

    return image_pyramid