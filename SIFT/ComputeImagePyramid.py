def compute_image_pyramid(image, number_octaves):

    image_pyramid = [image]
    for i in range (1, number_octaves):
        image_resized = image.resize((int(image.size[0] / 2), int(image.size[1] / 2)))
        image_pyramid.append(image_resized)
        image = image_resized

    return image_pyramid