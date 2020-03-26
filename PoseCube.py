from PIL import Image


def main():

    # load camera poses
    
    filename = "images_undistorted/img_0001.jpg"
    try:
        img = Image.open(filename)
        img.show()

    except IOError:
        pass


if __name__ == "__main__":
    main()