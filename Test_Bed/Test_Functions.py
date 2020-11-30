import numpy as np
import cv2 as cv

def main():

    img = cv.imread('images/img_1.jpg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create(nOctaveLayers=3, contrastThreshold=0.04, sigma=1)
    kp = sift.detect(gray, None)
    img = cv.drawKeypoints(gray, kp, img)
    cv.imshow("sift_keypoints.jpg", img)
    #cv.imwrite('sift_keypoints.jpg', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()