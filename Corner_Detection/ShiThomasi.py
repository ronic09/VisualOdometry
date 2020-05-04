import numpy as np
from scipy import signal

def shi_thomasi(img):
    # Sobel vectors
    sobel_Ix = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_Iy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    Ix = signal.convolve2d(img, sobel_Ix, mode='valid')
    Iy = signal.convolve2d(img, sobel_Iy, mode='valid')

    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix*Iy

    print(Ixy)
    return Ix, Iy