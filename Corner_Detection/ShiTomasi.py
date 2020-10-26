import logging
import numpy as np
from scipy import signal

def shi_tomasi(img, patch_size):
    # Sobel vectors
    sobel_Ix = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_Iy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Patch for which matrix parameters are calculated
    patch = np.ones((patch_size, patch_size))
    pr = patch_size//2

    # Get x and y dimension derivatives by convolution
    Ix = signal.convolve2d(img, sobel_Ix, mode='valid')
    Iy = signal.convolve2d(img, sobel_Iy, mode='valid')

    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix*Iy

    # Generate M matrix using a patch of all ones ->
    # equals summation over all values covered by the patch
    sumIxx = signal.convolve2d(Ixx, patch, mode='valid')
    sumIyy = signal.convolve2d(Iyy, patch, mode='valid')
    sumIxy = signal.convolve2d(Ixy, patch, mode='valid')

    # Calculate trace and determinant of M matrix
    trace = sumIxx + sumIyy
    determinant = sumIxx*sumIyy - sumIxy**2

    # Calculate Shi-Tomasi score and apply padding to get
    # same array size as original
    rShi = trace/2 - ((trace/2)**2 - determinant)**0.5
    rShi[rShi < 0] = 0

    rShi = np.pad(rShi, pr + 1, 'constant')

    logging.debug('The Shi-Tomasi scores of the image are: \n %s \n' % rShi)

    return rShi