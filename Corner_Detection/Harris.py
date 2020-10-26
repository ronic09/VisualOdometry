import logging
import numpy as np
from scipy import signal

def harris(img, patch_size, kappa):
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

    # Calculate Harris score and apply padding to get
    # same array size as original
    rHarris = determinant - kappa * trace**2
    rHarris[rHarris < 0] = 0
    rHarris = np.pad(rHarris, pr + 1, 'constant')

    logging.debug('The Harris scores of the image are: \n %s \n' % rHarris)

    return rHarris