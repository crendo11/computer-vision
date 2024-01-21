"""
This file contains the code and functions for a camera calibration
using a checkerboard pattern.
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
from os.path import realpath, join

def get_corners(img, pattern_size):
    """
    Find the corners of the checkerboard pattern in the image.

    Parameters
    ----------
    img : numpy array
        The image to find the corners in.
    pattern_size : tuple
        The size of the checkerboard pattern in the image.

    Returns
    -------
    corners : numpy array
        The corners of the checkerboard pattern in the image.
    """
    # Find the corners of the checkerboard pattern
    found, corners = cv.findChessboardCorners(img, pattern_size)
    if found:
        # Refine the corner positions
        corners = cv.cornerSubPix(img, corners, (11, 11), (-1, -1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    # return corners and index of the found
    return corners, found
