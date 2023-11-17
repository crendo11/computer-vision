import cv2
import matplotlib.pyplot as plt
import numpy as np


# to do:
# 1. Merge nearby corners.

def get_harris_matrix(I, k, Gx, Gy, w):
    # normalize the image
    I = np.float32(I)
    I /= I.max()

    # compute the gradients
    Ix, Iy = calcGradient(I, Gx, Gy)

    # create matrices of A, B, C and det
    # create matrices of A, B, C and det
    A = cv2.filter2D(Ix*Ix, ddepth=-1, kernel=w)
    B = cv2.filter2D(Iy*Iy, ddepth=-1, kernel=w)
    C = cv2.filter2D(Ix*Iy, ddepth=-1, kernel=w)
    det = A*B - C*C

    # calculate the determinant and trace
    det = A*B - C*C
    H = det - k*(A + B)**2

    return H

def get_harris_corners(H, threshold):
    H_bool = H > threshold
    # corners willl be a list of tuples (x,y) with the coordinates of the corners
    corners = []
    for i in range(1, H.shape[0]):
        for j in range(1, H.shape[1]):
            # mark the coreners 
            if H_bool[i][j] == True:
                corners.append((j,i))

    return corners


def calcGradient(M, kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]), ky = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).T):
    # use 2d filter function to calculate the gradient
    Ix = cv2.filter2D(M, ddepth=-1, kernel=kx)
    Iy = cv2.filter2D(M, ddepth=-1, kernel=ky)
    return Ix, Iy

def generate_gaussian_kernel(size=3, sigma=1):

    w = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            w[i][j] = np.exp(-((i-(size-1)/2)**2 + (j-(size-1)/2)**2)/(2*sigma**2))

    return w

# HOG histogram of oriented gradients

# 1. Calculate the gradient of the image
# 2. Calculate the magnitude and orientation of the gradient
# 3. Calculate the histogram of the gradient
# 4. Normalize the histogram
# 5. Concatenate the histograms

def calculate_hog(image, corners, num_bins=4):
    # normalize the image
    image = np.float32(image)
    image /= image.max()

    feature_vector = []
    
    for corner in corners:
        x, y = corner
        
        # Extract a small patch around the corner
        patch = image[y-8:y+8, x-8:x+8]
        
        if (patch.shape != (16, 16)):
            continue
        # Calculate gradient magnitude and orientation
        gradient_x, gradient_y = calcGradient(patch)
        magnitude, angle = cv2.cartToPolar(gradient_x, gradient_y, angleInDegrees=True)
        
        # Create histogram of gradient orientations
        histogram, _ = np.histogram(angle, bins=num_bins, range=(0, 180), weights=magnitude)
        
        # Normalize the histogram
        histogram /= np.sum(histogram)
        
        # Concatenate the histogram to the feature vector
        feature_vector.extend([histogram])
    
    return feature_vector
