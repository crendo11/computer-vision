import cv2
import matplotlib.pyplot as plt
import numpy as np


# importing sys
import sys
 
# adding Folder_2/subfolder to the system path
sys.path.insert(0, './')

import feature_detection as fd

# load the image
image = cv2.imread('./2023_11_10_Class_exercises/images/1.jpg')
# convert to gray scale image
Iog = image.copy()
I = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# define the size of the window
size = 3

# constants
k = 0.05
sigma = 0.5
threshold = 0.005

# create w matrix
w = fd.generate_gaussian_kernel(size)

# Gx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
Gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

Gy = Gx.T

H = fd.get_harris_matrix(I, k, Gx, Gy, w)
# define the radius of the circle based on the size of the image
radius = int(I.shape[0]*0.005)

corners = fd.get_harris_corners(H, threshold)

# calculate the harris matrix
for corner in corners:
    i, j = corner
    cv2.circle(Iog, (j,i), radius, (0,255,0), 1)

# show the image with matplotlib
plt.imshow(Iog)
plt.show()