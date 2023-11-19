import cv2
import matplotlib.pyplot as plt
import numpy as np


# importing sys
import sys
 
# adding Folder_2/subfolder to the system path
sys.path.insert(0, './')

import feature_detection as fd


# define the size of the window
size = 3

# constants
k = 0.04
sigma = 0.5
threshold = 0.5

# create w matrix
w = fd.generate_gaussian_kernel(size)

# Gx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
Gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

Gy = Gx.T


# create array to store the list of features of each image
features_list = []

# iterate over the 10 images
for z in range(10):

    # load the image
    image = cv2.imread('./2023_11_10_Class_practical/images/{}.jpg'.format(z))
    # convert to gray scale image
    Iog = image.copy()
    I = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    H = fd.get_harris_matrix(I, k, Gx, Gy, w)
    # define the radius of the circle based on the size of the image
    radius = int(I.shape[0]*0.0025)

    corners = fd.get_harris_corners(H, threshold)

    features = fd.calculate_hog(I, corners)

    # append the features to the list of features
    features_list.append(features)

    for corner in corners:
        i, j = corner
        cv2.circle(Iog, (i,j), radius, (0,255,0), 1)

    # save the image
    cv2.imwrite('./2023_11_10_Class_practical/images/{}_corners_t_5.jpg'.format(z), Iog)

    # console status
    print('Image {} processed.'.format(z))


# write the features to a file so it can be used later
with open('./2023_11_10_Class_practical/features_t_5.txt', 'w') as f:
    for item in features_list:
        f.write("%s\n" % item)


# calculate the harris matrix


# # show the image with matplotlib
# plt.imshow(Iog)
# plt.show()